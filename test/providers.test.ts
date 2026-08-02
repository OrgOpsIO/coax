import { describe, expect, it } from "vitest";
import { createRegistry } from "../src/registry";
import { anthropic } from "../src/providers/anthropic";
import { openai } from "../src/providers/openai";
import type { Message, ToolDefinition } from "../src/types";

const TOOLS: ToolDefinition[] = [{ name: "lookup", description: "Look something up.", jsonSchema: { type: "object", properties: {} } }];

type SentOptions = { headers?: Record<string, string>; signal?: AbortSignal };

/** Captures the body + request options an SDK client would have been called with. */
function captureAnthropic(reply: unknown[]) {
  const sent: { body: Record<string, unknown>; options?: SentOptions }[] = [];
  const client = {
    messages: {
      create: async () => ({ content: reply }),
      stream: (body: Record<string, unknown>, options?: SentOptions) => {
        sent.push({ body, options });
        return { finalMessage: async () => ({ content: reply, usage: { input_tokens: 5, output_tokens: 2 } }) };
      },
    },
  };
  return { client, sent };
}

function captureOpenai(message: unknown) {
  const sent: { body: Record<string, unknown>; options?: SentOptions }[] = [];
  const client = {
    chat: {
      completions: {
        create: async (body: Record<string, unknown>, options?: SentOptions) => {
          sent.push({ body, options });
          return { choices: [{ message }], usage: { prompt_tokens: 5, completion_tokens: 2 } };
        },
      },
    },
    audio: { transcriptions: { create: async () => ({ text: "" }) }, speech: { create: async () => new Response() } },
  };
  return { client, sent };
}

describe("provider tool mapping", () => {
  it("anthropic: reads tool_use blocks and replays calls + results as content blocks", async () => {
    const { client, sent } = captureAnthropic([
      { type: "text", text: "Ich schaue nach." },
      { type: "tool_use", id: "tu_1", name: "lookup", input: { q: "R-1" } },
    ]);
    const provider = anthropic({ model: "claude-x", client: client as never });

    const first = await provider.tools!({ messages: [{ role: "user", content: "?" }], tools: TOOLS });
    expect(first.text).toBe("Ich schaue nach.");
    expect(first.calls).toEqual([{ id: "tu_1", name: "lookup", input: { q: "R-1" } }]);

    // Replaying the transcript must produce Anthropic's block shapes.
    const transcript: Message[] = [
      { role: "user", content: "?" },
      { role: "assistant", content: "Ich schaue nach.", toolCalls: first.calls },
      { role: "user", content: "", toolResults: [{ id: "tu_1", name: "lookup", output: { total: 42 } }] },
    ];
    await provider.tools!({ messages: transcript, tools: TOOLS });
    const messages = sent[1]!.body.messages as { role: string; content: unknown }[];
    expect(messages[1]!.content).toEqual([
      { type: "text", text: "Ich schaue nach." },
      { type: "tool_use", id: "tu_1", name: "lookup", input: { q: "R-1" } },
    ]);
    // tool_result must come FIRST in the answering user turn.
    expect(messages[2]!.content).toEqual([{ type: "tool_result", tool_use_id: "tu_1", content: '{"total":42}' }]);
  });

  it("anthropic: marks failed tool results with is_error", async () => {
    const { client, sent } = captureAnthropic([{ type: "text", text: "ok" }]);
    const provider = anthropic({ model: "claude-x", client: client as never });
    await provider.tools!({
      messages: [{ role: "user", content: "", toolResults: [{ id: "tu_1", name: "lookup", output: "boom", isError: true }] }],
      tools: TOOLS,
    });
    const content = (sent[0]!.body.messages as { content: { is_error?: boolean }[] }[])[0]!.content;
    expect(content[0]!.is_error).toBe(true);
  });

  it("openai: reads tool_calls and expands results into one `tool` message each", async () => {
    const { client, sent } = captureOpenai({
      content: null,
      tool_calls: [
        { id: "call_1", function: { name: "lookup", arguments: '{"q":"R-1"}' } },
        { id: "call_2", function: { name: "lookup", arguments: "not json" } },
      ],
    });
    const provider = openai({ model: "gpt-x", client: client as never });

    const first = await provider.tools!({ messages: [{ role: "user", content: "?" }], tools: TOOLS });
    expect(first.calls[0]).toEqual({ id: "call_1", name: "lookup", input: { q: "R-1" } });
    // Malformed arguments become {} so schema validation reports it back to the model.
    expect(first.calls[1]!.input).toEqual({});

    await provider.tools!({
      messages: [
        { role: "assistant", content: "", toolCalls: first.calls },
        {
          role: "user",
          content: "",
          toolResults: [
            { id: "call_1", name: "lookup", output: { total: 42 } },
            { id: "call_2", name: "lookup", output: "nope", isError: true },
          ],
        },
      ],
      tools: TOOLS,
    });
    const messages = sent[1]!.body.messages as { role: string; tool_call_id?: string; content?: unknown }[];
    expect(messages[0]!.role).toBe("assistant");
    expect(messages.filter((m) => m.role === "tool")).toHaveLength(2);
    expect(messages[1]).toEqual({ role: "tool", tool_call_id: "call_1", content: '{"total":42}' });
    // A string output is passed through verbatim, not JSON-quoted.
    expect(messages[2]!.content).toBe("nope");
  });

  it("sends per-call headers merged over the provider's configured ones", async () => {
    const { client, sent } = captureOpenai({ content: "hi" });
    const provider = openai({ model: "gpt-x", client: client as never, headers: { "X-Tenant": "juhi", "X-Trace": "base" } });
    await provider.text({ messages: [{ role: "user", content: "?" }], headers: { Authorization: "Bearer user-token", "X-Trace": "call" } });
    expect(sent[0]!.options?.headers).toEqual({ "X-Tenant": "juhi", "X-Trace": "call", Authorization: "Bearer user-token" });
  });

  it("openai: hands the AbortSignal to the SDK so the HTTP request itself can die", async () => {
    const { client, sent } = captureOpenai({ content: "hi" });
    const provider = openai({ model: "gpt-x", client: client as never });
    const ac = new AbortController();
    await provider.text({ messages: [{ role: "user", content: "?" }], signal: ac.signal });
    expect(sent[0]!.options?.signal).toBe(ac.signal);
  });

  it("anthropic: hands the AbortSignal to the streaming call", async () => {
    const { client, sent } = captureAnthropic([{ type: "text", text: "ok" }]);
    const provider = anthropic({ model: "claude-x", client: client as never });
    const ac = new AbortController();
    await provider.text({ messages: [{ role: "user", content: "?" }], signal: ac.signal });
    expect(sent[0]!.options?.signal).toBe(ac.signal);
  });
});

describe("registry: compatible endpoints", () => {
  it("builds a freely named provider from an endpoint with an explicit api", () => {
    const registry = createRegistry({
      providers: { orgops: { apiKey: "sk-test", baseURL: "https://llm.example.io/v1", api: "openai" } },
      models: { local: "orgops:chat/Qwen/Qwen3-VL-32B-Instruct-AWQ" },
    });
    const { primary, ref } = registry.resolve("local");
    // The model id keeps its slashes and is not mangled by the provider split.
    expect(ref).toBe("orgops:chat/Qwen/Qwen3-VL-32B-Instruct-AWQ");
    expect(primary.model).toBe("chat/Qwen/Qwen3-VL-32B-Instruct-AWQ");
    expect(primary.name).toBe("openai");
    expect(primary.transcribe).toBeTypeOf("function"); // audio rides along with the openai wire adapter
  });

  it("routes an anthropic-compatible endpoint through the anthropic adapter", () => {
    const registry = createRegistry({
      providers: { orgops: { apiKey: "sk-test", baseURL: "https://llm.example.io/anthropic", api: "anthropic" } },
    });
    const { primary } = registry.resolve("orgops:chat/some-model");
    expect(primary.name).toBe("anthropic");
    expect(primary.tools).toBeTypeOf("function");
    expect(primary.speak).toBeUndefined(); // Anthropic has no TTS — the capability is honestly absent
  });

  it("still accepts a bare API key for the two built-in names", () => {
    const registry = createRegistry({ providers: { anthropic: "sk-ant", openai: "sk-oai" } });
    expect(registry.resolve("anthropic:claude-x").primary.name).toBe("anthropic");
    expect(registry.resolve("openai:gpt-x").primary.name).toBe("openai");
  });

  it("explains what a custom name is missing", () => {
    const registry = createRegistry({ providers: { orgops: { apiKey: "sk-test", baseURL: "https://x/v1" } } });
    expect(() => registry.resolve("orgops:m")).toThrow(/needs `api: "openai" \| "anthropic"`/);
  });

  it("keeps optional capabilities detectable through the retry wrapper", async () => {
    const registry = createRegistry({ providers: { openai: "sk", anthropic: "sk" } });
    const { retrying } = await import("../src/registry");
    expect(retrying(registry.resolve("openai:gpt-x").primary).speak).toBeTypeOf("function");
    expect(retrying(registry.resolve("anthropic:claude-x").primary).speak).toBeUndefined();
  });
});
