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

describe("conversation cache hints (anthropic)", () => {
  const EPHEMERAL = { type: "ephemeral" };

  it("marks the last block of the last message, wrapping string content into a text block", async () => {
    const { client, sent } = captureAnthropic([{ type: "text", text: "ok" }]);
    const provider = anthropic({ model: "claude-x", client: client as never });
    await provider.text({
      messages: [
        { role: "user", content: "turn 1" },
        { role: "assistant", content: "reply" },
        { role: "user", content: "turn 2" },
      ],
      cacheConversation: true,
    });
    const messages = sent[0]!.body.messages as { content: unknown }[];
    expect(messages[0]!.content).toBe("turn 1");
    expect(messages[1]!.content).toBe("reply");
    expect(messages[2]!.content).toEqual([{ type: "text", text: "turn 2", cache_control: EPHEMERAL }]);
  });

  it("marks the last block when the last message already carries blocks (tool results)", async () => {
    const { client, sent } = captureAnthropic([{ type: "text", text: "ok" }]);
    const provider = anthropic({ model: "claude-x", client: client as never });
    await provider.tools!({
      messages: [
        { role: "user", content: "?" },
        { role: "assistant", content: "", toolCalls: [{ id: "tu_1", name: "lookup", input: {} }] },
        { role: "user", content: "", toolResults: [{ id: "tu_1", name: "lookup", output: "42" }] },
      ],
      tools: TOOLS,
      cacheConversation: true,
    });
    const messages = sent[0]!.body.messages as { content: Record<string, unknown>[] }[];
    const lastBlocks = messages[2]!.content;
    expect(lastBlocks[lastBlocks.length - 1]!.cache_control).toEqual(EPHEMERAL);
    // Exactly one breakpoint in the whole request — only the newest message is marked.
    const breakpoints = messages.flatMap((m) => (Array.isArray(m.content) ? m.content : [])).filter((b) => (b as Record<string, unknown>).cache_control);
    expect(breakpoints).toHaveLength(1);
  });

  it("applies to structured calls too, and combines with a cached system prompt", async () => {
    const { client, sent } = captureAnthropic([{ type: "tool_use", input: { a: 1 } }]);
    const provider = anthropic({ model: "claude-x", client: client as never });
    await provider.structured({
      system: "stable rules",
      messages: [{ role: "user", content: "extract" }],
      jsonSchema: { type: "object" },
      schemaName: "output",
      cacheSystem: true,
      cacheConversation: true,
    });
    expect(sent[0]!.body.system).toEqual([{ type: "text", text: "stable rules", cache_control: EPHEMERAL }]);
    const messages = sent[0]!.body.messages as { content: unknown }[];
    expect(messages[0]!.content).toEqual([{ type: "text", text: "extract", cache_control: EPHEMERAL }]);
  });

  it("leaves messages untouched without the hint, and skips an empty-string last message", async () => {
    const { client, sent } = captureAnthropic([{ type: "text", text: "ok" }]);
    const provider = anthropic({ model: "claude-x", client: client as never });
    await provider.text({ messages: [{ role: "user", content: "plain" }] });
    await provider.text({ messages: [{ role: "user", content: "" }], cacheConversation: true });
    expect((sent[0]!.body.messages as { content: unknown }[])[0]!.content).toBe("plain");
    expect((sent[1]!.body.messages as { content: unknown }[])[0]!.content).toBe("");
  });

  it("is a no-op on the OpenAI wire (no cache_control ever reaches the body)", async () => {
    const { client, sent } = captureOpenai({ content: "ok" });
    const provider = openai({ model: "gpt-x", client: client as never });
    await provider.text({ messages: [{ role: "user", content: "turn" }], cacheConversation: true });
    expect(JSON.stringify(sent[0]!.body)).not.toContain("cache_control");
  });
});

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

describe("reasoningEffort", () => {
  it("openai: sends reasoning_effort literally, only when set", async () => {
    const { client, sent } = captureOpenai({ content: "hi" });
    const provider = openai({ model: "gpt-x", client: client as never });
    await provider.text({ messages: [{ role: "user", content: "?" }], reasoningEffort: "none" });
    await provider.text({ messages: [{ role: "user", content: "?" }] });
    expect(sent[0]!.body.reasoning_effort).toBe("none");
    expect(sent[1]!.body).not.toHaveProperty("reasoning_effort"); // an endpoint that doesn't know the field must never see it
  });

  it("openai: reasoningEffort rides on structured() and tools() too", async () => {
    const { client, sent } = captureOpenai({ content: "", tool_calls: [] });
    const provider = openai({ model: "gpt-x", client: client as never });
    await provider.structured({ messages: [{ role: "user", content: "?" }], jsonSchema: { type: "object" }, schemaName: "out", reasoningEffort: "high" });
    await provider.tools!({ messages: [{ role: "user", content: "?" }], tools: TOOLS, reasoningEffort: "low" });
    expect(sent[0]!.body.reasoning_effort).toBe("high");
    expect(sent[1]!.body.reasoning_effort).toBe("low");
  });

  it("anthropic: \"none\" sends no thinking field", async () => {
    const { client, sent } = captureAnthropic([{ type: "text", text: "ok" }]);
    const provider = anthropic({ model: "claude-x", client: client as never });
    await provider.text({ messages: [{ role: "user", content: "?" }], reasoningEffort: "none" });
    expect(sent[0]!.body).not.toHaveProperty("thinking");
  });

  it("anthropic: low/medium/high enable thinking with the matching budget", async () => {
    const { client, sent } = captureAnthropic([{ type: "text", text: "ok" }]);
    const provider = anthropic({ model: "claude-x", client: client as never });
    // A generous max_tokens so none of the three budgets get capped — isolates the per-effort mapping
    // from the max_tokens - 1024 cap, which has its own test below.
    await provider.text({ messages: [{ role: "user", content: "?" }], reasoningEffort: "low", maxTokens: 20_000 });
    await provider.text({ messages: [{ role: "user", content: "?" }], reasoningEffort: "medium", maxTokens: 20_000 });
    await provider.text({ messages: [{ role: "user", content: "?" }], reasoningEffort: "high", maxTokens: 20_000 });
    expect(sent[0]!.body.thinking).toEqual({ type: "enabled", budget_tokens: 2048 });
    expect(sent[1]!.body.thinking).toEqual({ type: "enabled", budget_tokens: 8192 });
    expect(sent[2]!.body.thinking).toEqual({ type: "enabled", budget_tokens: 16384 });
  });

  it("anthropic: caps the thinking budget at max_tokens - 1024", async () => {
    const { client, sent } = captureAnthropic([{ type: "text", text: "ok" }]);
    const provider = anthropic({ model: "claude-x", client: client as never });
    await provider.text({ messages: [{ role: "user", content: "?" }], reasoningEffort: "high", maxTokens: 4096 });
    expect(sent[0]!.body.thinking).toEqual({ type: "enabled", budget_tokens: 3072 }); // capped, not the full 16384
  });

  it("anthropic: rejects thinking when max_tokens can't fit budget + answer, instead of sending an invalid budget", async () => {
    const { client } = captureAnthropic([{ type: "text", text: "ok" }]);
    const provider = anthropic({ model: "claude-x", client: client as never });
    // 1500 - 1024 = 476, below Anthropic's 1024 minimum — the endpoint would 400. coax says why first.
    await expect(
      provider.text({ messages: [{ role: "user", content: "?" }], reasoningEffort: "low", maxTokens: 1500 }),
    ).rejects.toThrow(/maxTokens >= 2048/);
  });

  it("anthropic: reasoningEffort on the tools() path throws a clear error (v1 gap)", async () => {
    const { client } = captureAnthropic([{ type: "text", text: "ok" }]);
    const provider = anthropic({ model: "claude-x", client: client as never });
    await expect(
      provider.tools!({ messages: [{ role: "user", content: "?" }], tools: TOOLS, reasoningEffort: "low" }),
    ).rejects.toThrow(/tools\(\) path/);
  });

  it("anthropic: reasoningEffort \"none\" is fine on the tools() path — nothing to round-trip", async () => {
    const { client, sent } = captureAnthropic([{ type: "text", text: "ok" }]);
    const provider = anthropic({ model: "claude-x", client: client as never });
    await provider.tools!({ messages: [{ role: "user", content: "?" }], tools: TOOLS, reasoningEffort: "none" });
    expect(sent[0]!.body).not.toHaveProperty("thinking");
  });
});

describe("extraBody", () => {
  it("openai: merges endpoint under call, call wins, and either MAY override coax's own fields", async () => {
    const { client, sent } = captureOpenai({ content: "hi" });
    const provider = openai({ model: "gpt-x", client: client as never, extraBody: { temperature: 0.6, top_p: 0.95 } });
    await provider.text({ messages: [{ role: "user", content: "?" }], extraBody: { temperature: 0.2, max_tokens: 999 } });
    expect(sent[0]!.body.temperature).toBe(0.2); // call beats endpoint
    expect(sent[0]!.body.top_p).toBe(0.95); // endpoint alone still applies
    expect(sent[0]!.body.max_tokens).toBe(999); // extraBody may override coax's own field, on purpose
  });

  it("anthropic: merges endpoint under call across structured/text/tools", async () => {
    const { client, sent } = captureAnthropic([{ type: "text", text: "ok" }]);
    const provider = anthropic({ model: "claude-x", client: client as never, extraBody: { top_p: 0.9 } });
    await provider.text({ messages: [{ role: "user", content: "?" }], extraBody: { top_p: 0.5 } });
    await provider.tools!({ messages: [{ role: "user", content: "?" }], tools: TOOLS, extraBody: { top_p: 0.4 } });
    expect(sent[0]!.body.top_p).toBe(0.5);
    expect(sent[1]!.body.top_p).toBe(0.4);
  });
});

describe("toolChoice", () => {
  it("openai: sends the literal value, and defaults to \"auto\" when unset (unchanged behavior)", async () => {
    const { client, sent } = captureOpenai({ content: "", tool_calls: [] });
    const provider = openai({ model: "gpt-x", client: client as never });
    await provider.tools!({ messages: [{ role: "user", content: "?" }], tools: TOOLS, toolChoice: "required" });
    await provider.tools!({ messages: [{ role: "user", content: "?" }], tools: TOOLS });
    expect(sent[0]!.body.tool_choice).toBe("required");
    expect(sent[1]!.body.tool_choice).toBe("auto");
  });

  it("anthropic: maps required/auto/none to Anthropic's own vocabulary, and omits the field when unset", async () => {
    const { client, sent } = captureAnthropic([{ type: "text", text: "ok" }]);
    const provider = anthropic({ model: "claude-x", client: client as never });
    await provider.tools!({ messages: [{ role: "user", content: "?" }], tools: TOOLS, toolChoice: "required" });
    await provider.tools!({ messages: [{ role: "user", content: "?" }], tools: TOOLS, toolChoice: "auto" });
    await provider.tools!({ messages: [{ role: "user", content: "?" }], tools: TOOLS, toolChoice: "none" });
    await provider.tools!({ messages: [{ role: "user", content: "?" }], tools: TOOLS });
    expect(sent[0]!.body.tool_choice).toEqual({ type: "any" });
    expect(sent[1]!.body.tool_choice).toEqual({ type: "auto" });
    expect(sent[2]!.body.tool_choice).toEqual({ type: "none" });
    expect(sent[3]!.body).not.toHaveProperty("tool_choice"); // today's behavior: no field at all
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
