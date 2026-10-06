import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { z } from "zod";
import type { AIConfig } from "../src/config";
import { createAI } from "../src/ai";
import { createBudget } from "../src/budget";
import { CoaxAbortError, CoaxUnsupportedError, createClient } from "../src/client";
import { deepMerge, google, RESPONSE_KEYWORDS, TOOL_KEYWORDS, toGoogleSchema } from "../src/providers/google";
import { createRegistry, retrying } from "../src/registry";
import { isTransient } from "../src/retry";
import { CoaxToolError, runTools, tool } from "../src/tools";
import { CoaxRefusalError, type Message, type ToolDefinition } from "../src/types";

// Fixtures are copied unchanged from the board (.ziv/reference/fixtures/stage-01); each file's `source`
// names where its shape comes from (Agent Platform docs or the @google/genai 2.27.0 types).
const load = (name: string) => JSON.parse(readFileSync(new URL(`./fixtures/google/${name}`, import.meta.url), "utf8"));
const fixture = (name: string) => load(name).fixture;

type Params = { model: string; contents: any[]; config: Record<string, any> };
type EmbedParams = { model: string; contents: string[]; config?: Record<string, any> };

const copy = <T>(v: T): T => JSON.parse(JSON.stringify(v));

/**
 * A fake SDK client: records every params object, answers generateContent from `replies` and
 * generateContentStream from `streams` in order (the last one repeats). An Error in a reply is thrown;
 * an Error inside a stream is thrown mid-iteration.
 */
function fake(replies: unknown[] = [], streams: unknown[][] = []) {
  const sent: Params[] = [];
  const embedded: EmbedParams[] = [];
  let r = 0;
  let s = 0;
  const client = {
    models: {
      generateContent: async (params: Params) => {
        sent.push(params);
        const reply = replies[Math.min(r++, replies.length - 1)];
        if (reply instanceof Error) throw reply;
        return copy(reply);
      },
      generateContentStream: async (params: Params) => {
        sent.push(params);
        const chunks = streams[Math.min(s++, streams.length - 1)] ?? [];
        return (async function* () {
          for (const chunk of chunks) {
            if (chunk instanceof Error) throw chunk;
            yield copy(chunk);
          }
        })();
      },
      embedContent: async (params: EmbedParams) => {
        embedded.push(params);
        return { embeddings: [{ values: [embedded.length, 0.5], statistics: { tokenCount: 3 } }] };
      },
    },
  };
  return { client: client as never, sent, embedded };
}

const MODEL = "gemini-3.5-flash";
/** An ADC-configured provider (project set) over an injected fake client. */
const provider = (client: never, extra: Partial<Parameters<typeof google>[0]> = {}) => google({ model: MODEL, project: "p", client, ...extra });

const answer = (text: string, usageMetadata: Record<string, unknown> = { promptTokenCount: 1, candidatesTokenCount: 1, totalTokenCount: 2 }) => ({
  candidates: [{ content: { role: "model", parts: [{ text }] }, finishReason: "STOP", index: 0 }],
  usageMetadata,
});

const user = (content: string): Message => ({ role: "user", content });

async function drain<T, R>(gen: AsyncGenerator<T, R, void>): Promise<{ deltas: T[]; result: R }> {
  const deltas: T[] = [];
  let cur = await gen.next();
  while (!cur.done) {
    deltas.push(cur.value);
    cur = await gen.next();
  }
  return { deltas, result: cur.value };
}

/** Drains a stream that is expected to die: the deltas that made it out, and the error. */
async function drainToError<T, R>(gen: AsyncGenerator<T, R, void>): Promise<{ deltas: T[]; error: unknown }> {
  const deltas: T[] = [];
  try {
    let cur = await gen.next();
    while (!cur.done) {
      deltas.push(cur.value);
      cur = await gen.next();
    }
  } catch (error) {
    return { deltas, error };
  }
  throw new Error("expected the stream to fail");
}

const TEMP_TOOL: ToolDefinition = {
  name: "get_current_temperature",
  description: "Current temperature for a city.",
  jsonSchema: { type: "object", properties: { location: { type: "string" } }, required: ["location"], additionalProperties: false },
};
const LOOKUP: ToolDefinition = { name: "lookup", description: "Look something up.", jsonSchema: { type: "object", properties: {} } };

describe("google text (T1)", () => {
  it("returns every non-thought text part as the answer and sends only what was asked for", async () => {
    const { client, sent } = fake([fixture("text.response.json")]);
    const res = await provider(client).text({ system: "Write a haiku.", messages: [user("About types.")] });
    expect(res.text).toBe("Types guard the gate,\nerrors caught before they run,\nquiet builds at dawn.");
    expect(res.raw).toBe(res.text);
    expect(res.model).toBe(MODEL);
    expect(sent[0]).toEqual({
      model: MODEL,
      contents: [{ role: "user", parts: [{ text: "About types." }] }],
      config: { systemInstruction: "Write a haiku.", maxOutputTokens: 8192 },
    });
  });

  it("maxTokens: the call's, else the endpoint's, else 8192; no systemInstruction without a system prompt", async () => {
    const { client, sent } = fake([answer("ok")]);
    await provider(client, { maxTokens: 512 }).text({ messages: [user("?")] });
    await provider(client, { maxTokens: 512 }).text({ messages: [user("?")], maxTokens: 64 });
    expect(sent[0]!.config).toEqual({ maxOutputTokens: 512 });
    expect(sent[1]!.config).toEqual({ maxOutputTokens: 64 });
  });
});

describe("google registry and config (T3)", () => {
  it("a bare key under the name google is an Agent Platform API key — a provider without embed", () => {
    const registry = createRegistry({ providers: { google: "k" } });
    const { primary } = registry.resolve("google:gemini-3.5-flash");
    expect(primary.name).toBe("google");
    expect(primary.model).toBe("gemini-3.5-flash");
    expect(primary.tools).toBeTypeOf("function");
    expect(primary.toolsStream).toBeTypeOf("function");
    expect(primary.embed).toBeUndefined();
  });

  it('api: "google" works under any name; the ADC form has embed', () => {
    const registry = createRegistry({ providers: { "google-eu": { api: "google", project: "p", location: "eu" } } });
    const { primary } = registry.resolve("google-eu:x");
    expect(primary.name).toBe("google");
    expect(primary.model).toBe("x");
    expect(primary.embed).toBeTypeOf("function");
  });

  it("the unknown-name error also names the google wire", () => {
    const registry = createRegistry({ providers: { orgops: { apiKey: "sk-test", baseURL: "https://x/v1" } } });
    expect(() => registry.resolve("orgops:m")).toThrow(/api: "google"/);
    expect(() => registry.resolve("orgops:m")).toThrow(/"anthropic", "openai" and "google" are inferred/);
  });

  it("apiKey and project together are a config mistake, named on the first call", async () => {
    const ai = createAI({ providers: { google: { apiKey: "k", project: "p" } } });
    await expect(ai.text({ model: "google:m", prompt: "?" })).rejects.toThrow(/either apiKey .* or project/);
  });

  it("the old and the new config forms all type-check as AIConfig", () => {
    const configs = [
      { providers: { google: "k" } },
      { providers: { google: { project: "p" } } },
      { providers: { google: { project: "p", location: "eu", googleAuthOptions: { credentials: { client_email: "e" } } } } },
      { providers: { "google-eu": { api: "google" as const, project: "p" } } },
      { providers: { openai: { apiKey: "sk", baseURL: "https://x/v1" } } },
    ] satisfies AIConfig[];
    expect(configs).toHaveLength(5);
  });
});

describe("google contents and media (T4)", () => {
  it("an image and a PDF ride as inlineData after the text", async () => {
    const want = fixture("vision.request.sdk.json");
    const { client, sent } = fake([answer("ok")]);
    await provider(client).text({
      messages: [
        {
          role: "user",
          content: "Extract the fields.",
          media: [
            { kind: "image", mediaType: "image/png", dataBase64: want.contents[0].parts[1].inlineData.data },
            { kind: "pdf", mediaType: "application/pdf", dataBase64: "JVBERi0xLjQK" },
          ],
        },
      ],
    });
    expect(sent[0]!.contents).toEqual(want.contents);
  });

  it("a multi-turn transcript maps to user/model roles", async () => {
    const { client, sent } = fake([answer("ok")]);
    await provider(client).text({ messages: [user("a"), { role: "assistant", content: "b" }, user("c")] });
    expect(sent[0]!.contents).toEqual([
      { role: "user", parts: [{ text: "a" }] },
      { role: "model", parts: [{ text: "b" }] },
      { role: "user", parts: [{ text: "c" }] },
    ]);
  });
});

describe("google usage (T5)", () => {
  it("thinking and server-side tool tokens are counted; input + output equals Google's total", async () => {
    const meta = fixture("usage-metadata.json");
    const { client } = fake([answer("x", meta)]);
    const { usage } = await provider(client).text({ messages: [user("?")] });
    expect(usage).toEqual({ inputTokens: 10336, outputTokens: 76, cacheReadTokens: 0, cacheWriteTokens: 0 });
    expect(usage.inputTokens + usage.outputTokens).toBe(meta.totalTokenCount);
  });

  it("cache hits are the cached subset of the input", async () => {
    const { client } = fake([fixture("text.response.json")]);
    const { usage } = await provider(client).text({ messages: [user("?")] });
    expect(usage).toEqual({ inputTokens: 5000, outputTokens: 85, cacheReadTokens: 4096, cacheWriteTokens: 0 });
  });

  it("missing counters count as zero", async () => {
    const { client } = fake([{ candidates: [{ content: { parts: [{ text: "x" }] }, finishReason: "STOP" }] }]);
    const { usage } = await provider(client).text({ messages: [user("?")] });
    expect(usage).toEqual({ inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0 });
  });
});

describe("google structured output (T6)", () => {
  it("JSON text comes back raw and the client parses and validates it", async () => {
    const { client, sent } = fake([fixture("structured.response.json")]);
    const res = await createClient({ provider: provider(client) }).object({
      schema: z.object({ decision: z.object({ reason: z.string(), spam_type: z.string() }) }),
      prompt: "Moderate this message.",
    });
    expect(res.data.decision.spam_type).toBe("scam");
    expect(res.repairs).toBe(0);
    expect(res.usage).toEqual({ inputTokens: 120, outputTokens: 38, cacheReadTokens: 0, cacheWriteTokens: 0 });
    expect(sent[0]!.config.responseMimeType).toBe("application/json");
    expect(sent[0]!.config.responseJsonSchema).toBeTypeOf("object");
  });

  it("structured(): raw is the JSON text", async () => {
    const { client } = fake([fixture("structured.response.json")]);
    const res = await provider(client).structured({ messages: [user("?")], jsonSchema: { type: "object" }, schemaName: "out" });
    expect(typeof res.raw).toBe("string");
    expect(res.text).toBe(res.raw);
  });

  it("the fixture's Zod schema: minLength dropped, const → enum, the rest kept", () => {
    const sent = fixture("structured.request.sdk.json").config.responseJsonSchema;
    expect(toGoogleSchema(sent, RESPONSE_KEYWORDS)).toEqual({
      type: "object",
      properties: {
        title: { type: "string" },
        tags: { type: "array", items: { type: "string" }, minItems: 1 },
        kind: { type: "string", enum: ["article"] },
      },
      required: ["title", "tags", "kind"],
      additionalProperties: false,
    });
  });

  it("end to end, a z.literal tag reaches Google as an enum and min(1) on a string is not sent", async () => {
    const { client, sent } = fake([answer('{"title":"T","tags":["a"],"kind":"article"}')]);
    await createClient({ provider: provider(client) }).object({
      schema: z.object({ title: z.string().min(1), tags: z.array(z.string()).min(1), kind: z.literal("article") }),
      prompt: "Label this article: …",
    });
    const schema = sent[0]!.config.responseJsonSchema;
    expect(schema.properties.kind).toEqual({ type: "string", enum: ["article"] });
    expect(schema.properties.title).not.toHaveProperty("minLength");
  });

  it("property and $defs NAMES that look like keywords survive; their bodies are filtered", () => {
    const out = toGoogleSchema(
      {
        type: "object",
        properties: { pattern: { type: "string", pattern: "^a" }, default: { type: "number", default: 1 } },
        $defs: { const: { type: "string", minLength: 2 } },
      },
      RESPONSE_KEYWORDS,
    );
    expect(out).toEqual({
      type: "object",
      properties: { pattern: { type: "string" }, default: { type: "number" } },
      $defs: { const: { type: "string" } },
    });
  });

  it("a $ref keeps only its $-siblings", () => {
    expect(toGoogleSchema({ $ref: "#/$defs/node", description: "x", type: "object" }, RESPONSE_KEYWORDS)).toEqual({ $ref: "#/$defs/node" });
  });

  it("formats: date-time/date/time kept, the rest dropped; boolean const dropped; nullable anyOf kept", () => {
    const out = toGoogleSchema(
      {
        type: "object",
        properties: {
          email: { type: "string", format: "email" },
          at: { type: "string", format: "date-time" },
          flag: { type: "boolean", const: true },
          maybe: { anyOf: [{ type: "string", maxLength: 3 }, { type: "null" }] },
        },
      },
      RESPONSE_KEYWORDS,
    );
    expect(out.properties).toEqual({
      email: { type: "string" },
      at: { type: "string", format: "date-time" },
      flag: { type: "boolean" },
      maybe: { anyOf: [{ type: "string" }, { type: "null" }] },
    });
  });

  it("never mutates its input", () => {
    const input = { type: "object", properties: { kind: { type: "string", const: "a", minLength: 1 } }, additionalProperties: false };
    const before = JSON.stringify(input);
    toGoogleSchema(input, TOOL_KEYWORDS);
    toGoogleSchema(input, RESPONSE_KEYWORDS);
    expect(JSON.stringify(input)).toBe(before);
  });

  it("tool parameters: the smaller keyword list, oneOf spelled anyOf", () => {
    const out = toGoogleSchema(
      {
        type: "object",
        properties: { n: { type: "integer", minimum: 1 }, xs: { type: "array", items: { type: "string" }, minItems: 1 }, u: { oneOf: [{ type: "string" }, { type: "number" }] } },
        required: ["n"],
        additionalProperties: false,
      },
      TOOL_KEYWORDS,
    );
    expect(out).toEqual({
      type: "object",
      properties: { n: { type: "integer" }, xs: { type: "array", items: { type: "string" } }, u: { anyOf: [{ type: "string" }, { type: "number" }] } },
      required: ["n"],
    });
  });
});

describe("google reasoningEffort (T7)", () => {
  const LEVELS = { none: "MINIMAL", low: "LOW", medium: "MEDIUM", high: "HIGH" } as const;

  for (const [effort, level] of Object.entries(LEVELS) as [keyof typeof LEVELS, string][]) {
    const name = effort === "none" ? 'reasoningEffort "none" → MINIMAL (assumption, open question)' : `reasoningEffort "${effort}" → ${level}`;
    it(`${name} on text, structured and tools`, async () => {
      const { client, sent } = fake([answer("{}")]);
      const p = provider(client);
      await p.text({ messages: [user("?")], reasoningEffort: effort });
      await p.structured({ messages: [user("?")], jsonSchema: { type: "object" }, schemaName: "out", reasoningEffort: effort });
      await p.tools!({ messages: [user("?")], tools: [LOOKUP], reasoningEffort: effort });
      for (const params of sent) expect(params.config.thinkingConfig).toEqual({ thinkingLevel: level });
    });
  }

  it("unset sends no thinkingConfig at all", async () => {
    const { client, sent } = fake([answer("ok")]);
    await provider(client).text({ messages: [user("?")] });
    expect(sent[0]!.config).not.toHaveProperty("thinkingConfig");
  });
});

describe("google headers, extraBody, signal, cache hints (T8)", () => {
  it("headers: the call's over the endpoint's", async () => {
    const { client, sent } = fake([answer("ok")]);
    await provider(client, { headers: { a: "1", b: "1" } }).text({ messages: [user("?")], headers: { b: "2" } });
    expect(sent[0]!.config.httpOptions).toEqual({ headers: { a: "1", b: "2" } });
  });

  it("extraBody: endpoint and call merge deep; arrays replace", async () => {
    const { client, sent } = fake([answer("ok")]);
    const p = provider(client, { extraBody: { generationConfig: { temperature: 0.6, topP: 0.9 }, labels: { team: "x" }, stop: ["a", "b"] } });
    await p.text({
      messages: [user("?")],
      extraBody: { generationConfig: { temperature: 0.2 }, safetySettings: [{ category: "HARM_CATEGORY_HATE_SPEECH", threshold: "BLOCK_ONLY_HIGH" }], stop: ["c"] },
    });
    expect(sent[0]!.config.httpOptions).toEqual({
      extraBody: {
        generationConfig: { temperature: 0.2, topP: 0.9 },
        labels: { team: "x" },
        stop: ["c"],
        safetySettings: [{ category: "HARM_CATEGORY_HATE_SPEECH", threshold: "BLOCK_ONLY_HIGH" }],
      },
    });
  });

  it("deepMerge leaves both inputs untouched", () => {
    const base = { a: { b: 1 } };
    const over = { a: { c: 2 } };
    expect(deepMerge(base, over)).toEqual({ a: { b: 1, c: 2 } });
    expect(base).toEqual({ a: { b: 1 } });
    expect(deepMerge(undefined, undefined)).toEqual({});
  });

  it("signal: handed to the SDK as config.abortSignal", async () => {
    const { client, sent } = fake([answer("ok")]);
    const ctrl = new AbortController();
    await provider(client).text({ messages: [user("?")], signal: ctrl.signal });
    expect(sent[0]!.config.abortSignal).toBe(ctrl.signal);
  });

  it("an SDK abort surfaces as CoaxAbortError", async () => {
    const ctrl = new AbortController();
    const client = {
      models: {
        generateContent: async () => {
          ctrl.abort();
          throw new DOMException("This operation was aborted", "AbortError");
        },
      },
    };
    await expect(createClient({ provider: provider(client as never) }).text({ prompt: "?", signal: ctrl.signal })).rejects.toBeInstanceOf(CoaxAbortError);
  });

  it("cache hints change nothing on the wire — caching is implicit on Google", async () => {
    const { client, sent } = fake([answer("ok")]);
    const p = provider(client);
    await p.text({ system: "s", messages: [user("?")] });
    await p.text({ system: "s", messages: [user("?")], cacheSystem: true, cacheConversation: true });
    expect(sent[1]).toEqual(sent[0]);
  });
});

describe("google streaming (T9)", () => {
  it("textStream: text deltas, the empty final part yields nothing, usage is the LAST chunk's", async () => {
    const { client, sent } = fake([], [fixture("stream-text.chunks.json")]);
    const { deltas, result } = await drain(provider(client).textStream!({ messages: [user("?")] }));
    expect(deltas).toEqual(["Types guard ", "the gate,"]);
    expect(result.text).toBe("Types guard the gate,");
    // A summing implementation would report 27 input tokens here.
    expect(result.usage).toEqual({ inputTokens: 9, outputTokens: 46, cacheReadTokens: 0, cacheWriteTokens: 0 });
    expect(sent[0]!.model).toBe(MODEL);
  });

  it("structuredStream: fragments concatenate to the JSON text; streamObject validates it", async () => {
    const want = load("stream-structured.chunks.json").expect.raw;
    const { client } = fake([], [fixture("stream-structured.chunks.json")]);
    const { deltas, result } = await drain(provider(client).structuredStream!({ messages: [user("?")], jsonSchema: { type: "object" }, schemaName: "out" }));
    expect(deltas.join("")).toBe(want);
    expect(result.raw).toBe(want);

    const again = fake([], [fixture("stream-structured.chunks.json")]);
    const stream = createClient({ provider: provider(again.client) }).streamObject({
      schema: z.object({ title: z.string(), tags: z.array(z.string()), kind: z.literal("article") }),
      prompt: "?",
    });
    const { result: obj } = await drain(stream);
    expect(obj.data).toEqual({ title: "Types", tags: ["ts"], kind: "article" });
  });
});

describe("google tools: calls, ids, carrier (T10)", () => {
  it("parallel calls come back in order with Google's ids; the parts ride verbatim as providerData", async () => {
    const res0 = load("tools-parallel.response.json");
    const { client } = fake([res0.fixture]);
    const res = await provider(client).tools!({ messages: [user("?")], tools: [TEMP_TOOL] });
    expect(res.calls).toEqual(res0.expect.calls);
    expect(res.text).toBe("");
    expect(res.providerData).toEqual({ provider: "google", parts: res0.fixture.candidates[0].content.parts });
    expect(res.usage).toEqual({ inputTokens: 80, outputTokens: 136, cacheReadTokens: 0, cacheWriteTokens: 0 });
  });

  it("declarations use parametersJsonSchema with the tool keyword list; toolConfig only when toolChoice is set", async () => {
    const { client, sent } = fake([answer("ok")]);
    const p = provider(client);
    await p.tools!({ messages: [user("?")], tools: [TEMP_TOOL] });
    await p.tools!({ messages: [user("?")], tools: [TEMP_TOOL], toolChoice: "auto" });
    await p.tools!({ messages: [user("?")], tools: [TEMP_TOOL], toolChoice: "required" });
    await p.tools!({ messages: [user("?")], tools: [TEMP_TOOL], toolChoice: "none" });
    expect(sent[0]!.config.tools).toEqual([
      {
        functionDeclarations: [
          {
            name: "get_current_temperature",
            description: "Current temperature for a city.",
            parametersJsonSchema: { type: "object", properties: { location: { type: "string" } }, required: ["location"] },
          },
        ],
      },
    ]);
    expect(sent[0]!.config).not.toHaveProperty("toolConfig");
    expect(sent.slice(1).map((s) => s.config.toolConfig.functionCallingConfig.mode)).toEqual(["AUTO", "ANY", "NONE"]);
  });

  it("calls without ids get coax-internal ids call_0, call_1", async () => {
    const reply = fixture("tools-parallel.response.json");
    for (const part of reply.candidates[0].content.parts) delete part.functionCall.id;
    const { client } = fake([reply]);
    const res = await provider(client).tools!({ messages: [user("?")], tools: [TEMP_TOOL] });
    expect(res.calls.map((c) => c.id)).toEqual(["call_0", "call_1"]);
  });

  it("text and a call in one turn both surface; a turn without calls carries no providerData", async () => {
    const { client } = fake([
      { candidates: [{ content: { parts: [{ text: "Let me check." }, { functionCall: { id: "c1", name: "lookup", args: { q: "x" } } }] }, finishReason: "STOP" }] },
      answer("done"),
    ]);
    const p = provider(client);
    const turn = await p.tools!({ messages: [user("?")], tools: [LOOKUP] });
    expect(turn.text).toBe("Let me check.");
    expect(turn.calls).toEqual([{ id: "c1", name: "lookup", input: { q: "x" } }]);
    const final = await p.tools!({ messages: [user("?")], tools: [LOOKUP] });
    expect(final.providerData).toBeUndefined();
  });
});

describe("google tools: the round-trip (T11)", () => {
  async function parallelTurn() {
    const { client } = fake([fixture("tools-parallel.response.json")]);
    return provider(client).tools!({ messages: [user("Is it warmer in Paris or London?")], tools: [TEMP_TOOL] });
  }

  it("replays the carried parts verbatim and echoes Google's ids, results in ONE user content", async () => {
    const turn = await parallelTurn();
    const want = fixture("tools-followup.request.sdk.json");
    const { client, sent } = fake([answer("Paris.")]);
    await provider(client).tools!({
      messages: [
        user("Is it warmer in Paris or London?"),
        { role: "assistant", content: turn.text, toolCalls: turn.calls, providerData: turn.providerData },
        {
          role: "user",
          content: "",
          toolResults: [
            { id: "call-paris-1", name: "get_current_temperature", output: { celsius: 21 } },
            { id: "call-london-2", name: "get_current_temperature", output: 'The tool "get_current_temperature" failed: upstream timeout', isError: true },
          ],
        },
      ],
      tools: [TEMP_TOOL],
      toolChoice: "auto",
    });
    expect(sent[0]!.contents).toEqual(want.contents);
    expect(sent[0]!.config.toolConfig).toEqual(want.config.toolConfig);
  });

  it("id-less calls: the functionResponses go back without an id (never one Google did not issue)", async () => {
    const reply = fixture("tools-parallel.response.json");
    for (const part of reply.candidates[0].content.parts) delete part.functionCall.id;
    const first = fake([reply]);
    const turn = await provider(first.client).tools!({ messages: [user("?")], tools: [TEMP_TOOL] });
    const { client, sent } = fake([answer("ok")]);
    await provider(client).tools!({
      messages: [
        user("?"),
        { role: "assistant", content: "", toolCalls: turn.calls, providerData: turn.providerData },
        { role: "user", content: "", toolResults: turn.calls.map((c) => ({ id: c.id, name: c.name, output: 1 })) },
      ],
      tools: [TEMP_TOOL],
    });
    expect(sent[0]!.contents[2].parts).toEqual([
      { functionResponse: { name: "get_current_temperature", response: { output: 1 } } },
      { functionResponse: { name: "get_current_temperature", response: { output: 1 } } },
    ]);
  });

  it("foreign history gets the documented placeholder signature and no ids (deviates from the fixture's ids, spec §6)", async () => {
    const { client, sent } = fake([answer("ok")]);
    await provider(client).tools!({
      messages: [
        user("?"),
        { role: "assistant", content: "", toolCalls: [{ id: "toolu_01", name: "lookup", input: {} }] },
        { role: "user", content: "", toolResults: [{ id: "toolu_01", name: "lookup", output: "42" }] },
      ],
      tools: [LOOKUP],
    });
    expect(sent[0]!.contents).toEqual([
      { role: "user", parts: [{ text: "?" }] },
      { role: "model", parts: [{ functionCall: { name: "lookup", args: {} }, thoughtSignature: "skip_thought_signature_validator" }] },
      { role: "user", parts: [{ functionResponse: { name: "lookup", response: { output: "42" } } }] },
    ]);
  });

  it("only the FIRST foreign call carries the placeholder; assistant text leads the turn", async () => {
    const { client, sent } = fake([answer("ok")]);
    await provider(client).tools!({
      messages: [
        user("?"),
        { role: "assistant", content: "Checking both.", toolCalls: [{ id: "a", name: "lookup", input: { q: 1 } }, { id: "b", name: "lookup", input: { q: 2 } }] },
      ],
      tools: [LOOKUP],
    });
    expect(sent[0]!.contents[1]).toEqual({
      role: "model",
      parts: [
        { text: "Checking both." },
        { functionCall: { name: "lookup", args: { q: 1 } }, thoughtSignature: "skip_thought_signature_validator" },
        { functionCall: { name: "lookup", args: { q: 2 } } },
      ],
    });
  });

  it("an Anthropic thinking carrier (an array) is treated as foreign history", async () => {
    const { client, sent } = fake([answer("ok")]);
    await provider(client).tools!({
      messages: [
        user("?"),
        {
          role: "assistant",
          content: "",
          toolCalls: [{ id: "tu_1", name: "lookup", input: {} }],
          providerData: [{ type: "thinking", thinking: "hmm…", signature: "sig-abc" }],
        },
      ],
      tools: [LOOKUP],
    });
    expect(sent[0]!.contents[1]).toEqual({
      role: "model",
      parts: [{ functionCall: { name: "lookup", args: {} }, thoughtSignature: "skip_thought_signature_validator" }],
    });
  });

  it("sequential steps each keep their own signature", async () => {
    const history = fixture("tools-sequential.history.rest.json");
    const { client, sent } = fake([answer("Booked.")]);
    const carried = (content: { parts: unknown[] }) => ({ provider: "google", parts: content.parts });
    await provider(client).tools!({
      messages: [
        user(history[0].parts[0].text),
        { role: "assistant", content: "", toolCalls: [{ id: "call_0", name: "check_flight", input: { flight: "AA100" } }], providerData: carried(history[1]) },
        { role: "user", content: "", toolResults: [{ id: "call_0", name: "check_flight", output: { status: "delayed", departure_time: "12 PM" } }] },
        { role: "assistant", content: "", toolCalls: [{ id: "call_0", name: "book_taxi", input: { time: "10 AM" } }], providerData: carried(history[3]) },
        { role: "user", content: "", toolResults: [{ id: "call_0", name: "book_taxi", output: { booking_status: "success" } }] },
      ],
      tools: [LOOKUP],
    });
    expect(sent[0]!.contents[1]).toEqual(history[1]);
    expect(sent[0]!.contents[3]).toEqual(history[3]);
    expect(sent[0]!.contents[2].parts[0].functionResponse).toEqual({ name: "check_flight", response: { output: { status: "delayed", departure_time: "12 PM" } } });
  });

  it("end to end: runTools sends the carrier back on the next turn and ends with the answer", async () => {
    const parallel = fixture("tools-parallel.response.json");
    const { client, sent } = fake([parallel, answer("Paris is warmer.")]);
    const p = provider(client);
    const temperature = tool({
      name: "get_current_temperature",
      description: "Current temperature for a city.",
      input: z.object({ location: z.string() }),
      run: ({ location }) => ({ celsius: location === "Paris" ? 21 : 15 }),
    });
    const result = await runTools((req) => p.tools!(req), { tools: [temperature], messages: [user("Is it warmer in Paris or London?")] });
    expect(result.text).toBe("Paris is warmer.");
    expect(sent[1]!.contents[1]).toEqual({ role: "model", parts: parallel.candidates[0].content.parts });
    expect(sent[1]!.contents[2].parts.map((x: { functionResponse: { id?: string } }) => x.functionResponse.id)).toEqual(["call-paris-1", "call-london-2"]);
  });
});

describe("google toolsStream (T12)", () => {
  it("streams the text, keeps every part (signatures included) in order, usage from the last chunk", async () => {
    const want = load("tools-stream.chunks.json");
    const { client } = fake([], [want.fixture]);
    const { deltas, result } = await drain(provider(client).toolsStream!({ messages: [user("?")], tools: [LOOKUP] }));
    expect(deltas).toEqual(want.expect.deltas);
    expect(result.text).toBe(want.expect.text);
    expect(result.calls).toEqual(want.expect.calls);
    expect(result.providerData).toEqual({
      provider: "google",
      parts: [
        { text: "Ich schaue " },
        { text: "nach…", thoughtSignature: "c2lnLXRleHQ=" },
        { functionCall: { id: "call-1", name: "lookup", args: { q: "x" } }, thoughtSignature: "c2lnLWNhbGw=" },
        { text: "", thoughtSignature: "c2lnLWVuZA==" },
      ],
    });
    expect(result.usage).toEqual({ inputTokens: 60, outputTokens: 104, cacheReadTokens: 0, cacheWriteTokens: 0 });
  });

  it("thought text is never yielded", async () => {
    const chunks = [
      { candidates: [{ content: { parts: [{ text: "thinking…", thought: true }] } }] },
      { candidates: [{ content: { parts: [{ text: "Answer." }] }, finishReason: "STOP" }] },
    ];
    const { client } = fake([], [chunks]);
    const { deltas } = await drain(provider(client).toolsStream!({ messages: [user("?")], tools: [LOOKUP] }));
    expect(deltas).toEqual(["Answer."]);
  });
});

describe("google refusals and odd finishes (T13)", () => {
  const blocked = () => fixture("blocked-prompt.response.json");

  it("a blocked prompt rejects text/structured/tools with CoaxRefusalError carrying the billed usage", async () => {
    const { client } = fake([blocked()]);
    const p = provider(client);
    const refusal = { name: "CoaxRefusalError", model: MODEL, category: "PROHIBITED_CONTENT", usage: { inputTokens: 7, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0 } };
    await expect(p.text({ messages: [user("?")] })).rejects.toMatchObject(refusal);
    await expect(p.structured({ messages: [user("?")], jsonSchema: { type: "object" }, schemaName: "out" })).rejects.toMatchObject(refusal);
    await expect(p.tools!({ messages: [user("?")], tools: [LOOKUP] })).rejects.toMatchObject(refusal);
    await expect(p.text({ messages: [user("?")] })).rejects.toBeInstanceOf(CoaxRefusalError);
  });

  it("a candidate stopped by the safety filter is a refusal, not an empty success", async () => {
    const { client } = fake([fixture("safety-candidate.response.json")]);
    await expect(provider(client).text({ messages: [user("?")] })).rejects.toMatchObject({ name: "CoaxRefusalError", category: "SAFETY" });
  });

  it("textStream: deltas before the block arrive, then the iteration rejects", async () => {
    const { client } = fake([], [fixture("safety-stream.chunks.json")]);
    const { deltas, error } = await drainToError(provider(client).textStream!({ messages: [user("?")] }));
    expect(deltas).toEqual(["Here is how "]);
    expect(error).toBeInstanceOf(CoaxRefusalError);
    expect(error).toMatchObject({ category: "SAFETY", explanation: "Response stopped by the safety filter.", usage: { inputTokens: 12 } });
  });

  it("toolsStream and structuredStream reject on the same mid-stream block", async () => {
    const { client } = fake([], [fixture("safety-stream.chunks.json")]);
    const p = provider(client);
    expect((await drainToError(p.toolsStream!({ messages: [user("?")], tools: [LOOKUP] }))).error).toBeInstanceOf(CoaxRefusalError);
    expect((await drainToError(p.structuredStream!({ messages: [user("?")], jsonSchema: { type: "object" }, schemaName: "o" }))).error).toBeInstanceOf(
      CoaxRefusalError,
    );
  });

  it("a stream whose first chunk is a blocked prompt rejects before any delta", async () => {
    const { client } = fake([], [[blocked(), answer("never")]]);
    const { deltas, error } = await drainToError(provider(client).textStream!({ messages: [user("?")] }));
    expect(deltas).toEqual([]);
    expect(error).toMatchObject({ name: "CoaxRefusalError", category: "PROHIBITED_CONTENT", usage: { inputTokens: 7 } });
  });

  it("RECITATION is a refusal; MALFORMED_FUNCTION_CALL is a plain error; MAX_TOKENS returns what was produced", async () => {
    const odd = fixture("odd-finishes.response.json");
    const { client } = fake([odd.recitation, odd.malformedFunctionCall, odd.maxTokensEmpty]);
    const p = provider(client);
    await expect(p.text({ messages: [user("?")] })).rejects.toMatchObject({ name: "CoaxRefusalError", category: "RECITATION" });
    const malformed = p.tools!({ messages: [user("?")], tools: [LOOKUP] });
    await expect(malformed).rejects.toThrow(/finishReason "MALFORMED_FUNCTION_CALL"/);
    await expect(malformed).rejects.not.toBeInstanceOf(CoaxRefusalError);
    const truncated = await p.text({ messages: [user("?")] });
    expect(truncated.text).toBe("");
    expect(truncated.usage.outputTokens).toBe(8192);
  });

  it("a response with neither candidates nor a block is an error", async () => {
    const { client } = fake([{}]);
    await expect(provider(client).text({ messages: [user("?")] })).rejects.toThrow(/no candidates/);
    const empty = fake([], [[]]);
    expect((await drainToError(provider(empty.client).textStream!({ messages: [user("?")] }))).error).toMatchObject({ message: expect.stringMatching(/no candidates/) });
  });

  it("inside ai.run an odd finish becomes a CoaxToolError with the cause", async () => {
    const { client } = fake([fixture("odd-finishes.response.json").malformedFunctionCall]);
    const ai = createAI({ providers: { google: (model) => google({ model, project: "p", client }) } });
    const lookup = tool({ name: "lookup", description: "Look something up.", input: z.object({}), run: () => 42 });
    const err = await ai.run({ model: "google:gemini-3.5-flash", prompt: "?", tools: [lookup] }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(CoaxToolError);
    expect((err as CoaxToolError).cause).toMatchObject({ message: expect.stringMatching(/MALFORMED_FUNCTION_CALL/) });
  });
});

describe("google embed (T15)", () => {
  it("ADC: one request per input, in order; usage sums tokenCount; model is embedModel ?? model", async () => {
    const { client, embedded } = fake();
    const res = await provider(client).embed!({ input: ["a", "b"] });
    expect(embedded.map((e) => e.contents)).toEqual([["a"], ["b"]]);
    expect(embedded.every((e) => e.model === MODEL)).toBe(true);
    expect(res.embeddings).toEqual([
      [1, 0.5],
      [2, 0.5],
    ]);
    expect(res.usage).toEqual({ inputTokens: 6, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0 });
    expect(res.model).toBe(MODEL);

    const named = fake();
    const res2 = await provider(named.client, { embedModel: "gemini-embedding-001" }).embed!({ input: "a" });
    expect(named.embedded).toHaveLength(1);
    expect(named.embedded[0]!.model).toBe("gemini-embedding-001");
    expect(res2.model).toBe("gemini-embedding-001");
    expect(res2.embeddings).toHaveLength(1);
  });

  it("reads the vector and token count where the SDK puts them", async () => {
    const shape = fixture("embed.sdk.json");
    const client = { models: { embedContent: async () => copy(shape.response) } };
    const res = await provider(client as never).embed!({ input: "first chunk" });
    expect(res.embeddings).toEqual([[0.12, -0.03, 0.88]]);
    expect(res.usage.inputTokens).toBe(3);
  });

  it("an API-key provider has no embed → CoaxUnsupportedError naming the capability", async () => {
    const p = google({ model: MODEL, apiKey: "k" });
    expect(p.embed).toBeUndefined();
    await expect(createClient({ provider: p }).embed({ input: "a" })).rejects.toMatchObject({ name: "CoaxUnsupportedError", capability: "embeddings", provider: "google" });
    await expect(createClient({ provider: p }).embed({ input: "a" })).rejects.toBeInstanceOf(CoaxUnsupportedError);
  });

  it("the call's headers and extraBody reach the request; the endpoint's extraBody does not", async () => {
    const { client, embedded } = fake();
    await provider(client, { extraBody: { generationConfig: { temperature: 0.2 } }, headers: { a: "1" } }).embed!({
      input: "a",
      headers: { b: "2" },
      extraBody: { parameters: { autoTruncate: false } },
    });
    expect(embedded[0]!.config).toEqual({ httpOptions: { headers: { a: "1", b: "2" }, extraBody: { parameters: { autoTruncate: false } } } });
  });

  it("stops between inputs once the signal aborted", async () => {
    const ctrl = new AbortController();
    const embedded: unknown[] = [];
    const client = {
      models: {
        embedContent: async (params: unknown) => {
          embedded.push(params);
          ctrl.abort();
          return { embeddings: [{ values: [1] }] };
        },
      },
    };
    await expect(createClient({ provider: provider(client as never) }).embed({ input: ["a", "b"], signal: ctrl.signal })).rejects.toBeInstanceOf(CoaxAbortError);
    expect(embedded).toHaveLength(1);
  });

  it("an empty batch makes no request", async () => {
    const { client, embedded } = fake();
    const res = await provider(client).embed!({ input: [] });
    expect(res.embeddings).toEqual([]);
    expect(embedded).toHaveLength(0);
  });
});

describe("google errors and retries (T16)", () => {
  const apiError = (e: { name: string; status: number; message: string }) => Object.assign(new Error(e.message), { name: e.name, status: e.status });

  it("the SDK's ApiError statuses classify like every other vendor's", () => {
    const errors = load("errors.json");
    for (const key of errors.expect.transient) {
      const e = errors.fixture[key];
      expect(isTransient(apiError(e.thrown ?? e)), key).toBe(true);
    }
    for (const key of errors.expect.notTransient) expect(isTransient(apiError(errors.fixture[key])), key).toBe(false);
  });

  it("retrying(): a 503 then success → two generateContent calls", async () => {
    const { client, sent } = fake([apiError(fixture("errors.json").unavailable), answer("ok")]);
    const res = await retrying(provider(client), { attempts: 3, initialDelayMs: 1 }).text({ messages: [user("?")] });
    expect(res.text).toBe("ok");
    expect(sent).toHaveLength(2);
  });

  it("a mid-stream ApiError surfaces through the iteration and is not retried", async () => {
    const chunk = fixture("stream-text.chunks.json")[0];
    const { client, sent } = fake([], [[chunk, apiError(fixture("errors.json").midStreamErrorChunk.thrown)]]);
    const { deltas, error } = await drainToError(retrying(provider(client), { attempts: 3, initialDelayMs: 1 }).textStream!({ messages: [user("?")] }));
    expect(deltas).toEqual(["Types guard "]);
    expect(error).toMatchObject({ name: "ApiError", status: 500 });
    expect(sent).toHaveLength(1);
  });
});

describe("refusal usage in the vendor-neutral layer (T14)", () => {
  it("CoaxRefusalError: vendor-neutral message, usage defaults to empty", () => {
    const bare = new CoaxRefusalError("m");
    expect(bare.message).toBe("coax: m refused the request");
    expect(bare.usage).toEqual({ inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0 });
    expect(new CoaxRefusalError("m", "c", "e").message).toBe('coax: m refused the request (category "c") — e');
  });

  const USAGE = { inputTokens: 5, outputTokens: 1, cacheReadTokens: 0, cacheWriteTokens: 0 };
  const refusing = {
    name: "fake",
    model: "m",
    async structured(): Promise<never> {
      throw new CoaxRefusalError("m", "SAFETY", null, USAGE);
    },
    async text(): Promise<never> {
      throw new CoaxRefusalError("m", "SAFETY", null, USAGE);
    },
    async tools(): Promise<never> {
      throw new CoaxRefusalError("m", "SAFETY", null, USAGE);
    },
    async *textStream(): AsyncGenerator<string, never, void> {
      yield "a";
      throw new CoaxRefusalError("m", "SAFETY", null, USAGE);
    },
    async *structuredStream(): AsyncGenerator<string, never, void> {
      yield "{";
      throw new CoaxRefusalError("m", "SAFETY", null, USAGE);
    },
    async *toolsStream(): AsyncGenerator<string, never, void> {
      yield "a";
      throw new CoaxRefusalError("m", "SAFETY", null, USAGE);
    },
  };

  const calls = {
    object: (c: ReturnType<typeof createClient>) => c.object({ schema: z.object({}), prompt: "?" }),
    text: (c: ReturnType<typeof createClient>) => c.text({ prompt: "?" }),
    tools: (c: ReturnType<typeof createClient>) => c.tools({ messages: [user("?")], tools: [LOOKUP] }),
    streamObject: (c: ReturnType<typeof createClient>) => drain(c.streamObject({ schema: z.object({}), prompt: "?" })),
    stream: (c: ReturnType<typeof createClient>) => drain(c.stream({ prompt: "?" })),
    toolsStream: (c: ReturnType<typeof createClient>) => drain(c.toolsStream({ messages: [user("?")], tools: [LOOKUP] })),
  };

  for (const [name, call] of Object.entries(calls)) {
    it(`${name}(): a refused call is reported through onUsage once, then rethrown as is`, async () => {
      const seen: [unknown, string][] = [];
      const client = createClient({ provider: refusing, onUsage: (usage, model) => void seen.push([usage, model]) });
      const err = await (call(client) as Promise<unknown>).catch((e: unknown) => e);
      expect(err).toBeInstanceOf(CoaxRefusalError);
      expect(seen).toEqual([[USAGE, "m"]]);
    });
  }

  it("an aborted call is not reported as a refusal", async () => {
    const ctrl = new AbortController();
    const seen: unknown[] = [];
    const p = {
      ...refusing,
      async text(): Promise<never> {
        ctrl.abort();
        throw new CoaxRefusalError("m", "SAFETY", null, USAGE);
      },
    };
    const err = await createClient({ provider: p, onUsage: (u) => void seen.push(u) }).text({ prompt: "?", signal: ctrl.signal }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(CoaxAbortError);
    expect(seen).toEqual([]);
  });

  it("runTools: a refused step adds its usage to CoaxToolError.usage and the budget", async () => {
    const budget = createBudget(null);
    let step = 0;
    const lookup = tool({ name: "lookup", description: "Look something up.", input: z.object({}), run: () => 42 });
    const err = await runTools(
      async () => {
        if (step++ === 0) return { text: "", calls: [{ id: "c", name: "lookup", input: {} }], usage: { inputTokens: 10, outputTokens: 2, cacheReadTokens: 0, cacheWriteTokens: 0 }, model: "m" };
        throw new CoaxRefusalError("m", "SAFETY", null, USAGE);
      },
      { tools: [lookup], messages: [user("?")], budget },
    ).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(CoaxToolError);
    expect((err as CoaxToolError).cause).toBeInstanceOf(CoaxRefusalError);
    expect((err as CoaxToolError).usage).toEqual({ inputTokens: 15, outputTokens: 3, cacheReadTokens: 0, cacheWriteTokens: 0 });
    expect(budget.spent()).toBe(18);
  });

  it("through ai: a Google blocked prompt reaches onUsage with the billed tokens", async () => {
    const { client } = fake([fixture("blocked-prompt.response.json")]);
    const seen: unknown[] = [];
    const ai = createAI({ providers: { google: (model) => google({ model, project: "p", client }) }, onUsage: (u) => void seen.push(u) });
    await expect(ai.text({ model: "google:gemini-3.5-flash", prompt: "?" })).rejects.toBeInstanceOf(CoaxRefusalError);
    expect(seen).toEqual([{ inputTokens: 7, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0 }]);
  });
});
