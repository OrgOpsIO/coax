import { describe, expect, it } from "vitest";
import { z } from "zod";
import { createAI } from "../src/ai";
import { CoaxSchemaError } from "../src/client";
import { anthropic } from "../src/providers/anthropic";
import { openai } from "../src/providers/openai";
import { emptyUsage, type Provider, type ProviderResponse, type StructuredRequest } from "../src/types";

const Out = z.object({ title: z.string(), tags: z.array(z.string()).min(1) });

async function collect<T>(stream: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const v of stream) out.push(v);
  return out;
}

describe("provider structuredStream", () => {
  it("openai: yields the tool-call argument fragments and accumulates them as raw", async () => {
    const chunks = [
      { choices: [{ delta: { tool_calls: [{ function: { arguments: '{"title":' } }] } }] },
      { choices: [{ delta: { tool_calls: [{ function: { arguments: '"Hi"}' } }] } }] },
      { choices: [], usage: { prompt_tokens: 3, completion_tokens: 2 } },
    ];
    const client = {
      chat: {
        completions: {
          create: async (body: Record<string, unknown>) => {
            expect(body.stream).toBe(true);
            return (async function* () {
              yield* chunks;
            })();
          },
        },
      },
    };
    const provider = openai({ model: "gpt-x", client: client as never });
    const gen = provider.structuredStream!({ messages: [{ role: "user", content: "go" }], jsonSchema: { type: "object" }, schemaName: "output" });
    const deltas: string[] = [];
    let cur = await gen.next();
    while (!cur.done) {
      deltas.push(cur.value);
      cur = await gen.next();
    }
    expect(deltas.join("")).toBe('{"title":"Hi"}');
    expect(cur.value.raw).toBe('{"title":"Hi"}');
    expect(cur.value.usage.inputTokens).toBe(3);
  });

  it("anthropic: yields input_json_delta fragments and takes the final tool_use input as raw", async () => {
    const events = [
      { type: "content_block_start" },
      { type: "content_block_delta", delta: { type: "input_json_delta", partial_json: '{"title"' } },
      { type: "content_block_delta", delta: { type: "input_json_delta", partial_json: ':"Hi"}' } },
    ];
    const client = {
      messages: {
        create: async () => ({ content: [] }),
        stream: (body: Record<string, unknown>) => {
          expect(body.tool_choice).toEqual({ type: "tool", name: "output" });
          return {
            [Symbol.asyncIterator]: async function* () {
              yield* events;
            },
            finalMessage: async () => ({
              content: [{ type: "tool_use", id: "t1", name: "output", input: { title: "Hi" } }],
              usage: { input_tokens: 4, output_tokens: 1 },
            }),
          };
        },
      },
    };
    const provider = anthropic({ model: "claude-x", client: client as never });
    const gen = provider.structuredStream!({ messages: [{ role: "user", content: "go" }], jsonSchema: { type: "object" }, schemaName: "output" });
    const deltas: string[] = [];
    let cur = await gen.next();
    while (!cur.done) {
      deltas.push(cur.value);
      cur = await gen.next();
    }
    expect(deltas.join("")).toBe('{"title":"Hi"}');
    expect(cur.value.raw).toEqual({ title: "Hi" });
  });
});

describe("ai.streamObject()", () => {
  /** A provider whose structuredStream plays scripted JSON fragments per attempt. */
  function scripted(attempts: string[][], finals?: unknown[]): Provider {
    let attempt = 0;
    return {
      name: "mock",
      model: "m",
      async structured() {
        throw new Error("unused");
      },
      async text() {
        throw new Error("unused");
      },
      async *structuredStream(_req: StructuredRequest): AsyncGenerator<string, ProviderResponse, void> {
        const i = Math.min(attempt++, attempts.length - 1);
        const fragments = attempts[i]!;
        for (const f of fragments) yield f;
        const raw = finals?.[i] ?? fragments.join("");
        return { raw, text: typeof raw === "string" ? raw : JSON.stringify(raw), usage: { ...emptyUsage(), outputTokens: 2 }, model: "m" };
      },
    };
  }

  it("yields progressively completed partials and resolves the validated result", async () => {
    const ai = createAI({
      providers: { mock: () => scripted([['{"title":"Ha', 'llo","tags":["a"', "]}"]]) },
      models: { m: "mock:m" },
    });
    const { partials, result } = await ai.streamObject({ model: "m", schema: Out, prompt: "go" });
    const seen = await collect(partials);
    // jsonrepair closes the truncated JSON at each step — the object grows across snapshots.
    expect(seen[0]).toEqual({ title: "Ha" });
    expect(seen.at(-1)).toEqual({ title: "Hallo", tags: ["a"] });
    const final = await result;
    expect(final.data).toEqual({ title: "Hallo", tags: ["a"] });
    expect(final.repairs).toBe(0);
  });

  it("streams the repair round too — partials restart, result carries repairs: 1", async () => {
    const ai = createAI({
      providers: {
        mock: () => scripted([['{"title":"kaputt","tags":[]}'], ['{"title":"ok","tags":["x"]}']]),
      },
      models: { m: "mock:m" },
    });
    const { partials, result } = await ai.streamObject({ model: "m", schema: Out, prompt: "go" });
    const seen = await collect(partials);
    expect(seen).toContainEqual({ title: "kaputt", tags: [] });
    expect(seen.at(-1)).toEqual({ title: "ok", tags: ["x"] });
    const final = await result;
    expect(final.repairs).toBe(1);
    expect(final.data.tags).toEqual(["x"]);
  });

  it("exhausted repairs surface as CoaxSchemaError through iteration AND result", async () => {
    const ai = createAI({
      providers: { mock: () => scripted([['{"title":"x","tags":[]}']]) },
      models: { m: "mock:m" },
    });
    const { partials, result } = await ai.streamObject({ model: "m", schema: Out, prompt: "go", maxRepairs: 1 });
    await expect(collect(partials)).rejects.toBeInstanceOf(CoaxSchemaError);
    await expect(result).rejects.toBeInstanceOf(CoaxSchemaError);
  });

  it("unwraps the envelope for non-object root schemas in partials and result", async () => {
    const ai = createAI({
      providers: { mock: () => scripted([['{"value":["a","b"]}']], [{ value: ["a", "b"] }]) },
      models: { m: "mock:m" },
    });
    const { partials, result } = await ai.streamObject({ model: "m", schema: z.array(z.string()), prompt: "go" });
    const seen = await collect(partials);
    expect(seen.at(-1)).toEqual(["a", "b"]);
    expect((await result).data).toEqual(["a", "b"]);
  });

  it("degrades a provider without structuredStream to one final partial", async () => {
    const plain: Provider = {
      name: "mock",
      model: "m",
      async structured() {
        return { raw: { title: "einmal", tags: ["z"] }, text: "", usage: emptyUsage(), model: "m" };
      },
      async text() {
        throw new Error("unused");
      },
    };
    const ai = createAI({ providers: { mock: () => plain }, models: { m: "mock:m" } });
    const { partials, result } = await ai.streamObject({ model: "m", schema: Out, prompt: "go" });
    expect(await collect(partials)).toEqual([{ title: "einmal", tags: ["z"] }]);
    expect((await result).data.title).toBe("einmal");
  });

  it("falls back before the first fragment, like every other call", async () => {
    const dead: Provider = {
      name: "mock",
      model: "primary",
      async structured() {
        throw new Error("unused");
      },
      async text() {
        throw new Error("unused");
      },
      // eslint-disable-next-line require-yield
      async *structuredStream(): AsyncGenerator<string, ProviderResponse, void> {
        throw new Error("primary: dead on arrival");
      },
    };
    const ai = createAI({
      providers: { mock: (model) => (model === "primary" ? dead : scripted([['{"title":"ok","tags":["f"]}']])) },
      models: { m: { use: "mock:primary", fallback: "mock:backup" } },
    });
    const { partials, result } = await ai.streamObject({ model: "m", schema: Out, prompt: "go" });
    expect((await collect(partials)).at(-1)).toEqual({ title: "ok", tags: ["f"] });
    expect((await result).data.title).toBe("ok");
  });
});
