import { describe, expect, it } from "vitest";
import { createAI } from "../src/ai";
import { anthropic } from "../src/providers/anthropic";
import { openai } from "../src/providers/openai";
import { emptyUsage, type Provider } from "../src/types";

async function collect(stream: AsyncIterable<string>): Promise<string[]> {
  const out: string[] = [];
  for await (const delta of stream) out.push(delta);
  return out;
}

describe("provider textStream", () => {
  it("openai: streams deltas, accumulates the text, and reads usage from the final chunk", async () => {
    const sent: Record<string, unknown>[] = [];
    const chunks = [
      { choices: [{ delta: { content: "Hal" } }] },
      { choices: [{ delta: { content: "lo" } }] },
      { choices: [{ delta: {} }] },
      { choices: [], usage: { prompt_tokens: 9, completion_tokens: 4 } },
    ];
    const client = {
      chat: {
        completions: {
          create: async (body: Record<string, unknown>) => {
            sent.push(body);
            return (async function* () {
              yield* chunks;
            })();
          },
        },
      },
    };
    const provider = openai({ model: "gpt-x", client: client as never });
    const gen = provider.textStream!({ messages: [{ role: "user", content: "hi" }] });

    const deltas: string[] = [];
    let cur = await gen.next();
    while (!cur.done) {
      deltas.push(cur.value);
      cur = await gen.next();
    }
    expect(deltas).toEqual(["Hal", "lo"]);
    expect(cur.value.text).toBe("Hallo");
    expect(cur.value.usage.inputTokens).toBe(9);
    expect(sent[0]!.stream).toBe(true);
    expect(sent[0]!.stream_options).toEqual({ include_usage: true });
  });

  it("anthropic: surfaces text deltas (not thinking) and takes the final message for usage", async () => {
    const events = [
      { type: "content_block_delta", delta: { type: "thinking_delta", text: "hmm" } },
      { type: "content_block_delta", delta: { type: "text_delta", text: "Ha" } },
      { type: "content_block_delta", delta: { type: "text_delta", text: "llo" } },
      { type: "message_delta" },
    ];
    const client = {
      messages: {
        create: async () => ({ content: [] }),
        stream: () => ({
          [Symbol.asyncIterator]: async function* () {
            yield* events;
          },
          finalMessage: async () => ({ content: [{ type: "text", text: "Hallo" }], usage: { input_tokens: 5, output_tokens: 2 } }),
        }),
      },
    };
    const provider = anthropic({ model: "claude-x", client: client as never });
    const gen = provider.textStream!({ messages: [{ role: "user", content: "hi" }] });

    const deltas: string[] = [];
    let cur = await gen.next();
    while (!cur.done) {
      deltas.push(cur.value);
      cur = await gen.next();
    }
    expect(deltas).toEqual(["Ha", "llo"]);
    expect(cur.value.text).toBe("Hallo");
    expect(cur.value.usage.inputTokens).toBe(5);
  });
});

describe("ai.stream()", () => {
  function streamingProvider(model: string, deltas: string[], opts?: { failAfter?: number; failImmediately?: boolean }): Provider {
    return {
      name: "mock",
      model,
      async structured() {
        throw new Error("unused");
      },
      async text() {
        return { raw: "", text: deltas.join(""), usage: emptyUsage(), model };
      },
      async *textStream() {
        if (opts?.failImmediately) throw new Error(`${model}: dead on arrival`);
        let i = 0;
        for (const d of deltas) {
          if (opts?.failAfter !== undefined && i++ >= opts.failAfter) throw new Error(`${model}: died mid-stream`);
          yield d;
        }
        return { raw: "", text: deltas.join(""), usage: { ...emptyUsage(), outputTokens: 3 }, model };
      },
    };
  }

  it("streams deltas and resolves result with the full text once consumed", async () => {
    const ai = createAI({ providers: { mock: () => streamingProvider("m", ["a", "b", "c"]) }, models: { m: "mock:m" } });
    const { stream, result } = await ai.stream({ model: "m", prompt: "go" });
    expect(await collect(stream)).toEqual(["a", "b", "c"]);
    const final = await result;
    expect(final.text).toBe("abc");
    expect(final.usage.outputTokens).toBe(3);
  });

  it("falls back to the fallback model when the primary dies before its first delta", async () => {
    const ai = createAI({
      providers: {
        mock: (model) => (model === "primary" ? streamingProvider(model, ["x"], { failImmediately: true }) : streamingProvider(model, ["ok"])),
      },
      models: { m: { use: "mock:primary", fallback: "mock:backup" } },
    });
    const { stream, result } = await ai.stream({ model: "m", prompt: "go" });
    expect(await collect(stream)).toEqual(["ok"]);
    expect((await result).model).toBe("backup");
  });

  it("does NOT fall back after the first delta — the failure surfaces through iteration and result", async () => {
    const ai = createAI({
      providers: { mock: (model) => (model === "primary" ? streamingProvider(model, ["x", "y"], { failAfter: 1 }) : streamingProvider(model, ["never"])) },
      models: { m: { use: "mock:primary", fallback: "mock:backup" } },
    });
    const { stream, result } = await ai.stream({ model: "m", prompt: "go" });
    await expect(collect(stream)).rejects.toThrow(/died mid-stream/);
    await expect(result).rejects.toThrow(/died mid-stream/);
  });

  it("degrades a provider without textStream to a single yield of the whole text", async () => {
    const plain: Provider = {
      name: "mock",
      model: "m",
      async structured() {
        throw new Error("unused");
      },
      async text() {
        return { raw: "", text: "alles auf einmal", usage: { ...emptyUsage(), outputTokens: 5 }, model: "m" };
      },
    };
    const ai = createAI({ providers: { mock: () => plain }, models: { m: "mock:m" } });
    const { stream, result } = await ai.stream({ model: "m", prompt: "go" });
    expect(await collect(stream)).toEqual(["alles auf einmal"]);
    expect((await result).text).toBe("alles auf einmal");
  });

  it("onUsage fires once with the stream's final usage", async () => {
    const seen: { output: number }[] = [];
    const ai = createAI({
      providers: { mock: () => streamingProvider("m", ["a", "b"]) },
      models: { m: "mock:m" },
      onUsage: (usage) => void seen.push({ output: usage.outputTokens }),
    });
    const { stream } = await ai.stream({ model: "m", prompt: "go" });
    await collect(stream);
    expect(seen).toEqual([{ output: 3 }]);
  });
});
