import { describe, expect, it } from "vitest";
import { createAI } from "../src/ai";
import { CoaxUnsupportedError } from "../src/client";
import { openai } from "../src/providers/openai";
import { emptyUsage, type Provider } from "../src/types";

describe("embed", () => {
  function embedClient() {
    const sent: Record<string, unknown>[] = [];
    const client = {
      chat: { completions: { create: async () => ({ choices: [] }) } },
      audio: { transcriptions: { create: async () => ({ text: "" }) }, speech: { create: async () => new Response() } },
      embeddings: {
        create: async (body: Record<string, unknown>) => {
          sent.push(body);
          const n = Array.isArray(body.input) ? body.input.length : 1;
          return { data: Array.from({ length: n }, (_, i) => ({ embedding: [i, i + 0.5] })), usage: { prompt_tokens: 7 } };
        },
      },
    };
    return { client, sent };
  }

  it("openai: one vector per input, model from embedModel, usage mapped", async () => {
    const { client, sent } = embedClient();
    const provider = openai({ model: "gpt-x", embedModel: "text-embedding-3-small", client: client as never });
    const res = await provider.embed!({ input: ["a", "b"] });
    expect(res.embeddings).toEqual([
      [0, 0.5],
      [1, 1.5],
    ]);
    expect(res.model).toBe("text-embedding-3-small");
    expect(res.usage.inputTokens).toBe(7);
    expect(sent[0]!.model).toBe("text-embedding-3-small");
    expect(sent[0]!.input).toEqual(["a", "b"]);
  });

  it("without embedModel the error says exactly what to configure", async () => {
    const { client } = embedClient();
    const provider = openai({ model: "gpt-x", client: client as never });
    await expect(provider.embed!({ input: "x" })).rejects.toThrow(/embedModel/);
  });

  it("a provider without embed raises CoaxUnsupportedError through the ai layer", async () => {
    const bare: Provider = {
      name: "mock",
      model: "m",
      async structured() {
        throw new Error("unused");
      },
      async text() {
        throw new Error("unused");
      },
    };
    const ai = createAI({ providers: { mock: () => bare }, models: { m: "mock:m" } });
    await expect(ai.embed({ model: "m", input: "hallo" })).rejects.toBeInstanceOf(CoaxUnsupportedError);
  });

  it("ai.embed goes through alias resolution and fires onUsage", async () => {
    const seen: string[] = [];
    const withEmbed: Provider = {
      name: "mock",
      model: "m",
      async structured() {
        throw new Error("unused");
      },
      async text() {
        throw new Error("unused");
      },
      async embed(req) {
        return { embeddings: [Array.isArray(req.input) ? [1] : [2]], usage: { ...emptyUsage(), inputTokens: 3 }, model: "emb-1" };
      },
    };
    const ai = createAI({
      providers: { mock: () => withEmbed },
      models: { vectors: "mock:m" },
      onUsage: (usage, meta) => void seen.push(`${meta.purpose}:${usage.inputTokens}`),
    });
    const res = await ai.embed({ model: "vectors", input: "ein text" });
    expect(res.embeddings).toEqual([[2]]);
    expect(seen).toEqual(["embed:3"]);
  });
});
