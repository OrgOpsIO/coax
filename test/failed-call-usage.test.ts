import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { z } from "zod";
import { createAI } from "../src/ai";
import { createBudget } from "../src/budget";
import { CoaxAbortError, createClient } from "../src/client";
import { google } from "../src/providers/google";
import { retrying } from "../src/registry";
import { CoaxToolError, runTools, tool } from "../src/tools";
import { billedUsage, CoaxRefusalError, type Message, type Provider, type Usage, withBilledUsage } from "../src/types";

// Review R1.3 / human's answer O1 = (a): a failed call that was still billed reports its tokens through
// onUsage once, carries them on the thrown error (`billedUsage`), and counts in ai.run() and the budget —
// vendor-neutral (decisions/stage-01-failed-call-usage.md).

const fixture = (name: string) => JSON.parse(readFileSync(new URL(`./fixtures/google/${name}`, import.meta.url), "utf8")).fixture;
const copy = <T>(v: T): T => JSON.parse(JSON.stringify(v));
const usage = (inputTokens: number, outputTokens = 0): Usage => ({ inputTokens, outputTokens, cacheReadTokens: 0, cacheWriteTokens: 0 });
const user = (content: string): Message => ({ role: "user", content });
const LOOKUP = { name: "lookup", description: "Look something up.", jsonSchema: { type: "object", properties: {} } };

/** A fake @google/genai client: generateContent answers `replies` in order, generateContentStream `streams`. */
function fakeGoogle(replies: unknown[] = [], streams: unknown[][] = [], embed?: (call: number, params: { config?: { abortSignal?: AbortSignal } }) => unknown) {
  let r = 0;
  let s = 0;
  let e = 0;
  return {
    models: {
      generateContent: async () => copy(replies[Math.min(r++, replies.length - 1)]),
      generateContentStream: async () => {
        const chunks = streams[Math.min(s++, streams.length - 1)] ?? [];
        return (async function* () {
          for (const chunk of chunks) yield copy(chunk);
        })();
      },
      embedContent: async (params: { config?: { abortSignal?: AbortSignal } }) =>
        embed ? embed(++e, params) : { embeddings: [{ values: [e, 0.5], statistics: { tokenCount: 3 } }] },
    },
  } as never;
}

const gemini = (client: never) => google({ model: "gemini-3.5-flash", project: "p", client });

async function drain<T, R>(gen: AsyncGenerator<T, R, void>): Promise<R> {
  let cur = await gen.next();
  while (!cur.done) cur = await gen.next();
  return cur.value;
}

describe("google: a turn without a usable answer is billed (R1.3)", () => {
  const malformed = () => fixture("odd-finishes.response.json").malformedFunctionCall;

  it("text() and tools(): MALFORMED_FUNCTION_CALL reports its 60 prompt tokens once and carries them on the error", async () => {
    for (const call of ["text", "tools"] as const) {
      const seen: [Usage, string][] = [];
      const client = createClient({ provider: gemini(fakeGoogle([malformed()])), onUsage: (u, m) => void seen.push([u, m]) });
      const err = await (call === "text" ? client.text({ prompt: "?" }) : client.tools({ messages: [user("?")], tools: [LOOKUP] })).catch((e: unknown) => e);
      expect(err, call).toBeInstanceOf(Error);
      expect(err, call).not.toBeInstanceOf(CoaxRefusalError);
      expect((err as Error).message, call).toMatch(/finishReason "MALFORMED_FUNCTION_CALL"/);
      expect(billedUsage(err), call).toEqual(usage(60));
      expect(seen, call).toEqual([[usage(60), "gemini-3.5-flash"]]);
    }
  });

  it("the stream twin: a stream ending in MALFORMED_FUNCTION_CALL reports the last chunk's usage {5,2} once", async () => {
    const chunks = [
      { candidates: [{ content: { role: "model", parts: [{ text: "Let me " }] } }] },
      {
        candidates: [{ content: { role: "model", parts: [] }, finishReason: "MALFORMED_FUNCTION_CALL", finishMessage: "bad args" }],
        usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 2, totalTokenCount: 7 },
      },
    ];
    for (const call of ["stream", "toolsStream"] as const) {
      const seen: Usage[] = [];
      const client = createClient({ provider: gemini(fakeGoogle([], [chunks])), onUsage: (u) => void seen.push(u) });
      const gen = call === "stream" ? client.stream({ prompt: "?" }) : client.toolsStream({ messages: [user("?")], tools: [LOOKUP] });
      const err = await drain<string, unknown>(gen).catch((e: unknown) => e);
      expect(billedUsage(err), call).toEqual(usage(5, 2));
      expect(seen, call).toEqual([usage(5, 2)]);
    }
  });

  it("a response without candidates is billed too (its usageMetadata, when it has one)", async () => {
    const seen: Usage[] = [];
    const client = createClient({ provider: gemini(fakeGoogle([{ usageMetadata: { promptTokenCount: 9, totalTokenCount: 9 } }])), onUsage: (u) => void seen.push(u) });
    const err = await client.text({ prompt: "?" }).catch((e: unknown) => e);
    expect((err as Error).message).toMatch(/no candidates/);
    expect(seen).toEqual([usage(9)]);
  });

  it("inside ai.run(): CoaxToolError.usage and the budget include the 60, onUsage saw it once", async () => {
    const step0 = { candidates: [{ content: { role: "model", parts: [{ functionCall: { id: "c1", name: "lookup", args: {} } }] }, finishReason: "STOP" }], usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 2, totalTokenCount: 12 } };
    const seen: Usage[] = [];
    const ai = createAI({ providers: { google: (model) => google({ model, project: "p", client: fakeGoogle([step0, malformed()]) }) }, onUsage: (u) => void seen.push(u) });
    const budget = createBudget(null);
    const lookup = tool({ name: "lookup", description: "Look something up.", input: z.object({}), run: () => 42 });
    const err = await ai.run({ model: "google:gemini-3.5-flash", prompt: "?", tools: [lookup], budget }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(CoaxToolError);
    expect((err as CoaxToolError).usage).toEqual(usage(70, 2));
    expect(budget.spent()).toBe(72);
    expect(seen).toEqual([usage(10, 2), usage(60)]);
  });

  it("inside ai.object(): a repair round, then an odd finish — each turn reported once, the error carries only its own turn", async () => {
    const wrong = { candidates: [{ content: { role: "model", parts: [{ text: '{"n":"x"}' }] }, finishReason: "STOP" }], usageMetadata: { promptTokenCount: 4, candidatesTokenCount: 3, totalTokenCount: 7 } };
    const seen: Usage[] = [];
    const ai = createAI({ providers: { google: (model) => google({ model, project: "p", client: fakeGoogle([wrong, malformed()]) }) }, onUsage: (u) => void seen.push(u) });
    const err = await ai.object({ model: "google:gemini-3.5-flash", prompt: "?", schema: z.object({ n: z.number() }) }).catch((e: unknown) => e);
    expect(billedUsage(err)).toEqual(usage(60));
    expect(seen).toEqual([usage(4, 3), usage(60)]);
  });
});

describe("google embed: inputs finished before an abort or a failure are billed (R1.3)", () => {
  it("embed(['a','b']) aborted after the first input: CoaxAbortError carries the first input's tokens, reported once", async () => {
    const ctrl = new AbortController();
    const client = fakeGoogle([], [], (call) => {
      if (call === 1) ctrl.abort();
      return { embeddings: [{ values: [call], statistics: { tokenCount: 3 } }] };
    });
    const seen: Usage[] = [];
    const err = await createClient({ provider: gemini(client), onUsage: (u) => void seen.push(u) }).embed({ input: ["a", "b"], signal: ctrl.signal }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(CoaxAbortError);
    expect((err as CoaxAbortError).usage).toEqual(usage(3));
    expect(billedUsage(err)).toEqual(usage(3));
    expect(seen).toEqual([usage(3)]);
  });

  it("the SDK throwing on the abort mid-request keeps the tokens too (not replaced by an empty CoaxAbortError)", async () => {
    const ctrl = new AbortController();
    const client = fakeGoogle([], [], (call) => {
      if (call === 1) return { embeddings: [{ values: [1], statistics: { tokenCount: 3 } }] };
      ctrl.abort();
      throw new DOMException("This operation was aborted", "AbortError");
    });
    const seen: Usage[] = [];
    const err = await createClient({ provider: gemini(client), onUsage: (u) => void seen.push(u) }).embed({ input: ["a", "b"], signal: ctrl.signal }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(CoaxAbortError);
    expect((err as CoaxAbortError).usage).toEqual(usage(3));
    expect(seen).toEqual([usage(3)]);
  });

  it("a failure on the second input keeps the error as it is (status and all) and marks the first input's tokens", async () => {
    const failure = Object.assign(new Error("bad request"), { status: 400 });
    const client = fakeGoogle([], [], (call) => {
      if (call === 2) throw failure;
      return { embeddings: [{ values: [1], statistics: { tokenCount: 3 } }] };
    });
    const seen: Usage[] = [];
    const err = await createClient({ provider: gemini(client), onUsage: (u) => void seen.push(u) }).embed({ input: ["a", "b"] }).catch((e: unknown) => e);
    expect(err).toBe(failure);
    expect(billedUsage(err)).toEqual(usage(3));
    expect(seen).toEqual([usage(3)]);
  });

  it("a failure on the FIRST input cost nothing: no mark, no onUsage", async () => {
    const client = fakeGoogle([], [], () => {
      throw Object.assign(new Error("bad request"), { status: 400 });
    });
    const seen: Usage[] = [];
    const err = await createClient({ provider: gemini(client), onUsage: (u) => void seen.push(u) }).embed({ input: ["a", "b"] }).catch((e: unknown) => e);
    expect(billedUsage(err)).toBeUndefined();
    expect(seen).toEqual([]);
  });

  it("a retried batch re-bills the inputs it had finished — and the result counts both attempts", async () => {
    // Attempt 1: "a" (3 tokens) then a 503 on "b"; attempt 2: both inputs (6 tokens).
    const client = fakeGoogle([], [], (call) => {
      if (call === 2) throw Object.assign(new Error("unavailable"), { status: 503 });
      return { embeddings: [{ values: [call], statistics: { tokenCount: 3 } }] };
    });
    const seen: Usage[] = [];
    const c = createClient({ provider: retrying(gemini(client), { attempts: 2, initialDelayMs: 0 }), onUsage: (u) => void seen.push(u) });
    const res = await c.embed({ input: ["a", "b"] });
    expect(res.embeddings).toHaveLength(2);
    expect(res.usage).toEqual(usage(9));
    expect(seen).toEqual([usage(9)]);
  });
});

describe("billed failures are vendor-neutral (R1.3)", () => {
  const BILLED = usage(7, 1);
  const fail = () => withBilledUsage(new Error("the endpoint charged, then failed"), BILLED);
  // Not Google: any provider (a factory, a gateway adapter) uses the same public mark.
  const failing: Provider = {
    name: "other",
    model: "other-model",
    structured: async () => {
      throw fail();
    },
    text: async () => {
      throw fail();
    },
    async *textStream() {
      yield "a";
      throw fail();
    },
    async *structuredStream() {
      yield "{";
      throw fail();
    },
    tools: async () => {
      throw fail();
    },
    async *toolsStream() {
      yield "a";
      throw fail();
    },
    embed: async () => {
      throw fail();
    },
    transcribe: async () => {
      throw fail();
    },
    speak: async () => {
      throw fail();
    },
  };
  type C = ReturnType<typeof createClient>;
  const calls: Record<string, (c: C) => Promise<unknown>> = {
    object: (c) => c.object({ schema: z.object({}), prompt: "?" }),
    text: (c) => c.text({ prompt: "?" }),
    stream: (c) => drain(c.stream({ prompt: "?" })),
    streamObject: (c) => drain(c.streamObject({ schema: z.object({}), prompt: "?" })),
    tools: (c) => c.tools({ messages: [user("?")], tools: [LOOKUP] }),
    toolsStream: (c) => drain(c.toolsStream({ messages: [user("?")], tools: [LOOKUP] })),
    embed: (c) => c.embed({ input: "a" }),
    transcribe: (c) => c.transcribe({ audio: { data: new Uint8Array([1]) } }),
    speak: (c) => c.speak({ input: "hi" }),
  };

  for (const [name, call] of Object.entries(calls)) {
    it(`${name}(): a marked failure is reported through onUsage once (with the provider's model), then rethrown as is`, async () => {
      const seen: [Usage, string][] = [];
      const err = await call(createClient({ provider: failing, onUsage: (u, m) => void seen.push([u, m]) })).catch((e: unknown) => e);
      expect((err as Error).message).toBe("the endpoint charged, then failed");
      expect(billedUsage(err)).toEqual(BILLED);
      expect(seen).toEqual([[BILLED, "other-model"]]);
    });
  }

  it("an unmarked failure reports nothing, as before", async () => {
    const seen: Usage[] = [];
    const plain: Provider = { ...failing, text: async () => { throw new Error("500"); } };
    await createClient({ provider: plain, onUsage: (u) => void seen.push(u) }).text({ prompt: "?" }).catch(() => {});
    expect(seen).toEqual([]);
  });

  it("a marked failure that races an abort is a plain CoaxAbortError and reports nothing (abort keeps precedence)", async () => {
    const ctrl = new AbortController();
    const p: Provider = { ...failing, text: async () => { ctrl.abort(); throw fail(); } };
    const seen: Usage[] = [];
    const err = await createClient({ provider: p, onUsage: (u) => void seen.push(u) }).text({ prompt: "?", signal: ctrl.signal }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(CoaxAbortError);
    expect(seen).toEqual([]);
  });

  it("runTools: a marked failure adds its usage to CoaxToolError.usage and the budget, the CoaxToolError keeps it as cause", async () => {
    const budget = createBudget(null);
    let step = 0;
    const lookup = tool({ name: "lookup", description: "Look something up.", input: z.object({}), run: () => 42 });
    const err = await runTools(
      async () => {
        if (step++ === 0) return { text: "", calls: [{ id: "c", name: "lookup", input: {} }], usage: usage(10, 2), model: "m" };
        throw fail();
      },
      { tools: [lookup], messages: [user("?")], budget },
    ).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(CoaxToolError);
    expect((err as CoaxToolError).usage).toEqual(usage(17, 3));
    expect(budget.spent()).toBe(20);
    expect(billedUsage((err as CoaxToolError).cause)).toEqual(BILLED);
  });

  it("retrying(): billed attempts that were retried count in the result; an error that escapes carries all of them", async () => {
    let n = 0;
    const flaky: Provider = {
      ...failing,
      text: async () => {
        if (n++ < 1) throw withBilledUsage(Object.assign(new Error("overloaded"), { status: 503 }), usage(2));
        return { raw: "ok", text: "ok", usage: usage(5, 1), model: "other-model" };
      },
      tools: async () => {
        throw withBilledUsage(Object.assign(new Error("overloaded"), { status: 503 }), usage(2));
      },
    };
    const wrapped = retrying(flaky, { attempts: 3, initialDelayMs: 0 });
    expect((await wrapped.text({ messages: [user("?")] })).usage).toEqual(usage(7, 1));
    const err = await wrapped.tools!({ messages: [user("?")], tools: [LOOKUP] }).catch((e: unknown) => e);
    expect((err as { status: number }).status).toBe(503);
    expect(billedUsage(err)).toEqual(usage(6));
  });

  it("a refusal is still reported under the model it names (e.g. the dated id the API answered with), not the provider's", async () => {
    const p: Provider = { ...failing, text: async () => { throw new CoaxRefusalError("other-model-2026-10-01", "cyber", null, BILLED); } };
    const seen: string[] = [];
    await createClient({ provider: p, onUsage: (_, m) => void seen.push(m) }).text({ prompt: "?" }).catch(() => {});
    expect(seen).toEqual(["other-model-2026-10-01"]);
  });

  it("CoaxRefusalError carries the mark: billedUsage equals its usage", () => {
    expect(billedUsage(new CoaxRefusalError("m", "SAFETY", null, BILLED))).toEqual(BILLED);
  });

  it("the mark is invisible to JSON and to deep equality", () => {
    const err = withBilledUsage(new Error("x"), BILLED);
    expect(Object.keys(err)).toEqual([]);
    expect(err).toEqual(new Error("x"));
  });
});
