import { describe, expect, it } from "vitest";
import { z } from "zod";
import { createAI } from "../src/ai";
import { runToolsStream, tool, CoaxToolError, type RunEvent } from "../src/tools";
import { emptyUsage, type Provider, type ToolsRequest, type ToolsResponse } from "../src/types";

const lookup = tool({
  name: "lookup",
  description: "Look something up.",
  input: z.object({ q: z.string() }),
  run: ({ q }) => `result for ${q}`,
});

/** Scripted streaming turns: each entry is the deltas plus the complete turn response. */
function fakeTurns(queue: Array<{ deltas?: string[]; text?: string; calls?: { id: string; name: string; input: unknown }[] }>) {
  let i = 0;
  return (_req: Pick<ToolsRequest, "messages" | "tools" | "toolChoice">) =>
    (async function* (): AsyncGenerator<string, ToolsResponse, void> {
      const turn = queue[Math.min(i++, queue.length - 1)]!;
      for (const d of turn.deltas ?? []) yield d;
      return { text: turn.text ?? (turn.deltas ?? []).join(""), calls: turn.calls ?? [], usage: { ...emptyUsage(), outputTokens: 1 }, model: "mock-1" };
    })();
}

async function collect(events: AsyncIterable<RunEvent>): Promise<RunEvent[]> {
  const out: RunEvent[] = [];
  for await (const e of events) out.push(e);
  return out;
}

describe("runToolsStream", () => {
  it("emits deltas, calling and tool events in order, then returns the RunResult", async () => {
    const gen = runToolsStream(
      fakeTurns([
        { deltas: ["Ich schaue ", "nach…"], calls: [{ id: "c1", name: "lookup", input: { q: "x" } }] },
        { deltas: ["42 ", "ist die Antwort."] },
      ]),
      { messages: [{ role: "user", content: "?" }], tools: [lookup] },
    );
    const events: RunEvent[] = [];
    let cur = await gen.next();
    while (!cur.done) {
      events.push(cur.value);
      cur = await gen.next();
    }
    expect(events.map((e) => e.type)).toEqual(["delta", "delta", "calling", "tool", "delta", "delta"]);
    expect(events[2]).toMatchObject({ type: "calling", step: 0, call: { name: "lookup" } });
    expect(events[3]).toMatchObject({ type: "tool", step: 0, invocation: { name: "lookup", output: "result for x" } });
    expect(events[4]).toMatchObject({ type: "delta", step: 1 });
    expect(cur.value.text).toBe("42 ist die Antwort.");
    expect(cur.value.steps).toBe(2);
  });

  it("a mid-run model failure still carries the partial state on CoaxToolError", async () => {
    let n = 0;
    const dying = (_req: Pick<ToolsRequest, "messages" | "tools" | "toolChoice">) =>
      (async function* (): AsyncGenerator<string, ToolsResponse, void> {
        if (n++ === 0) return { text: "", calls: [{ id: "c1", name: "lookup", input: { q: "x" } }], usage: emptyUsage(), model: "m" };
        yield "halb…";
        throw new Error("boom at turn 2");
      })();
    const gen = runToolsStream(dying, { messages: [{ role: "user", content: "?" }], tools: [lookup] });
    const events: RunEvent[] = [];
    const err: CoaxToolError = await (async () => {
      let cur = await gen.next();
      while (!cur.done) {
        events.push(cur.value);
        cur = await gen.next();
      }
      return undefined as never;
    })().catch((e) => e);
    expect(err).toBeInstanceOf(CoaxToolError);
    expect(err.steps).toBe(1);
    expect(err.calls).toHaveLength(1);
    expect(events.some((e) => e.type === "delta" && e.text === "halb…")).toBe(true);
  });

  it("delivers a typed output through the stream too", async () => {
    const gen = runToolsStream(
      fakeTurns([{ calls: [{ id: "c1", name: "final_answer", input: { total: 7 } }] }]),
      { messages: [{ role: "user", content: "?" }], tools: [], output: z.object({ total: z.number() }) },
    );
    let cur = await gen.next();
    while (!cur.done) cur = await gen.next();
    expect(cur.value.data).toEqual({ total: 7 });
  });
});

describe("ai.runStream()", () => {
  function streamingToolsProvider(model: string): Provider {
    let turn = 0;
    return {
      name: "mock",
      model,
      async structured() {
        throw new Error("unused");
      },
      async text() {
        throw new Error("unused");
      },
      async *toolsStream(_req: ToolsRequest): AsyncGenerator<string, ToolsResponse, void> {
        if (turn++ === 0) {
          yield "Moment… ";
          return { text: "Moment… ", calls: [{ id: "c1", name: "lookup", input: { q: "a" } }], usage: emptyUsage(), model };
        }
        yield "Fertig.";
        return { text: "Fertig.", calls: [], usage: emptyUsage(), model };
      },
    };
  }

  it("streams a whole run end to end", async () => {
    const ai = createAI({ providers: { mock: () => streamingToolsProvider("m") }, models: { m: "mock:m" } });
    const { events, result } = await ai.runStream({ model: "m", prompt: "?", tools: [lookup] });
    const seen = await collect(events);
    expect(seen.map((e) => e.type)).toEqual(["delta", "calling", "tool", "delta"]);
    const final = await result;
    expect(final.text).toBe("Fertig.");
    expect(final.calls).toHaveLength(1);
  });

  it("degrades a provider with only non-streaming tools(): events still flow, just without deltas", async () => {
    let turn = 0;
    const plain: Provider = {
      name: "mock",
      model: "m",
      async structured() {
        throw new Error("unused");
      },
      async text() {
        throw new Error("unused");
      },
      async tools(): Promise<ToolsResponse> {
        if (turn++ === 0) return { text: "", calls: [{ id: "c1", name: "lookup", input: { q: "a" } }], usage: emptyUsage(), model: "m" };
        return { text: "Fertig.", calls: [], usage: emptyUsage(), model: "m" };
      },
    };
    const ai = createAI({ providers: { mock: () => plain }, models: { m: "mock:m" } });
    const { events, result } = await ai.runStream({ model: "m", prompt: "?", tools: [lookup] });
    const seen = await collect(events);
    expect(seen.map((e) => e.type)).toEqual(["calling", "tool"]);
    expect((await result).text).toBe("Fertig.");
  });
});
