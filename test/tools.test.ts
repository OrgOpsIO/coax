import { describe, expect, it } from "vitest";
import { z } from "zod";
import { runTools, tool, CoaxToolError } from "../src/tools";
import { CoaxAbortError } from "../src/client";
import { createBudget } from "../src/budget";
import { emptyUsage, type ToolsResponse, type Message, type ToolDefinition } from "../src/types";

// A scripted tool-calling provider: returns queued turns in order, recording what it was sent.
function fakeTools(queue: Array<{ text?: string; calls?: { id: string; name: string; input: unknown }[] }>) {
  const seen: { messages: Message[]; tools: ToolDefinition[] }[] = [];
  let i = 0;
  const fn = async (req: { messages: Message[]; tools: ToolDefinition[] }): Promise<ToolsResponse> => {
    // Snapshot the transcript — the driver mutates its own array between turns.
    seen.push({ messages: structuredClone(req.messages), tools: req.tools });
    const turn = queue[Math.min(i++, queue.length - 1)]!;
    return { text: turn.text ?? "", calls: turn.calls ?? [], usage: { ...emptyUsage(), inputTokens: 10, outputTokens: 4 }, model: "mock-1" };
  };
  return Object.assign(fn, { seen });
}

const lookup = tool({
  name: "lookup_invoice",
  description: "Fetch one invoice by number.",
  input: z.object({ number: z.string().min(1) }),
  run: ({ number }) => ({ number, total: 42.5 }),
});

describe("runTools with an output schema", () => {
  const Answer = z.object({ total: z.number(), currency: z.string().min(1) });

  it("ends the run through the answer tool and returns typed data", async () => {
    const callTools = fakeTools([
      { calls: [{ id: "c1", name: "lookup_invoice", input: { number: "R-1" } }] },
      { calls: [{ id: "c2", name: "final_answer", input: { total: 42.5, currency: "EUR" } }] },
    ]);
    const result = await runTools(callTools, { messages: [{ role: "user", content: "?" }], tools: [lookup], output: Answer });
    expect(result.data).toEqual({ total: 42.5, currency: "EUR" });
    // The answer tool is offered to the model alongside the real tools.
    expect(callTools.seen[0]!.tools.map((t) => t.name)).toEqual(["lookup_invoice", "final_answer"]);
  });

  it("feeds an answer that misses the schema back for correction", async () => {
    const callTools = fakeTools([
      { calls: [{ id: "c1", name: "final_answer", input: { total: "viel" } }] },
      { calls: [{ id: "c2", name: "final_answer", input: { total: 42.5, currency: "EUR" } }] },
    ]);
    const result = await runTools(callTools, { messages: [{ role: "user", content: "?" }], tools: [lookup], output: Answer });
    expect(result.data).toEqual({ total: 42.5, currency: "EUR" });
    const feedback = callTools.seen[1]!.messages.at(-1)!;
    expect(JSON.stringify(feedback.toolResults)).toMatch(/did not match the required schema/);
  });

  it("redirects a plain-text answer to the answer tool", async () => {
    const callTools = fakeTools([
      { text: "Die Rechnung beträgt 42,50 €." },
      { calls: [{ id: "c1", name: "final_answer", input: { total: 42.5, currency: "EUR" } }] },
    ]);
    const result = await runTools(callTools, { messages: [{ role: "user", content: "?" }], tools: [lookup], output: Answer });
    expect(result.data).toEqual({ total: 42.5, currency: "EUR" });
    const nudge = callTools.seen[1]!.messages.at(-1)!;
    expect(nudge.content).toMatch(/final_answer/);
  });

  it("wraps a non-object output schema transparently (same envelope as ai.object)", async () => {
    const callTools = fakeTools([{ calls: [{ id: "c1", name: "final_answer", input: { value: ["a", "b"] } }] }]);
    const result = await runTools(callTools, { messages: [{ role: "user", content: "?" }], tools: [], output: z.array(z.string()) });
    expect(result.data).toEqual(["a", "b"]);
  });

  it("rejects a user tool that collides with the reserved answer-tool name", async () => {
    const collider = tool({ name: "final_answer", description: "nope", input: z.object({}), run: () => "x" });
    const callTools = fakeTools([{ text: "unused" }]);
    await expect(runTools(callTools, { messages: [{ role: "user", content: "?" }], tools: [collider], output: Answer })).rejects.toThrow(/reserved/);
  });

  it("without an output schema nothing changes — no extra tool, text ends the run", async () => {
    const callTools = fakeTools([{ text: "fertig" }]);
    const result = await runTools(callTools, { messages: [{ role: "user", content: "?" }], tools: [lookup] });
    expect(result.text).toBe("fertig");
    expect(result.data).toBeUndefined();
    expect(callTools.seen[0]!.tools.map((t) => t.name)).toEqual(["lookup_invoice"]);
  });
});

describe("runTools", () => {
  it("runs a tool, feeds the result back, and returns the final answer", async () => {
    const callTools = fakeTools([
      { calls: [{ id: "c1", name: "lookup_invoice", input: { number: "R-2026-1" } }] },
      { text: "Die Rechnung R-2026-1 beträgt 42,50 €." },
    ]);
    const result = await runTools(callTools, { messages: [{ role: "user", content: "Wie hoch ist R-2026-1?" }], tools: [lookup] });

    expect(result.text).toBe("Die Rechnung R-2026-1 beträgt 42,50 €.");
    expect(result.steps).toBe(2);
    expect(result.calls).toHaveLength(1);
    expect(result.calls[0]!.output).toEqual({ number: "R-2026-1", total: 42.5 });
    // The transcript carries the call and its result, so it can be replayed to continue the conversation.
    expect(result.messages[1]).toMatchObject({ role: "assistant", toolCalls: [{ name: "lookup_invoice" }] });
    expect(result.messages[2]).toMatchObject({ role: "user", toolResults: [{ id: "c1", output: { total: 42.5 } }] });
  });

  it("hands the model JSON Schema derived from each tool's Zod schema", async () => {
    const callTools = fakeTools([{ text: "done" }]);
    await runTools(callTools, { messages: [{ role: "user", content: "?" }], tools: [lookup] });
    const sent = callTools.seen[0]!.tools[0]!;
    expect(sent.name).toBe("lookup_invoice");
    expect(sent.jsonSchema.type).toBe("object");
    expect((sent.jsonSchema.properties as Record<string, unknown>).number).toBeTruthy();
  });

  it("returns invalid arguments to the model as an error result instead of throwing", async () => {
    const callTools = fakeTools([
      { calls: [{ id: "c1", name: "lookup_invoice", input: { number: "" } }] }, // fails .min(1)
      { text: "Bitte nenne mir die Rechnungsnummer." },
    ]);
    const result = await runTools(callTools, { messages: [{ role: "user", content: "?" }], tools: [lookup] });

    expect(result.calls[0]!.error).toContain("number");
    const fedBack = callTools.seen[1]!.messages.at(-1)!;
    expect(fedBack.toolResults?.[0]!.isError).toBe(true);
    expect(String(fedBack.toolResults?.[0]!.output)).toContain("did not match the schema");
    expect(result.text).toBe("Bitte nenne mir die Rechnungsnummer.");
  });

  it("returns a handler failure to the model as an error result", async () => {
    const failing = tool({
      name: "fetch_client",
      description: "Load a client.",
      input: z.object({ id: z.number() }),
      run: () => {
        throw new Error("backend unreachable");
      },
    });
    const callTools = fakeTools([
      { calls: [{ id: "c1", name: "fetch_client", input: { id: 7 } }] },
      { text: "Die Daten sind gerade nicht abrufbar." },
    ]);
    const result = await runTools(callTools, { messages: [{ role: "user", content: "?" }], tools: [failing] });

    expect(result.calls[0]!.error).toBe("backend unreachable");
    expect(String(callTools.seen[1]!.messages.at(-1)!.toolResults?.[0]!.output)).toContain("backend unreachable");
  });

  it("tells the model when it invented a tool name", async () => {
    const callTools = fakeTools([{ calls: [{ id: "c1", name: "delete_everything", input: {} }] }, { text: "ok" }]);
    const result = await runTools(callTools, { messages: [{ role: "user", content: "?" }], tools: [lookup] });
    expect(result.calls[0]!.error).toBe("unknown tool");
    expect(String(result.calls[0]!.output)).toContain("lookup_invoice");
  });

  it("passes request-scoped context to every handler", async () => {
    const seen: unknown[] = [];
    const contextual = tool<{ q: string }, { clientId: string }>({
      name: "search",
      description: "Search.",
      input: z.object({ q: z.string() }),
      run: (_input, ctx) => {
        seen.push(ctx.context.clientId);
        return "ok";
      },
    });
    const callTools = fakeTools([{ calls: [{ id: "c1", name: "search", input: { q: "x" } }] }, { text: "done" }]);
    await runTools(callTools, { messages: [{ role: "user", content: "?" }], tools: [contextual], context: { clientId: "123" } });
    expect(seen).toEqual(["123"]);
  });

  it("runs several calls from one turn concurrently and keeps them paired by id", async () => {
    const slow = tool({
      name: "slow",
      description: "Waits.",
      input: z.object({ ms: z.number() }),
      run: async ({ ms }) => {
        await new Promise((r) => setTimeout(r, ms));
        return `waited ${ms}`;
      },
    });
    const callTools = fakeTools([
      {
        calls: [
          { id: "a", name: "slow", input: { ms: 30 } },
          { id: "b", name: "slow", input: { ms: 1 } },
        ],
      },
      { text: "both done" },
    ]);
    const started = Date.now();
    const result = await runTools(callTools, { messages: [{ role: "user", content: "?" }], tools: [slow] });

    expect(Date.now() - started).toBeLessThan(60); // concurrent, not 30 + 1 in series
    const results = result.messages.at(-1)!.toolResults!;
    expect(results.find((r) => r.id === "a")!.output).toBe("waited 30");
    expect(results.find((r) => r.id === "b")!.output).toBe("waited 1");
  });

  it("sums usage across turns and reports every invocation", async () => {
    const callTools = fakeTools([
      { calls: [{ id: "c1", name: "lookup_invoice", input: { number: "R-1" } }] },
      { calls: [{ id: "c2", name: "lookup_invoice", input: { number: "R-2" } }] },
      { text: "fertig" },
    ]);
    const seen: string[] = [];
    const result = await runTools(callTools, {
      messages: [{ role: "user", content: "?" }],
      tools: [lookup],
      onToolCall: (i) => void seen.push(i.name),
    });
    expect(result.usage.inputTokens).toBe(30); // three turns × 10
    expect(result.calls).toHaveLength(2);
    expect(seen).toEqual(["lookup_invoice", "lookup_invoice"]);
  });

  it("stops with CoaxToolError when the model never finishes", async () => {
    const callTools = fakeTools([{ calls: [{ id: "c1", name: "lookup_invoice", input: { number: "R-1" } }] }]);
    await expect(
      runTools(callTools, { messages: [{ role: "user", content: "?" }], tools: [lookup], maxSteps: 3 }),
    ).rejects.toBeInstanceOf(CoaxToolError);
  });

  it("stops before a turn once the token budget is exhausted", async () => {
    const callTools = fakeTools([{ calls: [{ id: "c1", name: "lookup_invoice", input: { number: "R-1" } }] }]);
    const budget = createBudget(20); // 14 tokens per turn → exhausted before turn 3
    await expect(
      runTools(callTools, { messages: [{ role: "user", content: "?" }], tools: [lookup], budget }),
    ).rejects.toThrow(/budget exhausted/);
  });

  it("books the completed turns on the error when maxSteps is exhausted", async () => {
    const callTools = fakeTools([{ calls: [{ id: "c1", name: "lookup_invoice", input: { number: "R-1" } }] }]);
    const err: CoaxToolError = await runTools(callTools, { messages: [{ role: "user", content: "?" }], tools: [lookup], maxSteps: 3 }).catch((e) => e);
    expect(err).toBeInstanceOf(CoaxToolError);
    expect(err.usage.inputTokens).toBe(30); // three turns × 10 — a dead run still cost tokens
    expect(err.usage.outputTokens).toBe(12);
    expect(err.steps).toBe(3);
    expect(err.calls).toHaveLength(3);
    // The transcript survives too — the audit trail of the run that died.
    expect(err.messages.at(-1)).toMatchObject({ role: "user", toolResults: [{ id: "c1" }] });
  });

  it("books the completed turns on the budget error too", async () => {
    const callTools = fakeTools([{ calls: [{ id: "c1", name: "lookup_invoice", input: { number: "R-1" } }] }]);
    const budget = createBudget(20);
    const err: CoaxToolError = await runTools(callTools, { messages: [{ role: "user", content: "?" }], tools: [lookup], budget }).catch((e) => e);
    expect(err).toBeInstanceOf(CoaxToolError);
    expect(err.usage.inputTokens).toBe(20); // two turns completed before the budget tripped
    expect(err.steps).toBe(2);
  });

  it("stops between steps once the signal aborts, booking what ran", async () => {
    const ac = new AbortController();
    const hangUp = tool({
      name: "lookup_invoice",
      description: "Fetch one invoice by number.",
      input: z.object({ number: z.string() }),
      run: () => {
        ac.abort(); // the caller hangs up while the tool is running
        return { total: 42.5 };
      },
    });
    const callTools = fakeTools([{ calls: [{ id: "c1", name: "lookup_invoice", input: { number: "R-1" } }] }]);
    const err: CoaxAbortError = await runTools(callTools, { messages: [{ role: "user", content: "?" }], tools: [hangUp], signal: ac.signal }).catch((e) => e);
    expect(err).toBeInstanceOf(CoaxAbortError);
    expect(err.usage.inputTokens).toBe(10); // the one completed turn
    expect(callTools.seen).toHaveLength(1); // no further model call was started
  });

  it("CoaxAbortError carries the transcript and completed tool calls so the run is resumable", async () => {
    const ac = new AbortController();
    const hangUp = tool({
      name: "lookup_invoice",
      description: "Fetch one invoice by number.",
      input: z.object({ number: z.string() }),
      run: () => {
        ac.abort();
        return { total: 42.5 };
      },
    });
    const callTools = fakeTools([{ calls: [{ id: "c1", name: "lookup_invoice", input: { number: "R-1" } }] }]);
    const err: CoaxAbortError = await runTools(callTools, { messages: [{ role: "user", content: "?" }], tools: [hangUp], signal: ac.signal }).catch((e) => e);
    expect(err.calls).toHaveLength(1);
    expect(err.calls[0]).toMatchObject({ name: "lookup_invoice", output: { total: 42.5 } });
    expect(err.messages.at(-1)).toMatchObject({ role: "user", toolResults: [{ id: "c1" }] });
  });

  it("wraps a provider failure in CoaxToolError, carrying the original as cause and the partial state", async () => {
    const boom = new Error("502 Bad Gateway");
    let calls = 0;
    const callTools = async () => {
      calls++;
      if (calls === 1) return { text: "", calls: [{ id: "c1", name: "lookup_invoice", input: { number: "R-1" } }], usage: { ...emptyUsage(), inputTokens: 10, outputTokens: 4 }, model: "mock-1" };
      throw boom;
    };
    const err: CoaxToolError = await runTools(callTools, { messages: [{ role: "user", content: "?" }], tools: [lookup] }).catch((e) => e);
    expect(err).toBeInstanceOf(CoaxToolError);
    expect(err.cause).toBe(boom);
    expect(err.message).toContain("502 Bad Gateway");
    expect(err.usage.inputTokens).toBe(10); // the one completed turn before the 502
    expect(err.steps).toBe(1);
    expect(err.calls).toHaveLength(1);
    expect(err.messages.at(-1)).toMatchObject({ role: "user", toolResults: [{ id: "c1" }] });
  });

  it("resolves the step-indexed toolChoice function and hands the provider only the plain value", async () => {
    const seen: (string | undefined)[] = [];
    const callTools = async (req: { messages: Message[]; tools: ToolDefinition[]; toolChoice?: string }) => {
      seen.push(req.toolChoice);
      if (seen.length === 1) return { text: "", calls: [{ id: "c1", name: "lookup_invoice", input: { number: "R-1" } }], usage: emptyUsage(), model: "mock-1" };
      return { text: "done", calls: [], usage: emptyUsage(), model: "mock-1" };
    };
    await runTools(callTools, {
      messages: [{ role: "user", content: "?" }],
      tools: [lookup],
      toolChoice: (step) => (step === 0 ? "required" : "auto"),
    });
    expect(seen).toEqual(["required", "auto"]);
  });

  it("passes a constant toolChoice through unchanged on every step", async () => {
    const seen: (string | undefined)[] = [];
    const callTools = async (req: { toolChoice?: string }) => {
      seen.push(req.toolChoice);
      return { text: "done", calls: [], usage: emptyUsage(), model: "mock-1" };
    };
    await runTools(callTools, { messages: [{ role: "user", content: "?" }], tools: [lookup], toolChoice: "none" });
    expect(seen).toEqual(["none"]);
  });

  it("leaves toolChoice undefined when the caller never sets one", async () => {
    const seen: (string | undefined)[] = [];
    const callTools = async (req: { toolChoice?: string }) => {
      seen.push(req.toolChoice);
      return { text: "done", calls: [], usage: emptyUsage(), model: "mock-1" };
    };
    await runTools(callTools, { messages: [{ role: "user", content: "?" }], tools: [lookup] });
    expect(seen).toEqual([undefined]);
  });

  describe("maxSteps: null (unlimited)", () => {
    it("throws immediately when neither a budget nor a signal brakes it", async () => {
      const callTools = fakeTools([{ text: "done" }]);
      await expect(
        runTools(callTools, { messages: [{ role: "user", content: "?" }], tools: [lookup], maxSteps: null }),
      ).rejects.toThrow(/needs a budget or a signal/);
      expect(callTools.seen).toHaveLength(0); // fails fast, before ever calling the model
    });

    it("runs unbounded with a budget, and stops once it is exhausted", async () => {
      const callTools = fakeTools([{ calls: [{ id: "c1", name: "lookup_invoice", input: { number: "R-1" } }] }]);
      const budget = createBudget(20); // 14 tokens/turn → exhausted before turn 3
      const err: CoaxToolError = await runTools(callTools, {
        messages: [{ role: "user", content: "?" }],
        tools: [lookup],
        maxSteps: null,
        budget,
      }).catch((e) => e);
      expect(err).toBeInstanceOf(CoaxToolError);
      expect(err.message).toMatch(/budget exhausted/);
      expect(err.steps).toBe(2); // stopped before starting a third turn — no artificial step ceiling applied
    });

    it("runs unbounded with only a signal, and stops once it aborts", async () => {
      const ac = new AbortController();
      const callTools = fakeTools([{ calls: [{ id: "c1", name: "lookup_invoice", input: { number: "R-1" } }] }]);
      let seen = 0;
      const countingCallTools = async (req: Parameters<typeof callTools>[0]) => {
        seen++;
        if (seen === 5) ac.abort();
        return callTools(req);
      };
      const err: CoaxAbortError = await runTools(countingCallTools, {
        messages: [{ role: "user", content: "?" }],
        tools: [lookup],
        maxSteps: null,
        signal: ac.signal,
      }).catch((e) => e);
      expect(err).toBeInstanceOf(CoaxAbortError);
      expect(seen).toBe(5); // ran well past any bounded default before the signal stopped it
    });
  });
});
