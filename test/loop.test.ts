import { describe, expect, it } from "vitest";
import { z } from "zod";
import { runLoop, CoaxLoopError } from "../src/loop";
import { createBudget } from "../src/budget";
import { emptyUsage } from "../src/types";
import type { ObjectResult } from "../src/client";

const Step = z.discriminatedUnion("action", [
  z.object({ action: z.literal("fetch"), query: z.string() }),
  z.object({ action: z.literal("answer"), text: z.string() }),
]);
type StepT = z.infer<typeof Step>;

// A fake `object` fn returning queued steps, so the loop is testable without a provider.
function fakeObject(queue: StepT[], usagePerCall = 10) {
  let i = 0;
  return async (): Promise<ObjectResult<StepT>> => ({
    data: queue[Math.min(i++, queue.length - 1)]!,
    usage: { ...emptyUsage(), inputTokens: usagePerCall },
    model: "mock",
    repairs: 0,
  });
}

describe("runLoop", () => {
  it("drives a fetch→answer loop and returns the value", async () => {
    const seen: string[] = [];
    const object = fakeObject([
      { action: "fetch", query: "weather" },
      { action: "answer", text: "sunny" },
    ]);
    const result = await runLoop<StepT, string>(object as never, {
      schema: Step,
      messages: [{ role: "user", content: "?" }],
      onStep: async (step) => {
        if (step.action === "answer") return { done: true, value: step.text };
        seen.push(step.query);
        return { done: false, reply: `result for ${step.query}` };
      },
    });
    expect(result).toBe("sunny");
    expect(seen).toEqual(["weather"]);
  });

  it("doom guard: throws when the same step repeats", async () => {
    const object = fakeObject([{ action: "fetch", query: "x" }]); // always the same
    await expect(
      runLoop<StepT, string>(object as never, {
        schema: Step,
        messages: [{ role: "user", content: "?" }],
        maxRepeat: 3,
        onStep: async () => ({ done: false, reply: "again" }),
      }),
    ).rejects.toBeInstanceOf(CoaxLoopError);
  });

  it("stops when the token budget is exhausted", async () => {
    const object = fakeObject([{ action: "fetch", query: "x" }], 60);
    const budget = createBudget(100);
    await expect(
      runLoop<StepT, string>(object as never, {
        schema: Step,
        messages: [{ role: "user", content: "?" }],
        budget,
        onStep: async () => ({ done: false, reply: "more" }),
      }),
    ).rejects.toThrow(/budget/);
  });

  it("throws after maxTurns without finishing", async () => {
    const object = fakeObject([{ action: "fetch", query: "x" }]);
    let n = 0;
    await expect(
      runLoop<StepT, string>(object as never, {
        schema: Step,
        messages: [{ role: "user", content: "?" }],
        maxTurns: 2,
        maxRepeat: 99,
        onStep: async () => ({ done: false, reply: `r${n++}` }), // distinct replies, never done
      }),
    ).rejects.toThrow(/within 2 turns/);
  });
});

describe("createBudget", () => {
  it("tracks spend and remaining", () => {
    const b = createBudget(100);
    b.record({ ...emptyUsage(), inputTokens: 30, outputTokens: 10 });
    expect(b.spent()).toBe(40);
    expect(b.remaining()).toBe(60);
    expect(b.over()).toBe(false);
    b.record({ ...emptyUsage(), inputTokens: 70 });
    expect(b.over()).toBe(true);
  });

  it("no limit → never over, infinite remaining", () => {
    const b = createBudget(null);
    b.record({ ...emptyUsage(), inputTokens: 1_000_000 });
    expect(b.over()).toBe(false);
    expect(b.remaining()).toBe(Infinity);
  });
});
