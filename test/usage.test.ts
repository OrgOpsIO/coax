import { describe, expect, it } from "vitest";
import { addUsage, emptyUsage, type Usage } from "../src/types";
import { createBudget } from "../src/budget";

const tokens = (inputTokens: number, outputTokens: number): Usage => ({ ...emptyUsage(), inputTokens, outputTokens });

describe("Usage beyond tokens (characters, audio seconds)", () => {
  it("emptyUsage() keeps exactly the four token fields", () => {
    expect(emptyUsage()).toStrictEqual({ inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0 });
  });

  it("a sum of token-only usages has no characters/audioSeconds keys", () => {
    expect(addUsage(tokens(3, 2), { inputTokens: 1, outputTokens: 1, cacheReadTokens: 4, cacheWriteTokens: 5 })).toStrictEqual({
      inputTokens: 4,
      outputTokens: 3,
      cacheReadTokens: 4,
      cacheWriteTokens: 5,
    });
  });

  it("carries characters when one side has them, and sums them when both do", () => {
    expect(addUsage({ ...emptyUsage(), characters: 12 }, emptyUsage())).toStrictEqual({ ...emptyUsage(), characters: 12 });
    expect(addUsage(emptyUsage(), { ...emptyUsage(), characters: 12 })).toStrictEqual({ ...emptyUsage(), characters: 12 });
    expect(addUsage({ ...emptyUsage(), characters: 12 }, { ...emptyUsage(), characters: 3 })).toStrictEqual({ ...emptyUsage(), characters: 15 });
  });

  it("carries audioSeconds the same way", () => {
    expect(addUsage({ ...emptyUsage(), audioSeconds: 1.25 }, emptyUsage())).toStrictEqual({ ...emptyUsage(), audioSeconds: 1.25 });
    expect(addUsage(emptyUsage(), { ...emptyUsage(), audioSeconds: 2 })).toStrictEqual({ ...emptyUsage(), audioSeconds: 2 });
    expect(addUsage({ ...emptyUsage(), audioSeconds: 1.25 }, { ...emptyUsage(), audioSeconds: 2 })).toStrictEqual({ ...emptyUsage(), audioSeconds: 3.25 });
  });

  it("keeps the two units apart and next to the tokens", () => {
    expect(addUsage({ ...tokens(1, 1), characters: 5 }, { ...tokens(2, 0), audioSeconds: 4 })).toStrictEqual({
      ...tokens(3, 1),
      characters: 5,
      audioSeconds: 4,
    });
  });

  it("a Budget counts tokens only — characters and seconds are not tokens", () => {
    const budget = createBudget(100);
    budget.record({ ...emptyUsage(), characters: 500, audioSeconds: 60 });
    expect(budget.spent()).toBe(0);
    budget.record({ ...tokens(3, 2), characters: 500 });
    expect(budget.spent()).toBe(5);
  });
});
