import { describe, expect, it } from "vitest";
import { createAI } from "../src/ai";
import { emptyUsage, type Provider } from "../src/types";

// Capture what the judge sends + return a scripted score.
function judgeProvider(score: number) {
  const seen: { system?: string; user?: string } = {};
  const factory = (model: string): Provider => ({
    name: "mock",
    model,
    async structured(req) {
      seen.system = req.system;
      seen.user = typeof req.messages[0]?.content === "string" ? req.messages[0]!.content : "";
      return { raw: { score, rationale: "because" }, text: "", usage: emptyUsage(), model };
    },
    async text() {
      return { raw: "", text: "", usage: emptyUsage(), model };
    },
  });
  return { factory, seen };
}

describe("ai.judge", () => {
  it("passes when the score meets the default threshold (midpoint of 1-5 = 3)", async () => {
    const { factory } = judgeProvider(4);
    const ai = createAI({ providers: { mock: factory }, defaults: { model: "mock:x" } });
    const j = await ai.judge({ output: "The answer is 42.", criteria: "Is it correct and concise?" });
    expect(j.score).toBe(4);
    expect(j.pass).toBe(true);
    expect(j.rationale).toBe("because");
  });

  it("fails below the threshold", async () => {
    const { factory } = judgeProvider(2);
    const ai = createAI({ providers: { mock: factory }, defaults: { model: "mock:x" } });
    const j = await ai.judge({ output: "meh", criteria: ["accurate", "helpful"] });
    expect(j.pass).toBe(false);
  });

  it("honors a custom passScore + scale, and formats multiple criteria", async () => {
    const { factory, seen } = judgeProvider(7);
    const ai = createAI({ providers: { mock: factory }, defaults: { model: "mock:x" } });
    const j = await ai.judge({ output: "x", criteria: ["clear", "on-brand", "no PII"], scale: [0, 10], passScore: 8 });
    expect(j.pass).toBe(false); // 7 < 8
    expect(seen.user).toContain("1. clear");
    expect(seen.user).toContain("3. no PII");
    expect(seen.system).toContain("0-10");
  });
});
