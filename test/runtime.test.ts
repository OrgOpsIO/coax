import { afterEach, describe, expect, it } from "vitest";
import { z } from "zod";
import { ai, configure, isConfigured, reset } from "../src/runtime";
import { emptyUsage, type Provider } from "../src/types";

const mock = (raw: unknown): ((model: string) => Provider) => (model) => ({
  name: "mock",
  model,
  async structured(req) {
    return { raw, text: JSON.stringify(raw), usage: { ...emptyUsage(), inputTokens: 3 }, model };
  },
  async text() {
    return { raw: "hi", text: "hi", usage: emptyUsage(), model };
  },
});

const Out = z.object({ ok: z.boolean() });

afterEach(() => reset());

describe("ambient ai", () => {
  it("throws a clear error before configure()", async () => {
    expect(isConfigured()).toBe(false);
    await expect(ai.object({ model: "m", schema: Out, prompt: "?" })).rejects.toThrow(/not configured/);
  });

  it("works app-wide after configure() — no instance threading", async () => {
    configure({ providers: { mock: mock({ ok: true }) }, models: { smart: "mock:x" }, defaults: { model: "smart" } });
    expect(isConfigured()).toBe(true);
    const { data } = await ai.object({ schema: Out, prompt: "?" });
    expect(data.ok).toBe(true);
  });

  it("reset() clears the ambient instance", async () => {
    configure({ providers: { mock: mock({ ok: true }) }, defaults: { model: "mock:x" } });
    reset();
    await expect(ai.text({ prompt: "?" })).rejects.toThrow(/not configured/);
  });
});
