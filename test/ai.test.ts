import { describe, expect, it, vi } from "vitest";
import { z } from "zod";
import { createAI } from "../src/ai";
import { parsePrompt, renderTemplate } from "../src/prompt-file";
import { emptyUsage, type Provider } from "../src/types";

// A custom provider factory — proves coax works with ANY provider (not just anthropic/openai) and lets
// us test the config layer without an API key. `queue` is keyed by model so fallback is observable.
function scripted(byModel: Record<string, unknown[]>) {
  const seen: string[] = [];
  const factory = (model: string): Provider => {
    let i = 0;
    return {
      name: "mock",
      model,
      async structured(req) {
        seen.push(`${model}:${req.schemaName}`);
        const q = byModel[model] ?? [];
        const raw = q[i++];
        if (raw instanceof Error) throw raw;
        return { raw, text: JSON.stringify(raw), usage: { ...emptyUsage(), inputTokens: 7 }, model };
      },
      async text() {
        seen.push(`${model}:text`);
        return { raw: "ok", text: "ok", usage: emptyUsage(), model };
      },
    };
  };
  return { factory, seen };
}

const Out = z.object({ answer: z.string().min(1) });

describe("createAI", () => {
  it("resolves a model alias and returns typed data", async () => {
    const { factory } = scripted({ "claude-sonnet-4-6": [{ answer: "42" }] });
    const ai = createAI({
      providers: { mock: factory },
      models: { smart: "mock:claude-sonnet-4-6" },
    });
    const { data } = await ai.object({ model: "smart", schema: Out, prompt: "?" });
    expect(data.answer).toBe("42");
  });

  it("uses defaults.model when none is given", async () => {
    const { factory } = scripted({ m: [{ answer: "hi" }] });
    const ai = createAI({ providers: { mock: factory }, models: { def: "mock:m" }, defaults: { model: "def" } });
    const { data } = await ai.object({ schema: Out, prompt: "?" });
    expect(data.answer).toBe("hi");
  });

  it("falls back to the fallback model when the primary throws", async () => {
    const { factory, seen } = scripted({
      primary: [new Error("boom")],
      backup: [{ answer: "recovered" }],
    });
    const ai = createAI({
      providers: { mock: factory },
      models: { smart: { use: "mock:primary", fallback: "mock:backup" } },
      defaults: { retries: { attempts: 1 } },
    });
    const { data } = await ai.object({ model: "smart", schema: Out, prompt: "?" });
    expect(data.answer).toBe("recovered");
    expect(seen).toContain("primary:output");
    expect(seen).toContain("backup:output");
  });

  it("fires onUsage with resolved meta", async () => {
    const onUsage = vi.fn();
    const { factory } = scripted({ m: [{ answer: "x" }] });
    const ai = createAI({ providers: { mock: factory }, models: { a: "mock:m" }, onUsage });
    await ai.object({ model: "a", schema: Out, prompt: "?", purpose: "extraction" });
    expect(onUsage).toHaveBeenCalledWith(expect.objectContaining({ inputTokens: 7 }), expect.objectContaining({ provider: "mock", alias: "a", purpose: "extraction" }));
  });

  it("retries transient errors then succeeds", async () => {
    const transient = Object.assign(new Error("rate limited"), { status: 429 });
    const { factory } = scripted({ m: [transient, { answer: "ok" }] });
    const ai = createAI({ providers: { mock: factory }, models: { a: "mock:m" }, defaults: { retries: { attempts: 3, initialDelayMs: 1 } } });
    const { data } = await ai.object({ model: "a", schema: Out, prompt: "?" });
    expect(data.answer).toBe("ok");
  });
});

describe("prompt files", () => {
  it("parses frontmatter + SYSTEM/USER sections", () => {
    const p = parsePrompt(`---\nmodel: smart\nmaxRepairs: 3\n---\n# SYSTEM\nYou are an expert at {{ domain }}.\n\n# USER\n{{ input }}\n`);
    expect(p.meta).toEqual({ model: "smart", maxRepairs: 3 });
    expect(p.system).toBe("You are an expert at {{ domain }}.");
    expect(p.user).toBe("{{ input }}");
  });

  it("treats a body with no sections as the user prompt", () => {
    const p = parsePrompt(`Summarize: {{ doc }}`);
    expect(p.system).toBeUndefined();
    expect(p.user).toBe("Summarize: {{ doc }}");
  });

  it("renders nested vars and blanks missing ones", () => {
    expect(renderTemplate("Hi {{ user.name }} / {{ missing }}", { user: { name: "Rebar" } })).toBe("Hi Rebar / ");
  });
});
