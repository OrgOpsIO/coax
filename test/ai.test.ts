import { describe, expect, it, vi } from "vitest";
import { z } from "zod";
import { createAI } from "../src/ai";
import { CoaxAbortError } from "../src/client";
import { tool } from "../src/tools";
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

  it("does not resurrect an aborted call on the fallback model", async () => {
    const ac = new AbortController();
    const asked: string[] = [];
    const factory = (model: string): Provider => ({
      name: "mock",
      model,
      async structured() {
        asked.push(model);
        ac.abort(); // the caller hangs up mid-call; the SDK throws its abort error
        throw Object.assign(new Error("Request was aborted."), { name: "APIUserAbortError" });
      },
      async text() {
        return { raw: "", text: "", usage: emptyUsage(), model };
      },
    });
    const ai = createAI({
      providers: { mock: factory },
      models: { smart: { use: "mock:primary", fallback: "mock:backup" } },
      defaults: { retries: { attempts: 1 } },
    });
    await expect(ai.object({ model: "smart", schema: Out, prompt: "?", signal: ac.signal })).rejects.toBeInstanceOf(CoaxAbortError);
    expect(asked).toEqual(["primary"]); // the backup was never asked to redo the cancelled work
  });
});

describe("createAI().run", () => {
  // The backend-for-frontend shape: the model asks for data, the tool fetches it with credentials the
  // model never sees, and the caller's identity rides along on the request headers.
  it("drives a tool-calling run and forwards identity headers to the endpoint", async () => {
    const turns = [
      { text: "", calls: [{ id: "c1", name: "open_invoices", input: { clientId: "123" } }] },
      { text: "Sie haben eine offene Rechnung über 42,50 €.", calls: [] },
    ];
    const headersSeen: (Record<string, string> | undefined)[] = [];
    let turn = 0;
    const provider: Provider = {
      name: "mock",
      model: "chat/Qwen3-VL-32B",
      structured: async () => ({ raw: {}, text: "{}", usage: emptyUsage(), model: "chat/Qwen3-VL-32B" }),
      text: async () => ({ raw: "", text: "", usage: emptyUsage(), model: "chat/Qwen3-VL-32B" }),
      async tools(req) {
        headersSeen.push(req.headers);
        return { ...turns[turn++]!, usage: { ...emptyUsage(), inputTokens: 9 }, model: "chat/Qwen3-VL-32B" };
      },
    };

    const openInvoices = tool<{ clientId: string }, { token: string }>({
      name: "open_invoices",
      description: "List a client's open invoices.",
      input: z.object({ clientId: z.string() }),
      run: ({ clientId }, ctx) => [{ id: "R-1", total: 42.5, requestedBy: clientId, auth: ctx.context.token }],
    });

    const ai = createAI({ providers: { orgops: () => provider }, models: { local: "orgops:chat/Qwen3-VL-32B" } });
    const result = await ai.run({
      model: "local",
      system: "Du bist der Assistent im Kundenportal.",
      prompt: "Habe ich offene Rechnungen?",
      tools: [openInvoices],
      context: { token: "server-side-secret" },
      headers: { Authorization: "Bearer end-user-token" },
    });

    expect(result.text).toContain("42,50");
    expect(result.calls).toHaveLength(1);
    expect(result.usage.inputTokens).toBe(18); // two turns × 9
    // The end user's token reached the gateway on every turn — so a policy engine behind it can decide.
    expect(headersSeen).toEqual([{ Authorization: "Bearer end-user-token" }, { Authorization: "Bearer end-user-token" }]);
  });

  it("reports a provider that cannot do tool calling", async () => {
    const { factory } = scripted({ m: [] });
    const ai = createAI({ providers: { mock: factory }, models: { a: "mock:m" } });
    await expect(
      ai.run({ model: "a", prompt: "?", tools: [tool({ name: "x", description: "x", input: z.object({}), run: () => 1 })] }),
    ).rejects.toThrow(/does not support tool calling/);
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
