import { describe, expect, it } from "vitest";
import { z } from "zod";
import { createClient, CoaxAbortError, CoaxSchemaError } from "../src/client";
import { emptyUsage, type Provider, type ProviderResponse, type StructuredRequest, type TextRequest } from "../src/types";

// A scripted provider: returns queued raw outputs in order, recording the requests it saw.
function mockProvider(queue: unknown[]): Provider & { calls: StructuredRequest[] } {
  const calls: StructuredRequest[] = [];
  let i = 0;
  return {
    name: "mock",
    model: "mock-1",
    calls,
    async structured(req: StructuredRequest): Promise<ProviderResponse> {
      calls.push(req);
      const raw = queue[i++];
      return { raw, text: typeof raw === "string" ? raw : JSON.stringify(raw), usage: { ...emptyUsage(), inputTokens: 10, outputTokens: 5 }, model: "mock-1" };
    },
    async text(_req: TextRequest): Promise<ProviderResponse> {
      return { raw: "hi", text: "hi", usage: emptyUsage(), model: "mock-1" };
    },
  };
}

const Field = z.object({ label: z.string(), helpText: z.string().min(1) });

describe("createClient().object", () => {
  it("returns typed, validated data on a valid first response", async () => {
    const provider = mockProvider([{ label: "Name", helpText: "Ihr Name." }]);
    const client = createClient({ provider });
    const { data, repairs } = await client.object({ schema: Field, prompt: "..." });
    expect(data).toEqual({ label: "Name", helpText: "Ihr Name." });
    expect(repairs).toBe(0);
  });

  it("aggressively parses a fenced + malformed response", async () => {
    const provider = mockProvider(["```json\n{label:'Name', helpText:'Ihr Name.',}\n```"]);
    const client = createClient({ provider });
    const { data } = await client.object({ schema: Field, prompt: "..." });
    expect(data).toEqual({ label: "Name", helpText: "Ihr Name." });
  });

  it("reprompts with the validation error, then succeeds (repairs=1)", async () => {
    const provider = mockProvider([
      { label: "Name", helpText: "" }, // helpText fails .min(1)
      { label: "Name", helpText: "Ihr Name." },
    ]);
    const client = createClient({ provider });
    const { data, repairs } = await client.object({ schema: Field, prompt: "..." });
    expect(data.helpText).toBe("Ihr Name.");
    expect(repairs).toBe(1);
    // The repair turn fed the concrete failure back to the model.
    const repairMsg = provider.calls[1]!.messages.at(-1)!;
    expect(repairMsg.role).toBe("user");
    expect(repairMsg.content).toContain("helpText");
  });

  it("sums usage across the initial call + repair rounds", async () => {
    const provider = mockProvider([{ label: "x", helpText: "" }, { label: "x", helpText: "ok" }]);
    const client = createClient({ provider });
    const { usage } = await client.object({ schema: Field, prompt: "..." });
    expect(usage.inputTokens).toBe(20); // two calls × 10
    expect(usage.outputTokens).toBe(10);
  });

  it("throws CoaxSchemaError after exhausting repairs", async () => {
    const provider = mockProvider([{ bad: 1 }, { bad: 2 }, { bad: 3 }]);
    const client = createClient({ provider });
    await expect(client.object({ schema: Field, prompt: "...", maxRepairs: 2 })).rejects.toBeInstanceOf(CoaxSchemaError);
  });

  it("discriminated unions work (agent-loop step)", async () => {
    const Step = z.discriminatedUnion("action", [
      z.object({ action: z.literal("fetch"), sections: z.array(z.string()) }),
      z.object({ action: z.literal("define"), value: z.number() }),
    ]);
    const provider = mockProvider([{ action: "fetch", sections: ["a", "b"] }]);
    const client = createClient({ provider });
    const { data } = await client.object({ schema: Step, prompt: "..." });
    expect(data).toEqual({ action: "fetch", sections: ["a", "b"] });
  });

  it("wraps a non-object root schema in an object envelope (provider tool roots must be objects)", async () => {
    const Step = z.discriminatedUnion("action", [
      z.object({ action: z.literal("fetch"), sections: z.array(z.string()) }),
      z.object({ action: z.literal("select"), candidate: z.number() }),
    ]);
    const provider = mockProvider([{ value: { action: "select", candidate: 2 } }]);
    const client = createClient({ provider });
    const { data } = await client.object({ schema: Step, prompt: "..." });
    // The tool schema handed to the provider is an object with a single `value` property...
    const sent = provider.calls[0]!.jsonSchema as Record<string, unknown>;
    expect(sent.type).toBe("object");
    expect((sent.properties as Record<string, unknown>).value).toBeTruthy();
    // ...and the envelope is unwrapped before validation.
    expect(data).toEqual({ action: "select", candidate: 2 });
  });

  it("leaves an object-root schema unwrapped", async () => {
    const provider = mockProvider([{ label: "Name", helpText: "Ihr Name." }]);
    const client = createClient({ provider });
    await client.object({ schema: Field, prompt: "..." });
    const sent = provider.calls[0]!.jsonSchema as Record<string, unknown>;
    expect(sent.type).toBe("object");
    expect((sent.properties as Record<string, unknown>).label).toBeTruthy();
    expect((sent.properties as Record<string, unknown>).value).toBeUndefined();
  });
});

describe("createClient().object — failure accounting & abort", () => {
  it("CoaxSchemaError carries the usage the failed attempts cost", async () => {
    const provider = mockProvider([{ bad: 1 }, { bad: 2 }, { bad: 3 }]);
    const client = createClient({ provider });
    const err: CoaxSchemaError = await client.object({ schema: Field, prompt: "...", maxRepairs: 2 }).catch((e) => e);
    expect(err).toBeInstanceOf(CoaxSchemaError);
    expect(err.usage.inputTokens).toBe(30); // three attempts × 10
    expect(err.usage.outputTokens).toBe(15);
  });

  it("CoaxSchemaError carries the transcript, including every repair reprompt", async () => {
    const provider = mockProvider([{ bad: 1 }, { bad: 2 }, { bad: 3 }]);
    const client = createClient({ provider });
    const err: CoaxSchemaError = await client.object({ schema: Field, prompt: "start here", maxRepairs: 2 }).catch((e) => e);
    expect(err.messages[0]).toMatchObject({ role: "user", content: "start here" });
    // maxRepairs: 2 → 3 attempts total, all invalid → 3 (assistant reply + reprompt) pairs appended
    // after the initial user turn.
    expect(err.messages).toHaveLength(7);
    expect(err.messages.filter((m) => m.role === "user" && m.content.includes("did not match"))).toHaveLength(3);
  });

  it("an already-aborted signal fails fast with CoaxAbortError, without calling the provider", async () => {
    const provider = mockProvider([{ label: "x", helpText: "y" }]);
    const client = createClient({ provider });
    const ac = new AbortController();
    ac.abort();
    await expect(client.object({ schema: Field, prompt: "...", signal: ac.signal })).rejects.toBeInstanceOf(CoaxAbortError);
    expect(provider.calls).toHaveLength(0);
  });

  it("normalizes an SDK abort mid-repair to CoaxAbortError carrying the usage spent so far", async () => {
    const ac = new AbortController();
    const inner = mockProvider([{ label: "x", helpText: "" }]); // first attempt: invalid → repair round
    const provider: typeof inner = {
      ...inner,
      async structured(req) {
        if (inner.calls.length === 1) {
          // Second attempt: the user hangs up; the SDK throws its own abort error.
          ac.abort();
          throw Object.assign(new Error("Request was aborted."), { name: "APIUserAbortError" });
        }
        return inner.structured(req);
      },
    };
    const client = createClient({ provider });
    const err: CoaxAbortError = await client.object({ schema: Field, prompt: "...", signal: ac.signal }).catch((e) => e);
    expect(err).toBeInstanceOf(CoaxAbortError);
    expect(err.usage.inputTokens).toBe(10); // the completed first attempt is still booked
  });
});
