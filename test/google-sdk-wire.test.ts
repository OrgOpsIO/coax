import { readFileSync } from "node:fs";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";
import { createAI } from "../src/ai";
import { CoaxAbortError } from "../src/client";
import { tool } from "../src/tools";
import { CoaxRefusalError, type Usage } from "../src/types";

// Measurer (stage 1, run 2): every other google test injects a fake SDK client, so what the REAL
// @google/genai makes of coax's params — URL, auth header, REST body, SSE parsing, retries, abort — was
// only ever read in the SDK's source. Here the installed SDK runs for real and only `fetch` is stubbed,
// answering from the same fixtures. No network: an unexpected request fails the test, and the ADC route
// gets a stub auth client (google-auth-library fetches tokens through node-fetch, not the global fetch).

const load = (name: string) => JSON.parse(readFileSync(new URL(`./fixtures/google/${name}`, import.meta.url), "utf8"));
const fixture = (name: string) => load(name).fixture;

type Sent = { url: string; headers: Record<string, string>; body: any; signal?: AbortSignal | null };
type Responder = (sent: Sent) => Response | Promise<Response>;

let sent: Sent[] = [];
let replies: Responder[] = [];
const reply = (...r: Responder[]) => void (replies = r);
const json = (body: unknown, status = 200): Responder => () => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
// Server-sent events as the SDK reads them (`data: <json>` blocks separated by a blank line).
const sse = (chunks: unknown[]): Responder => () =>
  new Response(chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join(""), { status: 200, headers: { "content-type": "text/event-stream" } });

beforeEach(() => {
  sent = [];
  replies = [];
  vi.stubGlobal("fetch", async (input: string | URL | Request, init: RequestInit = {}) => {
    const s: Sent = {
      url: String(input instanceof Request ? input.url : input),
      headers: Object.fromEntries(new Headers(init.headers).entries()),
      body: typeof init.body === "string" ? JSON.parse(init.body) : init.body,
      signal: init.signal,
    };
    sent.push(s);
    const next = replies.shift();
    if (!next) throw new Error(`test: unexpected request ${s.url}`);
    return next(s);
  });
});
afterEach(() => void vi.unstubAllGlobals());

const KEY = "test-key";
const keyAI = (extra: Record<string, unknown> = {}, onUsage?: (u: Usage) => void) =>
  createAI({
    providers: { google: { apiKey: KEY, ...extra } },
    models: { flash: "google:gemini-3.5-flash" },
    defaults: { retries: { attempts: 2, initialDelayMs: 1, maxDelayMs: 1 } },
    ...(onUsage ? { onUsage } : {}),
  });

describe("google through the real SDK (fetch stubbed)", () => {
  it("key route: global host, key in a header, REST body with extraBody merged deep under generationConfig", async () => {
    reply(json(fixture("text.response.json")));
    const ai = keyAI({ headers: { "x-team": "a", "x-both": "endpoint" }, extraBody: { generationConfig: { temperature: 0.6, topP: 0.9 }, labels: { team: "a" } } });
    const res = await ai.text({
      model: "flash",
      system: "Be brief.",
      prompt: "Haiku about types",
      reasoningEffort: "none",
      headers: { "x-both": "call" },
      extraBody: { generationConfig: { temperature: 0.2 } },
    });
    const [s] = sent;
    expect(s!.url).toMatch(/^https:\/\/aiplatform\.googleapis\.com\/v1[a-z0-9]*\/publishers\/google\/models\/gemini-3\.5-flash:generateContent$/);
    expect(s!.headers["x-goog-api-key"]).toBe(KEY);
    expect(s!.url).not.toContain(KEY);
    expect(s!.headers).toMatchObject({ "x-team": "a", "x-both": "call" });
    expect(s!.body).toEqual({
      contents: [{ role: "user", parts: [{ text: "Haiku about types" }] }],
      systemInstruction: { role: "user", parts: [{ text: "Be brief." }] },
      generationConfig: { maxOutputTokens: 8192, thinkingConfig: { thinkingLevel: "MINIMAL" }, temperature: 0.2, topP: 0.9 },
      labels: { team: "a" },
    });
    expect(res.text).toBe("Types guard the gate,\nerrors caught before they run,\nquiet builds at dawn.");
    expect(res.usage).toEqual({ inputTokens: 5000, outputTokens: 85, cacheReadTokens: 4096, cacheWriteTokens: 0 });
  });

  it("structured output lands as generationConfig.responseJsonSchema (not converted to responseSchema)", async () => {
    reply(json({ candidates: [{ content: { role: "model", parts: [{ text: '{"title":"Typed","kind":"article"}' }] }, finishReason: "STOP" }] }));
    const res = await keyAI().object({ model: "flash", schema: z.object({ title: z.string().min(3), kind: z.literal("article") }), prompt: "?" });
    expect(res.data).toEqual({ title: "Typed", kind: "article" });
    const g = sent[0]!.body.generationConfig;
    expect(g.responseMimeType).toBe("application/json");
    expect(g.responseSchema).toBeUndefined();
    expect(g.responseJsonSchema.properties).toEqual({ title: { type: "string" }, kind: { type: "string", enum: ["article"] } });
  });

  it("stream: streamGenerateContent?alt=sse, the SDK's SSE parser, usage from the last chunk", async () => {
    reply(sse(fixture("stream-text.chunks.json")));
    const { stream, result } = await keyAI().stream({ model: "flash", prompt: "?" });
    const deltas: string[] = [];
    for await (const d of stream) deltas.push(d);
    expect(sent[0]!.url).toMatch(/:streamGenerateContent\?alt=sse$/);
    expect(deltas).toEqual(["Types guard ", "the gate,"]);
    expect((await result).usage).toEqual({ inputTokens: 9, outputTokens: 46, cacheReadTokens: 0, cacheWriteTokens: 0 });
  });

  it("ai.run: the second request replays the signed model turn verbatim and answers both calls in one content", async () => {
    const temperature = tool({
      name: "get_current_temperature",
      description: "Current temperature for a city.",
      input: z.object({ location: z.string() }),
      run: ({ location }) => {
        if (location === "London") throw new Error("upstream timeout");
        return { celsius: 21 };
      },
    });
    reply(json(fixture("tools-parallel.response.json")), json(fixture("text.response.json")));
    const res = await keyAI().run({ model: "flash", prompt: "Paris and London?", tools: [temperature], toolChoice: "required" });
    expect(sent[0]!.body.toolConfig).toEqual({ functionCallingConfig: { mode: "ANY" } });
    expect(sent[0]!.body.tools[0].functionDeclarations[0].parametersJsonSchema).toEqual({
      type: "object",
      properties: { location: { type: "string" } },
      required: ["location"],
    });
    const contents = sent[1]!.body.contents;
    expect(contents[1]).toEqual(fixture("tools-parallel.response.json").candidates[0].content);
    expect(contents[2].role).toBe("user");
    expect(contents[2].parts.map((p: any) => p.functionResponse.id)).toEqual(["call-paris-1", "call-london-2"]);
    expect(contents[2].parts[0].functionResponse.response).toEqual({ output: { celsius: 21 } });
    expect(Object.keys(contents[2].parts[1].functionResponse.response)).toEqual(["error"]);
    expect(res.usage).toEqual({ inputTokens: 5080, outputTokens: 221, cacheReadTokens: 4096, cacheWriteTokens: 0 });
  });

  it("ai.runStream: every streamed part goes back, the empty-text signature part included", async () => {
    const lookup = tool({ name: "lookup", description: "Look something up.", input: z.object({ q: z.string() }), run: () => "42" });
    reply(sse(fixture("tools-stream.chunks.json")), sse(fixture("stream-text.chunks.json")));
    const { events, result } = await keyAI().runStream({ model: "flash", prompt: "?", tools: [lookup] });
    for await (const _ of events);
    expect((await result).text).toBe("Types guard the gate,");
    const parts = fixture("tools-stream.chunks.json").flatMap((c: any) => c.candidates[0].content.parts);
    expect(sent[1]!.body.contents[1]).toEqual({ role: "model", parts });
  });

  it("another vendor's history carries the placeholder signature and no ids on the wire", async () => {
    reply(json(fixture("text.response.json")));
    await keyAI().text({
      model: "flash",
      messages: [
        { role: "user", content: "q" },
        { role: "assistant", content: "", toolCalls: [{ id: "toolu_01", name: "lookup", input: {} }] },
        { role: "user", content: "", toolResults: [{ id: "toolu_01", name: "lookup", output: "42" }] },
      ],
    });
    expect(sent[0]!.body.contents.slice(1)).toEqual([
      { role: "model", parts: [{ functionCall: { name: "lookup", args: {} }, thoughtSignature: "skip_thought_signature_validator" }] },
      { role: "user", parts: [{ functionResponse: { name: "lookup", response: { output: "42" } } }] },
    ]);
  });

  it("a blocked prompt (HTTP 200) is a CoaxRefusalError whose billed usage reaches onUsage", async () => {
    const seen: Usage[] = [];
    reply(json(fixture("blocked-prompt.response.json")));
    const err = await keyAI({}, (u) => void seen.push(u)).text({ model: "flash", prompt: "?" }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(CoaxRefusalError);
    expect(err).toMatchObject({ category: "PROHIBITED_CONTENT", usage: { inputTokens: 7 } });
    expect(seen).toEqual([{ inputTokens: 7, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0 }]);
  });

  it("one retry layer: a 429 is retried by coax only (2 attempts → 2 requests); a 400 is not retried", async () => {
    const e = fixture("errors.json");
    const rateLimited = json(JSON.parse(e.rateLimited.message), 429);
    reply(rateLimited, rateLimited, rateLimited);
    await expect(keyAI().text({ model: "flash", prompt: "?" })).rejects.toMatchObject({ status: 429 });
    expect(sent).toHaveLength(2);

    sent = [];
    const invalid = json(JSON.parse(e.missingThoughtSignature.message), 400);
    reply(invalid, invalid);
    await expect(keyAI().text({ model: "flash", prompt: "?" })).rejects.toMatchObject({ status: 400 });
    expect(sent).toHaveLength(1);
  });

  it("an abort reaches fetch and the call rejects with CoaxAbortError", async () => {
    reply((s) => new Promise<Response>((_, reject) => s.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))));
    const ctrl = new AbortController();
    setTimeout(() => ctrl.abort(), 5);
    await expect(keyAI().text({ model: "flash", prompt: "?", signal: ctrl.signal })).rejects.toBeInstanceOf(CoaxAbortError);
    expect(sent[0]!.signal).toBeDefined();
  });

  it("ADC: generate and embed go to the regional host under projects/…/locations/…; predict's token_count is the usage", async () => {
    const authClient = { getRequestHeaders: async () => new Headers({ authorization: "Bearer test-token" }) };
    const ai = createAI({
      providers: { google: { project: "p", location: "europe-west4", googleAuthOptions: { authClient } } },
      models: { flash: "google:gemini-3.5-flash", emb: "google:gemini-embedding-001" },
    });
    const predict = fixture("embed-predict.rest.json").response;
    reply(json(fixture("text.response.json")), json(predict), json(predict));
    await ai.text({ model: "flash", prompt: "?" });
    const res = await ai.embed({ model: "emb", input: ["a", "b"] });
    const base = /^https:\/\/europe-west4-aiplatform\.googleapis\.com\/v1[a-z0-9]*\/projects\/p\/locations\/europe-west4\/publishers\/google\/models\//;
    expect(sent[0]!.url).toMatch(new RegExp(base.source + /gemini-3\.5-flash:generateContent$/.source));
    expect(sent[1]!.url).toMatch(new RegExp(base.source + /gemini-embedding-001:predict$/.source));
    expect(sent.map((s) => s.headers.authorization)).toEqual(["Bearer test-token", "Bearer test-token", "Bearer test-token"]);
    expect(sent.slice(1).map((s) => s.body.instances)).toEqual([[{ content: "a" }], [{ content: "b" }]]);
    expect(res.embeddings).toEqual([predict.predictions[0].embeddings.values, predict.predictions[0].embeddings.values]);
    expect(res.usage.inputTokens).toBe(8);
  });
});
