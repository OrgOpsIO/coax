import { beforeAll, afterEach, describe, expect, it, vi } from "vitest";
import { createAI } from "../src/ai";
import { elevenlabs } from "../src/providers/elevenlabs";

// Measurer (stage 2): the README configures ElevenLabs through the registry — `providers: { elevenlabs:
// { apiKey, voice } }`, a second account with `api: "elevenlabs"` + `baseURL` — not through a factory with an
// injected client. These tests run that path end to end: registry → provider → the REAL SDK the provider
// constructs itself → a stubbed global fetch (the SDK reads it per request; no network, fake key).
// Mutation sweep (decisions/stage-02-measurement.md): dropping the endpoint's voice, baseURL or headers in
// src/registry.ts survived the builder's suite.

type Seen = { url: string; headers: Record<string, string>; form?: [string, unknown][] };

function stubGlobalFetch() {
  const seen: Seen[] = [];
  vi.stubGlobal("fetch", async (url: string | URL | Request, init?: RequestInit) => {
    const entry: Seen = { url: String(url), headers: Object.fromEntries(new Headers(init?.headers).entries()) };
    if (init?.body instanceof FormData) entry.form = [...init.body.entries()].map(([k, v]) => [k, typeof v === "string" ? v : (v as File).name]);
    seen.push(entry);
    return String(url).includes("speech-to-text")
      ? Response.json({ language_code: "en", language_probability: 1, text: "hi", words: [], audio_duration_secs: 1 })
      : new Response(new Uint8Array([1, 2, 3]), { status: 200, headers: { "character-cost": "3" } });
  });
  return seen;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

// The SDK is 23k files; its cold import under a parallel full suite can exceed one test's 5 s on its own.
beforeAll(async () => {
  await import("@elevenlabs/elevenlabs-js");
}, 60_000);

describe("elevenlabs endpoint config reaches the wire through the registry (real SDK)", () => {
  it("the endpoint's voice is the voice of ai.speak(); a per-call voice wins", async () => {
    const seen = stubGlobalFetch();
    const ai = createAI({ providers: { elevenlabs: { apiKey: "test-key", voice: "v-endpoint" } } });
    await ai.speak({ model: "elevenlabs:eleven_flash_v2_5", input: "Hi." });
    await ai.speak({ model: "elevenlabs:eleven_flash_v2_5", input: "Hi.", voice: "v-call" });
    expect(seen.map((s) => new URL(s.url).pathname)).toStrictEqual(["/v1/text-to-speech/v-endpoint", "/v1/text-to-speech/v-call"]);
    expect(seen[0]!.headers["xi-api-key"]).toBe("test-key");
  });

  it("a second account under another name: api \"elevenlabs\" + baseURL (host root) is the host of both calls", async () => {
    const seen = stubGlobalFetch();
    const ai = createAI({ providers: { "labs-eu": { api: "elevenlabs", apiKey: "eu-key", voice: "v", baseURL: "https://api.eu.residency.elevenlabs.io" } } });
    await ai.speak({ model: "labs-eu:eleven_flash_v2_5", input: "Hi." });
    await ai.transcribe({ model: "labs-eu:scribe_v2", audio: { data: new Uint8Array([1]), mediaType: "audio/webm" } });
    expect(seen.map((s) => s.url.split("?")[0])).toStrictEqual([
      "https://api.eu.residency.elevenlabs.io/v1/text-to-speech/v",
      "https://api.eu.residency.elevenlabs.io/v1/speech-to-text",
    ]);
    expect(seen.every((s) => s.headers["xi-api-key"] === "eu-key")).toBe(true);
  });

  it("without baseURL the SDK's default host is used", async () => {
    const seen = stubGlobalFetch();
    await createAI({ providers: { elevenlabs: "test-key" } }).transcribe({ model: "elevenlabs:scribe_v2", audio: { data: new Uint8Array([1]) } });
    expect(seen[0]!.url).toBe("https://api.elevenlabs.io/v1/speech-to-text");
  });

  it("the endpoint's headers go out with every call, per-call headers merged over them", async () => {
    const seen = stubGlobalFetch();
    const ai = createAI({ providers: { elevenlabs: { apiKey: "test-key", voice: "v", headers: { "x-team": "voice", "x-trace": "endpoint" } } } });
    await ai.speak({ model: "elevenlabs:eleven_flash_v2_5", input: "Hi.", headers: { "x-trace": "call" } });
    await ai.transcribe({ model: "elevenlabs:scribe_v2", audio: { data: new Uint8Array([1]) } });
    expect(seen[0]!.headers).toMatchObject({ "x-team": "voice", "x-trace": "call" });
    expect(seen[1]!.headers).toMatchObject({ "x-team": "voice", "x-trace": "endpoint" });
  });
});

describe("elevenlabs words without timing (builder deviation 3)", () => {
  it("a word whose start/end is null is left out of words but stays in text", async () => {
    // The SDK's schema makes start/end nullable (api/types/SpeechToTextWordResponseModel.d.ts); a real
    // body with a null start parses, so the guard in toWords is reachable.
    const seen = stubGlobalFetch();
    vi.stubGlobal("fetch", async (url: string) => {
      seen.push({ url, headers: {} });
      return Response.json({
        language_code: "en",
        language_probability: 1,
        text: "Hello there",
        words: [
          { text: "Hello", start: 0, end: 0.4, type: "word", logprob: -0.1 },
          { text: " ", start: 0.4, end: 0.5, type: "spacing", logprob: 0 },
          { text: "there", start: null, end: null, type: "word", logprob: -0.2 },
        ],
      });
    });
    const p = elevenlabs({ model: "scribe_v2", apiKey: "test-key" });
    const res = await p.transcribe!({ audio: { data: new Uint8Array([1]) } });
    expect(res.text).toBe("Hello there");
    expect(res.words).toStrictEqual([{ text: "Hello", start: 0, end: 0.4 }]);
    expect(seen).toHaveLength(1);
  });
});
