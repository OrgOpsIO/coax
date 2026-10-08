import { ElevenLabsClient } from "@elevenlabs/elevenlabs-js";
import OpenAI from "openai";
import { describe, expect, it } from "vitest";
import { createAI, type AI } from "../src/ai";
import { CoaxAbortError, CoaxUnsupportedError } from "../src/client";
import { elevenlabs } from "../src/providers/elevenlabs";
import { openai } from "../src/providers/openai";
import { billedUsage, type Usage } from "../src/types";

// Measurer (stage 2): the same contract on both voice vendors. Each `ai.*` call below is written ONCE and
// run against the REAL vendor SDK (openai 6.x, @elevenlabs/elevenlabs-js 2.71.0) with a stubbed fetch —
// no network, fake keys. Only the model reference differs between the two runs, as a caller switching
// vendors would write it. Response bodies follow the shapes in test/fixtures/elevenlabs/
// (tts-convert.response, stt-convert.response a_minimal_rest_verbatim, openai-audio-usage.sdk).

type Reply = (url: string, signal?: AbortSignal | null) => Response | Promise<Response>;

/** Like real fetch: rejects at once on an already-aborted signal and when the signal aborts (stage 1, O11). */
function stubFetch(reply: Reply) {
  const urls: string[] = [];
  const signals: (AbortSignal | null | undefined)[] = [];
  const fetch = async (url: string | URL | Request, init?: RequestInit): Promise<Response> => {
    urls.push(String(url));
    const signal = init?.signal;
    signals.push(signal);
    const abortError = () => Object.assign(new Error("This operation was aborted"), { name: "AbortError" });
    if (signal?.aborted) throw abortError();
    return await new Promise<Response>((resolve, reject) => {
      signal?.addEventListener("abort", () => reject(abortError()), { once: true });
      Promise.resolve(reply(String(url), signal)).then(resolve, reject);
    });
  };
  return { fetch: fetch as typeof globalThis.fetch, urls, signals };
}

const AUDIO = new Uint8Array([0x49, 0x44, 0x33, 0x04, 0, 0, 0, 0]);
const ok: Reply = (url) =>
  url.includes("transcriptions")
    ? Response.json({ text: "Hello world!", usage: { type: "duration", seconds: 2 } })
    : url.includes("speech-to-text")
      ? Response.json({ language_code: "en", language_probability: 0.98, text: "Hello world!", words: [{ text: "Hello", start: 0, end: 0.5, type: "word", logprob: -0.1 }], audio_duration_secs: 2 })
      : new Response(AUDIO, { status: 200, headers: { "content-type": "audio/mpeg", "character-cost": "12" } });

interface Vendor {
  name: string;
  speakModel: string;
  transcribeModel: string;
  build(reply: Reply, onUsage?: (u: Usage) => void): { ai: AI; urls: string[]; signals: (AbortSignal | null | undefined)[] };
}

const vendors: Vendor[] = [
  {
    name: "openai",
    speakModel: "openai:gpt-4o-mini-tts",
    transcribeModel: "openai:whisper-1",
    build(reply, onUsage) {
      const { fetch, urls, signals } = stubFetch(reply);
      const client = new OpenAI({ apiKey: "test-key", maxRetries: 0, fetch });
      return { urls, signals, ai: createAI({ providers: { openai: (m) => openai({ model: m, client, voice: "v" }) }, onUsage, defaults: { retries: { attempts: 3, initialDelayMs: 1 } } }) };
    },
  },
  {
    name: "elevenlabs",
    speakModel: "elevenlabs:eleven_flash_v2_5",
    transcribeModel: "elevenlabs:scribe_v2",
    build(reply, onUsage) {
      const { fetch, urls, signals } = stubFetch(reply);
      // timeoutInSeconds 1: the SDK's per-request timer is not cleared on abort (decisions/stage-02-measurement.md).
      const client = new ElevenLabsClient({ apiKey: "test-key", maxRetries: 0, timeoutInSeconds: 1, fetch });
      return { urls, signals, ai: createAI({ providers: { elevenlabs: (m) => elevenlabs({ model: m, client, voice: "v" }) }, onUsage, defaults: { retries: { attempts: 3, initialDelayMs: 1 } } }) };
    },
  },
];

const TOKEN_KEYS = ["cacheReadTokens", "cacheWriteTokens", "inputTokens", "outputTokens"];
const audio = { data: new Uint8Array([1, 2, 3]), mediaType: "audio/webm" };

describe.each(vendors)("voice contract on $name (same caller code)", (v) => {
  it("ai.speak returns { audio, mediaType, usage, model } with the four token counters, reported once", async () => {
    const seen: Usage[] = [];
    const { ai } = v.build(ok, (u) => void seen.push(u));
    const res = await ai.speak({ model: v.speakModel, input: "Hello there.", format: "mp3" });
    expect(Object.keys(res).sort()).toStrictEqual(["audio", "mediaType", "model", "usage"]);
    expect(res.audio).toBeInstanceOf(Uint8Array);
    expect(Array.from(res.audio)).toStrictEqual(Array.from(AUDIO));
    expect(res.mediaType).toBe("audio/mpeg");
    expect(res.model).toBe(v.speakModel.split(":")[1]);
    for (const k of TOKEN_KEYS) expect(typeof res.usage[k as keyof Usage]).toBe("number");
    // Extra keys are only the billing units the vendor reported (never invented).
    expect(Object.keys(res.usage).filter((k) => !TOKEN_KEYS.includes(k)).every((k) => k === "characters")).toBe(true);
    expect(seen).toStrictEqual([res.usage]);
  });

  it("ai.transcribe returns { text, usage, model } (+ words where served) and reports audioSeconds once", async () => {
    const seen: Usage[] = [];
    const { ai } = v.build(ok, (u) => void seen.push(u));
    const res = await ai.transcribe({ model: v.transcribeModel, audio, language: "en" });
    expect(res.text).toBe("Hello world!");
    expect(Object.keys(res).filter((k) => k !== "words").sort()).toStrictEqual(["model", "text", "usage"]);
    expect(res.model).toBe(v.transcribeModel.split(":")[1]);
    expect(res.usage).toStrictEqual({ inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0, audioSeconds: 2 });
    expect(seen).toStrictEqual([res.usage]);
  });

  it("an aborted call is a CoaxAbortError, and an already-aborted one never reaches the wire", async () => {
    const hanging = v.build(() => new Promise<Response>(() => {}));
    const ac = new AbortController();
    const call = hanging.ai.speak({ model: v.speakModel, input: "Hi.", signal: ac.signal });
    setTimeout(() => ac.abort(), 5);
    await expect(call).rejects.toBeInstanceOf(CoaxAbortError);

    const fresh = v.build(ok);
    const done = new AbortController();
    done.abort();
    await expect(fresh.ai.transcribe({ model: v.transcribeModel, audio, signal: done.signal })).rejects.toBeInstanceOf(CoaxAbortError);
    expect(fresh.urls).toHaveLength(0);
  });

  it("the abort reaches the vendor's fetch at once — the HTTP request is cancelled, not left running", async () => {
    // Mutation sweep: with the signal not handed to the ElevenLabs SDK, the call still ended in
    // CoaxAbortError — via the SDK's own 1 s test timeout — while the request kept running (and billing).
    const { ai, signals } = v.build(() => new Promise<Response>(() => {}));
    const ac = new AbortController();
    const t0 = Date.now();
    const call = ai.speak({ model: v.speakModel, input: "Hi.", signal: ac.signal });
    setTimeout(() => ac.abort(), 5);
    await expect(call).rejects.toBeInstanceOf(CoaxAbortError);
    expect(Date.now() - t0).toBeLessThan(500);
    expect(signals).toHaveLength(1);
    expect(signals[0]?.aborted).toBe(true);
  });

  it("a 503 is retried by coax (3 attempts) and a 401 is not", async () => {
    const busy = v.build(() => Response.json({ error: { message: "busy" }, detail: { status: "busy", message: "busy" } }, { status: 503 }));
    await expect(busy.ai.speak({ model: v.speakModel, input: "Hi." })).rejects.toBeInstanceOf(Error);
    expect(busy.urls).toHaveLength(3);
    const denied = v.build(() => Response.json({ error: { message: "no" }, detail: { status: "invalid_api_key", message: "no" } }, { status: 401 }));
    await expect(denied.ai.speak({ model: v.speakModel, input: "Hi." })).rejects.toBeInstanceOf(Error);
    expect(denied.urls).toHaveLength(1);
  });

  it("what the vendor cannot serve is a CoaxUnsupportedError naming the provider, before the wire", async () => {
    const { ai, urls } = v.build(ok);
    // One field each wire cannot honour: speaker labels on the OpenAI wire, delivery instructions on ElevenLabs.
    const call = v.name === "openai" ? ai.transcribe({ model: v.transcribeModel, audio, speakers: true }) : ai.speak({ model: v.speakModel, input: "Hi.", instructions: "calm" });
    const err = await call.then(() => undefined, (e: unknown) => e);
    expect(err).toBeInstanceOf(CoaxUnsupportedError);
    expect((err as CoaxUnsupportedError).provider).toBe(v.name);
    expect(urls).toHaveLength(0);
  });
});

/** A 200 speech whose body sends one chunk and then waits; the fetch's abort errors it as undici does. */
const hangingSpeech: Reply = (_url, signal) => {
  let sent = false;
  const body = new ReadableStream<Uint8Array>(
    {
      start(controller) {
        signal?.addEventListener("abort", () => {
          try {
            controller.error(new DOMException("This operation was aborted", "AbortError"));
          } catch {
            // already closed
          }
        });
      },
      async pull(controller) {
        if (!sent) return (sent = true), controller.enqueue(AUDIO);
        if (!signal?.aborted) await new Promise<void>((r) => signal?.addEventListener("abort", () => r(), { once: true }));
      },
    },
    { highWaterMark: 0 },
  );
  return new Response(body, { status: 200, headers: { "content-type": "audio/mpeg", "character-cost": "12" } });
};

// Stage 3 (T5): ai.speakStream, written once, on both vendors.
describe.each(vendors)("streamed speech contract on $name (same caller code)", (v) => {
  it("ai.speakStream yields the audio and resolves result with { mediaType, usage, model }, reported once", async () => {
    const seen: Usage[] = [];
    const { ai } = v.build(ok, (u) => void seen.push(u));
    const { audio, mediaType, result } = await ai.speakStream({ model: v.speakModel, input: "Hello there.", format: "mp3" });
    expect(mediaType).toBe("audio/mpeg");
    const chunks: Uint8Array[] = [];
    for await (const chunk of audio) chunks.push(chunk);
    expect(chunks.length).toBeGreaterThan(0);
    expect(chunks.flatMap((c) => Array.from(c))).toStrictEqual(Array.from(AUDIO));
    const res = await result;
    expect(Object.keys(res).sort()).toStrictEqual(["mediaType", "model", "usage"]);
    expect(res.model).toBe(v.speakModel.split(":")[1]);
    for (const k of TOKEN_KEYS) expect(typeof res.usage[k as keyof Usage]).toBe("number");
    expect(seen).toStrictEqual([res.usage]);
  });

  it("a 401 rejects ai.speakStream() itself, before it resolves", async () => {
    const denied = v.build(() => Response.json({ error: { message: "no" }, detail: { status: "invalid_api_key", message: "no" } }, { status: 401 }));
    await expect(denied.ai.speakStream({ model: v.speakModel, input: "Hi." })).rejects.toBeInstanceOf(Error);
    expect(denied.urls).toHaveLength(1);
  });

  it("an abort mid-body is a CoaxAbortError through the iteration", async () => {
    const { ai } = v.build(hangingSpeech);
    const ac = new AbortController();
    const { audio, result } = await ai.speakStream({ model: v.speakModel, input: "Hi.", signal: ac.signal });
    const err = await (async () => {
      for await (const _ of audio) ac.abort();
    })().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(CoaxAbortError);
    await expect(result).rejects.toBe(err);
  });

  // Measurer (stage 3): the rest of the stream contract, written once.
  it("a 503 before the audio is retried by coax (3 attempts); an already-aborted call never reaches the wire", async () => {
    const busy = v.build(() => Response.json({ error: { message: "busy" }, detail: { status: "busy", message: "busy" } }, { status: 503 }));
    await expect(busy.ai.speakStream({ model: v.speakModel, input: "Hi." })).rejects.toBeInstanceOf(Error);
    expect(busy.urls).toHaveLength(3);

    const fresh = v.build(ok);
    const done = new AbortController();
    done.abort();
    await expect(fresh.ai.speakStream({ model: v.speakModel, input: "Hi.", signal: done.signal })).rejects.toBeInstanceOf(CoaxAbortError);
    expect(fresh.urls).toHaveLength(0);
  });

  it("an early break closes the response body on the wire (what it books differs by vendor: pinned below, O19)", async () => {
    const { cancelled } = await brokenOff(v);
    expect(cancelled).toBe(true);
  });
});

/** A speech whose body never ends (character-cost 12 in the header), left with `break` after the first chunk. */
async function brokenOff(v: Vendor): Promise<{ cancelled: boolean; seen: Usage[] }> {
  let cancelled = false;
  const endless: Reply = () =>
    new Response(
      new ReadableStream<Uint8Array>({ pull: (c) => c.enqueue(AUDIO), cancel: () => void (cancelled = true) }, { highWaterMark: 0 }),
      { status: 200, headers: { "content-type": "audio/mpeg", "character-cost": "12" } },
    );
  const seen: Usage[] = [];
  const { ai } = v.build(endless, (u) => void seen.push(u));
  const { audio } = await ai.speakStream({ model: v.speakModel, input: "Hi." });
  for await (const _ of audio) break;
  return { cancelled, seen };
}

describe("voice contract: what differs between the vendors, pinned (measured)", () => {
  it("an abort mid-speech carries what the vendor billed: characters on ElevenLabs, zero units on the OpenAI wire (measurer, stage 3)", async () => {
    const abortedUsage = async (v: Vendor) => {
      const seen: Usage[] = [];
      const { ai } = v.build(hangingSpeech, (u) => void seen.push(u));
      const ac = new AbortController();
      const { audio } = await ai.speakStream({ model: v.speakModel, input: "Hi.", signal: ac.signal });
      const err = (await (async () => {
        for await (const _ of audio) ac.abort();
      })().catch((e: unknown) => e)) as CoaxAbortError;
      expect(err).toBeInstanceOf(CoaxAbortError);
      return { usage: err.usage, billed: billedUsage(err), seen };
    };
    const zeros = { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0 };
    expect(await abortedUsage(vendors[0]!)).toStrictEqual({ usage: zeros, billed: undefined, seen: [] });
    expect(await abortedUsage(vendors[1]!)).toStrictEqual({ usage: { ...zeros, characters: 12 }, billed: { ...zeros, characters: 12 }, seen: [{ ...zeros, characters: 12 }] });
  });

  it("a break books what the vendor billed before the audio: characters on ElevenLabs; nothing on the OpenAI wire, which reports no usage for speech (O19)", async () => {
    const zeros = { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0 };
    expect(await brokenOff(vendors[0]!)).toStrictEqual({ cancelled: true, seen: [] });
    expect(await brokenOff(vendors[1]!)).toStrictEqual({ cancelled: true, seen: [{ ...zeros, characters: 12 }] });
  });

  it("ai.transcribeToken, same caller code: ElevenLabs issues a token, the OpenAI wire refuses before the wire (measurer, stage 3)", async () => {
    const tokenReply: Reply = (url) => (url.includes("single-use-token") ? Response.json({ token: "sutkn_parity" }) : Response.json({}, { status: 404 }));
    const results = [];
    for (const [v, model] of [[vendors[0]!, "openai:gpt-4o-transcribe"], [vendors[1]!, "elevenlabs:scribe_v2_realtime"]] as const) {
      const seen: Usage[] = [];
      const { ai, urls } = v.build(tokenReply, (u) => void seen.push(u));
      const res = await ai.transcribeToken({ model }).then(
        (r) => ({ keys: Object.keys(r).sort(), url: r.url, model: r.model }),
        (e: unknown) => ({ error: (e as Error).constructor.name, capability: (e as CoaxUnsupportedError).capability, provider: (e as CoaxUnsupportedError).provider }),
      );
      results.push({ res, requests: urls.length, reported: seen.length });
    }
    expect(results).toStrictEqual([
      { res: { error: "CoaxUnsupportedError", capability: "realtime transcription tokens", provider: "openai" }, requests: 0, reported: 0 },
      { res: { keys: ["model", "token", "url", "usage"], url: "wss://api.elevenlabs.io/v1/speech-to-text/realtime", model: "scribe_v2_realtime" }, requests: 1, reported: 1 },
    ]);
  });

  it("a vendor HTTP error carries `status` on the OpenAI wire but only `statusCode` on ElevenLabs — coax passes SDK errors through", async () => {
    const statusOf = async (v: Vendor) => {
      const { ai } = v.build(() => Response.json({ detail: { status: "invalid_api_key", message: "no" } }, { status: 401 }));
      const err = (await ai.speak({ model: v.speakModel, input: "Hi." }).catch((e: unknown) => e)) as { status?: number; statusCode?: number };
      return { status: err.status, statusCode: err.statusCode };
    };
    expect(await statusOf(vendors[0]!)).toStrictEqual({ status: 401, statusCode: undefined });
    expect(await statusOf(vendors[1]!)).toStrictEqual({ status: undefined, statusCode: 401 });
  });
});
