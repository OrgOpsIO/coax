import { readFileSync } from "node:fs";
import OpenAI from "openai";
import { describe, expect, it } from "vitest";
import { createAI } from "../src/ai";
import { CoaxAbortError } from "../src/client";
import { openai } from "../src/providers/openai";
import { emptyUsage, type Usage } from "../src/types";

// Streamed speech on the OpenAI wire (stage-03-spec §2.5, task T5): the REAL openai SDK with a stubbed fetch, no
// network, fake key. The request/response shapes are those of the fixture (a copy of
// .ziv/reference/fixtures/stage-03/openai-speech-stream.json, which names its sources).
const FIXTURE = JSON.parse(readFileSync(new URL("./fixtures/elevenlabs/openai-speech-stream.json", import.meta.url), "utf8")).fixture;
const RAW = FIXTURE.rawAudio;

type Seen = { url: string; body: Record<string, unknown> };

/**
 * A fetch that records the JSON body and answers 200 with a chunked body: `chunks` one per read, then "close" or
 * "hang" (calling `onHang`). An abort of the fetch's signal errors the body like undici (DOMException AbortError).
 */
function stubFetch(chunks: Uint8Array[], end: "close" | "hang" = "close", onHang?: () => void) {
  const seen: Seen[] = [];
  const fetch = async (url: string | URL | Request, init?: RequestInit): Promise<Response> => {
    seen.push({ url: String(url), body: JSON.parse(String(init?.body)) });
    const signal = init?.signal;
    if (signal?.aborted) throw Object.assign(new Error("This operation was aborted"), { name: "AbortError" });
    let i = 0;
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
          if (i < chunks.length) return controller.enqueue(chunks[i++]!);
          if (end === "close") return controller.close();
          onHang?.();
          if (!signal?.aborted) await new Promise<void>((r) => signal?.addEventListener("abort", () => r(), { once: true }));
        },
      },
      { highWaterMark: 0 },
    );
    return new Response(body, { status: 200, headers: RAW.response.headers });
  };
  return { fetch: fetch as typeof globalThis.fetch, seen };
}

const CHUNKS = [new Uint8Array([0x49, 0x44, 0x33]), new Uint8Array([4, 5]), new Uint8Array([6, 7, 8, 9])];

function provider(f: ReturnType<typeof stubFetch>, extra: { speakModel?: string; voice?: string } = {}) {
  const client = new OpenAI({ apiKey: "test-key", maxRetries: 0, fetch: f.fetch });
  return openai({ model: RAW.request.model, client, ...extra });
}

async function drain(audio: AsyncIterable<Uint8Array>) {
  const got: Uint8Array[] = [];
  for await (const c of audio) got.push(c);
  return got;
}

describe("openai speakStream through the real SDK", () => {
  it("posts to /audio/speech the same body speak sends — and no stream_format", async () => {
    const streamed = stubFetch(CHUNKS);
    const whole = stubFetch(CHUNKS);
    const call = { input: RAW.request.input, voice: RAW.request.voice, format: RAW.request.response_format, speed: 1.2, instructions: "calm" } as const;
    await drain((await provider(streamed).speakStream!(call)).audio);
    await provider(whole).speak!(call);
    expect(streamed.seen[0]!.url).toBe("https://api.openai.com/v1/audio/speech");
    expect(streamed.seen[0]!.body).toStrictEqual(whole.seen[0]!.body);
    expect(streamed.seen[0]!.body).toStrictEqual({ ...RAW.request, speed: 1.2, instructions: "calm" });
    expect("stream_format" in streamed.seen[0]!.body).toBe(false);
  });

  it("the endpoint's voice is the fallback, as for speak", async () => {
    const f = stubFetch(CHUNKS);
    await drain((await provider(f, { voice: "nova" }).speakStream!({ input: "Hi." })).audio);
    expect(f.seen[0]!.body.voice).toBe("nova");
  });

  it("yields the chunks in order, the media type per format, zero usage, on speakModel", async () => {
    for (const [format, mediaType] of [
      ["mp3", "audio/mpeg"],
      ["pcm", "audio/pcm"],
      ["wav", "audio/wav"],
    ] as const) {
      const f = stubFetch(CHUNKS);
      const usages: Usage[] = [];
      const ai = createAI({
        providers: { openai: (m) => openai({ model: m, client: new OpenAI({ apiKey: "test-key", maxRetries: 0, fetch: f.fetch }), speakModel: "tts-1" }) },
        onUsage: (u) => void usages.push(u),
      });
      const opened = await ai.speakStream({ model: "openai:gpt-x", input: "Hello there.", format });
      expect(opened.mediaType).toBe(mediaType);
      expect(await drain(opened.audio)).toStrictEqual(CHUNKS);
      expect(await opened.result).toStrictEqual({ mediaType, usage: emptyUsage(), model: "tts-1" });
      expect(f.seen[0]!.body.model).toBe("tts-1");
      expect(usages).toStrictEqual([emptyUsage()]);
    }
  });

  it("an empty body is 'returned no audio', never an empty success", async () => {
    const f = stubFetch([]);
    const res = await provider(f).speakStream!({ input: "Hi." });
    await expect(drain(res.audio)).rejects.toThrow("coax: openai returned no audio");
  });

  it("a 429 before the body is retried", async () => {
    let n = 0;
    const ok = stubFetch(CHUNKS);
    const fetch = (async (url: string | URL | Request, init?: RequestInit) =>
      ++n === 1 ? Response.json({ error: { message: "slow down" } }, { status: 429 }) : ok.fetch(url, init)) as typeof globalThis.fetch;
    const ai = createAI({
      providers: { openai: (m) => openai({ model: m, client: new OpenAI({ apiKey: "test-key", maxRetries: 0, fetch }) }) },
      defaults: { retries: { attempts: 3, initialDelayMs: 1 } },
    });
    expect(await drain((await ai.speakStream({ model: "openai:gpt-4o-mini-tts", input: "Hi." })).audio)).toStrictEqual(CHUNKS);
    expect(n).toBe(2);
  });

  it("an abort mid-body is a CoaxAbortError", async () => {
    const ac = new AbortController();
    const f = stubFetch(CHUNKS.slice(0, 1), "hang", () => ac.abort());
    const ai = createAI({ providers: { openai: (m) => openai({ model: m, client: new OpenAI({ apiKey: "test-key", maxRetries: 0, fetch: f.fetch }) }) } });
    const opened = await ai.speakStream({ model: "openai:gpt-4o-mini-tts", input: "Hi.", signal: ac.signal });
    const err = await drain(opened.audio).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(CoaxAbortError);
    expect((err as CoaxAbortError).usage).toStrictEqual(emptyUsage());
    await expect(opened.result).rejects.toBe(err);
  });
});
