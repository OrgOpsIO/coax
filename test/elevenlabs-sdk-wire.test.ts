import { readFileSync } from "node:fs";
import { ElevenLabsClient } from "@elevenlabs/elevenlabs-js";
import { beforeAll, describe, expect, it } from "vitest";
import { createAI } from "../src/ai";
import { CoaxAbortError, CoaxUnsupportedError } from "../src/client";
import { elevenlabs } from "../src/providers/elevenlabs";
import { emptyUsage, type Provider, type Usage } from "../src/types";

// The REAL SDK (@elevenlabs/elevenlabs-js 2.71.0) in the loop, against a stubbed fetch: no network,
// fake key. Fixtures are copies of .ziv/reference/fixtures/stage-02 (each names its source inside).
const fixture = (name: string) => JSON.parse(readFileSync(new URL(`./fixtures/elevenlabs/${name}.json`, import.meta.url), "utf8"));
const TTS_REQ = fixture("tts-convert.request"); // MEASURED wire of textToSpeech.convert
const TTS_RES = fixture("tts-convert.response"); // header names VERBATIM, values COMPOSED
const STT_REQ = fixture("stt-convert.request"); // MEASURED multipart wire of speechToText.convert
const STT_RES = fixture("stt-convert.response"); // VERBATIM doc bodies + one COMPOSED full body
const STT_OTHER = fixture("stt-other-shapes.response"); // multichannel / webhook / silence
const ERRORS = fixture("errors"); // documented error envelopes + MEASURED SDK error classes

type Seen = { url: string; method?: string; headers: Record<string, string>; body?: unknown; form?: [string, unknown][] };

const hex = (s: string) => new Uint8Array(s.match(/../g)!.map((b) => parseInt(b, 16)));
const AUDIO = hex(TTS_RES.fixture.bodyBytesHex as string);

/** A fetch that records what the SDK sent and answers with `reply`. Like real fetch, it rejects at once
 *  on an already-aborted signal and when the signal aborts mid-request (stage 1, O11). */
function stub(reply: (n: number) => Response | Promise<Response>) {
  const seen: Seen[] = [];
  const fetch = async (url: string | URL | Request, init?: RequestInit): Promise<Response> => {
    const entry: Seen = { url: String(url), method: init?.method, headers: Object.fromEntries(new Headers(init?.headers).entries()) };
    if (init?.body instanceof FormData) {
      entry.form = [];
      for (const [k, v] of init.body.entries()) entry.form.push([k, typeof v === "string" ? v : { blob: true, name: (v as File).name, type: v.type, size: v.size }]);
    } else if (typeof init?.body === "string") entry.body = JSON.parse(init.body);
    seen.push(entry);
    const signal = init?.signal;
    const abortError = () => Object.assign(new Error("This operation was aborted"), { name: "AbortError" });
    if (signal?.aborted) throw abortError();
    return await new Promise<Response>((resolve, reject) => {
      signal?.addEventListener("abort", () => reject(abortError()), { once: true });
      Promise.resolve(reply(seen.length)).then(resolve, reject);
    });
  };
  // timeoutInSeconds 1: the SDK arms its timer on every request and does not clear it on abort — a short
  // one cannot hold the test worker (decisions/stage-02-elevenlabs-sdk.md).
  const client = new ElevenLabsClient({ apiKey: "test-key", maxRetries: 0, timeoutInSeconds: 1, fetch: fetch as typeof globalThis.fetch });
  return { client, seen };
}

const audioReply = (headers: Record<string, string> = { "character-cost": TTS_RES.fixture.headers["character-cost"] }, body: Uint8Array = AUDIO) => () =>
  new Response(body, { status: 200, headers: { "content-type": "audio/mpeg", ...headers } });

const VOICE = TTS_REQ.fixture.sdkCall.voiceId as string;
const zeros = emptyUsage();

// The SDK is 23k files; its cold import under a parallel full suite can exceed one test's 5 s on its own.
beforeAll(async () => {
  await import("@elevenlabs/elevenlabs-js");
}, 60_000);

describe("elevenlabs speak through the real SDK", () => {
  it("sends the measured wire: voice in the path, output_format in the query, key header, JSON body", async () => {
    const { client, seen } = stub(audioReply());
    const p = elevenlabs({ model: "eleven_multilingual_v2", client });
    const res = await p.speak!({ input: "Hello there.", voice: VOICE, language: "en", speed: 1.1 });

    const wire = TTS_REQ.fixture.wire;
    expect(seen).toHaveLength(1);
    expect(seen[0]!.url).toBe(wire.url);
    expect(seen[0]!.method).toBe("POST");
    expect(seen[0]!.headers["xi-api-key"]).toBe("test-key");
    // The fixture's wire also carries extra_flag (from additionalBodyParameters) — coax sends no extras.
    const { extra_flag: _, ...body } = wire.body;
    expect(seen[0]!.body).toStrictEqual(body);

    expect(Array.from(res.audio)).toEqual(Array.from(AUDIO));
    expect(res.mediaType).toBe("audio/mpeg");
    expect(res.usage).toStrictEqual({ ...zeros, characters: 12 });
    expect(res.model).toBe("eleven_multilingual_v2");
  });

  it("sends no language_code / voice_settings keys when language and speed are unset", async () => {
    const { client, seen } = stub(audioReply());
    await elevenlabs({ model: "eleven_flash_v2_5", client }).speak!({ input: "Hi.", voice: VOICE });
    expect(seen[0]!.body).toStrictEqual({ text: "Hi.", model_id: "eleven_flash_v2_5" });
  });

  it("reports no characters when the header is missing or unparsable — never input.length, never 0", async () => {
    for (const headers of [{}, { "character-cost": "abc" }] as Record<string, string>[]) {
      const { client } = stub(audioReply(headers));
      const res = await elevenlabs({ model: "eleven_flash_v2_5", client, voice: VOICE }).speak!({ input: "Hello there." });
      expect(res.usage).toStrictEqual(emptyUsage());
    }
  });

  it("maps each served format onto one output_format variant and its media type", async () => {
    const cases = [
      ["mp3", "mp3_44100_128", "audio/mpeg"],
      ["opus", "opus_48000_128", "audio/opus"],
      ["wav", "wav_24000", "audio/wav"],
      ["pcm", "pcm_24000", "audio/pcm"],
    ] as const;
    for (const [format, outputFormat, mediaType] of cases) {
      const { client, seen } = stub(audioReply());
      const res = await elevenlabs({ model: "eleven_flash_v2_5", client, voice: VOICE }).speak!({ input: "Hi.", format });
      expect(new URL(seen[0]!.url).searchParams.get("output_format")).toBe(outputFormat);
      expect(res.mediaType).toBe(mediaType);
    }
  });

  it("merges per-call headers over the endpoint's", async () => {
    const { client, seen } = stub(audioReply());
    const p = elevenlabs({ model: "eleven_flash_v2_5", client, voice: VOICE, headers: { a: "1", b: "1" } });
    await p.speak!({ input: "Hi.", headers: { b: "2" } });
    expect(seen[0]!.headers.a).toBe("1");
    expect(seen[0]!.headers.b).toBe("2");
  });

  it("uses the endpoint's voice, and a per-call voice wins", async () => {
    const { client, seen } = stub(audioReply());
    const p = elevenlabs({ model: "eleven_flash_v2_5", client, voice: "v-end" });
    await p.speak!({ input: "Hi." });
    await p.speak!({ input: "Hi.", voice: "v-call" });
    expect(new URL(seen[0]!.url).pathname).toBe("/v1/text-to-speech/v-end");
    expect(new URL(seen[1]!.url).pathname).toBe("/v1/text-to-speech/v-call");
  });

  it("a per-call voice of \"\" counts as no voice: the endpoint's applies, else the voice-id error (review R2.7)", async () => {
    const { client, seen } = stub(audioReply());
    await elevenlabs({ model: "eleven_flash_v2_5", client, voice: "v-end" }).speak!({ input: "Hi.", voice: "" });
    expect(new URL(seen[0]!.url).pathname).toBe("/v1/text-to-speech/v-end");
    await expect(elevenlabs({ model: "eleven_flash_v2_5", client, voice: "" }).speak!({ input: "Hi.", voice: "" })).rejects.toThrow(/needs a voice id/);
    expect(seen).toHaveLength(1);
  });

  it("a negative or non-numeric character-cost is not a billed count (review R2.6)", async () => {
    for (const cost of ["-3", "-0.5", "Infinity", "NaN"]) {
      const { client } = stub(audioReply({ "character-cost": cost }));
      const res = await elevenlabs({ model: "eleven_flash_v2_5", client, voice: VOICE }).speak!({ input: "Hello there." });
      expect(res.usage).toStrictEqual(emptyUsage());
    }
    const { client } = stub(audioReply({ "character-cost": " 0 " }));
    const res = await elevenlabs({ model: "eleven_flash_v2_5", client, voice: VOICE }).speak!({ input: "Hi." });
    expect(res.usage).toStrictEqual({ ...zeros, characters: 0 });
  });

  it("a 200 with no audio bytes is a failure, never an empty success", async () => {
    const { client } = stub(audioReply(undefined, new Uint8Array(0)));
    await expect(elevenlabs({ model: "eleven_flash_v2_5", client, voice: VOICE }).speak!({ input: "Hi." })).rejects.toThrow(/returned no audio/);
  });

  it("refuses what it cannot honour before the wire", async () => {
    const { client, seen } = stub(audioReply());
    const p = elevenlabs({ model: "eleven_flash_v2_5", client });
    await expect(p.speak!({ input: "Hi." })).rejects.toThrow(/needs a voice id/);
    await expect(p.speak!({ input: "Hi.", voice: VOICE, instructions: "calm" })).rejects.toBeInstanceOf(CoaxUnsupportedError);
    await expect(p.speak!({ input: "Hi.", voice: VOICE, format: "aac" })).rejects.toThrow(/does not support aac output/);
    await expect(p.speak!({ input: "Hi.", voice: VOICE, format: "flac" })).rejects.toThrow(/does not support flac output/);
    await expect(p.speak!({ input: "Hi.", voice: VOICE, speed: 0.5 })).rejects.toThrow(/between 0.7 and 1.2/);
    await expect(p.speak!({ input: "Hi.", voice: VOICE, speed: 1.3 })).rejects.toThrow(/between 0.7 and 1.2/);
    expect(seen).toHaveLength(0);
  });

  it("through createAI, onUsage sees the billed characters once, with the reference's model", async () => {
    const { client } = stub(audioReply());
    const seenUsage: { usage: Usage; model: string }[] = [];
    const ai = createAI({
      providers: { elevenlabs: (m) => elevenlabs({ model: m, client, voice: "v" }) },
      onUsage: (usage, meta) => void seenUsage.push({ usage, model: meta.model }),
    });
    await ai.speak({ model: "elevenlabs:eleven_flash_v2_5", input: "Hello there." });
    expect(seenUsage).toStrictEqual([{ usage: { ...zeros, characters: 12 }, model: "eleven_flash_v2_5" }]);
  });
});

describe("elevenlabs transcribe through the real SDK", () => {
  const full = STT_RES.fixture.d_full_rest_composed;
  const webm = { data: new Uint8Array([1, 2, 3, 4]), mediaType: "audio/webm" };

  it("sends the measured multipart form, with tag_audio_events=false and no extras", async () => {
    const { client, seen } = stub(() => Response.json(full));
    await elevenlabs({ model: "scribe_v2", client }).transcribe!({ audio: webm, language: "de" });

    expect(seen[0]!.url).toBe(STT_REQ.fixture.wire.url);
    expect(seen[0]!.method).toBe("POST");
    expect(seen[0]!.headers["xi-api-key"]).toBe("test-key");
    const form = seen[0]!.form!;
    const measuredFile = (STT_REQ.fixture.wire.form as [string, unknown][]).find(([k]) => k === "file");
    expect(form).toContainEqual(["model_id", "scribe_v2"]);
    expect(form).toContainEqual(measuredFile);
    expect(form).toContainEqual(["language_code", "de"]);
    expect(form).toContainEqual(["tag_audio_events", "false"]);
    const keys = form.map(([k]) => k);
    for (const absent of ["diarize", "timestamps_granularity", "keyterms"]) expect(keys).not.toContain(absent);
  });

  it("asks for speaker labels only with speakers: true", async () => {
    const { client, seen } = stub(() => Response.json(full));
    await elevenlabs({ model: "scribe_v2", client }).transcribe!({ audio: webm, speakers: true });
    expect(seen[0]!.form).toContainEqual(["diarize", "true"]);
  });

  it("names the upload from mediaType for ArrayBuffer and Blob inputs; filename overrides", async () => {
    const { client, seen } = stub(() => Response.json(full));
    const p = elevenlabs({ model: "scribe_v2", client });
    await p.transcribe!({ audio: { data: new Uint8Array([1, 2]).buffer, mediaType: "audio/wav" } });
    await p.transcribe!({ audio: { data: new Blob([new Uint8Array([1, 2])], { type: "audio/webm" }), mediaType: "audio/webm" } });
    await p.transcribe!({ audio: { data: new Uint8Array([1, 2]), mediaType: "audio/mpeg", filename: "call-17.mp3" } });
    const file = (i: number) => seen[i]!.form!.find(([k]) => k === "file")![1];
    expect(file(0)).toStrictEqual({ blob: true, name: "audio.wav", type: "audio/wav", size: 2 });
    expect(file(1)).toStrictEqual({ blob: true, name: "audio.webm", type: "audio/webm", size: 2 });
    expect(file(2)).toMatchObject({ name: "call-17.mp3", type: "audio/mpeg" });
  });

  it("without a mediaType the upload is named 'audio' and the vendor sniffs the format — assumption, facts §6", async () => {
    const { client, seen } = stub(() => Response.json(full));
    await elevenlabs({ model: "scribe_v2", client }).transcribe!({ audio: { data: new Uint8Array([1, 2]) } });
    expect(seen[0]!.form!.find(([k]) => k === "file")![1]).toMatchObject({ name: "audio" });
  });

  it("returns text verbatim, spoken words only with their speaker, and the billed audio seconds", async () => {
    const { client } = stub(() => Response.json(full));
    const res = await elevenlabs({ model: "scribe_v2", client }).transcribe!({ audio: webm, speakers: true });
    expect(res.text).toBe(full.text);
    expect(res.words).toStrictEqual([
      { text: "Guten", start: 0.12, end: 0.41, speaker: "speaker_0" },
      { text: "Tag.", start: 0.45, end: 0.8, speaker: "speaker_0" },
      { text: "Ich", start: 1.7, end: 1.9, speaker: "speaker_1" },
    ]);
    expect(res.usage).toStrictEqual({ ...zeros, audioSeconds: 3.84 });
    expect(res.model).toBe("scribe_v2");
  });

  it("a body without audio_duration_secs reports no audio seconds (doc example, VERBATIM)", async () => {
    const { client } = stub(() => Response.json(STT_RES.fixture.a_minimal_rest_verbatim));
    const res = await elevenlabs({ model: "scribe_v2", client }).transcribe!({ audio: webm });
    expect(res.usage).toStrictEqual(emptyUsage());
    expect(res.words).toHaveLength(1);
  });

  it("a result without words has no words key (SDK-shaped fake: the real SDK 2.71.0 rejects such a body itself)", async () => {
    const { words: _, ...noWords } = full;
    // Measured: SDK 2.71.0's response schema requires `words` — a REST body without it throws the SDK's
    // ParseError before coax sees it. coax's own guard is proven with a client that returns the SDK shape.
    const real = stub(() => Response.json(noWords));
    await expect(elevenlabs({ model: "scribe_v2", client: real.client }).transcribe!({ audio: webm })).rejects.toThrow(/Missing required key "words"/);

    const fake = { speechToText: { convert: async () => ({ text: noWords.text, audioDurationSecs: 1 }) } };
    const res = await elevenlabs({ model: "scribe_v2", client: fake as never }).transcribe!({ audio: webm });
    expect("words" in res).toBe(false);
    expect(res).toStrictEqual({ text: noWords.text, usage: { ...zeros, audioSeconds: 1 }, model: "scribe_v2" });
  });

  it("a shape without top-level text (multichannel) is an error, not an empty transcript", async () => {
    const { client } = stub(() => Response.json(STT_OTHER.fixture.multichannel_separate));
    await expect(elevenlabs({ model: "scribe_v2", client }).transcribe!({ audio: webm })).rejects.toThrow(/no transcript text/);
  });

  it("a transcription prompt is refused before the wire", async () => {
    const { client, seen } = stub(() => Response.json(full));
    const err = await elevenlabs({ model: "scribe_v2", client })
      .transcribe!({ audio: webm, prompt: "Zod, coax" })
      .catch((e: unknown) => e);
    expect(err).toBeInstanceOf(CoaxUnsupportedError);
    expect((err as CoaxUnsupportedError).capability).toBe("a transcription prompt (`prompt`)");
    expect(seen).toHaveLength(0);
  });

  it("through createAI: speakers reaches the provider, words come back, onUsage sees audioSeconds", async () => {
    const { client, seen } = stub(() => Response.json(full));
    const usages: Usage[] = [];
    const ai = createAI({ providers: { elevenlabs: (m) => elevenlabs({ model: m, client }) }, onUsage: (u) => void usages.push(u) });
    const res = await ai.transcribe({ model: "elevenlabs:scribe_v2", audio: webm, speakers: true });
    expect(seen[0]!.form).toContainEqual(["diarize", "true"]);
    expect(res.words).toHaveLength(3);
    expect(usages).toStrictEqual([{ ...zeros, audioSeconds: 3.84 }]);
  });
});

describe("elevenlabs errors, retries and abort through the real SDK", () => {
  const env = ERRORS.fixture;
  const failing = (status: number, body: unknown) => () => Response.json(body, { status });

  function aiWith(client: ElevenLabsClient, extra: Record<string, Provider> = {}, models?: Record<string, { use: string; fallback: string }>) {
    return createAI({
      providers: { elevenlabs: (m) => elevenlabs({ model: m, client, voice: VOICE }), ...Object.fromEntries(Object.entries(extra).map(([k, p]) => [k, () => p])) },
      models,
      defaults: { retries: { attempts: 3, initialDelayMs: 1 } },
    });
  }

  for (const [status, body] of [
    [503, env.serverError503.body],
    [429, env.rateLimited429.body],
  ] as const) {
    it(`${status} is retried by coax (3 attempts), never by the SDK`, async () => {
      const { client, seen } = stub(failing(status, body));
      const err = await aiWith(client).speak({ model: "elevenlabs:eleven_flash_v2_5", input: "Hi." }).catch((e: unknown) => e);
      expect((err as { statusCode?: number }).statusCode).toBe(status);
      expect(seen).toHaveLength(3);
    });
  }

  for (const [status, body] of [
    [400, env.documentedEnvelope_VERBATIM.body],
    [401, env.unauthorized401.body],
    [402, env.paymentRequired402.body],
    [422, env.validation422_shape_VERBATIM.body],
  ] as const) {
    it(`${status} is not retried and surfaces with its statusCode`, async () => {
      const { client, seen } = stub(failing(status, body));
      const err = await aiWith(client).transcribe({ model: "elevenlabs:scribe_v2", audio: { data: new Uint8Array([1]) } }).catch((e: unknown) => e);
      expect((err as { statusCode?: number }).statusCode).toBe(status);
      expect(seen).toHaveLength(1);
    });
  }

  // What undici's fetch rejects with on a dropped connection: TypeError "fetch failed", the socket error as
  // its cause (Node 22). The SDK wraps it into ElevenLabsError("fetch failed") with no statusCode and no
  // code — the network code is only on err.cause.cause (SDK 2.71.0 core/fetcher/Fetcher.js, review R2.2).
  const dropped = (code = "ECONNRESET") => Object.assign(new TypeError("fetch failed"), { cause: Object.assign(new Error(`read ${code}`), { code }) });

  it("a dropped connection (network error) is retried by coax (3 attempts), as on the OpenAI wire", async () => {
    const { client, seen } = stub(() => Promise.reject(dropped()));
    const err = await aiWith(client).speak({ model: "elevenlabs:eleven_flash_v2_5", input: "Hi." }).catch((e: unknown) => e);
    expect((err as { code?: string }).code).toBe("ECONNRESET");
    expect((err as Error).message).toMatch(/fetch failed/);
    expect(seen).toHaveLength(3);

    const stt = stub(() => Promise.reject(dropped("ECONNREFUSED")));
    await aiWith(stt.client).transcribe({ model: "elevenlabs:scribe_v2", audio: { data: new Uint8Array([1]) } }).catch((e: unknown) => e);
    expect(stt.seen).toHaveLength(3);
  });

  it("a network failure that recovers serves the call and reports its usage once", async () => {
    const { client, seen } = stub((n) => (n === 1 ? Promise.reject(dropped()) : audioReply()()));
    const usages: Usage[] = [];
    const ai = createAI({ providers: { elevenlabs: (m) => elevenlabs({ model: m, client, voice: VOICE }) }, defaults: { retries: { attempts: 3, initialDelayMs: 1 } }, onUsage: (u) => void usages.push(u) });
    const res = await ai.speak({ model: "elevenlabs:eleven_flash_v2_5", input: "Hello there." });
    expect(seen).toHaveLength(2);
    expect(res.usage).toStrictEqual({ ...zeros, characters: 12 });
    expect(usages).toStrictEqual([{ ...zeros, characters: 12 }]);
  });

  it("an error's own code is never overwritten from its cause, and a non-Error rejection passes through as is", async () => {
    const own = Object.assign(new Error("own"), { code: "E_OWN", cause: dropped() });
    const fake = { speechToText: { convert: () => Promise.reject(own) }, textToSpeech: { convert: () => ({ withRawResponse: () => Promise.reject("plain") }) } };
    const p = elevenlabs({ model: "scribe_v2", client: fake as never, voice: VOICE });
    const err = await p.transcribe!({ audio: { data: new Uint8Array([1]) } }).catch((e: unknown) => e);
    expect(err).toBe(own);
    expect(own.code).toBe("E_OWN");
    await expect(p.speak!({ input: "Hi." })).rejects.toBe("plain");
  });

  it("a network error that lands after an abort is a CoaxAbortError and is not retried", async () => {
    const ac = new AbortController();
    let requests = 0;
    const fetch = async () => {
      requests++;
      ac.abort();
      throw dropped();
    };
    const client = new ElevenLabsClient({ apiKey: "test-key", maxRetries: 0, timeoutInSeconds: 1, fetch: fetch as typeof globalThis.fetch });
    await expect(aiWith(client).speak({ model: "elevenlabs:eleven_flash_v2_5", input: "Hi.", signal: ac.signal })).rejects.toBeInstanceOf(CoaxAbortError);
    expect(requests).toBe(1);
  });

  it("the SDK's own timeout is not a network error and is not retried", async () => {
    // The SDK aborts its own request at timeoutInSeconds (1 here) with the reason "timeout"; Node's fetch
    // rejects with that reason itself, a string (measured: .ziv/logs/stage-02/fix/probe-timeout.log), so the
    // SDK throws a plain ElevenLabsError "timeout" — not its ElevenLabsTimeoutError.
    let requests = 0;
    const fetch = (_url: unknown, init?: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        requests++;
        init?.signal?.addEventListener("abort", () => reject(init.signal!.reason), { once: true });
      });
    const client = new ElevenLabsClient({ apiKey: "test-key", maxRetries: 0, timeoutInSeconds: 1, fetch: fetch as typeof globalThis.fetch });
    const err = await aiWith(client).speak({ model: "elevenlabs:eleven_flash_v2_5", input: "Hi." }).catch((e: unknown) => e);
    expect((err as Error).name).toBe("ElevenLabsError");
    expect((err as Error).message).toBe('"timeout"');
    expect((err as { code?: unknown }).code).toBeUndefined();
    expect(requests).toBe(1);
  }, 10_000);

  it("an abort mid-request is a CoaxAbortError", async () => {
    const { client, seen } = stub(() => new Promise<Response>(() => {}));
    const ac = new AbortController();
    const call = aiWith(client).speak({ model: "elevenlabs:eleven_flash_v2_5", input: "Hi.", signal: ac.signal });
    setTimeout(() => ac.abort(), 5);
    await expect(call).rejects.toBeInstanceOf(CoaxAbortError);
    expect(seen).toHaveLength(1);
  });

  it("an already-aborted signal never reaches the wire", async () => {
    const { client, seen } = stub(audioReply());
    const ac = new AbortController();
    ac.abort();
    await expect(aiWith(client).speak({ model: "elevenlabs:eleven_flash_v2_5", input: "Hi.", signal: ac.signal })).rejects.toBeInstanceOf(CoaxAbortError);
    expect(seen).toHaveLength(0);
  });

  it("an abort does not run the alias' fallback", async () => {
    const { client } = stub(() => new Promise<Response>(() => {}));
    let fallbackCalls = 0;
    const backup: Provider = {
      name: "backup",
      model: "m",
      structured: async () => ({ raw: {}, text: "{}", usage: emptyUsage(), model: "m" }),
      text: async () => ({ raw: "", text: "", usage: emptyUsage(), model: "m" }),
      speak: async () => {
        fallbackCalls++;
        return { audio: new Uint8Array([1]), mediaType: "audio/mpeg", usage: emptyUsage(), model: "m" };
      },
    };
    const ai = aiWith(client, { backup }, { mouth: { use: "elevenlabs:eleven_flash_v2_5", fallback: "backup:m" } });
    const ac = new AbortController();
    const call = ai.speak({ model: "mouth", input: "Hi.", signal: ac.signal });
    setTimeout(() => ac.abort(), 5);
    await expect(call).rejects.toBeInstanceOf(CoaxAbortError);
    expect(fallbackCalls).toBe(0);
  });
});
