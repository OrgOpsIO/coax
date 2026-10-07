import { afterEach, describe, expect, it } from "vitest";
import { createAI, type AI } from "../src/ai";
import { CoaxAbortError, CoaxUnsupportedError } from "../src/client";
import type { CallMeta } from "../src/config";
import { ai as ambient, configure, reset } from "../src/runtime";
import { emptyUsage, withBilledUsage, type Provider, type SpeakRequest, type SpeakStreamResponse, type TranscribeTokenResponse, type Usage } from "../src/types";

// The plumbing of ai.speakStream() / ai.transcribeToken() (stage-03-spec §2.1–2.3, 2.6, task T1): fake
// providers only, no SDK, no network.

const zeros = emptyUsage();
const A = new Uint8Array([1, 2]);
const B = new Uint8Array([3]);
const C = new Uint8Array([4, 5, 6]);

type Gen = AsyncGenerator<Uint8Array, Usage, void>;

function provider(name: string, members: Partial<Pick<Provider, "speak" | "speakStream" | "transcribeToken">>): Provider {
  return {
    name,
    model: "m",
    structured: async () => ({ raw: {}, text: "{}", usage: zeros, model: "m" }),
    text: async () => ({ raw: "", text: "", usage: zeros, model: "m" }),
    ...members,
  };
}

/** A speakStream member whose audio is `gen(req)`; counts its openings. */
function streaming(gen: (req: SpeakRequest) => Gen, mediaType = "audio/mpeg") {
  const calls: SpeakRequest[] = [];
  const speakStream = async (req: SpeakRequest): Promise<SpeakStreamResponse> => {
    calls.push(req);
    return { audio: gen(req), mediaType, model: "m" };
  };
  return { speakStream, calls };
}

function aiWith(providers: Record<string, Provider>, opts: { models?: Record<string, { use: string; fallback: string }> } = {}) {
  const usages: { usage: Usage; meta: CallMeta }[] = [];
  const ai = createAI({
    providers: Object.fromEntries(Object.entries(providers).map(([k, p]) => [k, () => p])),
    models: opts.models,
    defaults: { retries: { attempts: 3, initialDelayMs: 1 } },
    onUsage: (usage, meta) => void usages.push({ usage, meta }),
  });
  return { ai, usages };
}

async function drain(audio: AsyncIterable<Uint8Array>): Promise<Uint8Array[]> {
  const out: Uint8Array[] = [];
  for await (const chunk of audio) out.push(chunk);
  return out;
}

function deferred<T = void>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => (resolve = r));
  return { promise, resolve };
}

describe("ai.speakStream (fake providers)", () => {
  it("yields the chunks in order, mediaType at open, the result once drained, onUsage once after the drain", async () => {
    const s = streaming(async function* () {
      yield A;
      yield B;
      yield C;
      return { ...zeros, characters: 7 };
    }, "audio/ogg");
    const { ai, usages } = aiWith({ voice: provider("voice", { speakStream: s.speakStream }) });

    const opened = await ai.speakStream({ model: "voice:m", input: "Hello.", voice: "v", format: "opus", speed: 1.1, language: "en", headers: { x: "1" } });
    expect(opened.mediaType).toBe("audio/ogg");
    expect(usages).toHaveLength(0);
    expect(await drain(opened.audio)).toStrictEqual([A, B, C]);
    expect(await opened.result).toStrictEqual({ mediaType: "audio/ogg", usage: { ...zeros, characters: 7 }, model: "m" });
    expect(usages).toStrictEqual([{ usage: { ...zeros, characters: 7 }, meta: { model: "m", provider: "voice", alias: undefined, purpose: "speakStream", fallback: false } }]);
    // The request carries exactly the fields ai.speak passes.
    expect(s.calls[0]).toStrictEqual({ input: "Hello.", voice: "v", format: "opus", speed: 1.1, instructions: undefined, language: "en", headers: { x: "1" }, signal: undefined });
  });

  it("sends no language key when none is given, as ai.speak", async () => {
    const s = streaming(async function* () {
      yield A;
      return zeros;
    });
    const { ai } = aiWith({ voice: provider("voice", { speakStream: s.speakStream }) });
    await drain((await ai.speakStream({ model: "voice:m", input: "Hi." })).audio);
    expect("language" in s.calls[0]!).toBe(false);
  });

  it("opens on the first chunk: the promise is pending until the first chunk is in", async () => {
    const gate = deferred();
    const s = streaming(async function* () {
      await gate.promise;
      yield A;
      return zeros;
    });
    const { ai } = aiWith({ voice: provider("voice", { speakStream: s.speakStream }) });
    let resolved = false;
    const call = ai.speakStream({ model: "voice:m", input: "Hi." }).then((r) => ((resolved = true), r));
    await new Promise((r) => setTimeout(r, 10));
    expect(resolved).toBe(false);
    gate.resolve();
    const opened = await call;
    expect(await drain(opened.audio)).toStrictEqual([A]);
  });

  it("retries the opening on a transient error (503 twice, then served: 3 openings)", async () => {
    let n = 0;
    const speakStream = async (): Promise<SpeakStreamResponse> => {
      if (++n <= 2) throw Object.assign(new Error("busy"), { status: 503 });
      return {
        audio: (async function* () {
          yield A;
          return zeros;
        })(),
        mediaType: "audio/mpeg",
        model: "m",
      };
    };
    const { ai } = aiWith({ voice: provider("voice", { speakStream }) });
    const opened = await ai.speakStream({ model: "voice:m", input: "Hi." });
    expect(await drain(opened.audio)).toStrictEqual([A]);
    expect(n).toBe(3);
  });

  it("never retries or falls back after the first chunk: the error surfaces through the iteration and result", async () => {
    const busy = Object.assign(new Error("busy"), { status: 503 });
    const s = streaming(async function* () {
      yield A;
      throw busy;
    });
    let backupCalls = 0;
    const backup = provider("backup", {
      speakStream: async () => {
        backupCalls++;
        throw new Error("never");
      },
    });
    const { ai } = aiWith({ voice: provider("voice", { speakStream: s.speakStream }), backup }, { models: { mouth: { use: "voice:m", fallback: "backup:m" } } });
    const opened = await ai.speakStream({ model: "mouth", input: "Hi." });
    const got: Uint8Array[] = [];
    const err = await (async () => {
      for await (const c of opened.audio) got.push(c);
    })().catch((e: unknown) => e);
    expect(err).toBe(busy);
    await expect(opened.result).rejects.toBe(busy);
    expect(got).toStrictEqual([A]);
    expect(s.calls).toHaveLength(1);
    expect(backupCalls).toBe(0);
  });

  it("falls back when the primary fails before its first chunk; the primary's billed failure is reported first", async () => {
    const billed5 = { ...zeros, characters: 5 };
    const primary = streaming(async function* () {
      throw withBilledUsage(new Error("dropped before the first chunk"), billed5);
    });
    const secondary = streaming(async function* () {
      yield B;
      yield C;
      return { ...zeros, characters: 9 };
    });
    const { ai, usages } = aiWith(
      { voice: provider("voice", { speakStream: primary.speakStream }), backup: provider("backup", { speakStream: secondary.speakStream }) },
      { models: { mouth: { use: "voice:m", fallback: "backup:m" } } },
    );
    const opened = await ai.speakStream({ model: "mouth", input: "Hi." });
    expect(await drain(opened.audio)).toStrictEqual([B, C]);
    expect((await opened.result).usage).toStrictEqual({ ...zeros, characters: 9 });
    expect(usages.map((u) => [u.usage, u.meta.provider, u.meta.fallback, u.meta.alias])).toStrictEqual([
      [billed5, "voice", false, "mouth"],
      [{ ...zeros, characters: 9 }, "backup", true, "mouth"],
    ]);
  });

  it("an already-aborted signal is a CoaxAbortError and never reaches the provider", async () => {
    const s = streaming(async function* () {
      yield A;
      return zeros;
    });
    const { ai } = aiWith({ voice: provider("voice", { speakStream: s.speakStream }) });
    const ac = new AbortController();
    ac.abort();
    await expect(ai.speakStream({ model: "voice:m", input: "Hi.", signal: ac.signal })).rejects.toBeInstanceOf(CoaxAbortError);
    expect(s.calls).toHaveLength(0);
  });

  it("an abort mid-speech: the provider's own marked CoaxAbortError passes through, is reported once, and never falls back", async () => {
    const billed = { ...zeros, characters: 12 };
    const own = withBilledUsage(new CoaxAbortError(billed), billed);
    const s = streaming(async function* (req) {
      yield A;
      if (!req.signal!.aborted) await new Promise((r) => req.signal!.addEventListener("abort", r, { once: true }));
      throw own;
    });
    let backupCalls = 0;
    const backup = provider("backup", {
      speakStream: async () => {
        backupCalls++;
        throw new Error("never");
      },
    });
    const { ai, usages } = aiWith({ voice: provider("voice", { speakStream: s.speakStream }), backup }, { models: { mouth: { use: "voice:m", fallback: "backup:m" } } });
    const ac = new AbortController();
    const opened = await ai.speakStream({ model: "mouth", input: "Hi.", signal: ac.signal });
    const err = await (async () => {
      for await (const _ of opened.audio) ac.abort();
    })().catch((e: unknown) => e);
    expect(err).toBe(own);
    await expect(opened.result).rejects.toBe(own);
    expect(usages.map((u) => u.usage)).toStrictEqual([billed]);
    expect(backupCalls).toBe(0);
  });

  it("an abort mid-speech that the SDK reports as a DOMException becomes a zero-usage CoaxAbortError, reported nowhere", async () => {
    const s = streaming(async function* (req) {
      yield A;
      if (!req.signal!.aborted) await new Promise((r) => req.signal!.addEventListener("abort", r, { once: true }));
      throw new DOMException("This operation was aborted", "AbortError");
    });
    const { ai, usages } = aiWith({ voice: provider("voice", { speakStream: s.speakStream }) });
    const ac = new AbortController();
    const opened = await ai.speakStream({ model: "voice:m", input: "Hi.", signal: ac.signal });
    const err = await (async () => {
      for await (const _ of opened.audio) ac.abort();
    })().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(CoaxAbortError);
    expect((err as CoaxAbortError).usage).toStrictEqual(zeros);
    expect((err as CoaxAbortError).cause).toBeInstanceOf(DOMException);
    expect(usages).toHaveLength(0);
  });

  it("an early break closes the provider's stream and reports nothing (assumption O19: break reports nothing)", async () => {
    let closed = false;
    let pulledPastBreak = false;
    const s = streaming(async function* () {
      try {
        yield A;
        yield B;
        pulledPastBreak = true;
        yield C;
        return { ...zeros, characters: 3 };
      } finally {
        closed = true;
      }
    });
    const { ai, usages } = aiWith({ voice: provider("voice", { speakStream: s.speakStream }) });
    const opened = await ai.speakStream({ model: "voice:m", input: "Hi." });
    for await (const _ of opened.audio) break;
    expect(closed).toBe(true);
    expect(pulledPastBreak).toBe(false);
    expect(usages).toHaveLength(0);
  });

  it("skips zero-length chunks", async () => {
    const s = streaming(async function* () {
      yield new Uint8Array(0);
      yield A;
      yield new Uint8Array(0);
      return zeros;
    });
    const { ai } = aiWith({ voice: provider("voice", { speakStream: s.speakStream }) });
    expect(await drain((await ai.speakStream({ model: "voice:m", input: "Hi." })).audio)).toStrictEqual([A]);
  });

  it("degrades to one speak call on a provider without speakStream: one chunk, speak's usage, reported once", async () => {
    const audio = new Uint8Array([9, 9, 9]);
    const usage = { ...zeros, characters: 4 };
    const { ai, usages } = aiWith({ voice: provider("voice", { speak: async () => ({ audio, mediaType: "audio/wav", usage, model: "m" }) }) });
    const opened = await ai.speakStream({ model: "voice:m", input: "Hi." });
    expect(opened.mediaType).toBe("audio/wav");
    expect(await drain(opened.audio)).toStrictEqual([audio]);
    expect(await opened.result).toStrictEqual({ mediaType: "audio/wav", usage, model: "m" });
    expect(usages.map((u) => u.usage)).toStrictEqual([usage]);
  });

  it("a provider with neither speak nor speakStream is a CoaxUnsupportedError for speech synthesis", async () => {
    const { ai } = aiWith({ text: provider("text", {}) });
    const err = await ai.speakStream({ model: "text:m", input: "Hi." }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(CoaxUnsupportedError);
    expect((err as CoaxUnsupportedError).capability).toBe("speech synthesis");
    expect((err as CoaxUnsupportedError).provider).toBe("text");
  });
});

describe("ai.transcribeToken (fake providers)", () => {
  const TOKEN: TranscribeTokenResponse = { token: "t", url: "wss://x/v1/speech-to-text/realtime", usage: zeros, model: "m" };

  it("returns the provider's token, url, usage and model (assumption O18: a token issue reports zero units once)", async () => {
    const { ai, usages } = aiWith({ ears: provider("ears", { transcribeToken: async () => ({ ...TOKEN }) }) });
    const ai2 = createAI({
      providers: { ears: () => provider("ears", { transcribeToken: async () => ({ ...TOKEN }) }) },
      models: { alias: "ears:m" },
      onUsage: (usage, meta) => void usages.push({ usage, meta }),
    });
    expect(await ai2.transcribeToken({ model: "alias" })).toStrictEqual(TOKEN);
    expect(usages).toStrictEqual([{ usage: zeros, meta: { model: "m", provider: "ears", alias: "alias", purpose: "transcribeToken", fallback: false } }]);
    expect(await ai.transcribeToken({ model: "ears:m", purpose: "listen" })).toStrictEqual(TOKEN);
    expect(usages[1]!.meta.purpose).toBe("listen");
  });

  it("passes headers and signal through", async () => {
    const seen: unknown[] = [];
    const { ai } = aiWith({ ears: provider("ears", { transcribeToken: async (req) => (seen.push(req), TOKEN) }) });
    const ac = new AbortController();
    await ai.transcribeToken({ model: "ears:m", headers: { a: "1" }, signal: ac.signal });
    expect(seen).toStrictEqual([{ headers: { a: "1" }, signal: ac.signal }]);
  });

  it("a 429 is retried", async () => {
    let n = 0;
    const { ai } = aiWith({
      ears: provider("ears", {
        transcribeToken: async () => {
          if (++n === 1) throw Object.assign(new Error("slow down"), { status: 429 });
          return TOKEN;
        },
      }),
    });
    expect((await ai.transcribeToken({ model: "ears:m" })).token).toBe("t");
    expect(n).toBe(2);
  });

  it("a failing primary goes to the fallback", async () => {
    const { ai, usages } = aiWith(
      {
        ears: provider("ears", {
          transcribeToken: async () => {
            throw Object.assign(new Error("denied"), { status: 401 });
          },
        }),
        backup: provider("backup", { transcribeToken: async () => ({ ...TOKEN, token: "b" }) }),
      },
      { models: { listen: { use: "ears:m", fallback: "backup:m" } } },
    );
    expect((await ai.transcribeToken({ model: "listen" })).token).toBe("b");
    expect(usages.map((u) => [u.meta.provider, u.meta.fallback])).toStrictEqual([["backup", true]]);
  });

  it("a provider without it is a CoaxUnsupportedError for realtime transcription tokens", async () => {
    const { ai } = aiWith({ text: provider("text", {}) });
    const err = await ai.transcribeToken({ model: "text:m" }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(CoaxUnsupportedError);
    expect((err as CoaxUnsupportedError).capability).toBe("realtime transcription tokens");
  });

  it("an already-aborted signal is a CoaxAbortError and never reaches the provider", async () => {
    let n = 0;
    const { ai } = aiWith({ ears: provider("ears", { transcribeToken: async () => (n++, TOKEN) }) });
    const ac = new AbortController();
    ac.abort();
    await expect(ai.transcribeToken({ model: "ears:m", signal: ac.signal })).rejects.toBeInstanceOf(CoaxAbortError);
    expect(n).toBe(0);
  });
});

describe("the ambient ai delegates the new calls", () => {
  afterEach(() => reset());

  it("speakStream and transcribeToken", async () => {
    const s = streaming(async function* () {
      yield A;
      return zeros;
    });
    configure({ providers: { v: () => provider("v", { speakStream: s.speakStream, transcribeToken: async () => ({ token: "t", url: "wss://x", usage: zeros, model: "m" }) }) } });
    const ai: AI = ambient;
    expect(await drain((await ai.speakStream({ model: "v:m", input: "Hi." })).audio)).toStrictEqual([A]);
    expect((await ai.transcribeToken({ model: "v:m" })).token).toBe("t");
  });
});

// Measurer (stage 3): proofs for what the measurer's mutation sweep showed unproven (logs/stage-03/measure/mutate.log,
// N14 N16).
describe("streamed speech and tokens: gaps closed by the measurer (stage 3, fake providers)", () => {
  it("the degrade path skips an empty speak result too: no zero-length chunk, speak's usage still booked", async () => {
    // A provider of your own whose speak returns 0 bytes (the OpenAI wire's speak accepts such a 200, spec §10).
    const usage = { ...zeros, characters: 2 };
    const { ai, usages } = aiWith({ voice: provider("voice", { speak: async () => ({ audio: new Uint8Array(0), mediaType: "audio/mpeg", usage, model: "m" }) }) });
    const opened = await ai.speakStream({ model: "voice:m", input: "Hi." });
    expect(await drain(opened.audio)).toStrictEqual([]);
    expect((await opened.result).usage).toStrictEqual(usage);
    expect(usages.map((u) => u.usage)).toStrictEqual([usage]);
  });

  it("a token's usage is the provider's, on the result and through onUsage — coax does not zero it", async () => {
    // ElevenLabs reports zero units (assumption O18); a provider of your own may bill an issue, and that is booked.
    const usage = { ...zeros, inputTokens: 3 };
    const { ai, usages } = aiWith({ ears: provider("ears", { transcribeToken: async () => ({ token: "t", url: "wss://x", usage, model: "m" }) }) });
    expect((await ai.transcribeToken({ model: "ears:m" })).usage).toStrictEqual(usage);
    expect(usages.map((u) => u.usage)).toStrictEqual([usage]);
  });
});
