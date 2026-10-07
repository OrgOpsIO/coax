import { describe, expect, it } from "vitest";
import { createClient, CoaxUnsupportedError } from "../src/client";
import { createAI } from "../src/ai";
import { emptyUsage, type Provider, type SpeakRequest, type TranscribeRequest } from "../src/types";
import { readFileSync } from "node:fs";
import { afterEach, vi } from "vitest";
import { openai } from "../src/providers/openai";
import { createRegistry } from "../src/registry";

/** A provider that serves audio, recording what it was asked for. */
function audioProvider(): Provider & { transcribeCalls: TranscribeRequest[]; speakCalls: SpeakRequest[] } {
  const transcribeCalls: TranscribeRequest[] = [];
  const speakCalls: SpeakRequest[] = [];
  return {
    name: "mock",
    model: "whisper-mock",
    transcribeCalls,
    speakCalls,
    structured: async () => ({ raw: {}, text: "{}", usage: emptyUsage(), model: "whisper-mock" }),
    text: async () => ({ raw: "", text: "", usage: emptyUsage(), model: "whisper-mock" }),
    async transcribe(req) {
      transcribeCalls.push(req);
      return { text: "Guten Tag, ich hätte eine Frage zu meiner Rechnung.", usage: { ...emptyUsage(), inputTokens: 3 }, model: "whisper-mock" };
    },
    async speak(req) {
      speakCalls.push(req);
      return { audio: new Uint8Array([1, 2, 3]), mediaType: "audio/mpeg", usage: emptyUsage(), model: "whisper-mock" };
    },
  };
}

/** A chat-only provider — the common case for a vendor API or a gateway without audio routes. */
const chatOnly: Provider = {
  name: "chat-only",
  model: "text-mock",
  structured: async () => ({ raw: {}, text: "{}", usage: emptyUsage(), model: "text-mock" }),
  text: async () => ({ raw: "hi", text: "hi", usage: emptyUsage(), model: "text-mock" }),
};

describe("audio capabilities", () => {
  it("transcribes and reports usage", async () => {
    const provider = audioProvider();
    const seen: number[] = [];
    const client = createClient({ provider, onUsage: (u) => void seen.push(u.inputTokens) });
    const { text, model } = await client.transcribe({ audio: { data: new Uint8Array([0]), mediaType: "audio/webm" }, language: "de" });

    expect(text).toContain("Rechnung");
    expect(model).toBe("whisper-mock");
    expect(provider.transcribeCalls[0]!.language).toBe("de");
    expect(seen).toEqual([3]);
  });

  it("synthesizes speech and returns bytes with a media type", async () => {
    const client = createClient({ provider: audioProvider() });
    const { audio, mediaType } = await client.speak({ input: "Guten Tag.", voice: "de-female", format: "mp3" });
    expect(Array.from(audio)).toEqual([1, 2, 3]);
    expect(mediaType).toBe("audio/mpeg");
  });

  it("names the missing capability instead of failing obscurely", async () => {
    const client = createClient({ provider: chatOnly });
    await expect(client.transcribe({ audio: { data: new Uint8Array([0]) } })).rejects.toBeInstanceOf(CoaxUnsupportedError);
    await expect(client.speak({ input: "x" })).rejects.toThrow(/does not support speech synthesis/);
  });

  it("routes ai.transcribe()/ai.speak() through the configured model alias", async () => {
    const provider = audioProvider();
    const meta: string[] = [];
    const ai = createAI({
      providers: { local: () => provider },
      models: { voice: "local:whisper-1" },
      onUsage: (_u, m) => void meta.push(`${m.alias}:${m.purpose}`),
    });

    await ai.transcribe({ model: "voice", audio: { data: new Uint8Array([0]) } });
    await ai.speak({ model: "voice", input: "Hallo" });
    expect(meta).toEqual(["voice:transcribe", "voice:speak"]);
  });

  it("falls back to the alias' fallback model when the primary endpoint has no audio", async () => {
    const working = audioProvider();
    const ai = createAI({
      providers: {
        primary: () => chatOnly,
        backup: () => working,
      },
      models: { voice: { use: "primary:none", fallback: "backup:whisper-1" } },
    });
    const { text } = await ai.transcribe({ model: "voice", audio: { data: new Uint8Array([0]) } });
    expect(text).toContain("Rechnung");
  });
});

describe("voice fields through ai.* (speakers, language, words)", () => {
  it("passes speakers and language through, and adds no key when they are unset", async () => {
    const provider = audioProvider();
    const ai = createAI({ providers: { local: () => provider } });
    await ai.transcribe({ model: "local:m", audio: { data: new Uint8Array([0]) }, speakers: true });
    await ai.transcribe({ model: "local:m", audio: { data: new Uint8Array([0]) } });
    await ai.speak({ model: "local:m", input: "Hallo", language: "de" });
    await ai.speak({ model: "local:m", input: "Hallo" });
    expect(provider.transcribeCalls[0]!.speakers).toBe(true);
    expect("speakers" in provider.transcribeCalls[1]!).toBe(false);
    expect(provider.speakCalls[0]!.language).toBe("de");
    expect("language" in provider.speakCalls[1]!).toBe(false);
  });

  it("a transcript without words keeps exactly today's result keys", async () => {
    const ai = createAI({ providers: { local: () => audioProvider() } });
    const res = await ai.transcribe({ model: "local:m", audio: { data: new Uint8Array([0]) } });
    expect(res).toStrictEqual({ text: "Guten Tag, ich hätte eine Frage zu meiner Rechnung.", usage: { ...emptyUsage(), inputTokens: 3 }, model: "whisper-mock" });
  });

  it("words from the provider reach the result", async () => {
    const words = [{ text: "Hallo", start: 0, end: 0.4, speaker: "speaker_0" }];
    const provider: Provider = { ...audioProvider(), transcribe: async () => ({ text: "Hallo", words, usage: emptyUsage(), model: "m" }) };
    const res = await createClient({ provider }).transcribe({ audio: { data: new Uint8Array([0]) } });
    expect(res.words).toStrictEqual(words);
  });
});

// openai SDK 6.46.0 response shapes — copy of .ziv/reference/fixtures/stage-02/openai-audio-usage.sdk.json.
const OPENAI_AUDIO = JSON.parse(readFileSync(new URL("./fixtures/elevenlabs/openai-audio-usage.sdk.json", import.meta.url), "utf8")).fixture;

/** A fake openai client recording the audio bodies it was sent. */
function fakeOpenaiAudio(transcription: unknown = { text: "Hallo." }) {
  const speech: Record<string, unknown>[] = [];
  const transcriptions: Record<string, unknown>[] = [];
  const client = {
    audio: {
      speech: {
        create: async (body: Record<string, unknown>) => {
          speech.push(body);
          return new Response(new Uint8Array([1, 2]));
        },
      },
      transcriptions: {
        create: async (body: Record<string, unknown>) => {
          transcriptions.push(body);
          return transcription;
        },
      },
    },
  };
  return { client: client as never, speech, transcriptions };
}

describe("openai wire: endpoint voice, language, speakers, whisper seconds", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("uses the endpoint's voice; a per-call voice wins; else alloy", async () => {
    const fake = fakeOpenaiAudio();
    await openai({ model: "tts-x", client: fake.client, voice: "de-female" }).speak!({ input: "Hallo" });
    await openai({ model: "tts-x", client: fake.client, voice: "de-female" }).speak!({ input: "Hallo", voice: "nova" });
    await openai({ model: "tts-x", client: fake.client }).speak!({ input: "Hallo" });
    expect(fake.speech.map((b) => b.voice)).toEqual(["de-female", "nova", "alloy"]);
  });

  it("a voice of \"\" counts as no voice, per call and on the endpoint, as on ElevenLabs (review R2.7)", async () => {
    const fake = fakeOpenaiAudio();
    await openai({ model: "tts-x", client: fake.client, voice: "de-female" }).speak!({ input: "Hallo", voice: "" });
    await openai({ model: "tts-x", client: fake.client, voice: "" }).speak!({ input: "Hallo" });
    expect(fake.speech.map((b) => b.voice)).toEqual(["de-female", "alloy"]);
  });

  it("the registry hands an endpoint's voice to the OpenAI wire", async () => {
    const bodies: Record<string, unknown>[] = [];
    vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
      bodies.push(JSON.parse(String(init.body)));
      return new Response(new Uint8Array([1]), { status: 200, headers: { "content-type": "audio/mpeg" } });
    });
    const registry = createRegistry({ providers: { orgops: { apiKey: "k", baseURL: "https://gateway.invalid/v1", api: "openai", voice: "de-female" } } });
    await registry.resolve("orgops:tts-x").primary.speak!({ input: "Hallo" });
    expect(bodies[0]!.voice).toBe("de-female");
  });

  it("does not send language — /audio/speech has no such field", async () => {
    const fake = fakeOpenaiAudio();
    await openai({ model: "tts-x", client: fake.client }).speak!({ input: "Hallo", language: "de" });
    expect("language" in fake.speech[0]!).toBe(false);
  });

  it("speaker labels are an error before the wire, not a transcript silently without them", async () => {
    const fake = fakeOpenaiAudio();
    const err = await openai({ model: "whisper-1", client: fake.client })
      .transcribe!({ audio: { data: new Uint8Array([0]) }, speakers: true })
      .catch((e: unknown) => e);
    expect(err).toBeInstanceOf(CoaxUnsupportedError);
    expect((err as CoaxUnsupportedError).capability).toBe("speaker labels (`speakers`)");
    expect(fake.transcriptions).toHaveLength(0);
  });

  it("whisper's duration usage becomes audioSeconds; token usage keeps exactly the four fields", async () => {
    const byDuration = await openai({ model: "whisper-1", client: fakeOpenaiAudio(OPENAI_AUDIO.transcription_duration).client })
      .transcribe!({ audio: { data: new Uint8Array([0]) } });
    expect(byDuration.usage).toStrictEqual({ ...emptyUsage(), audioSeconds: 4 });
    const byTokens = await openai({ model: "gpt-4o-transcribe", client: fakeOpenaiAudio(OPENAI_AUDIO.transcription_tokens).client })
      .transcribe!({ audio: { data: new Uint8Array([0]) } });
    expect(byTokens.usage).toStrictEqual({ inputTokens: 52, outputTokens: 3, cacheReadTokens: 0, cacheWriteTokens: 0 });
  });

  it("derives the upload's filename as before the move to providers/audio.ts", async () => {
    const fake = fakeOpenaiAudio();
    const p = openai({ model: "whisper-1", client: fake.client });
    await p.transcribe!({ audio: { data: new Uint8Array([0]), mediaType: "audio/webm" } });
    await p.transcribe!({ audio: { data: new Uint8Array([0]) } });
    const file = (i: number) => fake.transcriptions[i]!.file as File;
    expect([file(0).name, file(0).type]).toEqual(["audio.webm", "audio/webm"]);
    expect([file(1).name, file(1).type]).toEqual(["audio.wav", "audio/wav"]);
  });
});
