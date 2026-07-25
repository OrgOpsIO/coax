import { describe, expect, it } from "vitest";
import { createClient, CoaxUnsupportedError } from "../src/client";
import { createAI } from "../src/ai";
import { emptyUsage, type Provider, type SpeakRequest, type TranscribeRequest } from "../src/types";

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
