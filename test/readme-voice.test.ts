import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { AI } from "../src/ai";
import type { AIConfig } from "../src/config";
import { createRegistry } from "../src/registry";
import type { Tool } from "../src/tools";

// The config objects the README's "Any provider" and "Voice" sections show, kept here so a type change
// that breaks them fails the build instead of the reader.
const README_CONFIGS = {
  bareKey: { providers: { elevenlabs: "<ELEVENLABS_API_KEY>" } },
  residency: {
    providers: {
      elevenlabs: "<ELEVENLABS_API_KEY>",
      "labs-eu": { api: "elevenlabs", apiKey: "<ELEVENLABS_EU_API_KEY>", baseURL: "https://api.eu.residency.elevenlabs.io" },
    },
  },
  voice: {
    providers: {
      elevenlabs: { apiKey: "<ELEVENLABS_API_KEY>", voice: "<ELEVENLABS_VOICE_ID>" },
      openai: { apiKey: "<OPENAI_API_KEY>", voice: "alloy" },
      anthropic: "<ANTHROPIC_API_KEY>",
    },
    models: {
      ears: "elevenlabs:scribe_v2",
      listen: "elevenlabs:scribe_v2_realtime",
      mouth: { use: "elevenlabs:eleven_flash_v2_5", fallback: "openai:gpt-4o-mini-tts" },
      smart: "anthropic:claude-opus-4-8",
    },
  },
} satisfies Record<string, AIConfig>;

const README = readFileSync(new URL("../README.md", import.meta.url), "utf8");

// The README's server snippets for stage 3, as the type checker sees them (`npm run typecheck` covers test/).
// Never called: they only have to compile against the public types, with minimal stand-ins for the web framework.
type Res = { setHeader(name: string, value: string): void; write(chunk: Uint8Array): void; end(): void; on(event: "close", fn: () => void): void };
type App<C> = { post(path: string, handler: (c: C, res: Res) => Promise<unknown>): void };
declare const tools: Tool[];
export function readmeStreamedSpeech(ai: AI, res: Res, ac: AbortController, answer: { text: string }) {
  return async () => {
    const { audio, mediaType } = await ai.speakStream({ model: "mouth", input: answer.text, signal: ac.signal });
    res.setHeader("content-type", mediaType);
    for await (const chunk of audio) res.write(chunk);
    res.end();
  };
}
export function readmeTokenRoute(ai: AI, app: App<{ json(body: unknown): unknown }>) {
  app.post("/v1/listen", async (c) => {
    const { token, model, url } = await ai.transcribeToken({ model: "listen" });
    return c.json({ token, model, url });
  });
}
export function readmeVoiceLoop(ai: AI, app: App<{ body: { text: string } }>) {
  app.post("/v1/talk", async (req, res) => {
    const ac = new AbortController();
    res.on("close", () => ac.abort());
    const say = async (sentence: string) => {
      const { audio } = await ai.speakStream({ model: "mouth", input: sentence, format: "mp3", signal: ac.signal });
      for await (const chunk of audio) res.write(chunk);
    };
    res.setHeader("content-type", "audio/mpeg");
    const { events } = await ai.runStream({ model: "smart", prompt: req.body.text, tools, signal: ac.signal });
    let buffer = "";
    for await (const e of events) {
      if (e.type !== "delta") continue;
      buffer += e.text;
      for (let end = buffer.search(/[.!?]\s/); end >= 0; end = buffer.search(/[.!?]\s/)) {
        await say(buffer.slice(0, end + 1));
        buffer = buffer.slice(end + 2);
      }
    }
    if (buffer.trim()) await say(buffer);
    res.end();
  });
}

describe("README: ElevenLabs and voice configuration", () => {
  it("still shows these configs", () => {
    expect(README).toContain(`"labs-eu": { api: "elevenlabs", apiKey: process.env.ELEVENLABS_EU_API_KEY!, baseURL: "https://api.eu.residency.elevenlabs.io" }`);
    expect(README).toContain(`elevenlabs: { apiKey: process.env.ELEVENLABS_API_KEY!, voice: process.env.ELEVENLABS_VOICE_ID! }`);
    expect(README).toContain(`mouth: { use: "elevenlabs:eleven_flash_v2_5", fallback: "openai:gpt-4o-mini-tts" }`);
    expect(README).toContain(`ears: "elevenlabs:scribe_v2"`);
  });

  // Review R2.3 / measurement D3, D4: the SDK's fixed request timer, measured at 240.6 s after an aborted
  // speak (.ziv/logs/stage-02/measure/probe-exit-260.log). The README must say so, with the number the
  // installed SDK uses.
  const flat = README.replace(/\s+/g, " ");
  const section = (heading: string) => flat.slice(flat.indexOf(heading), flat.indexOf(" #", flat.indexOf(heading) + heading.length));

  it("says that an aborted or dropped ElevenLabs call leaves the SDK's 240 s timer, and what a script does about it", () => {
    const cancelling = section("### Cancelling a call");
    expect(cancelling).toContain("its SDK arms a 240-second timer for every request and does not clear it when the request is aborted or the connection drops");
    expect(cancelling).toContain("a short-lived process ends with `process.exit()` after such a call");
  });

  it("says that ElevenLabs batch transcription is capped by the same 240 s and is not retried", () => {
    const voice = section("### Voice");
    expect(voice).toContain("the SDK gives each request 240 seconds");
    // The error class is measured with Node's real fetch (.ziv/logs/stage-02/fix/probe-timeout.log); the
    // wire test "the SDK's own timeout is not a network error" pins it.
    expect(voice).toContain("fails with the SDK's `ElevenLabsError` `\"timeout\"`, which is not retried");
  });

  it("the 240 s is still the installed SDK's default for speak and transcribe", () => {
    const sdk = (path: string) => readFileSync(new URL(`../node_modules/@elevenlabs/elevenlabs-js/api/resources/${path}`, import.meta.url), "utf8");
    for (const client of [sdk("textToSpeech/client/Client.js"), sdk("speechToText/client/Client.js")]) {
      expect(client).toMatch(/timeoutInSeconds\) !== null && _\w+ !== void 0 \? _\w+ : 240\) \* 1000/);
    }
  });

  it("says that `language` on speak is ignored where the ElevenLabs model does not take it (review R2.5)", () => {
    expect(section("### Voice")).toContain("`eleven_multilingual_v2` (its default model) takes none");
  });

  // The merge with stage 1 (stage-02-spec §9): every place that names the SDKs coax ships names all of them.
  it("the README's SDK sentences and the package description name every vendor SDK in dependencies", () => {
    const VENDORS: Record<string, string> = { "@anthropic-ai/sdk": "Anthropic", openai: "OpenAI", "@google/genai": "Google", "@elevenlabs/elevenlabs-js": "ElevenLabs" };
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as { description: string; dependencies: Record<string, string> };
    const sdks = Object.keys(pkg.dependencies).filter((d) => d !== "jsonrepair");
    // A new SDK fails here until it is named below — and in the sentences this test reads.
    expect([...sdks].sort()).toStrictEqual(Object.keys(VENDORS).sort());
    const intro = flat.match(/The [^.]*SDKs ship \*inside\* coax/)?.[0] ?? "";
    const design = flat.match(/the provider SDKs \([^)]*\) ship inside coax/)?.[0] ?? "";
    for (const name of Object.values(VENDORS)) {
      expect(intro).toContain(name);
      expect(design).toContain(name);
      expect(pkg.description).toContain(name);
    }
  });

  it("names ElevenLabs among the failed calls that are still billed (review R2.4)", () => {
    expect(section("### Failed runs still cost tokens")).toContain("an ElevenLabs speech billed in characters that came back without audio");
  });

  it("a bare key and the residency name resolve to the elevenlabs provider", () => {
    expect(createRegistry(README_CONFIGS.bareKey).resolve("elevenlabs:scribe_v2").primary.name).toBe("elevenlabs");
    const eu = createRegistry(README_CONFIGS.residency).resolve("labs-eu:eleven_flash_v2_5");
    expect(eu.primary.name).toBe("elevenlabs");
    expect(eu.providerName).toBe("labs-eu");
  });

  // ---- Stage 3: streamed speech, realtime listening, the streamed voice loop ----

  it("shows the realtime alias, the streamed speech and the token route", () => {
    expect(README).toContain(`listen: "elevenlabs:scribe_v2_realtime"`);
    expect(README).toContain(`const { audio, mediaType } = await ai.speakStream({ model: "mouth", input: answer.text, signal: ac.signal });`);
    expect(README).toContain(`const { token, model, url } = await ai.transcribeToken({ model: "listen" });`);
    expect(README).toContain(`Scribe.connect({ token, modelId: model, baseUri: new URL(url).origin, commitStrategy: CommitStrategy.VAD, microphone: { echoCancellation: true } })`);
    expect(createRegistry(README_CONFIGS.voice).resolve("listen").primary.name).toBe("elevenlabs");
  });

  it("says what a token costs and why the OpenAI wire has none (assumption O18)", () => {
    const voice = section("### Voice");
    expect(voice).toContain("`onUsage` sees each issued token once, with zero units: the session itself is billed by ElevenLabs by audio duration, and coax never sees it");
    expect(voice).toContain("On the OpenAI wire `transcribeToken` raises `CoaxUnsupportedError`: its realtime secret can be used more than once until it expires, so it is not single-use.");
    expect(voice).toContain("The token is single-use and lives 15 minutes (the vendor's rule), and the key never leaves the server.");
  });

  it("says how streamed speech is billed and stopped (assumption O19: break books nothing)", () => {
    const voice = section("### Voice");
    expect(voice).toContain("Stop a speech early with the `signal`: the connection closes, and what was billed reaches `onUsage` and rides on the `CoaxAbortError`.");
    expect(voice).toContain("A `break` out of the loop closes the connection too, but books nothing.");
    expect(voice).toContain("On the OpenAI wire the audio streams as it is generated, with no usage, as `speak()`.");
  });

  it("narrows the 240 s timer caveat to calls stopped before ElevenLabs answered (measured, facts §3.4)", () => {
    expect(section("### Cancelling a call")).toContain("This holds only until ElevenLabs answers: once the answer has started (a speech is streaming, a body is arriving), the SDK has cleared the timer");
  });

  it("names a speech cut off after it was billed among the billed failures, and the new capabilities in Streaming and Design", () => {
    expect(section("### Failed runs still cost tokens")).toContain("an ElevenLabs speech cut off after it was billed (a dropped connection, an abort)");
    expect(section("### Streaming")).toContain("`speakStream()` for speech");
    expect(flat).toContain("`tools` / `transcribe` / `speak` / `speakStream` / `transcribeToken` are optional capabilities");
  });

  it("shows the streamed voice loop: sentence cut, one AbortController for both, the rest at the end", () => {
    const bff = section("## In a backend-for-frontend");
    expect(bff).toContain(`res.on("close", () => ac.abort());`);
    expect(bff).toContain("buffer.search(/[.!?]\\s/)");
    expect(bff).toContain(`await ai.runStream({ model: "smart", prompt: req.body.text, tools, signal: ac.signal })`);
    expect(bff).toContain(`await ai.speakStream({ model: "mouth", input: sentence, format: "mp3", signal: ac.signal })`);
    expect(bff).toContain("if (buffer.trim()) await say(buffer);");
    expect(bff).toContain("`mp3` and `pcm` concatenate cleanly into one response; `wav` carries a header per speech.");
  });

  it("the voice aliases resolve: ears on elevenlabs, mouth on elevenlabs with an openai fallback", () => {
    const registry = createRegistry(README_CONFIGS.voice);
    expect(registry.resolve("ears").primary.name).toBe("elevenlabs");
    const mouth = registry.resolve("mouth");
    expect(mouth.primary.name).toBe("elevenlabs");
    expect(mouth.primary.model).toBe("eleven_flash_v2_5");
    expect(mouth.fallback?.name).toBe("openai");
    expect(mouth.fallback?.speak).toBeTypeOf("function");
  });
});
