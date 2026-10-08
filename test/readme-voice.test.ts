import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createAI, type AI } from "../src/ai";
import { CoaxAbortError } from "../src/client";
import type { AIConfig } from "../src/config";
import { elevenlabs } from "../src/providers/elevenlabs";
import { createRegistry } from "../src/registry";
import type { Tool } from "../src/tools";
import { emptyUsage, withBilledUsage, type Provider, type Usage } from "../src/types";

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
type Middleware = (c: never, next: () => Promise<void>) => Promise<unknown>;
type App<C> = { post(path: string, ...handlers: [...Middleware[], (c: C, res: Res) => Promise<unknown>]): void };
declare const requireUser: Middleware;
export function readmeStreamedSpeech(ai: AI, res: Res, ac: AbortController, answer: { text: string }) {
  return async () => {
    const { audio, mediaType } = await ai.speakStream({ model: "mouth", input: answer.text, signal: ac.signal });
    res.setHeader("content-type", mediaType);
    for await (const chunk of audio) res.write(chunk);
    res.end();
  };
}
export function readmeTokenRoute(ai: AI, app: App<{ json(body: unknown): unknown }>) {
  app.post("/v1/listen", requireUser, async (c) => {
    const { token, model, url } = await ai.transcribeToken({ model: "listen" });
    return c.json({ token, model, url });
  });
}
// Also run, against fake providers, by the barge-in test below (review R3.2) — `tools` is a parameter only for that.
export function readmeVoiceLoop(ai: AI, app: App<{ body: { text: string } }>, tools: Tool[] = []) {
  app.post("/v1/talk", async (req, res) => {
    const ac = new AbortController();
    res.on("close", () => ac.abort());
    const say = async (sentence: string) => {
      const { audio } = await ai.speakStream({ model: "mouth", input: sentence, format: "mp3", signal: ac.signal });
      for await (const chunk of audio) res.write(chunk);
    };
    try {
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
    } catch (err) {
      if (!(err instanceof CoaxAbortError)) throw err;
    }
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
    expect(README).toContain(`Scribe.connect({ token, modelId: model, baseUri: url.slice(0, url.lastIndexOf("/v1/")), commitStrategy: CommitStrategy.VAD, microphone: { echoCancellation: true } })`);
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
    // Review R3.1: also for a speech opened ahead and not iterated yet. Review R3.3: what coax books, not what the vendor bills.
    expect(voice).toContain(
      "Stop a speech early with the `signal`, whether you are iterating it or opened it ahead: the connection closes, and the reported characters reach `onUsage` and ride on the `CoaxAbortError`.",
    );
    expect(voice).toContain("A `break` out of the loop closes the connection too, but books nothing.");
    expect(voice).toContain("On the OpenAI wire the audio streams as it is generated, with no usage, as `speak()`.");
  });

  it("narrows the 240 s timer caveat to calls stopped before ElevenLabs answered (measured, facts §3.4)", () => {
    expect(section("### Cancelling a call")).toContain("This holds only until ElevenLabs answers: once the answer has started (a speech is streaming, a body is arriving), the SDK has cleared the timer");
  });

  it("names a speech cut off after it was billed among the billed failures, and the new capabilities in Streaming and Design", () => {
    expect(section("### Failed runs still cost tokens")).toContain(
      "an ElevenLabs speech cut off after its header reported the characters (a dropped connection, an abort — coax books what the header said; whether ElevenLabs bills a cut-off speech is not documented)",
    );
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

  // ---- Fixer (stage 3): review R3.2–R3.5 ----

  it("names the voice loop's framework and catches the barge-in's CoaxAbortError (review R3.2)", () => {
    const bff = section("## In a backend-for-frontend");
    expect(bff).toContain("This one is an Express route, because it writes to the response as the audio arrives");
    expect(bff).toContain("if (!(err instanceof CoaxAbortError)) throw err;");
    expect(bff).toContain("A barge-in ends the handler with `CoaxAbortError`: catch it as above, or Express 4 and plain `node:http` leave it unhandled and Node ends the process.");
  });

  it("the voice loop, run against fake providers: a barge-in mid-speech settles the handler, and the speech in progress is booked once (review R3.2)", async () => {
    const billed: Usage = { ...emptyUsage(), characters: 12 };
    const zeros = emptyUsage();
    const base = { structured: async () => ({ raw: {}, text: "{}", usage: zeros, model: "m" }), text: async () => ({ raw: "", text: "", usage: zeros, model: "m" }) };
    const smart: Provider = {
      ...base,
      name: "smart",
      model: "m",
      tools: async () => ({ text: "", calls: [], usage: zeros, model: "m" }),
      toolsStream: async function* () {
        yield "Hello there. ";
        yield "How are you? ";
        return { text: "Hello there. How are you? ", calls: [], usage: zeros, model: "m" };
      },
    };
    // Shaped like ElevenLabs: billed at the header; after the first chunk the body fails on abort with the provider's marked error.
    const mouth: Provider = {
      ...base,
      name: "voice",
      model: "m",
      speakStream: async (req) => ({
        mediaType: "audio/mpeg",
        model: "m",
        audio: (async function* () {
          yield new Uint8Array([1]);
          if (!req.signal!.aborted) await new Promise((r) => req.signal!.addEventListener("abort", r, { once: true }));
          throw withBilledUsage(new CoaxAbortError(billed), billed);
        })(),
      }),
    };
    const usages: Usage[] = [];
    const ai = createAI({ providers: { smart: () => smart, voice: () => mouth }, models: { smart: "smart:m", mouth: "voice:m" }, onUsage: (u) => void usages.push(u) });
    let handler!: (c: { body: { text: string } }, res: Res) => Promise<unknown>;
    readmeVoiceLoop(ai, { post: (_path, ...hs) => void (handler = hs.at(-1) as typeof handler) });
    let hangUp = () => {};
    let writes = 0;
    let ended = false;
    const res: Res = {
      setHeader: () => {},
      on: (_e, fn) => void (hangUp = fn),
      write: () => {
        if (++writes === 1) hangUp(); // the user talks over the first sentence
      },
      end: () => void (ended = true),
    };
    await expect(handler({ body: { text: "Hi" } }, res)).resolves.toBeUndefined();
    expect(writes).toBe(1);
    expect(ended).toBe(false);
    expect(usages).toStrictEqual([billed]);
  });

  it("says what coax books for a speech, not what the vendor bills (review R3.3)", () => {
    const voice = section("### Voice");
    expect(voice).toContain("coax books the characters ElevenLabs reports in its response header, which arrives before the audio");
    expect(voice).toContain("Whether ElevenLabs bills a speech cut off early is not documented.");
    expect(section("## In a backend-for-frontend")).toContain("The abort books the speech in progress as the vendor reported it — the whole sentence, not just the part that played.");
    expect(flat).not.toContain("ElevenLabs bills a speech when it starts");
    expect(flat).not.toContain("The abort books what was already spoken");
  });

  it("puts the token route behind the app's auth and says why (review R3.4)", () => {
    const voice = section("### Voice");
    expect(voice).toContain(`app.post("/v1/listen", requireUser, async (c) => {`);
    expect(voice).toContain("Keep the route behind your auth: each token opens a realtime session billed to your account, so whoever can fetch one spends your money");
  });

  it("the browser's baseUri keeps a baseURL's path prefix: baseUri + the client's realtime path is coax's url (review R3.5)", async () => {
    // @elevenlabs/client 1.27.0 connects to `${baseUri}/v1/speech-to-text/realtime` (reviewer probe R3, logs/stage-03/review/probe.log).
    const client = { textToSpeech: { convert: () => undefined }, speechToText: { convert: () => undefined }, tokens: { singleUse: { create: async () => ({ token: "t" }) } } };
    expect(section("### Voice")).toContain("`baseUri` is `url` without its `/v1/…` path, so a `baseURL` with a path of its own (a proxy) carries over.");
    for (const [baseURL, expected] of [
      [undefined, "wss://api.elevenlabs.io"],
      ["https://api.eu.residency.elevenlabs.io", "wss://api.eu.residency.elevenlabs.io"],
      ["https://proxy.example.com/elevenlabs", "wss://proxy.example.com/elevenlabs"],
    ] as const) {
      const { url } = await elevenlabs({ model: "scribe_v2_realtime", client, baseURL }).transcribeToken!({});
      const baseUri = url.slice(0, url.lastIndexOf("/v1/"));
      expect(baseUri).toBe(expected);
      expect(`${baseUri}/v1/speech-to-text/realtime`).toBe(url);
    }
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
