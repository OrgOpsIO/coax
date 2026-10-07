import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { AIConfig } from "../src/config";
import { createRegistry } from "../src/registry";

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
      mouth: { use: "elevenlabs:eleven_flash_v2_5", fallback: "openai:gpt-4o-mini-tts" },
      smart: "anthropic:claude-opus-4-8",
    },
  },
} satisfies Record<string, AIConfig>;

const README = readFileSync(new URL("../README.md", import.meta.url), "utf8");

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
