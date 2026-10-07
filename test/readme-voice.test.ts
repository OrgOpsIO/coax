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
