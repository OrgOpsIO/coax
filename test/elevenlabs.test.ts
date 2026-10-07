import { describe, expect, expectTypeOf, it, vi } from "vitest";
import { z } from "zod";
import { createAI } from "../src/ai";
import { CoaxUnsupportedError } from "../src/client";
import type { ElevenLabsEndpoint } from "../src/config";
import { createRegistry, retrying } from "../src/registry";
import { elevenlabs } from "../src/providers/elevenlabs";
import { CoaxToolError, tool } from "../src/tools";
import { emptyUsage, type Provider } from "../src/types";

// The SDK is mocked here: `loads` counts how often the module is imported, `constructed` records what
// coax passed to the client constructor. The real SDK runs in elevenlabs-sdk-wire.test.ts.
const sdk = vi.hoisted(() => ({ loads: 0, constructed: [] as Record<string, unknown>[], speakCalls: 0 }));

vi.mock("@elevenlabs/elevenlabs-js", () => {
  sdk.loads++;
  class ElevenLabsClient {
    constructor(opts: Record<string, unknown>) {
      sdk.constructed.push(opts);
    }
    textToSpeech = {
      convert: () => ({
        withRawResponse: async () => {
          sdk.speakCalls++;
          return { data: new Response(new Uint8Array([1, 2, 3])).body, rawResponse: { headers: new Headers({ "character-cost": "3" }) } };
        },
      }),
    };
    speechToText = { convert: async () => ({ text: "hi", audioDurationSecs: 1 }) };
  }
  return { ElevenLabsClient };
});

const lookup = tool({ name: "lookup", description: "Look something up.", input: z.object({}), run: () => "42" });

/** Rejects with a CoaxUnsupportedError for `capability` on the elevenlabs provider. */
async function unsupported(p: Promise<unknown>, capability: string) {
  const err = await p.then(
    () => undefined,
    (e: unknown) => e,
  );
  expect(err).toBeInstanceOf(CoaxUnsupportedError);
  expect((err as CoaxUnsupportedError).capability).toBe(capability);
  expect((err as CoaxUnsupportedError).provider).toBe("elevenlabs");
}

// Order matters in this file: the SDK module is cached once imported, so every "never loads" case
// runs before the first speak/transcribe.
describe("elevenlabs is voice only — no SDK load for what it does not serve", () => {
  const ai = createAI({ providers: { elevenlabs: "k" }, defaults: { retries: { attempts: 1 } } });
  const model = "elevenlabs:eleven_v3";

  it("ai.text and ai.stream name text generation", async () => {
    await unsupported(ai.text({ model, prompt: "hi" }), "text generation");
    await unsupported(
      (async () => {
        const s = await ai.stream({ model, prompt: "hi" });
        for await (const _ of s.stream);
        return s.result;
      })(),
      "text generation",
    );
    expect(sdk.loads).toBe(0);
  });

  it("ai.object and ai.streamObject name structured output", async () => {
    const schema = z.object({ a: z.string() });
    await unsupported(ai.object({ model, schema, prompt: "hi" }), "structured output");
    await unsupported(
      (async () => {
        const s = await ai.streamObject({ model, schema, prompt: "hi" });
        for await (const _ of s.partials);
        return s.result;
      })(),
      "structured output",
    );
    expect(sdk.loads).toBe(0);
  });

  it("ai.run and ai.runStream name tool calling (as the cause of the run's CoaxToolError), ai.embed names embeddings", async () => {
    // A run wraps every failed model turn in CoaxToolError (transcript and usage so far) on every
    // vendor; the missing capability is its cause, and its message names it.
    const viaRun = async (p: Promise<unknown>) => {
      const err = await p.then(() => undefined, (e: unknown) => e);
      expect(err).toBeInstanceOf(CoaxToolError);
      expect((err as Error).message).toMatch(/provider "elevenlabs" does not support tool calling/);
      await unsupported(Promise.reject((err as Error).cause), "tool calling");
    };
    await viaRun(ai.run({ model, prompt: "hi", tools: [lookup] }));
    await viaRun(
      (async () => {
        const s = await ai.runStream({ model, prompt: "hi", tools: [lookup] });
        for await (const _ of s.events);
        return s.result;
      })(),
    );
    await unsupported(ai.embed({ model, input: "hi" }), "embeddings");
    expect(sdk.loads).toBe(0);
  });

  it("an alias falls back past the missing capability like past any model failure", async () => {
    const fallback: Provider = {
      name: "openai",
      model: "gpt-x",
      structured: async () => ({ raw: {}, text: "{}", usage: emptyUsage(), model: "gpt-x" }),
      text: async () => ({ raw: "from fallback", text: "from fallback", usage: emptyUsage(), model: "gpt-x" }),
    };
    const withAlias = createAI({
      providers: { elevenlabs: "k", openai: () => fallback },
      models: { both: { use: "elevenlabs:x", fallback: "openai:gpt-x" } },
    });
    const res = await withAlias.text({ model: "both", prompt: "hi" });
    expect(res.text).toBe("from fallback");
    expect(sdk.loads).toBe(0);
  });

  it("checks before the wire load nothing either (no voice, instructions, aac/flac, speed range)", async () => {
    const p = elevenlabs({ model: "eleven_flash_v2_5", apiKey: "k" });
    await expect(p.speak!({ input: "hi" })).rejects.toThrow(/needs a voice id/);
    await unsupported(p.speak!({ input: "hi", voice: "v", instructions: "calm" }), "delivery instructions (`instructions`)");
    await unsupported(p.speak!({ input: "hi", voice: "v", format: "aac" }), "aac output");
    await unsupported(p.speak!({ input: "hi", voice: "v", format: "flac" }), "flac output");
    await expect(p.speak!({ input: "hi", voice: "v", speed: 0.5 })).rejects.toThrow(/between 0.7 and 1.2/);
    await expect(p.speak!({ input: "hi", voice: "v", speed: 1.3 })).rejects.toThrow(/between 0.7 and 1.2/);
    await unsupported(p.transcribe!({ audio: { data: new Uint8Array([1]) }, prompt: "Zod, coax" }), "a transcription prompt (`prompt`)");
    expect(sdk.loads).toBe(0);
    expect(sdk.constructed).toEqual([]);
  });

  it("the first speak loads the SDK once, with the key, no SDK retries and no headers in the constructor", async () => {
    const p = elevenlabs({ model: "eleven_flash_v2_5", apiKey: "k", voice: "v", headers: { a: "1" } });
    await p.speak!({ input: "hi" });
    await p.speak!({ input: "again" });
    expect(sdk.loads).toBe(1);
    expect(sdk.constructed).toStrictEqual([{ apiKey: "k", maxRetries: 0 }]);
  });

  it("speed at the edges of the range passes", async () => {
    const p = elevenlabs({ model: "eleven_flash_v2_5", apiKey: "k", voice: "v" });
    const before = sdk.speakCalls;
    await p.speak!({ input: "hi", speed: 0.7 });
    await p.speak!({ input: "hi", speed: 1.2 });
    expect(sdk.speakCalls).toBe(before + 2);
  });

  it("baseURL reaches the SDK as baseUrl (data-residency host)", async () => {
    sdk.constructed.length = 0;
    const p = elevenlabs({ model: "scribe_v2", apiKey: "k", baseURL: "https://api.eu.residency.elevenlabs.io" });
    await p.transcribe!({ audio: { data: new Uint8Array([1]) } });
    expect(sdk.constructed).toStrictEqual([{ apiKey: "k", maxRetries: 0, baseUrl: "https://api.eu.residency.elevenlabs.io" }]);
  });
});

describe("elevenlabs configuration", () => {
  it("needs a non-empty apiKey unless a client is injected (never the SDK's env fallback)", () => {
    expect(() => elevenlabs({ model: "m" })).toThrow(/needs an apiKey/);
    expect(() => elevenlabs({ model: "m", apiKey: "" })).toThrow(/needs an apiKey/);
    expect(() => elevenlabs({ model: "m", client: {} as never })).not.toThrow();
  });

  it("a bare key under `elevenlabs` and `api: \"elevenlabs\"` under any other name resolve to the provider", () => {
    const registry = createRegistry({ providers: { elevenlabs: "k", "labs-eu": { api: "elevenlabs", apiKey: "k" } } });
    expect(registry.resolve("elevenlabs:scribe_v2").primary.name).toBe("elevenlabs");
    const eu = registry.resolve("labs-eu:eleven_flash_v2_5");
    expect(eu.primary.name).toBe("elevenlabs");
    expect(eu.primary.model).toBe("eleven_flash_v2_5");
  });

  for (const key of ["transcribeModel", "speakModel", "embedModel", "tokenParam", "strict", "extraBody"] as const) {
    it(`rejects the OpenAI-wire key \`${key}\` instead of dropping it`, () => {
      const value = key === "strict" ? true : key === "extraBody" ? { a: 1 } : "x";
      const registry = createRegistry({ providers: { elevenlabs: { apiKey: "k", [key]: value } as never } });
      expect(() => registry.resolve("elevenlabs:scribe_v2")).toThrow(new RegExp(`does not take \`${key}\``));
    });
  }

  // Added at the merge with stage 1 (stage-02-spec §9): Google's options are a type error on ElevenLabsEndpoint
  // and, for JavaScript and configs read at runtime, a config error rather than dropped.
  for (const [key, value] of [["project", "p"], ["location", "eu"], ["googleAuthOptions", {}]] as const) {
    it(`rejects the Google key \`${key}\` instead of dropping it`, () => {
      const registry = createRegistry({ providers: { elevenlabs: { apiKey: "k", [key]: value } as never, labs: { api: "elevenlabs", apiKey: "k", [key]: value } as never } });
      expect(() => registry.resolve("elevenlabs:scribe_v2")).toThrow(new RegExp(`does not take \`${key}\``));
      expect(() => registry.resolve("labs:scribe_v2")).toThrow(new RegExp(`provider "labs" \\(api "elevenlabs"\\) does not take \`${key}\``));
    });
  }

  it("a model key points at the reference instead", () => {
    const registry = createRegistry({ providers: { labs: { api: "elevenlabs", apiKey: "k", speakModel: "x" } as never } });
    expect(() => registry.resolve("labs:eleven_v3")).toThrow(/name the model in the reference, e\.g\. "labs:scribe_v2"/);
  });

  it("an unknown name without api still asks for api, and now names elevenlabs too", () => {
    const registry = createRegistry({ providers: { mystery: { apiKey: "k" } } });
    expect(() => registry.resolve("mystery:m")).toThrow(/needs `api: "openai" \| "anthropic"`/);
    expect(() => registry.resolve("mystery:m")).toThrow(/"elevenlabs"/);
  });

  it("retrying() forwards speak and transcribe, and adds no tools or embed", () => {
    const wrapped = retrying(elevenlabs({ model: "m", apiKey: "k" }));
    expect(typeof wrapped.speak).toBe("function");
    expect(typeof wrapped.transcribe).toBe("function");
    expect(wrapped.tools).toBeUndefined();
    expect(wrapped.embed).toBeUndefined();
  });

  it("OpenAI-wire keys are a type error on ElevenLabsEndpoint", () => {
    expectTypeOf<{ api: "elevenlabs"; apiKey: string; speakModel: string }>().not.toMatchTypeOf<ElevenLabsEndpoint>();
    // @ts-expect-error — speakModel is `never` on ElevenLabsEndpoint
    const bad: ElevenLabsEndpoint = { api: "elevenlabs", apiKey: "k", speakModel: "x" };
    expect(bad.apiKey).toBe("k");
  });
});
