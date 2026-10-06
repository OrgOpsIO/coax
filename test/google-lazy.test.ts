import { describe, expect, it, vi } from "vitest";
import { createAI } from "../src/ai";

// The SDK (and the google-auth-library + ws it pulls in) must load on the first Google CALL, never on
// importing coax or resolving a model. The mock counts how often the module is loaded and records the
// options every GoogleGenAI is constructed with.
const sdk = vi.hoisted(() => ({ loads: 0, constructed: [] as Record<string, unknown>[] }));

vi.mock("@google/genai", () => {
  sdk.loads++;
  class GoogleGenAI {
    models = {
      generateContent: async () => ({ candidates: [{ content: { parts: [{ text: "ok" }] }, finishReason: "STOP" }] }),
    };
    constructor(options: Record<string, unknown>) {
      sdk.constructed.push(options);
    }
  }
  return { GoogleGenAI };
});

describe("google: lazy SDK loading and constructor options (T2)", () => {
  it("loads nothing until the first call; a bare string is an Agent Platform API key", async () => {
    const ai = createAI({ providers: { google: "k" } });
    expect(sdk.loads).toBe(0);
    await ai.text({ model: "google:gemini-3.5-flash", prompt: "?" });
    expect(sdk.loads).toBe(1);
    expect(sdk.constructed.at(-1)).toEqual({ enterprise: true, apiKey: "k" });
  });

  it("the object form without a key is Application Default Credentials on the global location", async () => {
    const ai = createAI({ providers: { google: { project: "p" } } });
    await ai.text({ model: "google:gemini-3.5-flash", prompt: "?" });
    expect(sdk.constructed.at(-1)).toEqual({ enterprise: true, project: "p", location: "global" });
  });

  it("location and googleAuthOptions are passed through", async () => {
    const googleAuthOptions = { credentials: { client_email: "<SA_EMAIL>", private_key: "<PEM>" } };
    const ai = createAI({ providers: { google: { project: "p", location: "eu", googleAuthOptions } } });
    await ai.text({ model: "google:gemini-3.5-flash", prompt: "?" });
    expect(sdk.constructed.at(-1)).toEqual({ enterprise: true, project: "p", location: "eu", googleAuthOptions });
  });

  it("apiKey and project together reject the call instead of letting the SDK pick a route", async () => {
    const before = sdk.constructed.length;
    const ai = createAI({ providers: { google: { apiKey: "k", project: "p" } } });
    await expect(ai.text({ model: "google:gemini-3.5-flash", prompt: "?" })).rejects.toThrow(/either apiKey .* or project/);
    expect(sdk.constructed.length).toBe(before);
  });

  // Review R1.1: the SDK serves a key-without-project client from the global host and would drop these.
  it("apiKey with a regional location or with googleAuthOptions rejects the call; no client is constructed", async () => {
    const before = sdk.constructed.length;
    const regional = createAI({ providers: { google: { apiKey: "k", location: "eu" } } });
    await expect(regional.text({ model: "google:gemini-3.5-flash", prompt: "?" })).rejects.toThrow(
      /apiKey from the "global" location only — location "eu" needs project/,
    );
    const sa = createAI({ providers: { google: { apiKey: "k", googleAuthOptions: { credentials: { client_email: "<SA_EMAIL>" } } } } });
    await expect(sa.text({ model: "google:gemini-3.5-flash", prompt: "?" })).rejects.toThrow(/either apiKey .* or googleAuthOptions/);
    expect(sdk.constructed.length).toBe(before);
  });

  it('apiKey with location "global" is what the key route does anyway, so it is accepted', async () => {
    const ai = createAI({ providers: { google: { apiKey: "k", location: "global" } } });
    await ai.text({ model: "google:gemini-3.5-flash", prompt: "?" });
    expect(sdk.constructed.at(-1)).toEqual({ enterprise: true, apiKey: "k" });
  });

  it("a provider constructs its client once and reuses it", async () => {
    const before = sdk.constructed.length;
    const ai = createAI({ providers: { google: "k" } });
    await ai.text({ model: "google:gemini-3.5-flash", prompt: "?" });
    await ai.text({ model: "google:gemini-3.5-flash", prompt: "again" });
    expect(sdk.constructed.length).toBe(before + 1);
    expect(sdk.loads).toBe(1);
  });
});
