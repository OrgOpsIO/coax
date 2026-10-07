import { describe, expect, it, vi } from "vitest";
import { createAI } from "../src/ai";

// Measurer (stage 1): two proofs google-lazy.test.ts cannot give — its load counter is already 1 after its
// first test. Here a fresh module mock (Vitest isolates modules per file) records loads and every request.
const sdk = vi.hoisted(() => ({ loads: 0, sent: [] as Record<string, any>[], embedded: [] as Record<string, any>[] }));

vi.mock("@google/genai", () => {
  sdk.loads++;
  class GoogleGenAI {
    models = {
      generateContent: async (params: Record<string, any>) => {
        sdk.sent.push(params);
        return { candidates: [{ content: { parts: [{ text: "ok" }] }, finishReason: "STOP" }] };
      },
      embedContent: async (params: Record<string, any>) => {
        sdk.embedded.push(params);
        return { embeddings: [{ values: [0.1], statistics: { tokenCount: 2 } }] };
      },
    };
  }
  return { GoogleGenAI };
});

describe("google through the registry (measurer)", () => {
  it("resolving a google model and failing a key-only embed load no SDK (spec T2a)", async () => {
    const ai = createAI({ providers: { google: "k" } });
    // The registry builds the google provider here; the call fails on the missing capability before any request.
    await expect(ai.embed({ model: "google:gemini-embedding-001", input: "a" })).rejects.toMatchObject({ name: "CoaxUnsupportedError" });
    expect(sdk.loads).toBe(0);
  });

  it("an empty embed batch on an ADC provider loads no SDK either (measurer, run 2: M53)", async () => {
    const ai = createAI({ providers: { google: { project: "p" } } });
    const res = await ai.embed({ model: "google:gemini-embedding-001", input: [] });
    expect(res.embeddings).toEqual([]);
    expect(sdk.loads).toBe(0);
  });

  it("a GoogleEndpoint's headers, extraBody and embedModel reach the provider", async () => {
    const ai = createAI({
      providers: {
        google: { project: "p", headers: { "x-team": "a" }, extraBody: { labels: { team: "a" } }, embedModel: "gemini-embedding-001" },
      },
    });
    await ai.text({ model: "google:gemini-3.5-flash", prompt: "?", headers: { "x-user": "u" } });
    expect(sdk.sent.at(-1)!.config.httpOptions).toEqual({ headers: { "x-team": "a", "x-user": "u" }, extraBody: { labels: { team: "a" } } });

    const res = await ai.embed({ model: "google:gemini-3.5-flash", input: "a" });
    expect(sdk.embedded.at(-1)!.model).toBe("gemini-embedding-001");
    expect(res.model).toBe("gemini-embedding-001");
  });
});
