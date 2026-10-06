import { describe, expect, it } from "vitest";
import type { AIConfig } from "../src/config";
import { createRegistry } from "../src/registry";

// The config objects the README's Google sections show, kept here so a type change that breaks them
// fails the build instead of the reader.
const serviceAccountJson = { client_email: "<SA_EMAIL>", private_key: "<PEM>" };

const README_CONFIGS = {
  bareKey: { providers: { google: "<GOOGLE_API_KEY>" }, models: { flash: "google:gemini-3.5-flash" } },
  adc: { providers: { google: { project: "my-project", location: "global" } } },
  serviceAccount: { providers: { google: { project: "my-project", googleAuthOptions: { credentials: serviceAccountJson } } } },
  secondRegion: {
    providers: { google: "<GOOGLE_API_KEY>", "google-eu": { api: "google", project: "my-project", location: "eu" } },
    models: { flash: "google:gemini-3.5-flash", flashEu: "google-eu:gemini-3.5-flash" },
  },
  embeddings: { providers: { google: { project: "my-project" } } },
} satisfies Record<string, AIConfig>;

describe("README: Google configuration (T17)", () => {
  it("every form resolves to the google provider", () => {
    expect(createRegistry(README_CONFIGS.bareKey).resolve("flash").primary.name).toBe("google");
    expect(createRegistry(README_CONFIGS.adc).resolve("google:gemini-3.5-flash").primary.name).toBe("google");
    expect(createRegistry(README_CONFIGS.serviceAccount).resolve("google:gemini-3.5-flash").primary.name).toBe("google");
    expect(createRegistry(README_CONFIGS.secondRegion).resolve("flashEu").primary.name).toBe("google");
  });

  it("only the Application Default Credentials form has embed", () => {
    expect(createRegistry(README_CONFIGS.bareKey).resolve("google:gemini-embedding-001").primary.embed).toBeUndefined();
    expect(createRegistry(README_CONFIGS.embeddings).resolve("google:gemini-embedding-001").primary.embed).toBeTypeOf("function");
  });
});
