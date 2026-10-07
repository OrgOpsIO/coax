import { describe, expect, it } from "vitest";
import type { AIConfig } from "../src/config";

// Compile-time contract of the `providers` map: an option of another vendor is a type error, as it was
// before Google joined. `npm run typecheck` fails if an `@ts-expect-error` below stops being an error.
const factory = () => ({}) as never;

const valid = [
  { providers: { openai: "sk", anthropic: "sk", google: "key" } },
  { providers: { anthropic: { apiKey: "sk", baseURL: "https://x/anthropic" } } },
  { providers: { openai: { apiKey: "sk", transcribeModel: "whisper-1", tokenParam: "max_tokens", strict: true } } },
  { providers: { google: { project: "p", location: "europe-west4" } } },
  { providers: { google: { apiKey: "key", location: "global" } } },
  { providers: { orgops: { apiKey: "sk", baseURL: "https://x/v1", api: "openai", speakModel: "tts-1" } } },
  { providers: { "google-eu": { api: "google", project: "p", location: "eu" } } },
  { providers: { mine: factory, openai: factory, google: factory } },
] satisfies AIConfig[];

const invalid: AIConfig[] = [
  // @ts-expect-error project is Google's
  { providers: { openai: { apiKey: "sk", project: "p" } } },
  // @ts-expect-error location is Google's
  { providers: { anthropic: { apiKey: "sk", location: "eu" } } },
  // @ts-expect-error the Google form under openai
  { providers: { openai: { project: "p" } } },
  // @ts-expect-error baseURL is not a Google option
  { providers: { google: { project: "p", baseURL: "https://x/v1" } } },
  // @ts-expect-error speakModel is not a Google option
  { providers: { google: { apiKey: "key", speakModel: "tts-1" } } },
  // @ts-expect-error a compatible endpoint with a Google key
  { providers: { orgops: { apiKey: "sk", baseURL: "https://x/v1", api: "openai", project: "p" } } },
  // @ts-expect-error a compatible endpoint with a Google key, no api
  { providers: { orgops: { apiKey: "sk", baseURL: "https://x/v1", googleAuthOptions: {} } } },
];

describe("providers map types", () => {
  it("the valid forms are kept and the invalid ones are compile errors (checked by typecheck)", () => {
    expect(valid).toHaveLength(8);
    expect(invalid).toHaveLength(7);
  });
});
