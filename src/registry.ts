import type { AIConfig, ElevenLabsEndpoint, GoogleEndpoint, ProviderEndpoint, RetryConfig } from "./config";
import { anthropic } from "./providers/anthropic";
import { elevenlabs } from "./providers/elevenlabs";
import { google } from "./providers/google";
import { openai } from "./providers/openai";
import { withRetry } from "./retry";
import { addUsage, billedUsage, emptyUsage, withBilledUsage, type Provider, type ReasoningEffort, type Usage } from "./types";

/** OpenAI-wire endpoint keys that have no meaning on ElevenLabs. */
const ELEVENLABS_FOREIGN_KEYS = ["transcribeModel", "speakModel", "embedModel", "tokenParam", "strict", "extraBody", "project", "location", "googleAuthOptions"] as const;

/** Settings that ride on the model alias itself rather than the provider instance — see `resolve()`. */
export interface CallSettings {
  reasoningEffort?: ReasoningEffort;
}

export interface ResolvedModel {
  primary: Provider;
  fallback?: Provider;
  providerName: string;
  /** The resolved "provider:model". */
  ref: string;
  /**
   * Alias-level settings that must NOT become part of the provider cache key (`${providerName}:${model}`,
   * see `providerFor` below) — two aliases pointing at the same model but different `reasoningEffort`
   * would otherwise fight over one cached provider instance. The caller (`ai.ts`) merges this into the
   * request instead.
   */
  callSettings: CallSettings;
}

/**
 * `withRetry` for one provider call that keeps what failed attempts were billed (`billedUsage`, e.g. an
 * embed batch that died halfway on a 503): the retry re-bills that work, so the result that finally
 * succeeds carries both, and an error that finally escapes is marked with all of it.
 */
async function withBilledRetry<R extends { usage: Usage }>(fn: () => Promise<R>, cfg: RetryConfig | undefined, signal: AbortSignal | undefined): Promise<R> {
  let lost: Usage | undefined;
  const attempt = async (): Promise<R> => {
    try {
      return await fn();
    } catch (err) {
      const billed = billedUsage(err);
      if (billed) lost = addUsage(lost ?? emptyUsage(), billed);
      throw err;
    }
  };
  try {
    const res = await withRetry(attempt, cfg, signal);
    return lost ? { ...res, usage: addUsage(lost, res.usage) } : res;
  } catch (err) {
    if (lost && typeof err === "object" && err !== null) withBilledUsage(err, lost);
    throw err;
  }
}

/** Wrap a provider so its calls retry transient errors (rate limits / 5xx / network). */
export function retrying(provider: Provider, cfg?: RetryConfig): Provider {
  return {
    name: provider.name,
    model: provider.model,
    structured: (req) => withBilledRetry(() => provider.structured(req), cfg, req.signal),
    text: (req) => withBilledRetry(() => provider.text(req), cfg, req.signal),
    // A stream is never retried mid-flight (deltas already reached the consumer); failures before the
    // first delta are covered by the ai-layer fallback instead.
    ...(provider.textStream ? { textStream: (req: Parameters<NonNullable<Provider["textStream"]>>[0]) => provider.textStream!(req) } : {}),
    ...(provider.structuredStream ? { structuredStream: (req: Parameters<NonNullable<Provider["structuredStream"]>>[0]) => provider.structuredStream!(req) } : {}),
    ...(provider.toolsStream ? { toolsStream: (req: Parameters<NonNullable<Provider["toolsStream"]>>[0]) => provider.toolsStream!(req) } : {}),
    ...(provider.embed ? { embed: (req: Parameters<NonNullable<Provider["embed"]>>[0]) => withBilledRetry(() => provider.embed!(req), cfg, req.signal) } : {}),
    // Optional capabilities are forwarded only where the provider has them, so `capability is missing`
    // stays detectable through the wrapper.
    ...(provider.tools ? { tools: (req: Parameters<NonNullable<Provider["tools"]>>[0]) => withBilledRetry(() => provider.tools!(req), cfg, req.signal) } : {}),
    ...(provider.transcribe ? { transcribe: (req: Parameters<NonNullable<Provider["transcribe"]>>[0]) => withBilledRetry(() => provider.transcribe!(req), cfg, req.signal) } : {}),
    ...(provider.speak ? { speak: (req: Parameters<NonNullable<Provider["speak"]>>[0]) => withBilledRetry(() => provider.speak!(req), cfg, req.signal) } : {}),
  };
}

export function createRegistry(config: AIConfig) {
  const cache = new Map<string, Provider>();

  function fromEndpoint(providerName: string, endpoint: ProviderEndpoint | GoogleEndpoint | ElevenLabsEndpoint, model: string): Provider {
    // The provider NAME is free (`orgops`, `local`, …); `api` says which wire protocol to speak. It
    // defaults to the built-in of the same name so `anthropic`/`openai`/`google`/`elevenlabs` still work from a bare key.
    const api =
      endpoint.api ??
      (providerName === "anthropic" || providerName === "openai" || providerName === "google" || providerName === "elevenlabs"
        ? providerName
        : undefined);
    if (!api) {
      throw new Error(
        `coax: provider "${providerName}" needs \`api: "openai" | "anthropic"\` (for a compatible endpoint), \`api: "google"\`, ` +
          `\`api: "elevenlabs"\`, or a factory — only "anthropic", "openai", "google" and "elevenlabs" are inferred from the name`,
      );
    }
    if (api === "elevenlabs") {
      // These keys mean something on the OpenAI wire or on Google and nothing here. Dropping them silently would let a
      // config that names a model in `speakModel` run on another one, so each is a config error.
      for (const key of ELEVENLABS_FOREIGN_KEYS) {
        if ((endpoint as ProviderEndpoint)[key] !== undefined) {
          const hint = key === "transcribeModel" || key === "speakModel" ? ` — name the model in the reference, e.g. "${providerName}:scribe_v2"` : "";
          throw new Error(`coax: provider "${providerName}" (api "elevenlabs") does not take \`${key}\`${hint}`);
        }
      }
      const e = endpoint as ElevenLabsEndpoint;
      return elevenlabs({ model, apiKey: e.apiKey, baseURL: e.baseURL, headers: e.headers, voice: e.voice });
    }
    if (api === "google") {
      // A ProviderEndpoint-shaped config type-checks here too (it is a member of the ProviderConfig union),
      // but google has no use for these keys — dropping a baseURL would send the caller's traffic to
      // Google's public host instead of their proxy, so each is a config error, never ignored.
      for (const key of ["baseURL", "tokenParam", "strict", "transcribeModel", "speakModel", "voice"] as const) {
        if ((endpoint as ProviderEndpoint)[key] === undefined) continue;
        throw new Error(
          key === "baseURL"
            ? `coax: provider "${providerName}" (api "google") has no baseURL — the SDK talks to the Agent Platform directly; ` +
                `for an OpenAI-compatible gateway in front of Gemini, use \`api: "openai"\` with that baseURL`
            : `coax: provider "${providerName}" (api "google") does not take \`${key}\` — it belongs to the OpenAI wire`,
        );
      }
      const g = endpoint as GoogleEndpoint;
      return google({
        model,
        apiKey: g.apiKey,
        project: g.project,
        location: g.location,
        googleAuthOptions: g.googleAuthOptions,
        headers: g.headers,
        extraBody: g.extraBody,
        embedModel: g.embedModel,
      });
    }
    const spec = endpoint as ProviderEndpoint;
    const common = { model, apiKey: spec.apiKey, baseURL: spec.baseURL, headers: spec.headers, extraBody: spec.extraBody };
    return api === "anthropic"
      ? anthropic(common)
      : openai({ ...common, transcribeModel: spec.transcribeModel, speakModel: spec.speakModel, embedModel: spec.embedModel, tokenParam: spec.tokenParam, strict: spec.strict, voice: spec.voice });
  }

  function providerFor(providerName: string, model: string): Provider {
    const key = `${providerName}:${model}`;
    const hit = cache.get(key);
    if (hit) return hit;

    const spec = config.providers[providerName];
    if (spec === undefined) throw new Error(`coax: no provider configured for "${providerName}"`);

    const provider =
      typeof spec === "function"
        ? spec(model)
        : fromEndpoint(providerName, typeof spec === "string" ? { apiKey: spec } : spec, model);
    cache.set(key, provider);
    return provider;
  }

  function splitRef(ref: string): { providerName: string; model: string } {
    // Split on the FIRST colon only — gateway model ids routinely contain slashes and further colons
    // (e.g. "orgops:chat/Qwen/Qwen3-VL-32B-Instruct-AWQ").
    const i = ref.indexOf(":");
    if (i < 0) throw new Error(`coax: model "${ref}" must be "provider:model" or a configured alias`);
    return { providerName: ref.slice(0, i), model: ref.slice(i + 1) };
  }

  /** Resolve a model reference — a configured alias or a literal "provider:model". */
  function resolve(ref: string): ResolvedModel {
    const alias = config.models?.[ref];
    let use = ref;
    let fallbackRef: string | undefined;
    let reasoningEffort: ReasoningEffort | undefined;
    if (typeof alias === "string") use = alias;
    else if (alias) { use = alias.use; fallbackRef = alias.fallback; reasoningEffort = alias.reasoningEffort; }

    const p = splitRef(use);
    const primary = providerFor(p.providerName, p.model);
    let fallback: Provider | undefined;
    if (fallbackRef) { const f = splitRef(fallbackRef); fallback = providerFor(f.providerName, f.model); }
    return { primary, fallback, providerName: p.providerName, ref: use, callSettings: { reasoningEffort } };
  }

  return { resolve };
}
