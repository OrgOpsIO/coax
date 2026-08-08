import type { AIConfig, ProviderEndpoint, RetryConfig } from "./config";
import type { Provider, ReasoningEffort } from "./types";
import { anthropic } from "./providers/anthropic";
import { openai } from "./providers/openai";
import { withRetry } from "./retry";

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

/** Wrap a provider so its calls retry transient errors (rate limits / 5xx / network). */
export function retrying(provider: Provider, cfg?: RetryConfig): Provider {
  return {
    name: provider.name,
    model: provider.model,
    structured: (req) => withRetry(() => provider.structured(req), cfg, req.signal),
    text: (req) => withRetry(() => provider.text(req), cfg, req.signal),
    // A stream is never retried mid-flight (deltas already reached the consumer); failures before the
    // first delta are covered by the ai-layer fallback instead.
    ...(provider.textStream ? { textStream: (req: Parameters<NonNullable<Provider["textStream"]>>[0]) => provider.textStream!(req) } : {}),
    // Optional capabilities are forwarded only where the provider has them, so `capability is missing`
    // stays detectable through the wrapper.
    ...(provider.tools ? { tools: (req: Parameters<NonNullable<Provider["tools"]>>[0]) => withRetry(() => provider.tools!(req), cfg, req.signal) } : {}),
    ...(provider.transcribe ? { transcribe: (req: Parameters<NonNullable<Provider["transcribe"]>>[0]) => withRetry(() => provider.transcribe!(req), cfg, req.signal) } : {}),
    ...(provider.speak ? { speak: (req: Parameters<NonNullable<Provider["speak"]>>[0]) => withRetry(() => provider.speak!(req), cfg, req.signal) } : {}),
  };
}

export function createRegistry(config: AIConfig) {
  const cache = new Map<string, Provider>();

  function fromEndpoint(providerName: string, spec: ProviderEndpoint, model: string): Provider {
    // The provider NAME is free (`orgops`, `local`, …); `api` says which wire protocol to speak. It
    // defaults to the built-in of the same name so `anthropic`/`openai` still work from a bare key.
    const api = spec.api ?? (providerName === "anthropic" || providerName === "openai" ? providerName : undefined);
    if (!api) {
      throw new Error(
        `coax: provider "${providerName}" needs \`api: "openai" | "anthropic"\` (for a compatible endpoint) or a factory ` +
          `— only "anthropic" and "openai" are inferred from the name`,
      );
    }
    const common = { model, apiKey: spec.apiKey, baseURL: spec.baseURL, headers: spec.headers, extraBody: spec.extraBody };
    return api === "anthropic"
      ? anthropic(common)
      : openai({ ...common, transcribeModel: spec.transcribeModel, speakModel: spec.speakModel, tokenParam: spec.tokenParam, strict: spec.strict });
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
