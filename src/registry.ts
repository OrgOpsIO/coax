import type { AIConfig, RetryConfig } from "./config";
import type { Provider } from "./types";
import { anthropic } from "./providers/anthropic";
import { openai } from "./providers/openai";
import { withRetry } from "./retry";

export interface ResolvedModel {
  primary: Provider;
  fallback?: Provider;
  providerName: string;
  /** The resolved "provider:model". */
  ref: string;
}

/** Wrap a provider so its calls retry transient errors (rate limits / 5xx / network). */
export function retrying(provider: Provider, cfg?: RetryConfig): Provider {
  return {
    name: provider.name,
    model: provider.model,
    structured: (req) => withRetry(() => provider.structured(req), cfg),
    text: (req) => withRetry(() => provider.text(req), cfg),
  };
}

export function createRegistry(config: AIConfig) {
  const cache = new Map<string, Provider>();

  function providerFor(providerName: string, model: string): Provider {
    const key = `${providerName}:${model}`;
    const hit = cache.get(key);
    if (hit) return hit;

    const spec = config.providers[providerName];
    if (spec === undefined) throw new Error(`coax: no provider configured for "${providerName}"`);

    let provider: Provider;
    if (typeof spec === "function") {
      provider = spec(model);
    } else {
      const apiKey = typeof spec === "string" ? spec : spec.apiKey;
      const baseURL = typeof spec === "string" ? undefined : spec.baseURL;
      if (providerName === "anthropic") provider = anthropic({ apiKey, baseURL, model });
      else if (providerName === "openai") provider = openai({ apiKey, baseURL, model });
      else throw new Error(`coax: provider "${providerName}" needs a factory (only anthropic/openai are built in)`);
    }
    cache.set(key, provider);
    return provider;
  }

  function splitRef(ref: string): { providerName: string; model: string } {
    const i = ref.indexOf(":");
    if (i < 0) throw new Error(`coax: model "${ref}" must be "provider:model" or a configured alias`);
    return { providerName: ref.slice(0, i), model: ref.slice(i + 1) };
  }

  /** Resolve a model reference — a configured alias or a literal "provider:model". */
  function resolve(ref: string): ResolvedModel {
    const alias = config.models?.[ref];
    let use = ref;
    let fallbackRef: string | undefined;
    if (typeof alias === "string") use = alias;
    else if (alias) { use = alias.use; fallbackRef = alias.fallback; }

    const p = splitRef(use);
    const primary = providerFor(p.providerName, p.model);
    let fallback: Provider | undefined;
    if (fallbackRef) { const f = splitRef(fallbackRef); fallback = providerFor(f.providerName, f.model); }
    return { primary, fallback, providerName: p.providerName, ref: use };
  }

  return { resolve };
}
