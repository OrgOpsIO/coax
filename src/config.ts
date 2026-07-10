import type { Provider, Usage } from "./types";

/**
 * How a provider is configured. Either:
 *  - an API key string (for the built-in `anthropic` / `openai` providers), or
 *  - `{ apiKey, baseURL }` for the same, or
 *  - a factory `(model) => Provider` to plug in ANY provider (Gemini, a local model, a mock in tests).
 */
export type ProviderConfig = string | { apiKey: string; baseURL?: string } | ((model: string) => Provider);

/** A model alias resolves to `"provider:model"`, optionally with a fallback model on failure. */
export type ModelConfig = string | { use: string; fallback?: string };

export interface RetryConfig {
  /** Total attempts on transient errors (429/5xx/network). Default 3. */
  attempts?: number;
  initialDelayMs?: number;
  maxDelayMs?: number;
}

export interface CallDefaults {
  model?: string;
  maxRepairs?: number;
  maxTokens?: number;
  retries?: RetryConfig;
  /** Cache the system prompt by default (Anthropic cache_control; no-op on OpenAI). */
  cache?: boolean;
}

/** Metadata passed to observability hooks for every underlying model call. */
export interface CallMeta {
  /** The resolved "provider:model". */
  model: string;
  provider: string;
  /** The alias used, if the call referenced one. */
  alias?: string;
  /** Free-form label the caller passed (e.g. a role like "extraction"). */
  purpose?: string;
  /** True when this call ran on the fallback model after the primary failed. */
  fallback?: boolean;
}

export interface AIConfig {
  /** Provider keys/factories. Keys `anthropic` and `openai` are built in; any other name needs a factory. */
  providers: Record<string, ProviderConfig>;
  /** Named model aliases → "provider:model" (+ optional fallback). */
  models?: Record<string, ModelConfig>;
  defaults?: CallDefaults;
  /** Fired once per underlying model call (including repair + fallback rounds). */
  onUsage?: (usage: Usage, meta: CallMeta) => void | Promise<void>;
}
