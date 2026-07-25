import type { Provider, Usage } from "./types";

/**
 * A provider endpoint coax talks to with one of its built-in wire adapters. Name it whatever you like
 * — `orgops`, `local`, `staging` — and point `baseURL` at any compatible server (your own gateway,
 * vLLM, LM Studio). The name is yours; `api` says which protocol it speaks.
 */
export interface ProviderEndpoint {
  apiKey: string;
  /** Omit for the vendor's own API; set it for any compatible endpoint (usually ending in `/v1`). */
  baseURL?: string;
  /** Wire protocol. Defaults to the built-in matching the provider's name; required for any other name. */
  api?: "openai" | "anthropic";
  /** Headers sent with every call to this endpoint. Per-call `headers` are merged over these. */
  headers?: Record<string, string>;
  /** Model used for `ai.transcribe()` / `ai.speak()` where the endpoint names them separately from chat. */
  transcribeModel?: string;
  speakModel?: string;
}

/**
 * How a provider is configured. Either:
 *  - an API key string (for the built-in `anthropic` / `openai` providers),
 *  - a {@link ProviderEndpoint} — the way to reach your own OpenAI-/Anthropic-compatible server, or
 *  - a factory `(model) => Provider` to plug in ANY provider (Gemini, a local model, a mock in tests).
 */
export type ProviderConfig = string | ProviderEndpoint | ((model: string) => Provider);

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
  /** Cap on model turns in `ai.run()`. Default 8. */
  maxSteps?: number;
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
  /** Provider keys/endpoints/factories. Keys `anthropic` and `openai` work from a bare API key; any
   *  other name needs `api` (compatible endpoint) or a factory. */
  providers: Record<string, ProviderConfig>;
  /** Named model aliases → "provider:model" (+ optional fallback). */
  models?: Record<string, ModelConfig>;
  defaults?: CallDefaults;
  /** Fired once per underlying model call (including repair, tool and fallback rounds). */
  onUsage?: (usage: Usage, meta: CallMeta) => void | Promise<void>;
}
