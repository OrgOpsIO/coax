import type { Provider, ReasoningEffort, Usage } from "./types";

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
  /** Model for `ai.embed()` — embedding models are always named separately from chat (OpenAI wire). */
  embedModel?: string;
  /**
   * Merged flat into every request body this endpoint sends, under the per-call `extraBody` (which wins).
   * The place for an endpoint-wide quirk — e.g. Qwen's recommended `temperature`/`top_p`, or a gateway's
   * `chat_template_kwargs` — that every call through this endpoint should carry without repeating it
   * everywhere. See `BaseRequest.extraBody` for the merge order and the override caveat.
   */
  extraBody?: Record<string, unknown>;
  /** OpenAI wire only: which field carries the output-token cap. Default: `max_completion_tokens` on the
   *  vendor API (no `baseURL`), `max_tokens` on compatible endpoints. See `OpenAiOptions.tokenParam`. */
  tokenParam?: "max_tokens" | "max_completion_tokens";
  /** OpenAI wire only: `strict: true` structured output — the schema shape is grammar-guaranteed by the
   *  endpoint. Opt-in; needs strict-compatible schemas. See `OpenAiOptions.strict`. */
  strict?: boolean;
  /** Default voice for `ai.speak()` through this endpoint; a per-call `voice` wins. OpenAI wire: else "alloy". */
  voice?: string;
}

/**
 * ElevenLabs (voice only: `ai.speak()` and `ai.transcribe()`). Under any name other than `elevenlabs`,
 * set `api: "elevenlabs"` — e.g. a data-residency account, which has its own key and host.
 */
export interface ElevenLabsEndpoint {
  api?: "elevenlabs";
  apiKey: string;
  /** Host root, without `/v1` — e.g. "https://api.us.elevenlabs.io" or a data-residency host. Default: "https://api.elevenlabs.io". */
  baseURL?: string;
  /** Headers sent with every call to this endpoint. Per-call `headers` are merged over these. */
  headers?: Record<string, string>;
  /** Default voice id for `ai.speak()`; a per-call `voice` wins. ElevenLabs has no default voice of its own. */
  voice?: string;
  /** Not on ElevenLabs — name the model in the reference instead (`"elevenlabs:scribe_v2"`); here a type error. */
  transcribeModel?: never;
  speakModel?: never;
  embedModel?: never;
  tokenParam?: never;
  strict?: never;
  extraBody?: never;
}

/**
 * How a provider is configured. Either:
 *  - an API key string (for the built-in `anthropic` / `openai` / `elevenlabs` providers),
 *  - a {@link ProviderEndpoint} — the way to reach your own OpenAI-/Anthropic-compatible server,
 *  - an {@link ElevenLabsEndpoint} — ElevenLabs with more than a key (host, default voice, headers), or
 *  - a factory `(model) => Provider` to plug in ANY provider (Gemini, a local model, a mock in tests).
 */
export type ProviderConfig = string | ProviderEndpoint | ElevenLabsEndpoint | ((model: string) => Provider);

/**
 * A model alias resolves to `"provider:model"`, optionally with a fallback model on failure and a
 * `reasoningEffort` that applies to every call through this alias. The setting rides on the alias
 * rather than the provider so two aliases pointing at the same model can still think differently —
 * one config line instead of touching every call site (e.g. a "classification" alias with `"none"`
 * next to a "synthesis" alias with `"high"`, both on the same underlying model).
 */
export type ModelConfig = string | { use: string; fallback?: string; reasoningEffort?: ReasoningEffort };

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
  /** Cap on model turns in `ai.run()`. Default 8; `null` = unlimited (needs `budget` or `signal` — see
   *  `RunOptions.maxSteps`). */
  maxSteps?: number | null;
  /** Default reasoning effort for calls that don't set one directly or through their model alias.
   *  Precedence: per-call > alias > here. */
  reasoningEffort?: ReasoningEffort;
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
  /** Provider keys/endpoints/factories. Keys `anthropic`, `openai` and `elevenlabs` work from a bare
   *  API key; any other name needs `api` (compatible endpoint or `"elevenlabs"`) or a factory. */
  providers: Record<string, ProviderConfig>;
  /** Named model aliases → "provider:model" (+ optional fallback). */
  models?: Record<string, ModelConfig>;
  defaults?: CallDefaults;
  /** Fired once per underlying model call (including repair, tool and fallback rounds). */
  onUsage?: (usage: Usage, meta: CallMeta) => void | Promise<void>;
}
