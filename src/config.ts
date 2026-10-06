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
}

/**
 * Gemini on Google's Agent Platform. Two credentials, one per provider: an Agent Platform `apiKey`
 * (generate and stream only — no `embed`), or — without a key — Application Default Credentials for
 * `project` (everything, incl. `embed`). Under any name other than `google`, set `api: "google"`.
 */
export interface GoogleEndpoint {
  api?: "google";
  /** Agent Platform API key. Mutually exclusive with `project` and `googleAuthOptions`, and served from
   *  the `"global"` location only — any other `location` with a key is a config error. */
  apiKey?: string;
  /** Google Cloud project for Application Default Credentials (the SDK falls back to `GOOGLE_CLOUD_PROJECT`). */
  project?: string;
  /** Default `"global"` — the newest models are not served from every region. A region needs `project`. */
  location?: string;
  /** Passed verbatim to Google's auth library, e.g. `{ credentials: serviceAccountJson }`. Not with `apiKey`. */
  googleAuthOptions?: Record<string, unknown>;
  /** Headers sent with every call to this endpoint. Per-call `headers` are merged over these. */
  headers?: Record<string, string>;
  /** Deep-merged into every request body (REST field names, e.g. `{ generationConfig: { temperature: 0.2 } }`),
   *  under the per-call `extraBody`. */
  extraBody?: Record<string, unknown>;
  /** Model for `ai.embed()`. Default: the model of the reference (e.g. `google:gemini-embedding-001`).
   *  When set, it wins over the reference for every embed through this endpoint (as on the OpenAI wire). */
  embedModel?: string;
}

/**
 * How a provider is configured. Either:
 *  - an API key string (for the built-in `anthropic` / `openai` / `google` providers),
 *  - a {@link ProviderEndpoint} — the way to reach your own OpenAI-/Anthropic-compatible server,
 *  - a {@link GoogleEndpoint} — Gemini with Application Default Credentials, a region, or a second name, or
 *  - a factory `(model) => Provider` to plug in ANY provider (a local model, a mock in tests).
 */
export type ProviderConfig = string | ProviderEndpoint | GoogleEndpoint | ((model: string) => Provider);

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
  /** Cache the system prompt by default (Anthropic cache_control; no-op on OpenAI and Google). */
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
  /** Provider keys/endpoints/factories. Keys `anthropic`, `openai` and `google` work from a bare API key;
   *  any other name needs `api` (compatible endpoint, or `"google"`) or a factory. */
  providers: Record<string, ProviderConfig>;
  /** Named model aliases → "provider:model" (+ optional fallback). */
  models?: Record<string, ModelConfig>;
  defaults?: CallDefaults;
  /** Fired once per underlying model call (including repair, tool and fallback rounds). */
  onUsage?: (usage: Usage, meta: CallMeta) => void | Promise<void>;
}
