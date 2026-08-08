/** A binary attachment for vision-capable models. */
export interface Media {
  kind: "image" | "pdf";
  /** MIME type, e.g. "image/png" or "application/pdf". */
  mediaType: string;
  /** Base64-encoded bytes (no data: prefix). */
  dataBase64: string;
}

/** A tool call the model asked for. `id` must be echoed back with the matching result. */
export interface ToolCall {
  id: string;
  name: string;
  /** The model's arguments, already parsed from JSON where the provider sent a string. */
  input: unknown;
}

/**
 * How hard the model should think before answering. `"none"` turns thinking off outright — the biggest
 * lever for cutting latency/cost on calls that don't need it (classification, reformatting). Not every
 * endpoint understands this; it is only sent on the wire where explicitly set (see BaseRequest.reasoningEffort).
 */
export type ReasoningEffort = "none" | "low" | "medium" | "high";

/** Restricts how the model may respond a turn. `"required"` forbids a text answer — see `RunOptions.toolChoice`. */
export type ToolChoice = "auto" | "required" | "none";

/** One tool that actually ran, in order — the audit trail for a run and the resumable state on its errors. */
export interface ToolInvocation {
  name: string;
  input: unknown;
  output: unknown;
  /** Set when the tool failed; `output` then holds the message the model was shown. */
  error?: string;
  durationMs: number;
}

/** The outcome of running a tool, handed back to the model. Objects are JSON-encoded for the wire. */
export interface ToolResult {
  id: string;
  name: string;
  output: unknown;
  /** True when the tool failed — the model sees the message and can correct itself or try another path. */
  isError?: boolean;
}

/**
 * One turn in a conversation. `media` rides on user turns for vision models; `toolCalls` on the
 * assistant turn that requested them, `toolResults` on the user turn that answers it.
 */
export interface Message {
  role: "user" | "assistant";
  content: string;
  media?: Media[];
  toolCalls?: ToolCall[];
  toolResults?: ToolResult[];
  /**
   * Opaque provider state that must round-trip with this assistant turn — e.g. reasoning/thinking
   * blocks a provider requires to be replayed verbatim on the next request of a tool run. Set by the
   * provider, carried untouched by coax, never inspected. Treat it as a black box: don't read it,
   * don't fabricate it, and keep it on the message when persisting/replaying transcripts.
   */
  providerData?: unknown;
}

/** Token accounting, normalized across providers. */
export interface Usage {
  inputTokens: number;
  outputTokens: number;
  /** Tokens read from the prompt cache (0 if unsupported). */
  cacheReadTokens: number;
  /** Tokens written to the prompt cache (0 if unsupported). */
  cacheWriteTokens: number;
}

/** Fields every generation call accepts. */
interface BaseRequest {
  maxTokens?: number;
  /**
   * Cancel the call from outside — e.g. the BFF's incoming request died, so the upstream generation
   * should die with it. Passed through to the SDK, which aborts the HTTP request to the endpoint;
   * loops (repair rounds, tool runs) also stop between turns. Aborting surfaces as CoaxAbortError.
   */
  signal?: AbortSignal;
  /** Ask the provider to cache the (stable) system prompt — big savings across a fan-out of calls that
   *  share it. Provider-native where supported (Anthropic cache_control); a no-op where caching is
   *  automatic (OpenAI). */
  cacheSystem?: boolean;
  /**
   * Mark the conversation-so-far as reusable, so the NEXT call of a loop reads all prior turns from
   * cache — the textbook win for multi-turn agentic / validate→repair loops that re-send the whole
   * transcript every turn. Provider-native where supported (Anthropic: a cache breakpoint on the last
   * message); a no-op where caching is automatic (OpenAI).
   */
  cacheConversation?: boolean;
  /**
   * Extra HTTP headers for this one call, merged over the provider's configured headers. This is how
   * a caller's identity reaches a gateway that authorizes per user (e.g. forwarding the end user's
   * bearer token so the policy engine behind the endpoint decides) instead of every request looking
   * like one service account.
   */
  headers?: Record<string, string>;
  /**
   * How hard the model should think. Sent on the wire only when set (an endpoint that doesn't know the
   * field must never see it) — `"none"` is the big lever for calls that don't need it: classification,
   * reformatting, anything where thinking only burns tokens and latency. Precedence when resolved through
   * a model alias: per-call > alias > `defaults`.
   */
  reasoningEffort?: ReasoningEffort;
  /**
   * Merged flat into the wire body, last: `{ ...coax's own fields, ...endpoint.extraBody, ...this }` — it
   * MAY override coax's own fields (`max_tokens`, `tools`, …). That is the point: an escape hatch that
   * can't be overridden by anything isn't one. Use it for whatever the next gateway needs that coax
   * doesn't have a first-class field for yet (e.g. `temperature`, `top_p`, `chat_template_kwargs`) —
   * overriding a field coax itself relies on is your own risk.
   */
  extraBody?: Record<string, unknown>;
}

export interface StructuredRequest extends BaseRequest {
  system?: string;
  messages: Message[];
  /** JSON Schema the provider constrains its output to (from the caller's Zod schema). */
  jsonSchema: Record<string, unknown>;
  /** A stable name for the schema (tool name / json_schema name). */
  schemaName: string;
}

export interface TextRequest extends BaseRequest {
  system?: string;
  messages: Message[];
}

/** One tool as the provider sees it — a name, a description, and JSON Schema parameters. */
export interface ToolDefinition {
  name: string;
  description: string;
  jsonSchema: Record<string, unknown>;
}

export interface ToolsRequest extends BaseRequest {
  system?: string;
  messages: Message[];
  tools: ToolDefinition[];
  /** Already resolved to a plain value — `runTools` evaluates the step-indexed function form; a provider
   *  never sees anything but "auto" | "required" | "none" | undefined. */
  toolChoice?: ToolChoice;
}

/** Audio bytes going in (transcription). */
export interface AudioInput {
  data: Uint8Array | ArrayBuffer | Blob;
  /** MIME type, e.g. "audio/webm". Used to name the upload so the server can sniff the format. */
  mediaType?: string;
  /** Overrides the filename derived from `mediaType`. */
  filename?: string;
}

export type AudioFormat = "mp3" | "opus" | "aac" | "flac" | "wav" | "pcm";

export interface TranscribeRequest {
  audio: AudioInput;
  /** ISO-639-1 hint, e.g. "de". Improves accuracy and latency when the language is known. */
  language?: string;
  /** Context hint — domain vocabulary, names, expected spelling. */
  prompt?: string;
  headers?: Record<string, string>;
  signal?: AbortSignal;
}

export interface SpeakRequest {
  input: string;
  /** Voice id, as named by the endpoint's TTS service. */
  voice?: string;
  format?: AudioFormat;
  /** 0.25–4.0, 1.0 = normal. */
  speed?: number;
  /** Free-form delivery instruction where the service supports it (tone, pace, emotion). */
  instructions?: string;
  headers?: Record<string, string>;
  signal?: AbortSignal;
}

export interface ProviderResponse {
  /** The provider's structured output, already an object when the native mode returned JSON; a string
   *  otherwise. coax's aggressive parser handles either. */
  raw: unknown;
  /** The raw text form, used as the assistant turn when a repair round is needed. */
  text: string;
  usage: Usage;
  model: string;
}

export interface ToolsResponse {
  /** The assistant's free text this turn — often empty when it only called tools. */
  text: string;
  /** Empty when the model is done and answered in `text`. */
  calls: ToolCall[];
  usage: Usage;
  model: string;
  /** Opaque provider state to replay with this turn's assistant message — see `Message.providerData`. */
  providerData?: unknown;
}

export interface TranscribeResponse {
  text: string;
  usage: Usage;
  model: string;
}

export interface SpeakResponse {
  audio: Uint8Array;
  /** MIME type of `audio`, derived from the requested format. */
  mediaType: string;
  usage: Usage;
  model: string;
}

/**
 * A provider is the only vendor-specific surface. `structured` (native constrained-output mode:
 * Anthropic tool_use, OpenAI json_schema) and `text` are required. The rest are optional capabilities:
 * an endpoint that serves them implements them, and coax raises a precise error where it does not —
 * so "my gateway has no TTS" is a clear message, not a mystery 404. Swap providers = swap this object.
 */
export interface Provider {
  readonly name: string;
  readonly model: string;
  structured(req: StructuredRequest): Promise<ProviderResponse>;
  text(req: TextRequest): Promise<ProviderResponse>;
  /** Native tool calling — backs `ai.run()`. */
  tools?(req: ToolsRequest): Promise<ToolsResponse>;
  /** Speech-to-text — backs `ai.transcribe()`. */
  transcribe?(req: TranscribeRequest): Promise<TranscribeResponse>;
  /** Text-to-speech — backs `ai.speak()`. */
  speak?(req: SpeakRequest): Promise<SpeakResponse>;
}

export const emptyUsage = (): Usage => ({ inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0 });

export const addUsage = (a: Usage, b: Usage): Usage => ({
  inputTokens: a.inputTokens + b.inputTokens,
  outputTokens: a.outputTokens + b.outputTokens,
  cacheReadTokens: a.cacheReadTokens + b.cacheReadTokens,
  cacheWriteTokens: a.cacheWriteTokens + b.cacheWriteTokens,
});
