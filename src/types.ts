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
  /** Characters billed for speech synthesis (ElevenLabs: the `character-cost` response header). Absent
   *  when the vendor reported none. Not counted by a `Budget`, which counts tokens. */
  characters?: number;
  /** Seconds of audio billed for transcription (ElevenLabs `audio_duration_secs`, OpenAI whisper
   *  `usage.seconds`). Absent when the vendor reported none. Not counted by a `Budget`. */
  audioSeconds?: number;
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

export interface EmbedRequest {
  /** One text or a batch — a batch comes back as one vector per input, in order. */
  input: string | string[];
  headers?: Record<string, string>;
  signal?: AbortSignal;
  /** Merged flat into the wire body, last — same contract as `BaseRequest.extraBody`. */
  extraBody?: Record<string, unknown>;
}

export interface EmbedResponse {
  /** One vector per input, in input order. */
  embeddings: number[][];
  usage: Usage;
  model: string;
}

export interface TranscribeRequest {
  audio: AudioInput;
  /** Language hint, ISO-639-1 (e.g. "de"; ElevenLabs also takes ISO-639-3). Improves accuracy and latency when the language is known. */
  language?: string;
  /** Context hint — domain vocabulary, names, expected spelling. Not on ElevenLabs (CoaxUnsupportedError). */
  prompt?: string;
  /** Label who spoke each word (`words[].speaker`). ElevenLabs; CoaxUnsupportedError on the OpenAI wire. */
  speakers?: boolean;
  headers?: Record<string, string>;
  signal?: AbortSignal;
}

export interface SpeakRequest {
  input: string;
  /** Voice id, as named by the endpoint's TTS service. */
  voice?: string;
  format?: AudioFormat;
  /** 1.0 = normal. OpenAI wire 0.25–4.0; ElevenLabs 0.7–1.2 (outside → error before the call). */
  speed?: number;
  /** Free-form delivery instruction (tone, pace, emotion). Not on ElevenLabs (CoaxUnsupportedError). */
  instructions?: string;
  /** Language hint, ISO-639-1 (e.g. "de"). Sent where the service takes one (ElevenLabs — whose models
   *  ignore a language they don't support; `eleven_multilingual_v2` takes none); the OpenAI speech endpoint
   *  has no such field and reads the language from `input`. */
  language?: string;
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

/** One spoken word with its timing, in order. */
export interface TranscriptWord {
  text: string;
  /** Seconds from the start of the audio. */
  start: number;
  end: number;
  /** Speaker label (e.g. "speaker_0") where the vendor labelled one — ask for it with `speakers: true`. */
  speaker?: string;
}

export interface TranscribeResponse {
  text: string;
  /** Word timings where the vendor returns them (ElevenLabs); absent otherwise. */
  words?: TranscriptWord[];
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
 * A vendor that serves neither (a voice-only one) implements both by rejecting with `CoaxUnsupportedError`,
 * the same error coax raises for a missing optional capability.
 */
export interface Provider {
  readonly name: string;
  readonly model: string;
  structured(req: StructuredRequest): Promise<ProviderResponse>;
  text(req: TextRequest): Promise<ProviderResponse>;
  /**
   * Token streaming — backs `ai.stream()`. Yields text deltas as they arrive and returns the final
   * response (usage and all) when the stream ends. Optional: without it, coax degrades to one
   * non-streaming call whose whole text is yielded once — same contract, one big delta.
   */
  textStream?(req: TextRequest): AsyncGenerator<string, ProviderResponse, void>;
  /**
   * Structured-output streaming — backs `ai.streamObject()`. Yields RAW JSON text fragments of the
   * output as they arrive (coax turns them into partial objects centrally), returns the final
   * response when the stream ends. Optional: without it, coax degrades to one non-streaming call.
   */
  structuredStream?(req: StructuredRequest): AsyncGenerator<string, ProviderResponse, void>;
  /** Embeddings — backs `ai.embed()`. Optional: an endpoint that has none raises a precise error. */
  embed?(req: EmbedRequest): Promise<EmbedResponse>;
  /** Native tool calling — backs `ai.run()`. */
  tools?(req: ToolsRequest): Promise<ToolsResponse>;
  /**
   * One streaming tool-calling turn — backs `ai.runStream()`. Yields the turn's TEXT deltas as they
   * arrive (tool-call arguments are not surfaced mid-flight) and returns the complete `ToolsResponse`
   * when the turn ends. Optional: without it, a run streams nothing but still works turn by turn.
   */
  toolsStream?(req: ToolsRequest): AsyncGenerator<string, ToolsResponse, void>;
  /** Speech-to-text — backs `ai.transcribe()`. */
  transcribe?(req: TranscribeRequest): Promise<TranscribeResponse>;
  /** Text-to-speech — backs `ai.speak()`. */
  speak?(req: SpeakRequest): Promise<SpeakResponse>;
}

/**
 * Raised when the endpoint's safety layer declined the request (Anthropic `stop_reason: "refusal"`,
 * an HTTP 200 whose content is empty or a discarded partial). Without this, a refusal would surface as
 * a SUCCESSFUL call with empty text — booked as if the model had answered. Non-transient by design:
 * `withRetry` sees no status/code and rethrows immediately; the ai-layer model fallback still applies,
 * which is the right rescue for a false-positive classifier hit.
 */
export class CoaxRefusalError extends Error {
  constructor(
    readonly model: string,
    /** Anthropic `stop_details.category` (e.g. "cyber", "bio") — null when the endpoint gave none. */
    readonly category: string | null = null,
    readonly explanation: string | null = null,
  ) {
    super(
      `coax: ${model} refused the request (stop_reason "refusal"${category ? `, category "${category}"` : ""})` +
        (explanation ? ` — ${explanation}` : ""),
    );
    this.name = "CoaxRefusalError";
  }
}

export const emptyUsage = (): Usage => ({ inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0 });

/** Sums two usages. A billing unit beyond tokens (`characters`, `audioSeconds`) is summed when either side
 *  has it and left out when neither does — so token-only sums keep exactly the four token fields. */
export const addUsage = (a: Usage, b: Usage): Usage => ({
  inputTokens: a.inputTokens + b.inputTokens,
  outputTokens: a.outputTokens + b.outputTokens,
  cacheReadTokens: a.cacheReadTokens + b.cacheReadTokens,
  cacheWriteTokens: a.cacheWriteTokens + b.cacheWriteTokens,
  ...(a.characters != null || b.characters != null ? { characters: (a.characters ?? 0) + (b.characters ?? 0) } : {}),
  ...(a.audioSeconds != null || b.audioSeconds != null ? { audioSeconds: (a.audioSeconds ?? 0) + (b.audioSeconds ?? 0) } : {}),
});
