/** A binary attachment for vision-capable models. */
export interface Media {
  kind: "image" | "pdf";
  /** MIME type, e.g. "image/png" or "application/pdf". */
  mediaType: string;
  /** Base64-encoded bytes (no data: prefix). */
  dataBase64: string;
}

/** One turn in a conversation. `media` rides on user turns for vision models. */
export interface Message {
  role: "user" | "assistant";
  content: string;
  media?: Media[];
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

export interface StructuredRequest {
  system?: string;
  messages: Message[];
  /** JSON Schema the provider constrains its output to (from the caller's Zod schema). */
  jsonSchema: Record<string, unknown>;
  /** A stable name for the schema (tool name / json_schema name). */
  schemaName: string;
  maxTokens?: number;
}

export interface TextRequest {
  system?: string;
  messages: Message[];
  maxTokens?: number;
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

/**
 * A provider is the only vendor-specific surface. Implement `structured` (native constrained-output mode:
 * Anthropic tool_use, OpenAI json_schema) and `text` (free-form). Swap providers = swap this object.
 */
export interface Provider {
  readonly name: string;
  readonly model: string;
  structured(req: StructuredRequest): Promise<ProviderResponse>;
  text(req: TextRequest): Promise<ProviderResponse>;
}

export const emptyUsage = (): Usage => ({ inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0 });

export const addUsage = (a: Usage, b: Usage): Usage => ({
  inputTokens: a.inputTokens + b.inputTokens,
  outputTokens: a.outputTokens + b.outputTokens,
  cacheReadTokens: a.cacheReadTokens + b.cacheReadTokens,
  cacheWriteTokens: a.cacheWriteTokens + b.cacheWriteTokens,
});
