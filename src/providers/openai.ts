import type OpenAiSdk from "openai";
import {
  emptyUsage,
  type AudioFormat,
  type Message,
  type Provider,
  type ProviderResponse,
  type SpeakRequest,
  type SpeakResponse,
  type StructuredRequest,
  type TextRequest,
  type ToolCall,
  type ToolsRequest,
  type ToolsResponse,
  type TranscribeRequest,
  type TranscribeResponse,
  type Usage,
} from "../types";

export interface OpenAiOptions {
  model: string;
  apiKey?: string;
  /** Inject an existing SDK client. Otherwise coax lazily constructs one from `apiKey` (the SDK ships
   *  inside coax and is imported only then). */
  client?: OpenAiSdk;
  maxTokens?: number;
  /** Any OpenAI-compatible `/v1` endpoint — your own gateway, vLLM, LM Studio, a local runtime. */
  baseURL?: string;
  /** Headers sent with every request (per-call `headers` are merged over these). */
  headers?: Record<string, string>;
  /**
   * Model to use for `/audio/transcriptions` and `/audio/speech`. Gateways route audio by model name
   * just like chat, and it is rarely the same model as the chat one — default to the chat model so a
   * single-model endpoint works out of the box, override where the endpoint names them separately.
   */
  transcribeModel?: string;
  speakModel?: string;
  /** Merged into every request body from this endpoint, under the per-call `extraBody`. See `ProviderEndpoint.extraBody`. */
  extraBody?: Record<string, unknown>;
  /**
   * Which wire field carries the output-token cap. OpenAI's own API wants `max_completion_tokens`
   * (reasoning models reject the deprecated `max_tokens` outright); many compatible servers still only
   * know `max_tokens`. Default: `"max_completion_tokens"` when no `baseURL` is set (the vendor API),
   * `"max_tokens"` otherwise.
   */
  tokenParam?: "max_tokens" | "max_completion_tokens";
  /**
   * Set `strict: true` on the structured-output function definition, so OpenAI's constrained decoding
   * GUARANTEES the returned JSON matches the schema's shape — repair rounds then only ever fire for
   * what a grammar can't check (semantic refinements like `min(1)`). Opt-in because strict mode
   * rejects schemas it can't express: optional fields must be unions with `null`, and every property
   * must be required. Enable it when your schemas are strict-compatible.
   */
  strict?: boolean;
}

type ChatMessage = { content?: string | null; tool_calls?: { id: string; function: { name: string; arguments: string } }[] };
type ChatResponse = { choices: { message?: ChatMessage }[]; usage?: Record<string, unknown> };
type RequestOptions = { headers?: Record<string, string>; signal?: AbortSignal };
type AnyClient = {
  chat: { completions: { create(body: Record<string, unknown>, options?: RequestOptions): Promise<ChatResponse> } };
  audio: {
    transcriptions: { create(body: Record<string, unknown>, options?: RequestOptions): Promise<{ text: string; usage?: Record<string, unknown> }> };
    speech: { create(body: Record<string, unknown>, options?: RequestOptions): Promise<Response> };
  };
};

function mapUsage(u: Record<string, unknown> | undefined): Usage {
  const cached = (u?.prompt_tokens_details as { cached_tokens?: number } | undefined)?.cached_tokens ?? 0;
  return {
    inputTokens: (u?.prompt_tokens as number) ?? (u?.input_tokens as number) ?? 0,
    outputTokens: (u?.completion_tokens as number) ?? (u?.output_tokens as number) ?? 0,
    cacheReadTokens: cached,
    cacheWriteTokens: 0,
  };
}

function toContent(m: Message): unknown {
  if (!m.media?.length) return m.content;
  const parts: unknown[] = [{ type: "text", text: m.content }];
  for (const media of m.media) {
    // OpenAI chat vision takes images as data URIs. (PDF input needs the Files/Responses API — out of
    // scope for the chat provider; Anthropic handles PDF natively.)
    if (media.kind === "image") {
      parts.push({ type: "image_url", image_url: { url: `data:${media.mediaType};base64,${media.dataBase64}` } });
    }
  }
  return parts;
}

/** Tool results are serialized for the wire; a string stays a string so the model reads it verbatim. */
const toolOutput = (output: unknown): string => (typeof output === "string" ? output : JSON.stringify(output ?? null));

function toMessages(system: string | undefined, messages: Message[]): unknown[] {
  const out: unknown[] = [];
  if (system) out.push({ role: "system", content: system });
  for (const m of messages) {
    if (m.role === "assistant" && m.toolCalls?.length) {
      out.push({
        role: "assistant",
        content: m.content || null,
        tool_calls: m.toolCalls.map((c) => ({ id: c.id, type: "function", function: { name: c.name, arguments: JSON.stringify(c.input ?? {}) } })),
      });
      continue;
    }
    // Unlike Anthropic (one user turn carrying all result blocks), OpenAI wants ONE `role: "tool"`
    // message per result, keyed by tool_call_id.
    if (m.toolResults?.length) {
      for (const r of m.toolResults) out.push({ role: "tool", tool_call_id: r.id, content: toolOutput(r.output) });
      if (m.content) out.push({ role: "user", content: toContent(m) });
      continue;
    }
    out.push({ role: m.role, content: toContent(m) });
  }
  return out;
}

/**
 * Flat merge, call wins over endpoint wins over coax's own body — same rule as `headers`. This CAN
 * override coax's own fields (`max_tokens`, `tools`, …); that is the deliberate escape hatch described
 * on `BaseRequest.extraBody`, not a bug to guard against here.
 */
function withExtraBody(body: Record<string, unknown>, endpointExtraBody: Record<string, unknown> | undefined, callExtraBody: Record<string, unknown> | undefined): Record<string, unknown> {
  return { ...body, ...endpointExtraBody, ...callExtraBody };
}

/** Arguments arrive as a JSON string; a malformed one becomes `{}` so schema validation reports it. */
function parseArguments(raw: string): unknown {
  try {
    return JSON.parse(raw || "{}");
  } catch {
    return {};
  }
}

const AUDIO_MEDIA_TYPES: Record<AudioFormat, string> = {
  mp3: "audio/mpeg",
  opus: "audio/opus",
  aac: "audio/aac",
  flac: "audio/flac",
  wav: "audio/wav",
  pcm: "audio/pcm",
};

/** Whisper-style servers sniff the format from the upload's filename, so derive a sensible one. */
const EXTENSIONS: Record<string, string> = {
  "audio/mpeg": "mp3",
  "audio/mp3": "mp3",
  "audio/wav": "wav",
  "audio/x-wav": "wav",
  "audio/webm": "webm",
  "audio/ogg": "ogg",
  "audio/opus": "ogg",
  "audio/flac": "flac",
  "audio/mp4": "m4a",
  "audio/x-m4a": "m4a",
};

export function openai(opts: OpenAiOptions): Provider {
  let client: AnyClient | undefined = opts.client as AnyClient | undefined;

  // Vendor API (no baseURL): `max_completion_tokens` — the only cap ALL current OpenAI models accept
  // (reasoning models 400 on `max_tokens`). Compatible servers (vLLM, LM Studio, gateways): stay on
  // `max_tokens`, which they all know. `tokenParam` overrides either way.
  const tokenParam = opts.tokenParam ?? (opts.baseURL ? "max_tokens" : "max_completion_tokens");

  async function getClient(): Promise<AnyClient> {
    if (client) return client;
    const mod = await import("openai");
    const Ctor = (mod as unknown as { default: new (o: { apiKey?: string; baseURL?: string }) => AnyClient }).default;
    client = new Ctor({ apiKey: opts.apiKey, baseURL: opts.baseURL });
    return client;
  }

  const requestOptions = (headers: Record<string, string> | undefined, signal: AbortSignal | undefined): RequestOptions | undefined => {
    const merged = { ...opts.headers, ...headers };
    const out: RequestOptions = {
      ...(Object.keys(merged).length ? { headers: merged } : {}),
      // The SDK aborts the underlying HTTP request — the endpoint sees the disconnect.
      ...(signal ? { signal } : {}),
    };
    return Object.keys(out).length ? out : undefined;
  };

  return {
    name: "openai",
    model: opts.model,

    async structured(req: StructuredRequest): Promise<ProviderResponse> {
      const c = await getClient();
      const resp = await c.chat.completions.create(
        withExtraBody(
          {
            model: opts.model,
            [tokenParam]: req.maxTokens ?? opts.maxTokens ?? 8192,
            messages: toMessages(req.system, req.messages),
            tools: [{ type: "function", function: { name: req.schemaName, description: `Return a ${req.schemaName} object.`, parameters: req.jsonSchema, ...(opts.strict ? { strict: true } : {}) } }],
            tool_choice: { type: "function", function: { name: req.schemaName } },
            ...(req.reasoningEffort ? { reasoning_effort: req.reasoningEffort } : {}),
          },
          opts.extraBody,
          req.extraBody,
        ),
        requestOptions(req.headers, req.signal),
      );
      const args = resp.choices[0]?.message?.tool_calls?.[0]?.function?.arguments ?? "";
      return { raw: args, text: args, usage: mapUsage(resp.usage), model: opts.model };
    },

    async text(req: TextRequest): Promise<ProviderResponse> {
      const c = await getClient();
      const resp = await c.chat.completions.create(
        withExtraBody(
          {
            model: opts.model,
            [tokenParam]: req.maxTokens ?? opts.maxTokens ?? 8192,
            messages: toMessages(req.system, req.messages),
            ...(req.reasoningEffort ? { reasoning_effort: req.reasoningEffort } : {}),
          },
          opts.extraBody,
          req.extraBody,
        ),
        requestOptions(req.headers, req.signal),
      );
      const text = resp.choices[0]?.message?.content ?? "";
      return { raw: text, text, usage: mapUsage(resp.usage), model: opts.model };
    },

    async tools(req: ToolsRequest): Promise<ToolsResponse> {
      const c = await getClient();
      const resp = await c.chat.completions.create(
        withExtraBody(
          {
            model: opts.model,
            [tokenParam]: req.maxTokens ?? opts.maxTokens ?? 8192,
            messages: toMessages(req.system, req.messages),
            tools: req.tools.map((t) => ({ type: "function", function: { name: t.name, description: t.description, parameters: t.jsonSchema } })),
            // Literal values match the OpenAI wire directly; unset keeps today's hardcoded "auto".
            tool_choice: req.toolChoice ?? "auto",
            ...(req.reasoningEffort ? { reasoning_effort: req.reasoningEffort } : {}),
          },
          opts.extraBody,
          req.extraBody,
        ),
        requestOptions(req.headers, req.signal),
      );
      const message = resp.choices[0]?.message;
      const calls: ToolCall[] = (message?.tool_calls ?? []).map((tc) => ({
        id: tc.id,
        name: tc.function.name,
        input: parseArguments(tc.function.arguments),
      }));
      return { text: message?.content ?? "", calls, usage: mapUsage(resp.usage), model: opts.model };
    },

    async transcribe(req: TranscribeRequest): Promise<TranscribeResponse> {
      const c = await getClient();
      const { toFile } = await import("openai");
      const mediaType = req.audio.mediaType ?? "audio/wav";
      const filename = req.audio.filename ?? `audio.${EXTENSIONS[mediaType] ?? "wav"}`;
      const model = opts.transcribeModel ?? opts.model;
      const file = await toFile(req.audio.data as never, filename, { type: mediaType });
      const resp = await c.audio.transcriptions.create(
        { file, model, ...(req.language ? { language: req.language } : {}), ...(req.prompt ? { prompt: req.prompt } : {}) },
        requestOptions(req.headers, req.signal),
      );
      // Most self-hosted Whisper servers report no usage at all — report zeros rather than guessing.
      return { text: resp.text, usage: mapUsage(resp.usage), model };
    },

    async speak(req: SpeakRequest): Promise<SpeakResponse> {
      const c = await getClient();
      const format = req.format ?? "mp3";
      const model = opts.speakModel ?? opts.model;
      const resp = await c.audio.speech.create(
        {
          model,
          input: req.input,
          // Endpoints differ on whether `voice` is optional; send the OpenAI default only as a fallback
          // so a service with its own default voice still behaves.
          voice: req.voice ?? "alloy",
          response_format: format,
          ...(req.speed != null ? { speed: req.speed } : {}),
          ...(req.instructions ? { instructions: req.instructions } : {}),
        },
        requestOptions(req.headers, req.signal),
      );
      return { audio: new Uint8Array(await resp.arrayBuffer()), mediaType: AUDIO_MEDIA_TYPES[format], usage: emptyUsage(), model };
    },
  };
}
