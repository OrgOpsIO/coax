import type { ZodType } from "zod";
import { extractJson } from "./parse";
import { formatIssues, safeParse, toProviderSchema } from "./schema";
import {
  addUsage,
  emptyUsage,
  type Message,
  type Provider,
  type SpeakRequest,
  type SpeakResponse,
  type ToolsRequest,
  type ToolsResponse,
  type TranscribeRequest,
  type TranscribeResponse,
  type Usage,
} from "./types";

export class CoaxSchemaError extends Error {
  constructor(message: string, readonly lastError: string, readonly attempts: number) {
    super(message);
    this.name = "CoaxSchemaError";
  }
}

/** Raised when a call needs a capability the configured endpoint does not implement. */
export class CoaxUnsupportedError extends Error {
  constructor(readonly capability: string, readonly provider: string) {
    super(`coax: provider "${provider}" does not support ${capability}. Point this call at a model whose endpoint serves it.`);
    this.name = "CoaxUnsupportedError";
  }
}

export interface ObjectRequest<T> {
  schema: ZodType<T>;
  /** Stable name for the tool / json_schema. Defaults to "output". */
  schemaName?: string;
  system?: string;
  /** Shorthand for a single user message. Use `messages` for multi-turn / vision. */
  prompt?: string;
  messages?: Message[];
  maxTokens?: number;
  /** How many reprompt-on-validation-failure rounds. Default 2. */
  maxRepairs?: number;
  /** Cache the system prompt at the provider (Anthropic cache_control; no-op on OpenAI). */
  cache?: boolean;
  /** Extra HTTP headers for this call (e.g. forwarding the end user's identity to a gateway). */
  headers?: Record<string, string>;
}

export interface ObjectResult<T> {
  data: T;
  /** Summed usage across the initial call + any repair rounds. */
  usage: Usage;
  model: string;
  /** How many repair rounds were needed (0 = valid first try). */
  repairs: number;
}

export interface TextResult {
  text: string;
  usage: Usage;
  model: string;
}

export interface TranscribeResult {
  text: string;
  usage: Usage;
  model: string;
}

export interface SpeakResult {
  audio: Uint8Array;
  mediaType: string;
  usage: Usage;
  model: string;
}

export interface ClientOptions {
  provider: Provider;
  defaultMaxRepairs?: number;
  /** Observability hook, fired once per underlying model call (incl. repair rounds). */
  onUsage?: (usage: Usage, model: string) => void | Promise<void>;
}

export interface Client {
  readonly provider: Provider;
  /** Typed, validated, self-repairing structured output. */
  object<T>(req: ObjectRequest<T>): Promise<ObjectResult<T>>;
  /** Free-form text (HTML, prose, reasoning) — no schema. */
  text(req: { system?: string; prompt?: string; messages?: Message[]; maxTokens?: number; cache?: boolean; headers?: Record<string, string> }): Promise<TextResult>;
  /** One native tool-calling turn. `ai.run()` drives the loop over this. */
  tools(req: ToolsRequest): Promise<ToolsResponse>;
  /** Speech-to-text. Throws CoaxUnsupportedError where the endpoint has no transcription. */
  transcribe(req: TranscribeRequest): Promise<TranscribeResult>;
  /** Text-to-speech. Throws CoaxUnsupportedError where the endpoint has no speech synthesis. */
  speak(req: SpeakRequest): Promise<SpeakResult>;
}

function toMessages(prompt: string | undefined, messages: Message[] | undefined): Message[] {
  if (messages?.length) return [...messages];
  if (prompt != null) return [{ role: "user", content: prompt }];
  throw new Error("coax: provide either `prompt` or `messages`");
}

export function createClient(opts: ClientOptions): Client {
  const { provider, onUsage } = opts;

  /** Resolve an optional provider capability, or fail with a message that names the missing piece. */
  function capability<K extends "tools" | "transcribe" | "speak">(key: K, label: string): NonNullable<Provider[K]> {
    const fn = provider[key];
    if (!fn) throw new CoaxUnsupportedError(label, provider.name);
    return fn.bind(provider) as NonNullable<Provider[K]>;
  }

  return {
    provider,

    async object<T>(req: ObjectRequest<T>): Promise<ObjectResult<T>> {
      const schemaName = req.schemaName ?? "output";
      const { jsonSchema, unwrap } = toProviderSchema(req.schema);
      const maxRepairs = req.maxRepairs ?? opts.defaultMaxRepairs ?? 2;
      const messages = toMessages(req.prompt, req.messages);

      let usage = emptyUsage();
      let model = provider.model;
      let lastError = "";

      for (let attempt = 0; attempt <= maxRepairs; attempt++) {
        const res = await provider.structured({
          system: req.system,
          messages,
          jsonSchema,
          schemaName,
          maxTokens: req.maxTokens,
          cacheSystem: req.cache,
          headers: req.headers,
        });
        usage = addUsage(usage, res.usage);
        model = res.model;
        await onUsage?.(res.usage, res.model);

        const parsed = safeParse(req.schema, unwrap(extractJson(res.raw)));
        if (parsed.success) return { data: parsed.data, usage, model, repairs: attempt };

        lastError = formatIssues(parsed.error);
        // Reprompt with the exact validation failures — the model corrects far better with the concrete gaps.
        messages.push({ role: "assistant", content: res.text || JSON.stringify(res.raw) });
        messages.push({
          role: "user",
          content: `Your output did not match the required schema:\n${lastError}\n\nReturn a corrected result that matches the schema exactly.`,
        });
      }

      throw new CoaxSchemaError(`coax: could not produce a valid "${schemaName}" after ${maxRepairs + 1} attempt(s)`, lastError, maxRepairs + 1);
    },

    async text(req): Promise<TextResult> {
      const res = await provider.text({
        system: req.system,
        messages: toMessages(req.prompt, req.messages),
        maxTokens: req.maxTokens,
        cacheSystem: req.cache,
        headers: req.headers,
      });
      await onUsage?.(res.usage, res.model);
      return { text: res.text, usage: res.usage, model: res.model };
    },

    async tools(req: ToolsRequest): Promise<ToolsResponse> {
      const res = await capability("tools", "tool calling")(req);
      await onUsage?.(res.usage, res.model);
      return res;
    },

    async transcribe(req: TranscribeRequest): Promise<TranscribeResult> {
      const res: TranscribeResponse = await capability("transcribe", "transcription")(req);
      await onUsage?.(res.usage, res.model);
      return { text: res.text, usage: res.usage, model: res.model };
    },

    async speak(req: SpeakRequest): Promise<SpeakResult> {
      const res: SpeakResponse = await capability("speak", "speech synthesis")(req);
      await onUsage?.(res.usage, res.model);
      return { audio: res.audio, mediaType: res.mediaType, usage: res.usage, model: res.model };
    },
  };
}
