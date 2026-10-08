import type { ZodType } from "zod";
import { extractJson } from "./parse";
import { formatIssues, safeParse, toProviderSchema } from "./schema";
import {
  addUsage,
  billedUsage,
  CoaxRefusalError,
  emptyUsage,
  type EmbedRequest,
  type EmbedResponse,
  type Message,
  type Provider,
  type ProviderResponse,
  type ReasoningEffort,
  type SpeakRequest,
  type SpeakResponse,
  type SpeakStreamResponse,
  type ToolInvocation,
  type ToolsRequest,
  type ToolsResponse,
  type TranscribeRequest,
  type TranscribeResponse,
  type TranscribeTokenRequest,
  type TranscribeTokenResponse,
  type TranscriptWord,
  type Usage,
} from "./types";

export class CoaxSchemaError extends Error {
  constructor(
    message: string,
    readonly lastError: string,
    readonly attempts: number,
    /** Usage summed across all attempts — the failed run still cost these tokens. */
    readonly usage: Usage = emptyUsage(),
    /** The transcript up to the failure, including every repair reprompt — replayable, same idea as
     *  `CoaxToolError.messages`. */
    readonly messages: Message[] = [],
  ) {
    super(message);
    this.name = "CoaxSchemaError";
  }
}

/**
 * Raised when a call is cancelled through its AbortSignal — the one error to check for "the user hung
 * up", regardless of which layer (SDK, repair loop, tool run) the abort landed in. `usage` carries what
 * the completed turns before the abort cost; the aborted call itself reports nothing — unless the provider
 * raised the abort itself and marked what the call was already billed (`billedUsage`, e.g. the inputs an
 * embed batch finished), which is then in `usage` and reported through `onUsage` once. `messages`/`calls`
 * are populated by `runTools` so an aborted `ai.run()` is resumable exactly like a `CoaxToolError` —
 * everywhere else (object/text/…) there is no transcript to carry, so they stay empty.
 */
export class CoaxAbortError extends Error {
  readonly usage: Usage;
  readonly messages: Message[];
  readonly calls: ToolInvocation[];
  constructor(usage?: Usage, cause?: unknown, messages?: Message[], calls?: ToolInvocation[]) {
    super("coax: the call was aborted by its AbortSignal", cause === undefined ? undefined : { cause });
    this.name = "CoaxAbortError";
    this.usage = usage ?? emptyUsage();
    this.messages = messages ?? [];
    this.calls = calls ?? [];
  }
}

/**
 * The CoaxAbortError for whatever was thrown after an abort. The abort takes precedence: anything else
 * (a refusal that raced the abort included) becomes a plain CoaxAbortError that reports nothing. A
 * CoaxAbortError the provider raised itself passes through untouched — with what it says the aborted call
 * was billed (`billedUsage`), which is then reported once.
 */
function abortedBy(spent: Usage, err: unknown): CoaxAbortError {
  return err instanceof CoaxAbortError ? err : new CoaxAbortError(spent, err);
}

/**
 * Run one provider call under the caller's AbortSignal: fail fast when already aborted, and normalize
 * whatever the SDK throws after an abort (APIUserAbortError, DOMException, …) to CoaxAbortError.
 * `spent` supplies the usage accumulated so far in the surrounding loop.
 */
async function aborting<T>(signal: AbortSignal | undefined, spent: () => Usage, fn: () => Promise<T>): Promise<T> {
  if (signal?.aborted) throw new CoaxAbortError(spent());
  try {
    return await fn();
  } catch (err) {
    throw signal?.aborted ? abortedBy(spent(), err) : err;
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
  /** Cache the system prompt at the provider (Anthropic cache_control; no-op on OpenAI and Google). */
  cache?: boolean;
  /** Mark the conversation-so-far as reusable for the loop's next call. See `BaseRequest.cacheConversation`. */
  cacheConversation?: boolean;
  /** Extra HTTP headers for this call (e.g. forwarding the end user's identity to a gateway). */
  headers?: Record<string, string>;
  /** Cancel the call (incl. repair rounds) — surfaces as CoaxAbortError. */
  signal?: AbortSignal;
  /** How hard the model should think. Sent on the wire only when set. See `BaseRequest.reasoningEffort`. */
  reasoningEffort?: ReasoningEffort;
  /** Merged into the wire body, last (flat; deep on Google) — MAY override coax's own fields. See `BaseRequest.extraBody`. */
  extraBody?: Record<string, unknown>;
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
  /** Word timings where the vendor returns them (ElevenLabs); absent otherwise. */
  words?: TranscriptWord[];
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
  /**
   * Structured output as a stream of PARTIAL objects: each yield is the current parse of the JSON
   * generated so far (unvalidated snapshots — only the final result is schema-checked). Repair rounds
   * stream too: a failed attempt's reprompt restarts the partials. Returns the validated ObjectResult.
   */
  streamObject<T>(req: ObjectRequest<T>): AsyncGenerator<unknown, ObjectResult<T>, void>;
  /** Embeddings — one vector per input. Throws CoaxUnsupportedError where the endpoint has none. */
  embed(req: EmbedRequest): Promise<EmbedResponse>;
  /** Free-form text (HTML, prose, reasoning) — no schema. */
  text(req: {
    system?: string;
    prompt?: string;
    messages?: Message[];
    maxTokens?: number;
    cache?: boolean;
    cacheConversation?: boolean;
    headers?: Record<string, string>;
    signal?: AbortSignal;
    reasoningEffort?: ReasoningEffort;
    extraBody?: Record<string, unknown>;
  }): Promise<TextResult>;
  /**
   * Token streaming: an async generator yielding text deltas, returning the final `TextResult` (usage
   * and all) when the stream ends. Providers without `textStream` degrade to one non-streaming call
   * whose whole text is yielded once. Abort surfaces as CoaxAbortError, mid-stream included.
   */
  stream(req: {
    system?: string;
    prompt?: string;
    messages?: Message[];
    maxTokens?: number;
    cache?: boolean;
    cacheConversation?: boolean;
    headers?: Record<string, string>;
    signal?: AbortSignal;
    reasoningEffort?: ReasoningEffort;
    extraBody?: Record<string, unknown>;
  }): AsyncGenerator<string, TextResult, void>;
  /** One native tool-calling turn. `ai.run()` drives the loop over this. */
  tools(req: ToolsRequest): Promise<ToolsResponse>;
  /** One STREAMING tool-calling turn: yields the turn's text deltas, returns the complete response.
   *  Providers without `toolsStream` degrade to a non-streaming turn that yields nothing. */
  toolsStream(req: ToolsRequest): AsyncGenerator<string, ToolsResponse, void>;
  /** Speech-to-text. Throws CoaxUnsupportedError where the endpoint has no transcription. */
  transcribe(req: TranscribeRequest): Promise<TranscribeResult>;
  /** Text-to-speech. Throws CoaxUnsupportedError where the endpoint has no speech synthesis. */
  speak(req: SpeakRequest): Promise<SpeakResult>;
  /** Streamed text-to-speech: resolves once the vendor accepted the request. The audio generator reports
   *  usage when drained. Providers without `speakStream` degrade to one `speak` call, yielded as one chunk. */
  speakStream(req: SpeakRequest): Promise<SpeakStreamResponse>;
  /** A single-use token for realtime transcription in the browser. Throws CoaxUnsupportedError where the
   *  endpoint has none. */
  transcribeToken(req: TranscribeTokenRequest): Promise<TranscribeTokenResponse>;
}

function toMessages(prompt: string | undefined, messages: Message[] | undefined): Message[] {
  if (messages?.length) return [...messages];
  if (prompt != null) return [{ role: "user", content: prompt }];
  throw new Error("coax: provide either `prompt` or `messages`");
}

export function createClient(opts: ClientOptions): Client {
  const { provider, onUsage } = opts;

  // A failed call that was still billed (a refused prompt, a turn without a usable answer, an embed batch
  // cut short — whatever the provider marked, see `billedUsage`) is reported like any completed call, then
  // the error goes through unchanged. An abort keeps precedence — `abortedBy()` has already turned anything
  // thrown after `signal.aborted` into a CoaxAbortError (see there for the one that carries a mark).
  async function reportBilled(err: unknown): Promise<void> {
    const usage = billedUsage(err);
    if (usage) await onUsage?.(usage, err instanceof CoaxRefusalError ? err.model : provider.model);
  }

  async function billed<T>(signal: AbortSignal | undefined, spent: () => Usage, fn: () => Promise<T>): Promise<T> {
    try {
      return await aborting(signal, spent, fn);
    } catch (err) {
      await reportBilled(err);
      throw err;
    }
  }

  /** Resolve an optional provider capability, or fail with a message that names the missing piece. */
  function capability<K extends "tools" | "transcribe" | "speak" | "embed" | "transcribeToken">(key: K, label: string): NonNullable<Provider[K]> {
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
        const res = await billed(req.signal, () => usage, () =>
          provider.structured({
            system: req.system,
            messages,
            jsonSchema,
            schemaName,
            maxTokens: req.maxTokens,
            cacheSystem: req.cache,
            cacheConversation: req.cacheConversation,
            headers: req.headers,
            signal: req.signal,
            reasoningEffort: req.reasoningEffort,
            extraBody: req.extraBody,
          }),
        );
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

      throw new CoaxSchemaError(`coax: could not produce a valid "${schemaName}" after ${maxRepairs + 1} attempt(s)`, lastError, maxRepairs + 1, usage, messages);
    },

    async text(req): Promise<TextResult> {
      const res = await billed(req.signal, emptyUsage, () =>
        provider.text({
          system: req.system,
          messages: toMessages(req.prompt, req.messages),
          maxTokens: req.maxTokens,
          cacheSystem: req.cache,
          cacheConversation: req.cacheConversation,
          headers: req.headers,
          signal: req.signal,
          reasoningEffort: req.reasoningEffort,
          extraBody: req.extraBody,
        }),
      );
      await onUsage?.(res.usage, res.model);
      return { text: res.text, usage: res.usage, model: res.model };
    },

    async *streamObject<T>(req: ObjectRequest<T>): AsyncGenerator<unknown, ObjectResult<T>, void> {
      const schemaName = req.schemaName ?? "output";
      const { jsonSchema, unwrap } = toProviderSchema(req.schema);
      const maxRepairs = req.maxRepairs ?? opts.defaultMaxRepairs ?? 2;
      const messages = toMessages(req.prompt, req.messages);

      // The current parse of the JSON-so-far, or undefined while it doesn't parse yet. jsonrepair
      // (inside extractJson) closes truncated structures, so partials appear early and often.
      const partialValue = (acc: string): unknown => {
        const v = extractJson(acc);
        return typeof v === "string" ? undefined : v;
      };

      let usage = emptyUsage();
      let model = provider.model;
      let lastError = "";

      for (let attempt = 0; attempt <= maxRepairs; attempt++) {
        const wire = {
          system: req.system,
          messages,
          jsonSchema,
          schemaName,
          maxTokens: req.maxTokens,
          cacheSystem: req.cache,
          cacheConversation: req.cacheConversation,
          headers: req.headers,
          signal: req.signal,
          reasoningEffort: req.reasoningEffort,
          extraBody: req.extraBody,
        };

        let res: ProviderResponse;
        if (provider.structuredStream) {
          const gen = provider.structuredStream(wire);
          let acc = "";
          let lastYield = "";
          try {
            if (req.signal?.aborted) throw new CoaxAbortError(usage);
            let cur = await gen.next();
            while (!cur.done) {
              acc += cur.value;
              const partial = partialValue(acc);
              if (partial !== undefined) {
                // Deduplicate — a delta that only extends a string value mid-token can parse identically.
                const snapshot = JSON.stringify(partial);
                if (snapshot !== lastYield) {
                  lastYield = snapshot;
                  yield unwrap(partial);
                }
              }
              cur = await gen.next();
            }
            res = cur.value;
          } catch (err) {
            const failure = req.signal?.aborted ? abortedBy(usage, err) : err;
            await reportBilled(failure);
            throw failure;
          }
        } else {
          // No native structured streaming → one non-streaming call; the final object is the only partial.
          res = await billed(req.signal, () => usage, () => provider.structured(wire));
          const whole = unwrap(extractJson(res.raw));
          if (whole !== undefined) yield whole;
        }

        usage = addUsage(usage, res.usage);
        model = res.model;
        await onUsage?.(res.usage, res.model);

        const parsed = safeParse(req.schema, unwrap(extractJson(res.raw)));
        if (parsed.success) return { data: parsed.data, usage, model, repairs: attempt };

        lastError = formatIssues(parsed.error);
        messages.push({ role: "assistant", content: res.text || JSON.stringify(res.raw) });
        messages.push({
          role: "user",
          content: `Your output did not match the required schema:\n${lastError}\n\nReturn a corrected result that matches the schema exactly.`,
        });
      }

      throw new CoaxSchemaError(`coax: could not produce a valid "${schemaName}" after ${maxRepairs + 1} attempt(s)`, lastError, maxRepairs + 1, usage, messages);
    },

    async embed(req: EmbedRequest): Promise<EmbedResponse> {
      const res = await billed(req.signal, emptyUsage, () => capability("embed", "embeddings")(req));
      await onUsage?.(res.usage, res.model);
      return { embeddings: res.embeddings, usage: res.usage, model: res.model };
    },

    async *stream(req): AsyncGenerator<string, TextResult, void> {
      const wire = {
        system: req.system,
        messages: toMessages(req.prompt, req.messages),
        maxTokens: req.maxTokens,
        cacheSystem: req.cache,
        cacheConversation: req.cacheConversation,
        headers: req.headers,
        signal: req.signal,
        reasoningEffort: req.reasoningEffort,
        extraBody: req.extraBody,
      };

      // No native streaming on this provider → one non-streaming call, its whole text as one delta.
      if (!provider.textStream) {
        const res = await billed(req.signal, emptyUsage, () => provider.text(wire));
        await onUsage?.(res.usage, res.model);
        if (res.text) yield res.text;
        return { text: res.text, usage: res.usage, model: res.model };
      }

      const gen = provider.textStream(wire);
      try {
        if (req.signal?.aborted) throw new CoaxAbortError();
        let cur = await gen.next();
        while (!cur.done) {
          yield cur.value;
          cur = await gen.next();
        }
        const res = cur.value;
        await onUsage?.(res.usage, res.model);
        return { text: res.text, usage: res.usage, model: res.model };
      } catch (err) {
        // Same normalization as `aborting()` — but around iteration, so a mid-stream abort lands here too.
        const failure = req.signal?.aborted ? abortedBy(emptyUsage(), err) : err;
        await reportBilled(failure);
        throw failure;
      }
    },

    async tools(req: ToolsRequest): Promise<ToolsResponse> {
      const res = await billed(req.signal, emptyUsage, () => capability("tools", "tool calling")(req));
      await onUsage?.(res.usage, res.model);
      return res;
    },

    async *toolsStream(req: ToolsRequest): AsyncGenerator<string, ToolsResponse, void> {
      if (!provider.toolsStream) {
        // Non-streaming degrade — the turn still needs the tools capability to exist at all.
        const res = await billed(req.signal, emptyUsage, () => capability("tools", "tool calling")(req));
        await onUsage?.(res.usage, res.model);
        return res;
      }
      const gen = provider.toolsStream(req);
      try {
        if (req.signal?.aborted) throw new CoaxAbortError();
        let cur = await gen.next();
        while (!cur.done) {
          yield cur.value;
          cur = await gen.next();
        }
        await onUsage?.(cur.value.usage, cur.value.model);
        return cur.value;
      } catch (err) {
        const failure = req.signal?.aborted ? abortedBy(emptyUsage(), err) : err;
        await reportBilled(failure);
        throw failure;
      }
    },

    async transcribe(req: TranscribeRequest): Promise<TranscribeResult> {
      const res: TranscribeResponse = await billed(req.signal, emptyUsage, () => capability("transcribe", "transcription")(req));
      await onUsage?.(res.usage, res.model);
      // `words` only when the vendor returned them, so a result without them keeps exactly today's keys.
      return { text: res.text, ...(res.words ? { words: res.words } : {}), usage: res.usage, model: res.model };
    },

    async speak(req: SpeakRequest): Promise<SpeakResult> {
      const res: SpeakResponse = await billed(req.signal, emptyUsage, () => capability("speak", "speech synthesis")(req));
      await onUsage?.(res.usage, res.model);
      return { audio: res.audio, mediaType: res.mediaType, usage: res.usage, model: res.model };
    },

    async speakStream(req: SpeakRequest): Promise<SpeakStreamResponse> {
      // No native streaming on this provider → one speak call, its whole audio as one chunk.
      if (!provider.speakStream) {
        const res: SpeakResponse = await billed(req.signal, emptyUsage, () => capability("speak", "speech synthesis")(req));
        await onUsage?.(res.usage, res.model);
        return {
          mediaType: res.mediaType,
          model: res.model,
          audio: (async function* () {
            if (res.audio.byteLength) yield res.audio;
            return res.usage;
          })(),
        };
      }

      const opened = await billed(req.signal, emptyUsage, () => provider.speakStream!(req));
      async function* audio(): AsyncGenerator<Uint8Array, Usage, void> {
        let finished = false;
        try {
          let cur = await opened.audio.next();
          while (!cur.done) {
            if (cur.value.byteLength > 0) yield cur.value;
            cur = await opened.audio.next();
          }
          finished = true;
          await onUsage?.(cur.value, opened.model);
          return cur.value;
        } catch (err) {
          finished = true;
          // Same normalization as `stream()` — around iteration, so a mid-speech abort lands here too.
          const failure = req.signal?.aborted ? abortedBy(emptyUsage(), err) : err;
          await reportBilled(failure);
          throw failure;
        } finally {
          // The consumer stopped early: close the provider's stream and with it the connection (an open one
          // holds a vendor concurrency slot). Nothing is reported for it, as for every coax stream left early.
          if (!finished) await opened.audio.return(emptyUsage());
        }
      }
      return { mediaType: opened.mediaType, model: opened.model, audio: audio() };
    },

    async transcribeToken(req: TranscribeTokenRequest): Promise<TranscribeTokenResponse> {
      const res = await billed(req.signal, emptyUsage, () => capability("transcribeToken", "realtime transcription tokens")(req));
      await onUsage?.(res.usage, res.model);
      return { token: res.token, url: res.url, usage: res.usage, model: res.model };
    },
  };
}
