import { z, type ZodType } from "zod";
import type { AIConfig } from "./config";
import type { AudioFormat, AudioInput, EmbedResponse, Media, Message, Provider, ReasoningEffort, Usage } from "./types";
import { CoaxAbortError, createClient, type ObjectResult, type SpeakResult, type TextResult, type TranscribeResult } from "./client";
import { createRegistry, retrying, type CallSettings } from "./registry";
import { parsePrompt, renderTemplate, type ParsedPrompt } from "./prompt-file";
import { runLoop, type LoopOptions } from "./loop";
import { runTools, type RunOptions, type RunResult, type Tool } from "./tools";

export interface ObjectCall<T> {
  /** Model alias (from config.models) or a literal "provider:model". Falls back to defaults.model. */
  model?: string;
  schema: ZodType<T>;
  schemaName?: string;
  system?: string;
  prompt?: string;
  messages?: Message[];
  maxTokens?: number;
  maxRepairs?: number;
  /** Cache the system prompt (Anthropic cache_control; no-op on OpenAI). */
  cache?: boolean;
  /** Mark the conversation-so-far as reusable for the loop's next call. See `BaseRequest.cacheConversation`. */
  cacheConversation?: boolean;
  /** Extra HTTP headers — e.g. forwarding the end user's token so the gateway authorizes per user. */
  headers?: Record<string, string>;
  /** Cancel the call from outside (e.g. the BFF request died) — surfaces as CoaxAbortError. */
  signal?: AbortSignal;
  /** Free-form label for observability (e.g. a role like "extraction"). */
  purpose?: string;
  /** How hard the model should think. Precedence: here > the model alias > `defaults.reasoningEffort`. */
  reasoningEffort?: ReasoningEffort;
  /** Merged flat into the wire body, last — MAY override coax's own fields. See `BaseRequest.extraBody`. */
  extraBody?: Record<string, unknown>;
}

export interface TextCall {
  model?: string;
  system?: string;
  prompt?: string;
  messages?: Message[];
  maxTokens?: number;
  cache?: boolean;
  /** Mark the conversation-so-far as reusable for the loop's next call. See `BaseRequest.cacheConversation`. */
  cacheConversation?: boolean;
  headers?: Record<string, string>;
  signal?: AbortSignal;
  purpose?: string;
  /** How hard the model should think. Precedence: here > the model alias > `defaults.reasoningEffort`. */
  reasoningEffort?: ReasoningEffort;
  /** Merged flat into the wire body, last — MAY override coax's own fields. See `BaseRequest.extraBody`. */
  extraBody?: Record<string, unknown>;
}

export interface JudgeCall {
  model?: string;
  /** The thing to evaluate — a string or any object (JSON-stringified for the judge). */
  output: unknown;
  /** Acceptance criteria / rubric. Multiple criteria are numbered for the judge. */
  criteria: string | string[];
  /** Scoring scale, inclusive. Default [1, 5]. */
  scale?: [number, number];
  /** Minimum score to pass. Default: the scale midpoint, rounded up. */
  passScore?: number;
  /** Override the judge's system instruction. */
  system?: string;
  /** For multimodal judging — e.g. a screenshot of the rendered artifact. */
  media?: Media[];
  headers?: Record<string, string>;
  signal?: AbortSignal;
  purpose?: string;
}

export interface Judgement {
  score: number;
  pass: boolean;
  rationale: string;
}

export interface EmbedCall {
  model?: string;
  /** One text or a batch — one vector per input, in order. */
  input: string | string[];
  headers?: Record<string, string>;
  signal?: AbortSignal;
  purpose?: string;
  /** Merged flat into the wire body, last — same contract as `BaseRequest.extraBody`. */
  extraBody?: Record<string, unknown>;
}

export interface TranscribeCall {
  model?: string;
  /** The audio bytes. A browser upload's Blob/File works directly. */
  audio: AudioInput;
  /** ISO-639-1 hint, e.g. "de". */
  language?: string;
  /** Context hint — domain vocabulary, names, expected spelling. */
  prompt?: string;
  headers?: Record<string, string>;
  signal?: AbortSignal;
  purpose?: string;
}

export interface SpeakCall {
  model?: string;
  input: string;
  voice?: string;
  format?: AudioFormat;
  speed?: number;
  instructions?: string;
  headers?: Record<string, string>;
  signal?: AbortSignal;
  purpose?: string;
}

export interface RunCall<C = unknown, T = unknown> extends Omit<RunOptions<C, T>, "tools"> {
  model?: string;
  system?: string;
  prompt?: string;
  messages?: Message[];
  tools: Tool<any, C>[];
  maxTokens?: number;
  cache?: boolean;
  /** Mark the transcript-so-far as reusable each turn — prior turns of the run read from cache. See `BaseRequest.cacheConversation`. */
  cacheConversation?: boolean;
  headers?: Record<string, string>;
  purpose?: string;
  /** How hard the model should think. Precedence: here > the model alias > `defaults.reasoningEffort`. */
  reasoningEffort?: ReasoningEffort;
  /** Merged flat into the wire body, last — MAY override coax's own fields. See `BaseRequest.extraBody`. */
  extraBody?: Record<string, unknown>;
}

/** What `ai.stream()` opens: the delta stream, and the final result once the stream is consumed. */
export interface TextStream {
  /** Text deltas in arrival order. Iterate exactly once. */
  stream: AsyncIterable<string>;
  /** Resolves with the final `TextResult` (full text + usage) once `stream` has been fully consumed;
   *  rejects if the stream dies mid-flight. If you only want the final text, use `ai.text()`. */
  result: Promise<TextResult>;
}

/** Recursively optional — the honest type of an object still being generated. */
export type DeepPartial<T> = T extends (infer U)[] ? DeepPartial<U>[] : T extends object ? { [K in keyof T]?: DeepPartial<T[K]> } : T;

/** What `ai.streamObject()` opens: progressively completed partials, and the validated final result. */
export interface ObjectStream<T> {
  /** Snapshots of the object as it is generated — UNVALIDATED (only the final result is schema-checked).
   *  A repair round restarts the snapshots from scratch. Iterate exactly once. */
  partials: AsyncIterable<DeepPartial<T>>;
  /** Resolves with the validated `ObjectResult` once `partials` has been fully consumed; rejects (e.g.
   *  CoaxSchemaError after exhausted repairs) if the stream dies. */
  result: Promise<ObjectResult<T>>;
}

/**
 * Split an opened generator into a re-yielding stream and a promise of its return value. `opened.first`
 * was already pulled by the caller (inside the fallback boundary); the result promise is pre-caught so
 * a caller that only iterates the stream never sees an unhandled rejection — the same error already
 * surfaces through the iteration.
 */
function split<Y, R>(opened: { gen: AsyncGenerator<Y, R, void>; first: IteratorResult<Y, R> }): { stream: AsyncIterable<Y>; result: Promise<R> } {
  let resolveResult!: (r: R) => void;
  let rejectResult!: (e: unknown) => void;
  const result = new Promise<R>((res, rej) => {
    resolveResult = res;
    rejectResult = rej;
  });
  result.catch(() => {});
  async function* pump(): AsyncGenerator<Y, void, void> {
    try {
      let cur = opened.first;
      while (!cur.done) {
        yield cur.value;
        cur = await opened.gen.next();
      }
      resolveResult(cur.value);
    } catch (err) {
      rejectResult(err);
      throw err;
    }
  }
  return { stream: pump(), result };
}

export interface AI {
  /** Typed, validated, self-repairing structured output. */
  object<T>(call: ObjectCall<T>): Promise<ObjectResult<T>>;
  /** Free-form text. */
  text(call: TextCall): Promise<TextResult>;
  /**
   * Token streaming for text — same call shape as `text()`. The returned promise resolves once the
   * stream has STARTED (the first delta is in), so retryable failures and model fallback still apply
   * to a primary that dies before producing anything; after the first delta the stream is committed —
   * no fallback mid-stream, a later failure surfaces through the iteration and `result`.
   */
  stream(call: TextCall): Promise<TextStream>;
  /**
   * Structured output as a stream of partial objects — same call shape as `object()`. Each partial is
   * the current parse of the JSON generated so far (unvalidated; the final `result` is validated and
   * self-repairing exactly like `object()`, with repair rounds restarting the partials). Same
   * opening/fallback semantics as `stream()`.
   */
  streamObject<T>(call: ObjectCall<T>): Promise<ObjectStream<T>>;
  /** Embeddings — one vector per input, with the same alias/fallback/usage plumbing as every call. */
  embed(call: EmbedCall): Promise<EmbedResponse>;
  /**
   * LLM-as-judge: score an output against a rubric. Returns a numeric score, a pass/fail against the
   * threshold, and a rationale. Use it to verify non-deterministic output that a schema can't catch —
   * intent satisfaction, quality, tone — including multimodal (judge a rendered screenshot).
   */
  judge(call: JudgeCall): Promise<Judgement>;
  /**
   * Tool-calling run: hand the model a set of tools and let it work until it answers. coax validates
   * every tool's arguments against its Zod schema, runs your handler, feeds the result back, and
   * returns the final text plus the full transcript. With an `output` schema, the run ends through a
   * validated answer tool instead — the typed object lands on `RunResult.data`.
   */
  run<C = unknown, T = unknown>(call: RunCall<C, T>): Promise<RunResult<T>>;
  /**
   * Agent loop: each turn returns a typed step (usually a discriminated union); your `onStep` handler
   * either finishes or feeds back the next user message. Built-in doom guard + optional token budget.
   * Prefer `run()` when the model should choose from a set of actions; use `loop()` when YOU own the
   * control flow between turns.
   */
  loop<T, R>(opts: LoopOptions<T, R>): Promise<R>;
  /** Speech-to-text against an endpoint that serves it. */
  transcribe(call: TranscribeCall): Promise<TranscribeResult>;
  /** Text-to-speech against an endpoint that serves it. */
  speak(call: SpeakCall): Promise<SpeakResult>;
  /**
   * Load a `.prompt.md` file and return a callable. Pass `schema` for structured output, else text.
   * The returned function fills the file's `{{ vars }}` and runs the call with the file's config;
   * its second parameter takes per-invocation options (an AbortSignal for cancellation).
   */
  prompt<T = string>(
    path: string,
    opts?: { schema?: ZodType<T>; model?: string },
  ): (vars?: Record<string, unknown>, call?: { signal?: AbortSignal }) => Promise<T extends string ? TextResult : ObjectResult<T>>;
}

async function readFile(path: string): Promise<string> {
  const { readFile: rf } = await import("node:fs/promises");
  return rf(path, "utf8");
}

export function createAI(config: AIConfig): AI {
  const registry = createRegistry(config);
  const d = config.defaults ?? {};

  function clientFor(provider: Provider, alias: string | undefined, purpose: string | undefined, fallback: boolean) {
    const onUsage = config.onUsage
      ? (usage: Usage, model: string) => config.onUsage!(usage, { model, provider: provider.name, alias, purpose, fallback })
      : undefined;
    return createClient({ provider: retrying(provider, d.retries), onUsage });
  }

  function aliasOf(modelRef: string | undefined): string | undefined {
    return modelRef && config.models?.[modelRef] ? modelRef : undefined;
  }

  /**
   * Resolve the model, run `call` against the primary, and fall back to the alias' fallback model on
   * ANY failure. Shared by every capability so fallback behaves identically across them. `call` also
   * receives the alias' `callSettings` (currently just `reasoningEffort`) so the caller can apply the
   * per-call > alias > defaults precedence without the registry's cache key ever seeing it (see
   * `ResolvedModel.callSettings`).
   */
  async function withFallback<R>(
    modelRef: string | undefined,
    requested: string | undefined,
    purpose: string | undefined,
    call: (client: ReturnType<typeof clientFor>, callSettings: CallSettings) => Promise<R>,
  ): Promise<R> {
    const ref = modelRef ?? d.model;
    if (!ref) throw new Error("coax: no model given and no defaults.model configured");
    const { primary, fallback, callSettings } = registry.resolve(ref);
    const alias = aliasOf(requested);
    try {
      return await call(clientFor(primary, alias, purpose, false), callSettings);
    } catch (err) {
      // A user abort is not a model failure — retrying it on the fallback would resurrect the very
      // request the caller just cancelled.
      if (!fallback || err instanceof CoaxAbortError) throw err;
      return await call(clientFor(fallback, alias, purpose, true), callSettings);
    }
  }

  /** Per-call > alias > `defaults.reasoningEffort` — the precedence every capability applies identically. */
  function reasoningEffortFor(callValue: ReasoningEffort | undefined, callSettings: CallSettings): ReasoningEffort | undefined {
    return callValue ?? callSettings.reasoningEffort ?? d.reasoningEffort;
  }

  const api: AI = {
    object<T>(call: ObjectCall<T>): Promise<ObjectResult<T>> {
      return withFallback(call.model, call.model, call.purpose, (client, callSettings) =>
        client.object({
          schema: call.schema,
          schemaName: call.schemaName,
          system: call.system,
          prompt: call.prompt,
          messages: call.messages,
          maxTokens: call.maxTokens ?? d.maxTokens,
          maxRepairs: call.maxRepairs ?? d.maxRepairs,
          cache: call.cache ?? d.cache,
          cacheConversation: call.cacheConversation,
          headers: call.headers,
          signal: call.signal,
          reasoningEffort: reasoningEffortFor(call.reasoningEffort, callSettings),
          extraBody: call.extraBody,
        }),
      );
    },

    text(call: TextCall): Promise<TextResult> {
      return withFallback(call.model, call.model, call.purpose, (client, callSettings) =>
        client.text({
          system: call.system,
          prompt: call.prompt,
          messages: call.messages,
          maxTokens: call.maxTokens ?? d.maxTokens,
          cache: call.cache ?? d.cache,
          cacheConversation: call.cacheConversation,
          headers: call.headers,
          signal: call.signal,
          reasoningEffort: reasoningEffortFor(call.reasoningEffort, callSettings),
          extraBody: call.extraBody,
        }),
      );
    },

    async stream(call: TextCall): Promise<TextStream> {
      // Opening = create the generator AND pull the first item. Anything that fails up to there (a 401,
      // an immediate 500, a model that can't even start) goes through withFallback like any other call.
      const opened = await withFallback(call.model, call.model, call.purpose ?? "stream", async (client, callSettings) => {
        const gen = client.stream({
          system: call.system,
          prompt: call.prompt,
          messages: call.messages,
          maxTokens: call.maxTokens ?? d.maxTokens,
          cache: call.cache ?? d.cache,
          cacheConversation: call.cacheConversation,
          headers: call.headers,
          signal: call.signal,
          reasoningEffort: reasoningEffortFor(call.reasoningEffort, callSettings),
          extraBody: call.extraBody,
        });
        return { gen, first: await gen.next() };
      });

      return split(opened);
    },

    async streamObject<T>(call: ObjectCall<T>): Promise<ObjectStream<T>> {
      const opened = await withFallback(call.model, call.model, call.purpose ?? "streamObject", async (client, callSettings) => {
        const gen = client.streamObject<T>({
          schema: call.schema,
          schemaName: call.schemaName,
          system: call.system,
          prompt: call.prompt,
          messages: call.messages,
          maxTokens: call.maxTokens ?? d.maxTokens,
          maxRepairs: call.maxRepairs ?? d.maxRepairs,
          cache: call.cache ?? d.cache,
          cacheConversation: call.cacheConversation,
          headers: call.headers,
          signal: call.signal,
          reasoningEffort: reasoningEffortFor(call.reasoningEffort, callSettings),
          extraBody: call.extraBody,
        });
        return { gen, first: await gen.next() };
      });
      const { stream, result } = split(opened);
      return { partials: stream as AsyncIterable<DeepPartial<T>>, result };
    },

    async embed(call: EmbedCall): Promise<EmbedResponse> {
      return withFallback(call.model, call.model, call.purpose ?? "embed", (client) =>
        client.embed({ input: call.input, headers: call.headers, signal: call.signal, extraBody: call.extraBody }),
      );
    },

    async judge(call: JudgeCall): Promise<Judgement> {
      const [min, max] = call.scale ?? [1, 5];
      const passScore = call.passScore ?? Math.ceil((min + max) / 2);
      const schema = z.object({
        score: z.number().min(min).max(max).describe(`Score from ${min} (fails the criteria) to ${max} (fully meets them).`),
        rationale: z.string().describe("One or two sentences: concretely why this score, citing the criteria."),
      });
      const criteria = Array.isArray(call.criteria) ? call.criteria.map((c, i) => `${i + 1}. ${c}`).join("\n") : call.criteria;
      const output = typeof call.output === "string" ? call.output : JSON.stringify(call.output, null, 2);
      const system = call.system
        ?? `You are a strict, fair evaluator. Score the OUTPUT against the CRITERIA on a ${min}-${max} scale where ${max} fully meets them and ${min} fails. Judge only against the criteria; be specific in the rationale.`;
      const messages: Message[] = [{ role: "user", content: `CRITERIA:\n${criteria}\n\nOUTPUT:\n${output}`, ...(call.media ? { media: call.media } : {}) }];
      const { data } = await api.object({ model: call.model, schema, system, messages, headers: call.headers, signal: call.signal, purpose: call.purpose ?? "judge" });
      return { score: data.score, pass: data.score >= passScore, rationale: data.rationale };
    },

    // `async` so a bad call rejects rather than throwing synchronously — same contract as every other method.
    async run<C = unknown, T = unknown>(call: RunCall<C, T>): Promise<RunResult<T>> {
      const messages = call.messages?.length ? call.messages : call.prompt != null ? [{ role: "user" as const, content: call.prompt }] : undefined;
      if (!messages) throw new Error("coax: provide either `prompt` or `messages`");
      return withFallback(call.model, call.model, call.purpose, (client, callSettings) =>
        runTools<C, T>(
          (req) =>
            client.tools({
              system: call.system,
              messages: req.messages,
              tools: req.tools,
              toolChoice: req.toolChoice,
              maxTokens: call.maxTokens ?? d.maxTokens,
              cacheSystem: call.cache ?? d.cache,
              cacheConversation: call.cacheConversation,
              headers: call.headers,
              signal: call.signal,
              reasoningEffort: reasoningEffortFor(call.reasoningEffort, callSettings),
              extraBody: call.extraBody,
            }),
          {
            messages,
            tools: call.tools,
            context: call.context,
            output: call.output,
            // `call.maxSteps` may legitimately BE `null` (unlimited) — `??` would treat that as "not
            // set" and fall through to `d.maxSteps`, which is wrong; only `undefined` defers to defaults.
            maxSteps: call.maxSteps !== undefined ? call.maxSteps : d.maxSteps,
            budget: call.budget,
            signal: call.signal,
            onToolCall: call.onToolCall,
            toolChoice: call.toolChoice,
          },
        ),
      );
    },

    loop<T, R>(opts: LoopOptions<T, R>): Promise<R> {
      return runLoop<T, R>((call) => api.object(call), opts);
    },

    transcribe(call: TranscribeCall): Promise<TranscribeResult> {
      return withFallback(call.model, call.model, call.purpose ?? "transcribe", (client) =>
        client.transcribe({ audio: call.audio, language: call.language, prompt: call.prompt, headers: call.headers, signal: call.signal }),
      );
    },

    speak(call: SpeakCall): Promise<SpeakResult> {
      return withFallback(call.model, call.model, call.purpose ?? "speak", (client) =>
        client.speak({
          input: call.input,
          voice: call.voice,
          format: call.format,
          speed: call.speed,
          instructions: call.instructions,
          headers: call.headers,
          signal: call.signal,
        }),
      );
    },

    prompt<T = string>(path: string, opts?: { schema?: ZodType<T>; model?: string }) {
      let parsed: ParsedPrompt | undefined;
      return (async (vars: Record<string, unknown> = {}, call?: { signal?: AbortSignal }) => {
        if (!parsed) parsed = parsePrompt(await readFile(path));
        const system = parsed.system ? renderTemplate(parsed.system, vars) : undefined;
        const user = renderTemplate(parsed.user, vars);
        const model = opts?.model ?? parsed.meta.model;
        if (opts?.schema) {
          return api.object({ model, schema: opts.schema, system, prompt: user, maxRepairs: parsed.meta.maxRepairs, maxTokens: parsed.meta.maxTokens, signal: call?.signal, purpose: parsed.meta.purpose });
        }
        return api.text({ model, system, prompt: user, maxTokens: parsed.meta.maxTokens, signal: call?.signal, purpose: parsed.meta.purpose });
      }) as (vars?: Record<string, unknown>, call?: { signal?: AbortSignal }) => Promise<T extends string ? TextResult : ObjectResult<T>>;
    },
  };

  return api;
}
