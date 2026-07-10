import { z, type ZodType } from "zod";
import type { AIConfig } from "./config";
import type { Media, Message, Provider, Usage } from "./types";
import { createClient, type ObjectResult, type TextResult } from "./client";
import { createRegistry, retrying } from "./registry";
import { parsePrompt, renderTemplate, type ParsedPrompt } from "./prompt-file";
import { runLoop, type LoopOptions } from "./loop";

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
  /** Free-form label for observability (e.g. a role like "extraction"). */
  purpose?: string;
}

export interface TextCall {
  model?: string;
  system?: string;
  prompt?: string;
  messages?: Message[];
  maxTokens?: number;
  cache?: boolean;
  purpose?: string;
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
  /** For multimodal judging — e.g. a screenshot of the rendered artifact (Day 4: judge the artifact, not the code). */
  media?: Media[];
  purpose?: string;
}

export interface Judgement {
  score: number;
  pass: boolean;
  rationale: string;
}

export interface AI {
  /** Typed, validated, self-repairing structured output. */
  object<T>(call: ObjectCall<T>): Promise<ObjectResult<T>>;
  /** Free-form text. */
  text(call: TextCall): Promise<TextResult>;
  /**
   * LLM-as-judge: score an output against a rubric (Day 4). Returns a numeric score, a pass/fail against
   * the threshold, and a rationale. Use it to verify non-deterministic output that a schema can't catch —
   * intent satisfaction, quality, tone — including multimodal (judge a rendered screenshot).
   */
  judge(call: JudgeCall): Promise<Judgement>;
  /**
   * Agent loop: each turn returns a typed step (usually a discriminated union); your `onStep` handler
   * either finishes or feeds back the next user message. Built-in doom guard + optional token budget.
   */
  loop<T, R>(opts: LoopOptions<T, R>): Promise<R>;
  /**
   * Load a `.prompt.md` file and return a callable. Pass `schema` for structured output, else text.
   * The returned function fills the file's `{{ vars }}` and runs the call with the file's config.
   */
  prompt<T = string>(
    path: string,
    opts?: { schema?: ZodType<T>; model?: string },
  ): (vars?: Record<string, unknown>) => Promise<T extends string ? TextResult : ObjectResult<T>>;
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

  const api: AI = {
    async object<T>(call: ObjectCall<T>): Promise<ObjectResult<T>> {
      const modelRef = call.model ?? d.model;
      if (!modelRef) throw new Error("coax: no model given and no defaults.model configured");
      const { primary, fallback } = registry.resolve(modelRef);
      const alias = aliasOf(call.model);
      const req = {
        schema: call.schema,
        schemaName: call.schemaName,
        system: call.system,
        prompt: call.prompt,
        messages: call.messages,
        maxTokens: call.maxTokens ?? d.maxTokens,
        maxRepairs: call.maxRepairs ?? d.maxRepairs,
        cache: call.cache ?? d.cache,
      };
      try {
        return await clientFor(primary, alias, call.purpose, false).object(req);
      } catch (err) {
        if (!fallback) throw err;
        return await clientFor(fallback, alias, call.purpose, true).object(req);
      }
    },

    async text(call: TextCall): Promise<TextResult> {
      const modelRef = call.model ?? d.model;
      if (!modelRef) throw new Error("coax: no model given and no defaults.model configured");
      const { primary, fallback } = registry.resolve(modelRef);
      const alias = aliasOf(call.model);
      const req = { system: call.system, prompt: call.prompt, messages: call.messages, maxTokens: call.maxTokens ?? d.maxTokens, cache: call.cache ?? d.cache };
      try {
        return await clientFor(primary, alias, call.purpose, false).text(req);
      } catch (err) {
        if (!fallback) throw err;
        return await clientFor(fallback, alias, call.purpose, true).text(req);
      }
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
      const { data } = await api.object({ model: call.model, schema, system, messages, purpose: call.purpose ?? "judge" });
      return { score: data.score, pass: data.score >= passScore, rationale: data.rationale };
    },

    loop<T, R>(opts: LoopOptions<T, R>): Promise<R> {
      return runLoop<T, R>((call) => api.object(call), opts);
    },

    prompt<T = string>(path: string, opts?: { schema?: ZodType<T>; model?: string }) {
      let parsed: ParsedPrompt | undefined;
      return (async (vars: Record<string, unknown> = {}) => {
        if (!parsed) parsed = parsePrompt(await readFile(path));
        const system = parsed.system ? renderTemplate(parsed.system, vars) : undefined;
        const user = renderTemplate(parsed.user, vars);
        const model = opts?.model ?? parsed.meta.model;
        if (opts?.schema) {
          return api.object({ model, schema: opts.schema, system, prompt: user, maxRepairs: parsed.meta.maxRepairs, maxTokens: parsed.meta.maxTokens, purpose: parsed.meta.purpose });
        }
        return api.text({ model, system, prompt: user, maxTokens: parsed.meta.maxTokens, purpose: parsed.meta.purpose });
      }) as (vars?: Record<string, unknown>) => Promise<T extends string ? TextResult : ObjectResult<T>>;
    },
  };

  return api;
}
