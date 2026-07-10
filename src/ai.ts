import type { ZodType } from "zod";
import type { AIConfig } from "./config";
import type { Message, Provider, Usage } from "./types";
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

export interface AI {
  /** Typed, validated, self-repairing structured output. */
  object<T>(call: ObjectCall<T>): Promise<ObjectResult<T>>;
  /** Free-form text. */
  text(call: TextCall): Promise<TextResult>;
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
