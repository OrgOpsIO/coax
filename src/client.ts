import { zodToJsonSchema } from "zod-to-json-schema";
import type { ZodType } from "zod";
import { extractJson } from "./parse";
import { addUsage, emptyUsage, type Message, type Provider, type Usage } from "./types";

export class CoaxSchemaError extends Error {
  constructor(message: string, readonly lastError: string, readonly attempts: number) {
    super(message);
    this.name = "CoaxSchemaError";
  }
}

/** Minimal Zod surface coax relies on — kept loose so both Zod 3 and Zod 4 satisfy it. */
type SafeParseResult<T> =
  | { success: true; data: T }
  | { success: false; error: { issues?: { path: (string | number)[]; message: string }[]; message?: string } };

function formatIssues(error: { issues?: { path: (string | number)[]; message: string }[]; message?: string }): string {
  if (!error.issues?.length) return error.message ?? "output did not match the schema";
  return error.issues
    .map((i) => `- ${i.path.length ? i.path.join(".") : "(root)"}: ${i.message}`)
    .join("\n");
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
  text(req: { system?: string; prompt?: string; messages?: Message[]; maxTokens?: number }): Promise<TextResult>;
}

function toMessages(prompt: string | undefined, messages: Message[] | undefined): Message[] {
  if (messages?.length) return [...messages];
  if (prompt != null) return [{ role: "user", content: prompt }];
  throw new Error("coax: provide either `prompt` or `messages`");
}

export function createClient(opts: ClientOptions): Client {
  const { provider, onUsage } = opts;

  return {
    provider,

    async object<T>(req: ObjectRequest<T>): Promise<ObjectResult<T>> {
      const schemaName = req.schemaName ?? "output";
      const jsonSchema = zodToJsonSchema(req.schema as never, schemaName) as Record<string, unknown>;
      const maxRepairs = req.maxRepairs ?? opts.defaultMaxRepairs ?? 2;
      const messages = toMessages(req.prompt, req.messages);

      let usage = emptyUsage();
      let model = provider.model;
      let lastError = "";

      for (let attempt = 0; attempt <= maxRepairs; attempt++) {
        const res = await provider.structured({ system: req.system, messages, jsonSchema, schemaName, maxTokens: req.maxTokens });
        usage = addUsage(usage, res.usage);
        model = res.model;
        await onUsage?.(res.usage, res.model);

        const parsed = (req.schema as unknown as { safeParse(d: unknown): SafeParseResult<T> }).safeParse(extractJson(res.raw));
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
      const res = await provider.text({ system: req.system, messages: toMessages(req.prompt, req.messages), maxTokens: req.maxTokens });
      await onUsage?.(res.usage, res.model);
      return { text: res.text, usage: res.usage, model: res.model };
    },
  };
}
