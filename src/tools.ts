import type { ZodType } from "zod";
import { addUsage, emptyUsage, type Message, type ToolCall, type ToolResult, type ToolsRequest, type ToolsResponse, type Usage } from "./types";
import { formatIssues, safeParse, toProviderSchema } from "./schema";
import { CoaxAbortError } from "./client";
import type { Budget } from "./budget";

/**
 * A tool the model may call. The Zod schema is the contract in both directions: it becomes the JSON
 * Schema the provider constrains arguments to, AND it validates what actually arrives before your
 * handler sees it — so `run` receives typed, checked input, never `any` from the wire.
 *
 * Put the tools where the data is. In a backend-for-frontend, a tool is the natural place to call your
 * own API: the model never sees your credentials, only the result you choose to return.
 */
export interface Tool<I = any, C = any> {
  /** Must match `^[a-zA-Z0-9_-]+$` — providers reject other names. */
  name: string;
  /** The model picks tools by this text. Say what it does AND when to use it. */
  description: string;
  input: ZodType<I>;
  run(input: I, ctx: ToolContext<C>): unknown | Promise<unknown>;
}

export interface ToolContext<C = unknown> {
  /** Whatever you passed as `context` — request-scoped values (the caller's id, an auth header, a db handle). */
  context: C;
  /** 0-based model turn this call came from. */
  step: number;
  call: ToolCall;
}

/** Identity helper: defines a tool while inferring the handler's input type from the schema. */
export function tool<I, C = unknown>(def: Tool<I, C>): Tool<I, C> {
  return def;
}

/** One tool that actually ran, in order — your audit trail for the turn. */
export interface ToolInvocation {
  name: string;
  input: unknown;
  output: unknown;
  /** Set when the tool failed; `output` then holds the message the model was shown. */
  error?: string;
  durationMs: number;
}

export interface RunOptions<C = unknown> {
  tools: Tool<any, C>[];
  /** Passed to every handler as `ctx.context`. */
  context?: C;
  /** Hard cap on model turns (a turn = one model call + the tools it asked for). Default 8. */
  maxSteps?: number;
  /** Optional token budget — the run stops before a turn once it is exhausted. */
  budget?: Budget;
  /** Cancel the run — the in-flight model call aborts, and no further turn starts. */
  signal?: AbortSignal;
  /** Fired after each tool runs — for logging, tracing, or streaming progress to a UI. */
  onToolCall?: (invocation: ToolInvocation) => void | Promise<void>;
}

export interface RunResult {
  /** The model's final answer, once it stopped calling tools. */
  text: string;
  /** The full transcript, including tool calls and results — feed it back in to continue the conversation. */
  messages: Message[];
  /** Every tool that ran, in order. */
  calls: ToolInvocation[];
  /** Model turns used. */
  steps: number;
  usage: Usage;
  model: string;
}

/**
 * A tool run that died at a limit (maxSteps, budget) still cost tokens and produced a transcript —
 * both ride on the error so a failed run can be booked and debugged, not just mourned.
 */
export class CoaxToolError extends Error {
  /** Usage summed across the turns that completed before the failure. */
  readonly usage: Usage;
  /** The transcript up to the failure — replayable, and gold for debugging a dead agent run. */
  readonly messages: Message[];
  /** Every tool that ran before the failure, in order. */
  readonly calls: ToolInvocation[];
  /** Model turns completed. */
  readonly steps: number;
  constructor(message: string, state?: { usage?: Usage; messages?: Message[]; calls?: ToolInvocation[]; steps?: number }) {
    super(message);
    this.name = "CoaxToolError";
    this.usage = state?.usage ?? emptyUsage();
    this.messages = state?.messages ?? [];
    this.calls = state?.calls ?? [];
    this.steps = state?.steps ?? 0;
  }
}

type ToolsFn = (req: Pick<ToolsRequest, "messages" | "tools">) => Promise<ToolsResponse>;

/**
 * The tool-calling driver behind `ai.run`. Kept separate from the provider so it is testable with a
 * fake `tools` fn.
 *
 * Failures are fed back, not thrown: an unknown tool, arguments that miss the schema, or a handler that
 * throws all come back to the model as an error result, so it can correct itself — the same
 * validate-then-repair idea as structured output. Only exhausting `maxSteps` or the budget is fatal.
 */
export async function runTools<C = unknown>(
  callTools: ToolsFn,
  opts: RunOptions<C> & { messages: Message[] },
): Promise<RunResult> {
  const messages = [...opts.messages];
  const maxSteps = opts.maxSteps ?? 8;
  const byName = new Map(opts.tools.map((t) => [t.name, t]));
  const definitions = opts.tools.map((t) => ({
    name: t.name,
    description: t.description,
    jsonSchema: toProviderSchema(t.input).jsonSchema,
  }));

  let usage = emptyUsage();
  let model = "";
  const calls: ToolInvocation[] = [];

  for (let step = 0; step < maxSteps; step++) {
    if (opts.signal?.aborted) throw new CoaxAbortError(usage);
    if (opts.budget?.over()) throw new CoaxToolError(`coax: token budget exhausted after ${step} step(s)`, { usage, messages, calls, steps: step });

    let res: ToolsResponse;
    try {
      res = await callTools({ messages, tools: definitions });
    } catch (err) {
      // An abort mid-call carries only that call's (empty) usage — add what this run spent before it.
      throw err instanceof CoaxAbortError ? new CoaxAbortError(addUsage(usage, err.usage), err) : err;
    }
    usage = addUsage(usage, res.usage);
    model = res.model;
    opts.budget?.record(res.usage);

    // No tool calls = the model is answering. Done.
    if (!res.calls.length) return { text: res.text, messages, calls, steps: step + 1, usage, model };

    messages.push({ role: "assistant", content: res.text, toolCalls: res.calls });

    // Independent calls in one turn run concurrently — the model asked for them together precisely
    // because they don't depend on each other.
    const results = await Promise.all(
      res.calls.map(async (call): Promise<ToolResult> => {
        const started = Date.now();
        const record = (output: unknown, error?: string): ToolResult => {
          const invocation: ToolInvocation = { name: call.name, input: call.input, output, durationMs: Date.now() - started };
          if (error !== undefined) invocation.error = error;
          calls.push(invocation);
          void opts.onToolCall?.(invocation);
          return { id: call.id, name: call.name, output, ...(error !== undefined ? { isError: true } : {}) };
        };

        const t = byName.get(call.name);
        if (!t) {
          const known = opts.tools.map((x) => x.name).join(", ") || "(none)";
          return record(`No tool named "${call.name}". Available tools: ${known}.`, "unknown tool");
        }

        const parsed = safeParse(t.input, call.input);
        if (!parsed.success) {
          const issues = formatIssues(parsed.error);
          return record(`Your arguments did not match the schema for "${call.name}":\n${issues}\n\nCall it again with corrected arguments.`, issues);
        }

        try {
          return record(await t.run(parsed.data, { context: opts.context as C, step, call }));
        } catch (err) {
          // The message goes to the model, so keep it useful but free of internals — a handler that
          // wants to hide a cause should throw a message it is happy for the model to read.
          const message = err instanceof Error ? err.message : String(err);
          return record(`The tool "${call.name}" failed: ${message}`, message);
        }
      }),
    );

    messages.push({ role: "user", content: "", toolResults: results });
  }

  throw new CoaxToolError(`coax: the model was still calling tools after ${maxSteps} steps`, { usage, messages, calls, steps: maxSteps });
}
