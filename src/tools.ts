import type { ZodType } from "zod";
import {
  addUsage,
  emptyUsage,
  type Message,
  type ToolCall,
  type ToolChoice,
  type ToolInvocation,
  type ToolResult,
  type ToolsRequest,
  type ToolsResponse,
  type Usage,
} from "./types";
import { formatIssues, safeParse, toProviderSchema } from "./schema";
import { CoaxAbortError } from "./client";
import type { Budget } from "./budget";

// Re-exported for back-compat. The type itself now lives in types.ts: CoaxAbortError (client.ts) needs
// it to carry a run's partial state on abort, and client.ts can't import from tools.ts — tools.ts
// already imports CoaxAbortError from client.ts, and that would be a cycle.
export type { ToolInvocation };

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

export interface RunOptions<C = unknown> {
  tools: Tool<any, C>[];
  /** Passed to every handler as `ctx.context`. */
  context?: C;
  /**
   * Hard cap on model turns (a turn = one model call + the tools it asked for). Default 8.
   * `null` = unlimited — but ONLY together with a real brake: `budget` or `signal`. `null` without
   * either throws immediately. A step count you picked because you had to pick something is a
   * pseudo-limit; a budget or a signal is an honest reason to stop.
   */
  maxSteps?: number | null;
  /** Optional token budget — the run stops before a turn once it is exhausted. */
  budget?: Budget;
  /** Cancel the run — the in-flight model call aborts, and no further turn starts. */
  signal?: AbortSignal;
  /** Fired after each tool runs — for logging, tracing, or streaming progress to a UI. */
  onToolCall?: (invocation: ToolInvocation) => void | Promise<void>;
  /**
   * Restrict how the model may respond this turn. `"required"` forces a tool call — the model can
   * never answer with text — so a run pinned to a constant `"required"` necessarily ends in the
   * `maxSteps` error, since the loop's only other exit (`!res.calls.length`) can never fire. The
   * step-indexed function form is the recommended way to use `"required"`: force step 0 to search,
   * then fall back to `"auto"` once there is something to answer from. Default `"auto"` (today's
   * behavior).
   */
  toolChoice?: ToolChoice | ((step: number) => ToolChoice);
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
 * A tool run that died still cost tokens and produced a transcript — both ride on the error so a
 * failed run can be booked and debugged, not just mourned. Covers every way a run can die: exhausting
 * `maxSteps` or the budget (coax's own errors, no `cause`), and — since 0.6 — any error the underlying
 * model call itself throws (a 500, a timeout, …), wrapped here with the original as `cause` so a dead
 * run is exactly as resumable as hitting a limit. This is a behavior change from 0.5: code that used to
 * catch a raw provider error around `ai.run()` now sees `CoaxToolError` — check `err.cause` for it.
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
  constructor(message: string, state?: { usage?: Usage; messages?: Message[]; calls?: ToolInvocation[]; steps?: number }, cause?: unknown) {
    super(message, cause === undefined ? undefined : { cause });
    this.name = "CoaxToolError";
    this.usage = state?.usage ?? emptyUsage();
    this.messages = state?.messages ?? [];
    this.calls = state?.calls ?? [];
    this.steps = state?.steps ?? 0;
  }
}

type ToolsFn = (req: Pick<ToolsRequest, "messages" | "tools" | "toolChoice">) => Promise<ToolsResponse>;

/** Resolves the possibly step-indexed `toolChoice` option to the plain value a provider sees. */
function resolveToolChoice(toolChoice: RunOptions["toolChoice"], step: number): ToolChoice | undefined {
  return typeof toolChoice === "function" ? toolChoice(step) : toolChoice;
}

/**
 * The tool-calling driver behind `ai.run`. Kept separate from the provider so it is testable with a
 * fake `tools` fn.
 *
 * Failures are fed back, not thrown: an unknown tool, arguments that miss the schema, or a handler that
 * throws all come back to the model as an error result, so it can correct itself — the same
 * validate-then-repair idea as structured output. Only exhausting `maxSteps` / the budget, or the
 * underlying model call itself failing, is fatal.
 */
export async function runTools<C = unknown>(
  callTools: ToolsFn,
  opts: RunOptions<C> & { messages: Message[] },
): Promise<RunResult> {
  const messages = [...opts.messages];
  // `undefined` (never set) defaults to 8; `null` (set on purpose) means unlimited. `?? 8` alone can't
  // tell those apart — it would also replace an explicit `null` with 8.
  const maxSteps = opts.maxSteps === undefined ? 8 : opts.maxSteps;
  if (maxSteps === null && !opts.budget && !opts.signal) {
    throw new Error("coax: maxSteps: null needs a budget or a signal — unlimited without a brake is not a limit, it's a prayer");
  }
  const byName = new Map(opts.tools.map((t) => [t.name, t]));
  const definitions = opts.tools.map((t) => ({
    name: t.name,
    description: t.description,
    jsonSchema: toProviderSchema(t.input).jsonSchema,
  }));

  let usage = emptyUsage();
  let model = "";
  const calls: ToolInvocation[] = [];

  for (let step = 0; maxSteps === null || step < maxSteps; step++) {
    if (opts.signal?.aborted) throw new CoaxAbortError(usage, undefined, messages, calls);
    if (opts.budget?.over()) throw new CoaxToolError(`coax: token budget exhausted after ${step} step(s)`, { usage, messages, calls, steps: step });

    let res: ToolsResponse;
    try {
      res = await callTools({ messages, tools: definitions, toolChoice: resolveToolChoice(opts.toolChoice, step) });
    } catch (err) {
      // An abort mid-call carries only that call's (empty) usage — add what this run spent before it,
      // plus the transcript/calls so far, so an aborted run is resumable exactly like a CoaxToolError.
      if (err instanceof CoaxAbortError) throw new CoaxAbortError(addUsage(usage, err.usage), err, messages, calls);
      // Any other provider failure (a 500, a timeout, …) is wrapped the same way — see CoaxToolError's
      // doc comment for why this is a behavior change from 0.5.
      const message = err instanceof Error ? err.message : String(err);
      throw new CoaxToolError(`coax: the model call failed at step ${step}: ${message}`, { usage, messages, calls, steps: step }, err);
    }
    usage = addUsage(usage, res.usage);
    model = res.model;
    opts.budget?.record(res.usage);

    // No tool calls = the model is answering. Done.
    if (!res.calls.length) return { text: res.text, messages, calls, steps: step + 1, usage, model };

    messages.push({ role: "assistant", content: res.text, toolCalls: res.calls, ...(res.providerData !== undefined ? { providerData: res.providerData } : {}) });

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

  // Unreachable when maxSteps is null — that loop only exits through a `return` or `throw` above, never
  // by falling out the bottom, so this is the bounded case's error only (as the maxSteps doc says).
  throw new CoaxToolError(`coax: the model was still calling tools after ${maxSteps} steps`, { usage, messages, calls, steps: maxSteps as number });
}
