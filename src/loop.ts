import type { ZodType } from "zod";
import type { Message } from "./types";
import type { Budget } from "./budget";
import type { ObjectResult } from "./client";

/** What `onStep` returns: stop with a value, or continue (optionally appending the next user message). */
export type LoopControl<R> = { done: true; value: R } | { done: false; reply?: string | Message[] };

export interface LoopOptions<T, R> {
  model?: string;
  schema: ZodType<T>;
  schemaName?: string;
  system?: string;
  cache?: boolean;
  /** Initial conversation — must include the first user turn. */
  messages: Message[];
  /** Hard cap on turns. Default 8. */
  maxTurns?: number;
  /** Doom guard: stop if the model returns the same step this many times in a row. Default 3. */
  maxRepeat?: number;
  /** Optional token budget — the loop stops before a turn once it is exhausted. */
  budget?: Budget;
  purpose?: string;
  /**
   * Handle one typed step. Return `{ done: true, value }` to stop, or `{ done: false, reply }` to continue
   * — coax appends the model's step as an assistant turn, then your `reply` as the next user turn.
   */
  onStep: (step: T, ctx: { turn: number; messages: Message[] }) => LoopControl<R> | Promise<LoopControl<R>>;
}

export class CoaxLoopError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CoaxLoopError";
  }
}

type ObjectFn = <T>(call: { model?: string; schema: ZodType<T>; schemaName?: string; system?: string; cache?: boolean; messages: Message[]; purpose?: string }) => Promise<ObjectResult<T>>;

/** The agent-loop driver behind `ai.loop`. Kept separate so it is testable with a fake `object` fn. */
export async function runLoop<T, R>(object: ObjectFn, opts: LoopOptions<T, R>): Promise<R> {
  const messages = [...opts.messages];
  const maxTurns = opts.maxTurns ?? 8;
  const maxRepeat = opts.maxRepeat ?? 3;
  let lastFingerprint = "";
  let repeats = 0;

  for (let turn = 0; turn < maxTurns; turn++) {
    if (opts.budget?.over()) throw new CoaxLoopError(`coax: token budget exhausted after ${turn} turn(s)`);

    const { data, usage } = await object<T>({
      model: opts.model, schema: opts.schema, schemaName: opts.schemaName,
      system: opts.system, cache: opts.cache, messages, purpose: opts.purpose,
    });
    opts.budget?.record(usage);

    const fingerprint = JSON.stringify(data);
    if (fingerprint === lastFingerprint) {
      if (++repeats + 1 >= maxRepeat) throw new CoaxLoopError(`coax: loop stuck — the model returned the same step ${repeats + 1} times`);
    } else {
      repeats = 0;
      lastFingerprint = fingerprint;
    }

    const control = await opts.onStep(data, { turn, messages });
    if (control.done) return control.value;

    messages.push({ role: "assistant", content: fingerprint });
    if (control.reply != null) {
      if (typeof control.reply === "string") messages.push({ role: "user", content: control.reply });
      else messages.push(...control.reply);
    }
  }

  throw new CoaxLoopError(`coax: loop did not finish within ${maxTurns} turns`);
}
