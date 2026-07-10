import type { Usage } from "./types";

/** A running token budget — cap the total spend of a fan-out or an agent loop. */
export interface Budget {
  readonly limit: number | null;
  /** Add a call's usage to the running total. */
  record(usage: Usage): void;
  /** Total tokens spent so far (input + output). */
  spent(): number;
  /** Remaining tokens (Infinity when no limit). */
  remaining(): number;
  /** True once spent ≥ limit × threshold (default 1). Always false when there is no limit. */
  over(threshold?: number): boolean;
}

export function createBudget(limit: number | null): Budget {
  let spent = 0;
  return {
    limit,
    record(usage: Usage) {
      spent += usage.inputTokens + usage.outputTokens;
    },
    spent: () => spent,
    remaining: () => (limit == null ? Infinity : Math.max(0, limit - spent)),
    over: (threshold = 1) => limit != null && spent >= limit * threshold,
  };
}
