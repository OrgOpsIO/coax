import type { RetryConfig } from "./config";

/** Transient = worth retrying: rate limits, server errors, overloaded, network hiccups. */
export function isTransient(err: unknown): boolean {
  const status = (err as { status?: number; statusCode?: number } | undefined)?.status ?? (err as { statusCode?: number } | undefined)?.statusCode;
  if (typeof status === "number") return status === 408 || status === 409 || status === 429 || (status >= 500 && status < 600);
  const code = (err as { code?: string } | undefined)?.code;
  return code === "ECONNRESET" || code === "ETIMEDOUT" || code === "ECONNREFUSED" || code === "EPIPE";
}

/** Waits `ms`, but wakes up immediately when `signal` aborts — a cancel must not sit out a 30s backoff. */
const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve) => {
    const timer = setTimeout(finish, ms);
    function finish() {
      signal?.removeEventListener("abort", finish);
      clearTimeout(timer);
      resolve();
    }
    signal?.addEventListener("abort", finish, { once: true });
  });

/**
 * Run `fn`, retrying transient failures with exponential backoff. Non-transient errors throw
 * immediately, and an aborted `signal` stops retrying — the last error is rethrown as-is (the
 * client layer normalizes post-abort errors to CoaxAbortError).
 */
export async function withRetry<T>(fn: () => Promise<T>, cfg?: RetryConfig, signal?: AbortSignal): Promise<T> {
  const attempts = Math.max(1, cfg?.attempts ?? 3);
  const initial = cfg?.initialDelayMs ?? 500;
  const max = cfg?.maxDelayMs ?? 30_000;
  let lastErr: unknown;
  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      if (attempt >= attempts || !isTransient(err) || signal?.aborted) throw err;
      await sleep(Math.min(max, initial * 2 ** (attempt - 1)), signal);
      if (signal?.aborted) throw err;
    }
  }
  throw lastErr;
}
