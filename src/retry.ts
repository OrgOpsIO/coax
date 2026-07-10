import type { RetryConfig } from "./config";

/** Transient = worth retrying: rate limits, server errors, overloaded, network hiccups. */
export function isTransient(err: unknown): boolean {
  const status = (err as { status?: number; statusCode?: number } | undefined)?.status ?? (err as { statusCode?: number } | undefined)?.statusCode;
  if (typeof status === "number") return status === 408 || status === 409 || status === 429 || (status >= 500 && status < 600);
  const code = (err as { code?: string } | undefined)?.code;
  return code === "ECONNRESET" || code === "ETIMEDOUT" || code === "ECONNREFUSED" || code === "EPIPE";
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/** Run `fn`, retrying transient failures with exponential backoff. Non-transient errors throw immediately. */
export async function withRetry<T>(fn: () => Promise<T>, cfg?: RetryConfig): Promise<T> {
  const attempts = Math.max(1, cfg?.attempts ?? 3);
  const initial = cfg?.initialDelayMs ?? 500;
  const max = cfg?.maxDelayMs ?? 30_000;
  let lastErr: unknown;
  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      if (attempt >= attempts || !isTransient(err)) throw err;
      await sleep(Math.min(max, initial * 2 ** (attempt - 1)));
    }
  }
  throw lastErr;
}
