import { describe, expect, it } from "vitest";
import { withRetry } from "../src/retry";

const transient = () => Object.assign(new Error("rate limited"), { status: 429 });

describe("withRetry + AbortSignal", () => {
  it("wakes from the backoff immediately when the signal aborts, rethrowing the original error", async () => {
    const boom = transient();
    const ac = new AbortController();
    const started = Date.now();
    const pending = withRetry(
      async () => {
        throw boom;
      },
      { attempts: 3, initialDelayMs: 5_000 },
      ac.signal,
    );
    setTimeout(() => ac.abort(), 10);
    await expect(pending).rejects.toBe(boom);
    expect(Date.now() - started).toBeLessThan(1_000); // did not sit out the 5s backoff
  });

  it("does not start another attempt once aborted", async () => {
    const boom = transient();
    const ac = new AbortController();
    let attempts = 0;
    const pending = withRetry(
      async () => {
        attempts++;
        throw boom;
      },
      { attempts: 5, initialDelayMs: 50 },
      ac.signal,
    );
    setTimeout(() => ac.abort(), 5);
    await expect(pending).rejects.toBe(boom);
    expect(attempts).toBe(1);
  });
});
