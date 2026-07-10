import { jsonrepair } from "jsonrepair";

/**
 * The "aggressive" layer — the thing a bare JSON.parse + validate lacks. Turn whatever the model produced
 * into a JS value the schema can validate:
 *   1. If the provider already returned an object (native structured mode), use it as-is.
 *   2. Strip a ```json … ``` markdown fence.
 *   3. JSON.parse; on failure, run jsonrepair (fixes unquoted keys, trailing commas, single quotes,
 *      truncation, comments, …) and parse again.
 *   4. Give up and return the original — the schema validation will produce a precise error to reprompt with.
 * Pure + exported so it is unit-testable without a model.
 */
export function extractJson(raw: unknown): unknown {
  if (raw === null || typeof raw !== "string") return raw;

  let s = raw.trim();
  const fence = s.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  if (fence) s = fence[1]!.trim();

  try {
    return JSON.parse(s);
  } catch {
    // fall through to repair
  }
  try {
    return JSON.parse(jsonrepair(s));
  } catch {
    return raw;
  }
}
