import { z, type ZodType } from "zod";

/**
 * Zod ↔ provider glue, shared by structured output (`client.object`) and tool inputs (`ai.run`).
 * Pure and unit-testable — no provider, no network.
 */

/** Minimal Zod surface coax relies on — kept loose so both Zod 3 and Zod 4 satisfy it. */
export type ZodIssues = { issues?: { path: (string | number)[]; message: string }[]; message?: string };

export type SafeParseResult<T> = { success: true; data: T } | { success: false; error: ZodIssues };

/** `safeParse` through the loose surface above, so a Zod 3 schema passed by a caller still works. */
export function safeParse<T>(schema: ZodType<T>, data: unknown): SafeParseResult<T> {
  return (schema as unknown as { safeParse(d: unknown): SafeParseResult<T> }).safeParse(data);
}

/** Render validation failures as concrete, repromptable lines — the model corrects far better with them. */
export function formatIssues(error: ZodIssues): string {
  if (!error.issues?.length) return error.message ?? "output did not match the schema";
  return error.issues.map((i) => `- ${i.path.length ? i.path.join(".") : "(root)"}: ${i.message}`).join("\n");
}

export interface ProviderSchema {
  /** A JSON Schema with an object root — what provider tool/json_schema parameters must be. */
  jsonSchema: Record<string, unknown>;
  /** Reverses the envelope (if one was needed) before validation. Identity for object-root schemas. */
  unwrap(raw: unknown): unknown;
}

/**
 * Zod → a provider-safe JSON Schema.
 *
 * Provider tool roots MUST be an object: Anthropic `input_schema` and OpenAI `json_schema` both reject
 * a bare union/array/primitive root ("input_schema.type: Field required"). When the caller's schema
 * isn't an object at the root (e.g. `z.discriminatedUnion` / `z.array` / `z.string`), wrap it under a
 * single `value` property and unwrap the model's output again. Transparent: the caller's schema and
 * result are unchanged.
 */
export function toProviderSchema(schema: ZodType): ProviderSchema {
  // Zod 4's native JSON Schema. Drop `$schema` — providers want a bare parameters object.
  const { $schema: _drop, ...userSchema } = z.toJSONSchema(schema as never) as Record<string, unknown>;
  if (userSchema.type === "object") return { jsonSchema: userSchema, unwrap: (raw) => raw };

  return {
    jsonSchema: { type: "object", properties: { value: userSchema }, required: ["value"], additionalProperties: false },
    // Defensive: if the model returned the bare value anyway (no `value` key), accept it as-is rather
    // than forcing undefined.
    unwrap: (raw) => (raw && typeof raw === "object" && "value" in raw ? (raw as { value: unknown }).value : raw),
  };
}
