import type { GoogleGenAI } from "@google/genai";
import { CoaxAbortError } from "../client";
import {
  CoaxRefusalError,
  emptyUsage,
  type EmbedRequest,
  type EmbedResponse,
  type Message,
  type Provider,
  type ProviderResponse,
  type ReasoningEffort,
  type StructuredRequest,
  type TextRequest,
  type ToolCall,
  type ToolChoice,
  type ToolsRequest,
  type ToolsResponse,
  type Usage,
  withBilledUsage,
} from "../types";

export interface GoogleOptions {
  model: string;
  /** An Agent Platform API key. Reaches generate/stream only — a key-only provider has no `embed`. */
  apiKey?: string;
  /** Google Cloud project for Application Default Credentials (the route used when no `apiKey` is set). */
  project?: string;
  /** Agent Platform location, e.g. "global", "eu", "europe-west4". Default "global". With `apiKey`, only
   *  "global" — a key is served from there, so any other location is a config error, not dropped. */
  location?: string;
  /** Passed verbatim to the SDK's auth (e.g. `{ credentials: serviceAccountJson }`). Not with `apiKey`. */
  googleAuthOptions?: Record<string, unknown>;
  /** Inject an existing SDK client; otherwise coax lazily constructs one (the SDK ships inside coax). */
  client?: GoogleGenAI;
  maxTokens?: number;
  /** Headers sent with every request (per-call `headers` are merged over these). */
  headers?: Record<string, string>;
  /** Deep-merged into every request body (REST field names), under the per-call `extraBody`. */
  extraBody?: Record<string, unknown>;
  /** Model for `embed()`, over the reference's when set. Default: the model of the reference itself (e.g. `google:gemini-embedding-001`). */
  embedModel?: string;
}

type FunctionCall = { id?: string; name?: string; args?: Record<string, unknown> };
type Part = { text?: string; thought?: boolean; thoughtSignature?: string; functionCall?: FunctionCall; [key: string]: unknown };
type Candidate = { content?: { parts?: Part[] }; finishReason?: string; finishMessage?: string };
type GoogleResponse = {
  candidates?: Candidate[];
  promptFeedback?: { blockReason?: string; blockReasonMessage?: string };
  usageMetadata?: Record<string, unknown>;
};
type Params = { model: string; contents: unknown[]; config: Record<string, unknown> };
type EmbedParams = { model: string; contents: string[]; config?: Record<string, unknown> };
type AnyClient = {
  models: {
    generateContent(params: Params): Promise<GoogleResponse>;
    generateContentStream(params: Params): Promise<AsyncIterable<GoogleResponse>>;
    embedContent(params: EmbedParams): Promise<{ embeddings?: { values?: number[]; statistics?: { tokenCount?: number } }[] }>;
  };
};

/** The opaque state a Google tool turn hands back (see `ToolsResponse.providerData`). */
type Carrier = { provider: "google"; parts: Part[] };

// The Agent Platform's documented signature for history the model did not produce itself (another
// vendor's turns, a coax fallback mid-run, hand-built transcripts). Gemini 3 answers 400 when the first
// functionCall of a step comes back without any signature.
const FOREIGN_SIGNATURE = "skip_thought_signature_validator";

// Output withheld by a filter: the safety family, plus recitation (copyright) — the same caller-visible
// situation and the same rescue (a fallback model). Compared as strings: the SDK enum lacks MODEL_ARMOR.
const REFUSAL_FINISH = new Set([
  "SAFETY",
  "PROHIBITED_CONTENT",
  "BLOCKLIST",
  "SPII",
  "IMAGE_SAFETY",
  "IMAGE_PROHIBITED_CONTENT",
  "MODEL_ARMOR",
  "RECITATION",
  "IMAGE_RECITATION",
]);
// Everything else that ends a turn (MALFORMED_FUNCTION_CALL, UNEXPECTED_TOOL_CALL, …) is a failure, not
// a decline — an Error, never an empty success. MAX_TOKENS returns what was produced, as on the other wires.
const USABLE_FINISH = new Set(["STOP", "MAX_TOKENS", "FINISH_REASON_UNSPECIFIED"]);

// Gemini 3 has no "off": MINIMAL is the lowest level it offers (and some models reject it with a 400 —
// the endpoint's honest answer, as with OpenAI models that reject "none").
const THINKING_LEVELS: Record<ReasoningEffort, string> = { none: "MINIMAL", low: "LOW", medium: "MEDIUM", high: "HIGH" };

// Google spells "required" as ANY.
const FUNCTION_CALLING_MODES: Record<ToolChoice, string> = { auto: "AUTO", required: "ANY", none: "NONE" };

/** Keywords `responseJsonSchema` supports (Google's structured-output docs). */
export const RESPONSE_KEYWORDS: ReadonlySet<string> = new Set([
  "$id", "$defs", "$ref", "$anchor", "type", "format", "title", "description", "enum", "items", "prefixItems",
  "minItems", "maxItems", "minimum", "maximum", "anyOf", "oneOf", "properties", "additionalProperties", "required",
  "propertyOrdering",
]);

/** Keywords `parametersJsonSchema` supports (the Agent Platform function-calling docs) — a smaller set. */
export const TOOL_KEYWORDS: ReadonlySet<string> = new Set(["$ref", "$defs", "type", "format", "description", "properties", "items", "enum", "anyOf", "required"]);

const DATE_FORMATS = new Set(["date-time", "date", "time"]);

const isPlainObject = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

/**
 * Zod's JSON Schema → the subset Google's grammar accepts. Unlisted keywords are dropped rather than sent:
 * dropping is safe whether the platform ignores or rejects them, passing through is safe in only one of
 * the two. What the grammar no longer enforces, coax still does — every result is validated against the
 * caller's Zod schema and repaired. Never mutates its input.
 */
export function toGoogleSchema(schema: Record<string, unknown>, allowed: ReadonlySet<string>): Record<string, unknown> {
  const sub = (v: unknown): unknown => (isPlainObject(v) ? toGoogleSchema(v, allowed) : v);
  // Keys of `properties`/`$defs` are names, not keywords — never filtered; their bodies are.
  const named = (v: unknown): unknown =>
    isPlainObject(v) ? Object.fromEntries(Object.entries(v).map(([name, s]) => [name, sub(s)])) : v;

  // Google: "If $ref is set on a sub-schema, no other properties, except for than those starting as a $, may be set".
  const refOnly = typeof schema.$ref === "string";
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(schema)) {
    if (refOnly && !key.startsWith("$")) continue;
    if (key === "const") {
      // `const` carries every z.literal / discriminated-union tag; `enum: [v]` is Google's spelling of it.
      if ((typeof value === "string" || typeof value === "number") && allowed.has("enum") && !("enum" in schema)) out.enum = [value];
      continue;
    }
    // Google: oneOf is "interpreted the same" as anyOf; the function-calling list only knows anyOf.
    const name = key === "oneOf" && !allowed.has("oneOf") ? "anyOf" : key;
    if (!allowed.has(name)) continue;
    if (name === "format" && !(typeof value === "string" && DATE_FORMATS.has(value))) continue;
    if (name === "properties" || name === "$defs") out[name] = named(value);
    else if (name === "items" || name === "additionalProperties") out[name] = sub(value);
    else if (name === "anyOf" || name === "oneOf" || name === "prefixItems") out[name] = Array.isArray(value) ? value.map(sub) : value;
    else out[name] = value;
  }
  return out;
}

/**
 * Objects merge recursively; arrays and primitives replace — the same rule the SDK uses when it merges
 * `httpOptions.extraBody` into the REST body, so endpoint ⊕ call ⊕ coax's body all compose alike.
 */
export function deepMerge(base: Record<string, unknown> | undefined, over: Record<string, unknown> | undefined): Record<string, unknown> {
  const out: Record<string, unknown> = { ...base };
  for (const [key, value] of Object.entries(over ?? {})) {
    const current = out[key];
    out[key] = isPlainObject(current) && isPlainObject(value) ? deepMerge(current, value) : value;
  }
  return out;
}

function count(u: Record<string, unknown> | undefined, key: string): number {
  const v = u?.[key];
  return typeof v === "number" ? v : 0;
}

// Thinking is billed as output, so it counts as output (as OpenAI's reasoning tokens do); the prompt count
// includes cache hits, with `cacheReadTokens` as that subset — input + output equals Google's total.
function mapUsage(u: Record<string, unknown> | undefined): Usage {
  return {
    inputTokens: count(u, "promptTokenCount") + count(u, "toolUsePromptTokenCount"),
    outputTokens: count(u, "candidatesTokenCount") + count(u, "thoughtsTokenCount"),
    cacheReadTokens: count(u, "cachedContentTokenCount"),
    // Implicit caching writes are not billed per request.
    cacheWriteTokens: 0,
  };
}

function carrierOf(data: unknown): Carrier | undefined {
  return isPlainObject(data) && data.provider === "google" && Array.isArray(data.parts) ? (data as Carrier) : undefined;
}

/** Answer text: every non-thought text part. Not the SDK's `.text` getter — it warns on any non-text part. */
const textOf = (parts: Part[]): string =>
  parts
    .filter((p) => typeof p.text === "string" && p.thought !== true)
    .map((p) => p.text)
    .join("");

/** Calls in order. Gemini 3 issues ids; for one that didn't, `call_<i>` is coax-internal and never sent back. */
const callsOf = (parts: Part[]): ToolCall[] =>
  parts
    .filter((p): p is Part & { functionCall: FunctionCall } => isPlainObject(p.functionCall))
    .map((p, i) => ({ id: p.functionCall.id ?? `call_${i}`, name: p.functionCall.name ?? "", input: p.functionCall.args ?? {} }));

function toContents(messages: Message[]): unknown[] {
  const out: unknown[] = [];
  // functionCall ids of the model turn just before — a functionResponse echoes an id only if Google issued it.
  let issued = new Set<string>();
  for (const m of messages) {
    if (m.role === "assistant") {
      const carrier = carrierOf(m.providerData);
      let parts: Part[];
      if (carrier) {
        // Verbatim: signatures are positional ("don't merge a Part containing a signature with one that
        // does not"), so the parts go back exactly as Google sent them.
        parts = carrier.parts;
      } else if (m.toolCalls?.length) {
        parts = m.content ? [{ text: m.content }] : [];
        for (const [i, c] of m.toolCalls.entries()) {
          parts.push({
            functionCall: { name: c.name, args: (c.input ?? {}) as Record<string, unknown> },
            ...(i === 0 ? { thoughtSignature: FOREIGN_SIGNATURE } : {}),
          });
        }
      } else {
        parts = [{ text: m.content }];
      }
      issued = new Set(parts.map((p) => p.functionCall?.id).filter((id): id is string => typeof id === "string"));
      out.push({ role: "model", parts });
      continue;
    }

    const parts: unknown[] = [];
    if (m.toolResults?.length) {
      // All results of a step go back in ONE user content, in call order, under the SDK's documented keys.
      for (const r of m.toolResults) {
        parts.push({
          functionResponse: {
            ...(issued.has(r.id) ? { id: r.id } : {}),
            name: r.name,
            response: r.isError ? { error: r.output } : { output: r.output ?? null },
          },
        });
      }
      if (m.content) parts.push({ text: m.content });
    } else {
      if (m.content || !m.media?.length) parts.push({ text: m.content });
      for (const media of m.media ?? []) {
        parts.push({ inlineData: { mimeType: media.kind === "pdf" ? "application/pdf" : media.mediaType, data: media.dataBase64 } });
      }
    }
    issued = new Set();
    out.push({ role: "user", parts });
  }
  return out;
}

export function google(opts: GoogleOptions): Provider {
  if (opts.apiKey && opts.project) {
    throw new Error(
      'coax: provider "google" needs either apiKey (Agent Platform API key) or project (Application Default Credentials), not both',
    );
  }
  // The SDK serves every key-without-project client from the global host, whatever location it is given —
  // a region is data residency, so dropping it silently would change what the caller configured.
  if (opts.apiKey && opts.location !== undefined && opts.location !== "global") {
    throw new Error(
      `coax: provider "google" serves an apiKey from the "global" location only — location "${opts.location}" needs project (Application Default Credentials) instead of apiKey`,
    );
  }
  if (opts.apiKey && opts.googleAuthOptions) {
    throw new Error(
      'coax: provider "google" needs either apiKey (Agent Platform API key) or googleAuthOptions (Google auth library credentials), not both',
    );
  }
  let client: AnyClient | undefined = opts.client as unknown as AnyClient | undefined;

  async function getClient(): Promise<AnyClient> {
    if (client) return client;
    const mod = await import("@google/genai");
    const Ctor = (mod as unknown as { GoogleGenAI: new (o: Record<string, unknown>) => AnyClient }).GoogleGenAI;
    // `enterprise: true` is the SDK's flag for the Agent Platform. No SDK retries: coax's withRetry
    // already retries the SDK's ApiError statuses — two layers would multiply attempts.
    client = new Ctor(
      opts.apiKey
        ? { enterprise: true, apiKey: opts.apiKey }
        : {
            enterprise: true,
            ...(opts.project ? { project: opts.project } : {}),
            location: opts.location ?? "global",
            ...(opts.googleAuthOptions ? { googleAuthOptions: opts.googleAuthOptions } : {}),
          },
    );
    return client;
  }

  const httpOptions = (headers: Record<string, string> | undefined, extraBody: Record<string, unknown>): Record<string, unknown> => {
    const mergedHeaders = { ...opts.headers, ...headers };
    const out: Record<string, unknown> = {
      ...(Object.keys(mergedHeaders).length ? { headers: mergedHeaders } : {}),
      ...(Object.keys(extraBody).length ? { extraBody } : {}),
    };
    return Object.keys(out).length ? { httpOptions: out } : {};
  };

  /** The generation config every path shares. Cache hints are not read: caching is implicit on Google. */
  function config(req: TextRequest): Record<string, unknown> {
    return {
      ...(req.system ? { systemInstruction: req.system } : {}),
      maxOutputTokens: req.maxTokens ?? opts.maxTokens ?? 8192,
      ...(req.reasoningEffort ? { thinkingConfig: { thinkingLevel: THINKING_LEVELS[req.reasoningEffort] } } : {}),
      // extraBody is deep-merged: every Google generation knob sits under `generationConfig`, and a flat
      // merge would wipe coax's own (structured output, token cap, thinking) the moment a caller sets one.
      ...httpOptions(req.headers, deepMerge(opts.extraBody, req.extraBody)),
      // The SDK drops the connection; the service itself is not cancelled (SDK doc).
      ...(req.signal ? { abortSignal: req.signal } : {}),
    };
  }

  const params = (req: TextRequest, extra: Record<string, unknown> = {}): Params => ({
    model: opts.model,
    contents: toContents(req.messages),
    config: { ...config(req), ...extra },
  });

  const structuredConfig = (req: StructuredRequest) => ({
    responseMimeType: "application/json",
    // Google has no name field for the schema; `schemaName` has nowhere to go.
    responseJsonSchema: toGoogleSchema(req.jsonSchema, RESPONSE_KEYWORDS),
  });

  const toolsConfig = (req: ToolsRequest) => ({
    tools: [
      {
        functionDeclarations: req.tools.map((t) => ({
          name: t.name,
          description: t.description,
          parametersJsonSchema: toGoogleSchema(t.jsonSchema, TOOL_KEYWORDS),
        })),
      },
    ],
    ...(req.toolChoice ? { toolConfig: { functionCallingConfig: { mode: FUNCTION_CALLING_MODES[req.toolChoice] } } } : {}),
  });

  // The SDK throws nothing for a safety block — it is an HTTP 200. Every path runs through these so a
  // block surfaces as CoaxRefusalError (with what the refused call was billed), never as an empty success.
  function assertNotBlocked(resp: GoogleResponse, usage: Usage): void {
    const feedback = resp.promptFeedback;
    if (feedback?.blockReason) throw new CoaxRefusalError(opts.model, feedback.blockReason, feedback.blockReasonMessage ?? null, usage);
    const candidate = resp.candidates?.[0];
    if (candidate?.finishReason && REFUSAL_FINISH.has(candidate.finishReason)) {
      throw new CoaxRefusalError(opts.model, candidate.finishReason, candidate.finishMessage ?? null, usage);
    }
  }

  // A turn without a usable answer (MALFORMED_FUNCTION_CALL & co., or no candidate at all) was still billed:
  // its usage is marked on the error, so coax reports and counts it like a refusal's.
  function assertFinished(sawCandidate: boolean, finishReason: string | undefined, finishMessage: string | undefined, usage: Usage): void {
    if (!sawCandidate) throw withBilledUsage(new Error(`coax: google model ${opts.model} returned no candidates`), usage);
    if (finishReason && !USABLE_FINISH.has(finishReason)) {
      throw withBilledUsage(
        new Error(
          `coax: google model ${opts.model} ended the turn without a usable answer (finishReason "${finishReason}"${finishMessage ? `: ${finishMessage}` : ""})`,
        ),
        usage,
      );
    }
  }

  async function generate(p: Params): Promise<{ parts: Part[]; usage: Usage }> {
    const c = await getClient();
    const resp = await c.models.generateContent(p);
    const usage = mapUsage(resp.usageMetadata);
    assertNotBlocked(resp, usage);
    const candidate = resp.candidates?.[0];
    assertFinished(candidate !== undefined, candidate?.finishReason, candidate?.finishMessage, usage);
    return { parts: [...(candidate?.content?.parts ?? [])], usage };
  }

  /**
   * One streamed turn: yields answer-text deltas, returns every part of every chunk in arrival order —
   * unmerged, because a signature may ride in an empty-text part of the last chunk. Usage is the LAST
   * chunk's, never a sum (whether intermediate chunks are cumulative is not documented).
   */
  async function* generateStream(p: Params): AsyncGenerator<string, { parts: Part[]; usage: Usage }, void> {
    const c = await getClient();
    const stream = await c.models.generateContentStream(p);
    const parts: Part[] = [];
    let usage = emptyUsage();
    let sawCandidate = false;
    let finishReason: string | undefined;
    let finishMessage: string | undefined;
    for await (const chunk of stream) {
      if (chunk.usageMetadata) usage = mapUsage(chunk.usageMetadata);
      // A blocked prompt arrives in the first chunk — rejected before anything is yielded.
      if (chunk.promptFeedback?.blockReason) assertNotBlocked(chunk, usage);
      const candidate = chunk.candidates?.[0];
      for (const part of candidate?.content?.parts ?? []) {
        parts.push(part);
        if (typeof part.text === "string" && part.text && part.thought !== true) yield part.text;
      }
      // A filter stop is checked after the chunk's text: deltas already yielded stay yielded, then the
      // iteration throws.
      assertNotBlocked(chunk, usage);
      if (!candidate) continue;
      sawCandidate = true;
      if (candidate.finishReason) {
        finishReason = candidate.finishReason;
        finishMessage = candidate.finishMessage;
      }
    }
    assertFinished(sawCandidate, finishReason, finishMessage, usage);
    return { parts, usage };
  }

  function toToolsResponse(parts: Part[], usage: Usage): ToolsResponse {
    const calls = callsOf(parts);
    // The turn's parts, verbatim, carry the thought signatures the next request must send back. An
    // object (not an array), so the anthropic provider — which replays arrays only — ignores it after a
    // cross-vendor fallback.
    const carrier: Carrier = { provider: "google", parts };
    return { text: textOf(parts), calls, usage, model: opts.model, ...(calls.length ? { providerData: carrier } : {}) };
  }

  const provider: Provider = {
    name: "google",
    model: opts.model,

    async structured(req: StructuredRequest): Promise<ProviderResponse> {
      const { parts, usage } = await generate(params(req, structuredConfig(req)));
      // Structured output arrives as JSON text; the client parses it like any other raw string.
      const text = textOf(parts);
      return { raw: text, text, usage, model: opts.model };
    },

    async text(req: TextRequest): Promise<ProviderResponse> {
      const { parts, usage } = await generate(params(req));
      const text = textOf(parts);
      return { raw: text, text, usage, model: opts.model };
    },

    async *structuredStream(req: StructuredRequest): AsyncGenerator<string, ProviderResponse, void> {
      const { parts, usage } = yield* generateStream(params(req, structuredConfig(req)));
      const text = textOf(parts);
      return { raw: text, text, usage, model: opts.model };
    },

    async *textStream(req: TextRequest): AsyncGenerator<string, ProviderResponse, void> {
      const { parts, usage } = yield* generateStream(params(req));
      const text = textOf(parts);
      return { raw: text, text, usage, model: opts.model };
    },

    async tools(req: ToolsRequest): Promise<ToolsResponse> {
      const { parts, usage } = await generate(params(req, toolsConfig(req)));
      return toToolsResponse(parts, usage);
    },

    async *toolsStream(req: ToolsRequest): AsyncGenerator<string, ToolsResponse, void> {
      const { parts, usage } = yield* generateStream(params(req, toolsConfig(req)));
      return toToolsResponse(parts, usage);
    },
  };

  // An Agent Platform API key reaches generateContent/streamGenerateContent only — an embed that always
  // 403s would be a capability that pretends, so a key-only provider has none and coax names what's missing.
  if (!(opts.apiKey && !opts.client)) {
    provider.embed = async (req: EmbedRequest): Promise<EmbedResponse> => {
      const model = opts.embedModel ?? opts.model;
      const inputs = typeof req.input === "string" ? [req.input] : req.input;
      if (!inputs.length) return { embeddings: [], usage: emptyUsage(), model };
      const c = await getClient();
      // Only the call's extraBody: an endpoint-wide generationConfig must not land in an embedding request.
      const embedConfig = {
        ...(req.signal ? { abortSignal: req.signal } : {}),
        ...httpOptions(req.headers, req.extraBody ?? {}),
      };
      const embeddings: number[][] = [];
      let tokens = 0;
      const spent = (): Usage => ({ ...emptyUsage(), inputTokens: tokens });
      try {
        // One request per input, in order: gemini-embedding-001 takes a single input per request and the
        // SDK refuses more than one content for gemini-embedding-2.
        for (const input of inputs) {
          if (req.signal?.aborted) throw new CoaxAbortError(spent());
          const resp = await c.models.embedContent({ model, contents: [input], ...(Object.keys(embedConfig).length ? { config: embedConfig } : {}) });
          const embedding = resp.embeddings?.[0];
          if (!embedding?.values) throw new Error(`coax: google model ${model} returned no embedding`);
          embeddings.push(embedding.values);
          tokens += embedding.statistics?.tokenCount ?? 0;
        }
      } catch (err) {
        // The inputs embedded before an abort or a failure were billed — they ride on what is thrown. An
        // abort becomes coax's own CoaxAbortError here, which the client keeps (and reports) as it is.
        const failure = req.signal?.aborted && !(err instanceof CoaxAbortError) ? new CoaxAbortError(spent(), err) : err;
        if (tokens && typeof failure === "object" && failure !== null) withBilledUsage(failure, spent());
        throw failure;
      }
      return { embeddings, usage: spent(), model };
    };
  }

  return provider;
}
