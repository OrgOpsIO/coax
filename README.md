# coax

**A clean, provider-agnostic way to put LLMs into your software — with the good patterns built in.**
Pure TypeScript. No native modules, no codegen, no DSL, no vendor lock-in.

Configure your provider keys once, pick a model by name, hand it a [Zod](https://zod.dev) schema (or a
prompt file), and get back typed, validated data — with retries, model fallback, and self-repair handled
for you. The same configuration also gives you **tools**, **voice** (speech-to-text and back), and any
**OpenAI-/Anthropic-compatible endpoint** — a frontier API and your own self-hosted model side by side.

```bash
npm install @orgops/coax zod
```

The Anthropic and OpenAI SDKs ship *inside* coax — nothing else to install.

## Configure once, use `ai` everywhere

Set your keys and models in **one place** at startup. Then anywhere in your app just import `ai` — no
threading an instance around, no `createAI` at every call site.

```ts
// coax.setup.ts — run once at startup (a Nuxt/Nitro server plugin, a Vite entry, your main.ts)
import { configure } from "@orgops/coax";

configure({
  providers: {
    anthropic: process.env.ANTHROPIC_API_KEY!,   // string key, or { apiKey, baseURL }
    openai: process.env.OPENAI_API_KEY!,
  },
  models: {
    default: "anthropic:claude-sonnet-4-6",
    smart:   { use: "anthropic:claude-opus-4-8", fallback: "anthropic:claude-sonnet-4-6" },
    fast:    "anthropic:claude-haiku-4-5",
    cheap:   "openai:gpt-5-mini",
  },
  defaults: { model: "default", maxRepairs: 2, retries: { attempts: 3 } },
  onUsage: (usage, meta) => track(usage, meta),   // one hook for all your LLM cost/latency
});
```

```ts
// anywhere else — no setup, no imports of an instance
import { ai } from "@orgops/coax";
import { z } from "zod";

const { data } = await ai.object({
  model: "smart",
  schema: z.object({ title: z.string(), tags: z.array(z.string()).min(1) }),
  system: "You label articles.",
  prompt: article,
});
data.tags; // string[] — guaranteed

const { text } = await ai.text({ model: "fast", prompt: "Write a haiku about TypeScript." });
```

Using `ai` before `configure()` throws a clear error — so the setup is explicit and enforced. Switching
provider is one word (`"anthropic:…"` → `"openai:…"`); everything else stays the same.

### Nuxt / Nitro

Keys live in `runtimeConfig` (from env); wire coax once in a server plugin, use `ai` in any route/service:

```ts
// server/plugins/coax.ts
import { configure } from "@orgops/coax";
export default defineNitroPlugin(() => {
  const c = useRuntimeConfig();
  configure({
    providers: { anthropic: c.anthropicApiKey, openai: c.openaiApiKey },
    models: { smart: "anthropic:claude-opus-4-8", fast: "anthropic:claude-haiku-4-5" },
    onUsage: (usage, meta) => appendUsageEvent(usage, meta),
  });
});
```

### Explicit instance (libraries, tests, multiple configs)

Prefer no global state? `createAI(config)` returns the same interface as an instance:

```ts
import { createAI } from "@orgops/coax";
const ai = createAI({ providers: { … }, models: { … } });
```

## Prompt files

Keep prompts out of your code, versioned and reviewable, in a `.prompt.md`:

```md
---
model: smart
maxRepairs: 2
---
# SYSTEM
You are an expert at {{ domain }}.

# USER
{{ input }}
```

```ts
const classify = ai.prompt("./prompts/classify.prompt.md", { schema: LabelSchema });
const { data } = await classify({ domain: "insurance", input: text });
```

No frontmatter/sections needed — a plain `.md` is just the user prompt. Pass `schema` for structured
output, omit it for text.

## Any provider — including your own

`anthropic` and `openai` are built in from a bare API key. For **anything speaking one of those two wire
protocols** — your own gateway, vLLM, LM Studio, a local runtime — name the provider whatever you like
and say which protocol it speaks:

```ts
configure({
  providers: {
    anthropic: process.env.ANTHROPIC_API_KEY!,                   // the vendor API
    orgops: {                                                    // your own server, any name you want
      apiKey: process.env.ORGOPS_API_KEY!,
      baseURL: "https://llm.example.io/v1",
      api: "openai",                                             // or "anthropic" for an /anthropic route
      transcribeModel: "whisper-1",                              // where audio is a separate model
      speakModel: "tts-1",
    },
  },
  models: {
    smart: "anthropic:claude-opus-4-8",
    local: "orgops:chat/Qwen/Qwen3-VL-32B-Instruct-AWQ",         // slashes and colons in the id are fine
  },
});
```

Two OpenAI-wire details are handled for you, overridable per endpoint: the output-token cap goes out as
`max_completion_tokens` on the vendor API (reasoning models reject the deprecated `max_tokens`) and as
`max_tokens` on compatible servers (`tokenParam` overrides). And `strict: true` opts a strict-compatible
endpoint into grammar-guaranteed structured output — the schema's *shape* can then never miss, repair
rounds only fire for what a grammar can't check (`min(1)`, formats, refinements).

Both live side by side — switching a call from a frontier model to your own box is one word. Anything
that speaks neither protocol still plugs in with a factory implementing the small `Provider` interface:

```ts
createAI({
  providers: { gemini: (model) => myGeminiProvider(model) },      // (model: string) => Provider
  models: { flash: "gemini:gemini-2.5-flash" },
});
```

### Forwarding the caller's identity

Every call takes `headers`, merged over the provider's own. That is how a request keeps the **end user's**
identity instead of looking like one service account — so a gateway that authorizes per user (OIDC token →
policy engine) can still decide:

```ts
await ai.object({ model: "local", schema, prompt, headers: { Authorization: req.header("authorization")! } });
```

### Reasoning effort

`reasoningEffort: "none" | "low" | "medium" | "high"` controls how hard the model thinks — the biggest
lever for cost and latency on calls that don't need it (classification, reformatting, anything that isn't
genuinely hard). It hangs on the model alias (so two aliases on the *same* underlying model can think
differently), on `defaults`, and per call — precedence is **call > alias > defaults**:

```ts
configure({
  models: {
    classify:  { use: "orgops:qwen3-vl", reasoningEffort: "none" },  // never thinks
    synthesize: { use: "orgops:qwen3-vl", reasoningEffort: "high" }, // same model, different alias
  },
});

await ai.text({ model: "classify", prompt });                          // "none", from the alias
await ai.text({ model: "classify", prompt, reasoningEffort: "low" });  // overridden for this one call
```

Sent on the wire only where set — an endpoint that has never heard of it never sees the field. OpenAI
gets `reasoning_effort` with the literal value; Anthropic gets `thinking: { type: "enabled", budget_tokens }`
(budget derived from the effort, capped below `max_tokens`), or no `thinking` field at all for `"none"`.

On Anthropic tool runs, thinking blocks round-trip automatically: they ride opaquely on the assistant
message (`Message.providerData`) and are replayed verbatim — signature intact — on the next turn, so
`reasoningEffort` works on `ai.run()` too. Persist that field if you store transcripts. One combination
is impossible by the provider's rules and fails clearly: `toolChoice: "required"` while thinking.

On OpenAI, coax sends the effort literally (`reasoning_effort`); which values a given model accepts is
the endpoint's business — older reasoning models know `low`–`high` but not `"none"`, non-reasoning
models reject the field entirely.

### The `extraBody` escape hatch

Whatever the next gateway needs that coax has no first-class field for — `temperature`, `top_p`, a
vendor-specific `chat_template_kwargs` — goes straight into the wire body, per endpoint and per call:

```ts
configure({
  providers: { orgops: { apiKey, baseURL, api: "openai", extraBody: { temperature: 0.6, top_p: 0.95 } } },
});

await ai.text({ model: "local", prompt, extraBody: { chat_template_kwargs: { enable_thinking: false } } });
```

Flat merge, same rule as `headers`: call over endpoint over coax's own body. No whitelisting — and it
**may** override coax's own fields (`max_tokens`, `tools`, …). That is deliberate: an escape hatch that
can't be overridden by anything isn't one. Overriding a field coax itself relies on is your own risk.

### Cancelling a call

Every call takes a `signal`. When it aborts, the HTTP request to the endpoint is aborted (the SDKs
abort the underlying fetch, so the server sees the disconnect), and loops — repair rounds, `run()`,
`loop()` — stop before starting another turn. Retries stop too, including waking early out of a
backoff pause. The abort surfaces as one error, `CoaxAbortError`, regardless of which layer it landed
in — and it is **never** retried on a fallback model.

```ts
const ac = new AbortController();
req.on("close", () => ac.abort());   // the BFF request died → the upstream generation dies with it

await ai.run({ model: "local", prompt: question, tools, signal: ac.signal });
```

One honest caveat: coax can only drop the connection to the endpoint. Anthropic, OpenAI, and the usual
self-hosted runtimes (vLLM, …) stop generating on disconnect; a gateway in between must pass the
disconnect upstream for the cancellation to reach the model server.

### Failed runs still cost tokens

A run that dies is booked, not lost: `CoaxToolError` and `CoaxLoopError` carry the `usage` summed over the
completed turns plus the transcript up to the failure (`messages`, and `calls` for tool runs),
`CoaxSchemaError` carries the usage *and* the transcript (including every repair reprompt) of its failed
attempts, and `CoaxAbortError` carries what was spent before the abort — plus `messages`/`calls` when the
abort happened inside `ai.run()`, so an aborted run is exactly as resumable as one that hit a limit. The
`onUsage` hook and any `Budget` see every turn as it happens either way — the error fields close the gap
for callers who account from results.

**Breaking in 0.6:** `ai.run()` now wraps *every* failed model call — not just hitting `maxSteps` or the
budget — in `CoaxToolError`, with the original error as `.cause` and the same partial state riding along.
Code that used to catch a raw provider error (a 502, a timeout, …) around `ai.run()` now sees
`CoaxToolError`; check `err.cause` for the original, and `err.messages`/`err.calls` to resume:

```ts
try {
  return await ai.run({ model, prompt, tools });
} catch (err) {
  if (err instanceof CoaxToolError) {
    log(err.cause);                                            // the 502, the timeout, whatever it was
    return ai.run({ model, messages: err.messages, tools });    // pick up exactly where it died
  }
  throw err;
}
```

## What's built in (so you don't hand-roll it every time)

- **Typed contracts** — your Zod schema *is* the spec. `z.string().min(1)` means the model can't return empty.
- **Aggressive parsing** — strips ```` ```json ```` fences and repairs malformed JSON (unquoted keys, trailing
  commas, truncation) before validating.
- **Validate → repair** — on a schema miss, coax reprompts with the exact errors (up to `maxRepairs`).
- **Retries** — transient errors (429/5xx/network) retried with exponential backoff.
- **Model fallback** — a model alias can name a `fallback`; used automatically when the primary fails.
- **Cancellation** — every call takes an `AbortSignal`; the HTTP request aborts, loops stop between
  turns, and the abort surfaces as `CoaxAbortError` (never retried, never sent to the fallback).
- **Prompt caching** — `cache: true` caches the system prompt at the provider (Anthropic `cache_control`;
  a no-op where caching is automatic). Big savings across a fan-out that shares a stable system prompt.
  `cacheConversation: true` marks the conversation-so-far as reusable, so a loop's next turn reads all
  prior turns from cache instead of re-billing the whole transcript.
- **Streaming** — `ai.stream()` yields text deltas as they arrive; `result` resolves with the full
  text and usage once the stream is drained. Model fallback still covers a primary that dies before
  its first token.
- **Tools** — `ai.run()` hands the model typed tools and runs the whole call/validate/reply loop. With
  an `output` schema the run ends through a validated answer tool: typed data on `result.data`.
- **Agent loops** — `ai.loop()` drives a typed multi-turn loop with a built-in doom guard + token budget.
- **Token budget** — `createBudget(limit)` caps the total spend of a loop, a run, or a fan-out.
- **Usage** — one `onUsage(usage, meta)` hook across every call, plus summed `usage` on each result.
- **Vision** — image/pdf media are first-class.
- **Voice** — `ai.transcribe()` / `ai.speak()` where the endpoint serves them, with a precise error where it doesn't.
- **Reasoning effort** — `reasoningEffort: "none" | "low" | "medium" | "high"` per call, per model alias,
  or as a default; mapped to each provider's native wire field, sent only where set.
- **`extraBody`** — a generic, un-whitelisted escape hatch into the wire body, per endpoint and per call,
  for whatever the next gateway needs that coax doesn't have a first-class field for.
- **`toolChoice`** — force/forbid tool calls per run, constant or step-indexed (`"auto" | "required" | "none"`).

```ts
const { data, usage, repairs } = await ai.object({ schema, prompt, maxRepairs: 3 });
// repairs === 0 → valid first try
```

### Tools

`ai.run()` gives the model a set of tools and drives the whole exchange: it hands over the JSON Schema
derived from each tool's Zod schema, **validates the arguments that come back**, runs your handler, feeds
the result in, and repeats until the model answers.

The Zod schema is the contract in both directions — your handler receives typed, checked input, never
`any` off the wire. And a tool is the natural place to reach your own API: the model sees the result you
return, never your credentials.

```ts
import { tool, ai } from "@orgops/coax";

const openInvoices = tool({
  name: "open_invoices",
  description: "List the signed-in client's open invoices. Use for any question about money owed.",
  input: z.object({ since: z.string().date().optional() }),
  run: async ({ since }, ctx) => fetchFromBackend(ctx.context.clientId, since),   // your API, your token
});

const { text, calls, usage } = await ai.run({
  model: "local",
  system: "Du bist der Assistent im Kundenportal. Antworte knapp und auf Deutsch.",
  prompt: question,
  tools: [openInvoices],
  context: { clientId },                 // request-scoped, reaches every handler as ctx.context
  maxSteps: 6,
  budget: createBudget(50_000),
  onToolCall: (c) => log(c.name, c.durationMs),
});
```

A run can also end in **typed data instead of prose**: pass `output` and the model delivers its final
answer through one extra validated tool — schema misses go back for correction, exactly like structured
output:

```ts
const { data } = await ai.run({
  model: "smart",
  prompt: "Wie hoch ist Rechnung R-2026-1, in welcher Währung?",
  tools: [openInvoices],
  output: z.object({ total: z.number(), currency: z.string() }),
});
data.total; // number — validated, not parsed out of prose
```

Failures are **fed back, not thrown**: an invented tool name, arguments that miss the schema, or a handler
that throws all return to the model as an error result so it can correct itself — the same
validate-then-repair idea as structured output. Exhausting `maxSteps`/the budget, or the model call itself
failing, is fatal — and that error carries the run's `usage`, `messages`, and `calls`, so the dead run is
booked and debuggable (see [Failed runs still cost tokens](#failed-runs-still-cost-tokens)).

`result.messages` is the full transcript including calls and results — pass it back in as `messages` to
continue the conversation on the next request.

**Forcing a tool call.** `toolChoice: "required"` makes a turn call a tool instead of answering —
useful for step 0 ("search before you answer"), pointless as a constant (the model can then *never*
answer, so the run always ends in the `maxSteps` error). Pass a function instead, evaluated per step:

```ts
await ai.run({
  model: "local", prompt: question, tools: [search, answer],
  toolChoice: (step) => (step === 0 ? "required" : "auto"),   // force the first lookup, then let it decide
});
```

**Unlimited steps.** `maxSteps` also accepts `null` — but only together with a `budget` or a `signal`;
`null` alone throws immediately (`"unlimited without a brake is not a limit, it's a prayer"`). A step
count picked because *something* had to be picked is a pseudo-limit; a budget or a signal is an honest
reason to stop.

### Voice

`ai.transcribe()` and `ai.speak()` reach an endpoint's `/audio/transcriptions` and `/audio/speech`. Chain
them with `run()` and you have a voice assistant that never leaves your own server:

```ts
const { text } = await ai.transcribe({ model: "local", audio: { data: await file.arrayBuffer(), mediaType: "audio/webm" }, language: "de" });
const answer = await ai.run({ model: "local", prompt: text, tools, context });
const { audio, mediaType } = await ai.speak({ model: "local", input: answer.text, voice: "de-female" });
```

`audio.data` takes a `Uint8Array`, `ArrayBuffer`, or a browser `Blob`/`File`. A provider without these
routes throws `CoaxUnsupportedError` naming the missing capability — not a mystery 404.

### Agent loops

`ai.loop()` runs the multi-turn conversation for you — you just handle each typed step. It appends the
model's step and your reply automatically, guards against a stuck model (doom guard), and can enforce a
token budget.

```ts
import { createBudget } from "@orgops/coax";

const Step = z.discriminatedUnion("action", [
  z.object({ action: z.literal("search"), query: z.string() }),
  z.object({ action: z.literal("answer"), text: z.string() }),
]);

const answer = await ai.loop<z.infer<typeof Step>, string>({
  model: "smart",
  schema: Step,
  system: "Answer the question, searching when you need to.",
  messages: [{ role: "user", content: task }],
  maxTurns: 8,
  budget: createBudget(100_000),
  onStep: async (step) => {
    if (step.action === "answer") return { done: true, value: step.text };
    return { done: false, reply: await runSearch(step.query) };
  },
});
```

### Evaluation — LLM-as-judge

Schemas catch shape; they can't catch quality. `ai.judge()` scores an output against a rubric — for
intent satisfaction, tone, correctness, or (multimodally) a rendered screenshot.

```ts
const { score, pass, rationale } = await ai.judge({
  model: "smart",
  output: draft,
  criteria: ["Answers the question", "Cites a source", "No PII"],
  scale: [1, 5],        // default
  passScore: 4,         // default: scale midpoint
});
if (!pass) await regenerate(rationale);
```

### Prompt caching

```ts
await ai.object({ model: "fast", schema, system: bigStableSystemPrompt, prompt, cache: true });
// …the same system prompt across a fan-out is billed once at the cache rate.
```

A multi-turn loop re-sends the whole transcript every turn; `cacheConversation: true` marks the
conversation-so-far as reusable, so the *next* call bills prior turns at the cache-read rate:

```ts
await ai.run({ model: "smart", messages, tools, cacheConversation: true });
// …turn N+1 reads turns 1…N from cache instead of paying full price for them again.
```

Both flags are provider-neutral: on Anthropic they place `cache_control` breakpoints (the two combine
to at most two breakpoints per request); on endpoints where prefix caching is automatic (OpenAI)
they're a no-op.

### Streaming

`ai.stream()` takes the same call as `ai.text()` and yields deltas as they arrive — the shape a BFF
pipes straight into an SSE response:

```ts
const { stream, result } = await ai.stream({ model: "fast", prompt: question, signal: ac.signal });
for await (const delta of stream) res.write(`data: ${JSON.stringify(delta)}\n\n`);
const { text, usage } = await result;   // the full text + the bill, once the stream is drained
```

The returned promise resolves when the stream has **started** — so retries and model fallback still
cover a primary that dies before producing anything. After the first delta the stream is committed:
no fallback mid-stream (your user already saw those tokens), a later failure surfaces through the
iteration and `result`. A custom provider without native streaming degrades to one big delta.

### Vision

```ts
await ai.object({
  schema,
  messages: [{ role: "user", content: "Extract the fields.", media: [{ kind: "image", mediaType: "image/png", dataBase64 }] }],
});
```

## In a backend-for-frontend

Everything above composes into the pattern coax is really for: the browser sends a question (or audio),
the server owns the credentials and the tools, and the model only ever sees what a tool chose to return.

```ts
// One Hono route: voice in → answer out, against your own model.
app.post("/v1/assistant", async (c) => {
  const form = await c.req.formData();
  const audio = form.get("audio") as File | null;
  const identity = { Authorization: c.req.header("authorization")! };

  const question = audio
    ? (await ai.transcribe({ model: "local", audio: { data: audio, mediaType: audio.type }, language: "de", headers: identity })).text
    : String(form.get("text"));

  const { text, calls } = await ai.run({
    model: "local",
    system: SYSTEM_PROMPT,
    prompt: question,
    tools: [openInvoices, nextAppointment],
    context: { clientId: c.get("clientId") },   // never sent to the model, only to your handlers
    headers: identity,                           // the end user's identity reaches the gateway
  });

  return c.json({ question, answer: text, used: calls.map((x) => x.name) });
});
```

Three boundaries stay intact: the browser holds no key, the model holds no credentials, and the gateway
still sees who is asking.

## Design

Small and unopinionated. The only vendor-specific surface is the `Provider` interface: `structured` and
`text` are required, `tools` / `transcribe` / `speak` are optional capabilities an endpoint either serves
or honestly doesn't. Everything else — schema handling, aggressive parsing, the repair/retry/fallback
loop, the tool driver, prompt files — is pure and unit-tested against fakes, no network. Zod is a peer
dependency (you write the schemas); the provider SDKs ship inside coax and load lazily. The high-level
`createAI` is the recommended entry point; `createClient` (single provider, no config) is available for
lower-level use.

## License

MIT © orgops
