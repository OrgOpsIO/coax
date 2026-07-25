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

## What's built in (so you don't hand-roll it every time)

- **Typed contracts** — your Zod schema *is* the spec. `z.string().min(1)` means the model can't return empty.
- **Aggressive parsing** — strips ```` ```json ```` fences and repairs malformed JSON (unquoted keys, trailing
  commas, truncation) before validating.
- **Validate → repair** — on a schema miss, coax reprompts with the exact errors (up to `maxRepairs`).
- **Retries** — transient errors (429/5xx/network) retried with exponential backoff.
- **Model fallback** — a model alias can name a `fallback`; used automatically when the primary fails.
- **Prompt caching** — `cache: true` caches the system prompt at the provider (Anthropic `cache_control`;
  a no-op where caching is automatic). Big savings across a fan-out that shares a stable system prompt.
- **Tools** — `ai.run()` hands the model typed tools and runs the whole call/validate/reply loop.
- **Agent loops** — `ai.loop()` drives a typed multi-turn loop with a built-in doom guard + token budget.
- **Token budget** — `createBudget(limit)` caps the total spend of a loop, a run, or a fan-out.
- **Usage** — one `onUsage(usage, meta)` hook across every call, plus summed `usage` on each result.
- **Vision** — image/pdf media are first-class.
- **Voice** — `ai.transcribe()` / `ai.speak()` where the endpoint serves them, with a precise error where it doesn't.

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

Failures are **fed back, not thrown**: an invented tool name, arguments that miss the schema, or a handler
that throws all return to the model as an error result so it can correct itself — the same
validate-then-repair idea as structured output. Only exhausting `maxSteps` or the budget is fatal.

`result.messages` is the full transcript including calls and results — pass it back in as `messages` to
continue the conversation on the next request.

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
