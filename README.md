# coax

**A clean, provider-agnostic way to put LLMs into your software — with the good patterns built in.**
Pure TypeScript. No native modules, no codegen, no DSL, no vendor lock-in.

Configure your provider keys once, pick a model by name, hand it a [Zod](https://zod.dev) schema (or a
prompt file), and get back typed, validated data — with retries, model fallback, and self-repair handled
for you.

```bash
npm install @orgops/coax zod
npm install @anthropic-ai/sdk   # and/or: npm install openai
```

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

## Any provider

`anthropic` and `openai` are built in. Plug in anything else (Gemini, a local model, a test mock) with a
factory — it just implements the small `Provider` interface:

```ts
createAI({
  providers: {
    anthropic: process.env.ANTHROPIC_API_KEY!,
    gemini: (model) => myGeminiProvider(model),   // (model: string) => Provider
  },
  models: { flash: "gemini:gemini-2.5-flash" },
});
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
- **Agent loops** — `ai.loop()` drives a typed multi-turn loop with a built-in doom guard + token budget.
- **Token budget** — `createBudget(limit)` caps the total spend of a loop or a fan-out.
- **Usage** — one `onUsage(usage, meta)` hook across every call, plus summed `usage` on each result.
- **Vision** — image/pdf media are first-class.

```ts
const { data, usage, repairs } = await ai.object({ schema, prompt, maxRepairs: 3 });
// repairs === 0 → valid first try
```

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

## Design

Small and unopinionated. The only vendor-specific surface is the `Provider` interface (`structured` +
`text`); everything else — schema handling, aggressive parsing, the repair/retry/fallback loop, prompt
files — is pure and unit-tested. Zod is a peer dependency; the SDKs are optional peers, imported lazily
only when used. The high-level `createAI` is the recommended entry point; `createClient` (single provider,
no config) is available for lower-level use.

## License

MIT © orgops
