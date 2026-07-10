# coax

**Typed, provider-agnostic, self-repairing structured output from LLMs.**
Pure TypeScript — no native modules, no codegen, no DSL, no vendor lock-in.

You define a [Zod](https://zod.dev) schema, coax coaxes the model into it: it uses the provider's native
constrained-output mode, aggressively parses whatever comes back (markdown fences, malformed JSON, …), and
if it still doesn't fit, it reprompts the model with the exact validation errors until it does.

```bash
npm install @orgops/coax zod
# plus the SDK for the provider you use:
npm install @anthropic-ai/sdk   # and/or: npm install openai
```

## Quick start

```ts
import { createClient, anthropic } from "@orgops/coax";
import { z } from "zod";

const client = createClient({
  provider: anthropic({ apiKey: process.env.ANTHROPIC_API_KEY!, model: "claude-sonnet-4-6" }),
});

const Field = z.object({
  label: z.string(),
  helpText: z.string().min(1), // ← a hard contract: the model cannot omit this
});

const { data } = await client.object({
  schema: Field,
  system: "You design insurance calculator fields.",
  prompt: "A field for the customer's date of birth.",
});

data.helpText; // string — typed and guaranteed non-empty
```

## Provider-agnostic — the provider is data

Swap one line. Nothing else changes.

```ts
import { openai } from "@orgops/coax";

const client = createClient({
  provider: openai({ apiKey: process.env.OPENAI_API_KEY!, model: "gpt-5" }),
});
```

Bring your own SDK instance (recommended when your app already has one):

```ts
import Anthropic from "@anthropic-ai/sdk";
anthropic({ client: new Anthropic(), model: "claude-sonnet-4-6" });
```

## What makes it robust

- **Native constrained output** — Anthropic tool-use / OpenAI function-calling, so the model aims at the schema.
- **Aggressive parsing** — strips ```` ```json ```` fences and repairs malformed JSON (unquoted keys, trailing
  commas, single quotes, truncation) via [`jsonrepair`](https://github.com/josdejong/jsonrepair) before validation.
- **Validate → repair** — on a Zod failure, coax reprompts with the concrete errors (up to `maxRepairs`, default 2).
  Modern models correct almost always on the first repair.

```ts
await client.object({ schema, prompt, maxRepairs: 3 });
// → { data, usage, model, repairs }   (repairs = 0 means valid first try)
```

## Agent loops & unions

A discriminated union is a typed "which tool" per turn — drive the loop yourself, coax types every step.

```ts
const Step = z.discriminatedUnion("action", [
  z.object({ action: z.literal("fetch"), sections: z.array(z.string()) }),
  z.object({ action: z.literal("answer"), text: z.string() }),
]);

const messages = [{ role: "user" as const, content: "..." }];
while (true) {
  const { data } = await client.object({ schema: Step, messages });
  if (data.action === "answer") break;
  // handle fetch, append the result to messages, loop…
}
```

## Vision

```ts
await client.object({
  schema,
  messages: [{
    role: "user",
    content: "Extract the fields from this form.",
    media: [{ kind: "image", mediaType: "image/png", dataBase64: "…" }], // or kind: "pdf" (Anthropic)
  }],
});
```

## Free-form text

```ts
const { text } = await client.text({ system, prompt: "Write the offer HTML." });
```

## Observability

```ts
createClient({
  provider,
  onUsage: (usage, model) => record(usage, model), // fired per model call, incl. repair rounds
});
// object() also returns summed `usage` across all rounds.
```

## Design

coax is intentionally small. The only vendor-specific surface is the `Provider` interface
(`structured` + `text`) — implement it to add a provider. Everything else (schema, parsing, repair loop)
is pure and unit-tested. Zod is a peer dependency; the SDKs are optional peers, imported lazily only when used.

## License

MIT © orgops
