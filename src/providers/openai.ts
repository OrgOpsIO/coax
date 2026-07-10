import type OpenAiSdk from "openai";
import type { Message, Provider, ProviderResponse, StructuredRequest, TextRequest, Usage } from "../types";

export interface OpenAiOptions {
  model: string;
  apiKey?: string;
  /** Inject an existing SDK client. Otherwise coax lazily constructs one from `apiKey` (the SDK is an
   *  optional peer dependency, imported only then). */
  client?: OpenAiSdk;
  maxTokens?: number;
  baseURL?: string;
}

type AnyClient = {
  chat: { completions: { create(body: Record<string, unknown>): Promise<{ choices: { message: { content?: string | null; tool_calls?: { function: { arguments: string } }[] } }[]; usage?: Record<string, unknown> }> } };
};

function mapUsage(u: Record<string, unknown> | undefined): Usage {
  const cached = (u?.prompt_tokens_details as { cached_tokens?: number } | undefined)?.cached_tokens ?? 0;
  return {
    inputTokens: (u?.prompt_tokens as number) ?? 0,
    outputTokens: (u?.completion_tokens as number) ?? 0,
    cacheReadTokens: cached,
    cacheWriteTokens: 0,
  };
}

function toContent(m: Message): unknown {
  if (!m.media?.length) return m.content;
  const parts: unknown[] = [{ type: "text", text: m.content }];
  for (const media of m.media) {
    // OpenAI chat vision takes images as data URIs. (PDF input needs the Files/Responses API — out of
    // scope for the chat provider; Anthropic handles PDF natively.)
    if (media.kind === "image") {
      parts.push({ type: "image_url", image_url: { url: `data:${media.mediaType};base64,${media.dataBase64}` } });
    }
  }
  return parts;
}

function toMessages(system: string | undefined, messages: Message[]): unknown[] {
  const out: unknown[] = [];
  if (system) out.push({ role: "system", content: system });
  for (const m of messages) out.push({ role: m.role, content: toContent(m) });
  return out;
}

export function openai(opts: OpenAiOptions): Provider {
  let client: AnyClient | undefined = opts.client as AnyClient | undefined;

  async function getClient(): Promise<AnyClient> {
    if (client) return client;
    const mod = await import("openai");
    const Ctor = (mod as unknown as { default: new (o: { apiKey?: string; baseURL?: string }) => AnyClient }).default;
    client = new Ctor({ apiKey: opts.apiKey, baseURL: opts.baseURL });
    return client;
  }

  return {
    name: "openai",
    model: opts.model,

    async structured(req: StructuredRequest): Promise<ProviderResponse> {
      const c = await getClient();
      const resp = await c.chat.completions.create({
        model: opts.model,
        max_tokens: req.maxTokens ?? opts.maxTokens ?? 8192,
        messages: toMessages(req.system, req.messages),
        tools: [{ type: "function", function: { name: req.schemaName, description: `Return a ${req.schemaName} object.`, parameters: req.jsonSchema } }],
        tool_choice: { type: "function", function: { name: req.schemaName } },
      });
      const args = resp.choices[0]?.message?.tool_calls?.[0]?.function?.arguments ?? "";
      return { raw: args, text: args, usage: mapUsage(resp.usage), model: opts.model };
    },

    async text(req: TextRequest): Promise<ProviderResponse> {
      const c = await getClient();
      const resp = await c.chat.completions.create({
        model: opts.model,
        max_tokens: req.maxTokens ?? opts.maxTokens ?? 8192,
        messages: toMessages(req.system, req.messages),
      });
      const text = resp.choices[0]?.message?.content ?? "";
      return { raw: text, text, usage: mapUsage(resp.usage), model: opts.model };
    },
  };
}
