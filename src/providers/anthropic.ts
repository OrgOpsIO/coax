import type AnthropicSdk from "@anthropic-ai/sdk";
import type { Message, Provider, ProviderResponse, StructuredRequest, TextRequest, Usage } from "../types";

export interface AnthropicOptions {
  model: string;
  apiKey?: string;
  /** Inject an existing SDK client (recommended in apps that already have one). If omitted, coax lazily
   *  constructs one from `apiKey` — the SDK is an optional peer dependency, imported only then. */
  client?: AnthropicSdk;
  maxTokens?: number;
  baseURL?: string;
}

type AnyClient = {
  messages: { create(body: Record<string, unknown>): Promise<{ content: unknown[]; usage?: Record<string, number> }> };
};

function mapUsage(u: Record<string, number> | undefined): Usage {
  return {
    inputTokens: u?.input_tokens ?? 0,
    outputTokens: u?.output_tokens ?? 0,
    cacheReadTokens: u?.cache_read_input_tokens ?? 0,
    cacheWriteTokens: u?.cache_creation_input_tokens ?? 0,
  };
}

function toContent(m: Message): unknown {
  if (!m.media?.length) return m.content;
  const blocks: unknown[] = [{ type: "text", text: m.content }];
  for (const media of m.media) {
    blocks.push(
      media.kind === "pdf"
        ? { type: "document", source: { type: "base64", media_type: "application/pdf", data: media.dataBase64 } }
        : { type: "image", source: { type: "base64", media_type: media.mediaType, data: media.dataBase64 } },
    );
  }
  return blocks;
}

const toMessages = (messages: Message[]) => messages.map((m) => ({ role: m.role, content: toContent(m) }));

export function anthropic(opts: AnthropicOptions): Provider {
  let client: AnyClient | undefined = opts.client as AnyClient | undefined;

  async function getClient(): Promise<AnyClient> {
    if (client) return client;
    const mod = await import("@anthropic-ai/sdk");
    const Ctor = (mod as unknown as { default: new (o: { apiKey?: string; baseURL?: string }) => AnyClient }).default;
    client = new Ctor({ apiKey: opts.apiKey, baseURL: opts.baseURL });
    return client;
  }

  return {
    name: "anthropic",
    model: opts.model,

    async structured(req: StructuredRequest): Promise<ProviderResponse> {
      const c = await getClient();
      const resp = await c.messages.create({
        model: opts.model,
        max_tokens: req.maxTokens ?? opts.maxTokens ?? 8192,
        ...(req.system ? { system: req.system } : {}),
        tools: [{ name: req.schemaName, description: `Return a ${req.schemaName} object.`, input_schema: req.jsonSchema }],
        tool_choice: { type: "tool", name: req.schemaName },
        messages: toMessages(req.messages),
      });
      const tool = resp.content.find((b): b is { type: "tool_use"; input: unknown } => (b as { type?: string }).type === "tool_use");
      const raw = tool?.input;
      return { raw, text: raw === undefined ? "" : JSON.stringify(raw), usage: mapUsage(resp.usage), model: opts.model };
    },

    async text(req: TextRequest): Promise<ProviderResponse> {
      const c = await getClient();
      const resp = await c.messages.create({
        model: opts.model,
        max_tokens: req.maxTokens ?? opts.maxTokens ?? 8192,
        ...(req.system ? { system: req.system } : {}),
        messages: toMessages(req.messages),
      });
      const text = resp.content
        .filter((b): b is { type: "text"; text: string } => (b as { type?: string }).type === "text")
        .map((b) => b.text)
        .join("");
      return { raw: text, text, usage: mapUsage(resp.usage), model: opts.model };
    },
  };
}
