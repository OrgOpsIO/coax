import type AnthropicSdk from "@anthropic-ai/sdk";
import type {
  Message,
  Provider,
  ProviderResponse,
  StructuredRequest,
  TextRequest,
  ToolCall,
  ToolsRequest,
  ToolsResponse,
  Usage,
} from "../types";

export interface AnthropicOptions {
  model: string;
  apiKey?: string;
  /** Inject an existing SDK client (recommended in apps that already have one). If omitted, coax lazily
   *  constructs one from `apiKey` — the SDK ships inside coax and is imported only then. */
  client?: AnthropicSdk;
  maxTokens?: number;
  /** Any Anthropic-compatible endpoint — e.g. a gateway exposing `/anthropic`. */
  baseURL?: string;
  /** Headers sent with every request (per-call `headers` are merged over these). */
  headers?: Record<string, string>;
}

type AnthropicResponse = { content: unknown[]; usage?: Record<string, number> };
type RequestOptions = { headers?: Record<string, string> };
type AnyClient = {
  messages: {
    create(body: Record<string, unknown>, options?: RequestOptions): Promise<AnthropicResponse>;
    stream(body: Record<string, unknown>, options?: RequestOptions): { finalMessage(): Promise<AnthropicResponse> };
  };
};

// The SDK REFUSES non-streaming requests whose max_tokens imply a >10-minute response ("Streaming is
// required for operations that may take longer than 10 minutes"), which large structured outputs hit.
// Stream under the hood and return the accumulated final message — identical result, no ceiling.
async function createMessage(c: AnyClient, body: Record<string, unknown>, options?: RequestOptions): Promise<AnthropicResponse> {
  return c.messages.stream(body, options).finalMessage();
}

function mapUsage(u: Record<string, number> | undefined): Usage {
  return {
    inputTokens: u?.input_tokens ?? 0,
    outputTokens: u?.output_tokens ?? 0,
    cacheReadTokens: u?.cache_read_input_tokens ?? 0,
    cacheWriteTokens: u?.cache_creation_input_tokens ?? 0,
  };
}

/** Tool results are serialized for the wire; a string stays a string so the model reads it verbatim. */
const toolOutput = (output: unknown): string => (typeof output === "string" ? output : JSON.stringify(output ?? null));

function toContent(m: Message): unknown {
  if (m.role === "assistant" && m.toolCalls?.length) {
    const blocks: unknown[] = m.content ? [{ type: "text", text: m.content }] : [];
    for (const c of m.toolCalls) blocks.push({ type: "tool_use", id: c.id, name: c.name, input: c.input ?? {} });
    return blocks;
  }
  if (m.toolResults?.length) {
    // Anthropic requires tool_result blocks at the START of the user turn that answers a tool_use.
    const blocks: unknown[] = m.toolResults.map((r) => ({
      type: "tool_result",
      tool_use_id: r.id,
      content: toolOutput(r.output),
      ...(r.isError ? { is_error: true } : {}),
    }));
    if (m.content) blocks.push({ type: "text", text: m.content });
    return blocks;
  }
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

// A cached system prompt is sent as a content block carrying cache_control; Anthropic then reuses the
// prefix across calls that share it (a big saving on a fan-out with a stable system prompt).
function systemParam(system: string | undefined, cache: boolean | undefined): Record<string, unknown> {
  if (!system) return {};
  if (!cache) return { system };
  return { system: [{ type: "text", text: system, cache_control: { type: "ephemeral" } }] };
}

const isBlock = (b: unknown, type: string): boolean => (b as { type?: string }).type === type;

export function anthropic(opts: AnthropicOptions): Provider {
  let client: AnyClient | undefined = opts.client as AnyClient | undefined;

  async function getClient(): Promise<AnyClient> {
    if (client) return client;
    const mod = await import("@anthropic-ai/sdk");
    const Ctor = (mod as unknown as { default: new (o: { apiKey?: string; baseURL?: string }) => AnyClient }).default;
    client = new Ctor({ apiKey: opts.apiKey, baseURL: opts.baseURL });
    return client;
  }

  const requestOptions = (headers: Record<string, string> | undefined): RequestOptions | undefined => {
    const merged = { ...opts.headers, ...headers };
    return Object.keys(merged).length ? { headers: merged } : undefined;
  };

  const textOf = (content: unknown[]): string =>
    content
      .filter((b): b is { type: "text"; text: string } => isBlock(b, "text"))
      .map((b) => b.text)
      .join("");

  return {
    name: "anthropic",
    model: opts.model,

    async structured(req: StructuredRequest): Promise<ProviderResponse> {
      const c = await getClient();
      const resp = await createMessage(
        c,
        {
          model: opts.model,
          max_tokens: req.maxTokens ?? opts.maxTokens ?? 8192,
          ...systemParam(req.system, req.cacheSystem),
          tools: [{ name: req.schemaName, description: `Return a ${req.schemaName} object.`, input_schema: req.jsonSchema }],
          tool_choice: { type: "tool", name: req.schemaName },
          messages: toMessages(req.messages),
        },
        requestOptions(req.headers),
      );
      const tool = resp.content.find((b): b is { type: "tool_use"; input: unknown } => isBlock(b, "tool_use"));
      const raw = tool?.input;
      return { raw, text: raw === undefined ? "" : JSON.stringify(raw), usage: mapUsage(resp.usage), model: opts.model };
    },

    async text(req: TextRequest): Promise<ProviderResponse> {
      const c = await getClient();
      const resp = await createMessage(
        c,
        {
          model: opts.model,
          max_tokens: req.maxTokens ?? opts.maxTokens ?? 8192,
          ...systemParam(req.system, req.cacheSystem),
          messages: toMessages(req.messages),
        },
        requestOptions(req.headers),
      );
      const text = textOf(resp.content);
      return { raw: text, text, usage: mapUsage(resp.usage), model: opts.model };
    },

    async tools(req: ToolsRequest): Promise<ToolsResponse> {
      const c = await getClient();
      const resp = await createMessage(
        c,
        {
          model: opts.model,
          max_tokens: req.maxTokens ?? opts.maxTokens ?? 8192,
          ...systemParam(req.system, req.cacheSystem),
          tools: req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.jsonSchema })),
          messages: toMessages(req.messages),
        },
        requestOptions(req.headers),
      );
      const calls: ToolCall[] = resp.content
        .filter((b): b is { type: "tool_use"; id: string; name: string; input: unknown } => isBlock(b, "tool_use"))
        .map((b) => ({ id: b.id, name: b.name, input: b.input }));
      return { text: textOf(resp.content), calls, usage: mapUsage(resp.usage), model: opts.model };
    },
  };
}
