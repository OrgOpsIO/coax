import type AnthropicSdk from "@anthropic-ai/sdk";
import type {
  Message,
  Provider,
  ProviderResponse,
  ReasoningEffort,
  StructuredRequest,
  TextRequest,
  ToolCall,
  ToolChoice,
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
  /** Merged into every request body from this endpoint, under the per-call `extraBody`. See `ProviderEndpoint.extraBody`. */
  extraBody?: Record<string, unknown>;
}

type AnthropicResponse = { content: unknown[]; usage?: Record<string, number> };
type RequestOptions = { headers?: Record<string, string>; signal?: AbortSignal };
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
    // Thinking blocks (carried opaquely as providerData) must lead the assistant turn — Anthropic
    // rejects a tool_use turn whose thinking arrives after other blocks, and verifies the signature.
    const blocks: unknown[] = Array.isArray(m.providerData) ? [...m.providerData] : [];
    if (m.content) blocks.push({ type: "text", text: m.content });
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

// A conversation cache hint becomes a breakpoint on the LAST block of the LAST message: everything
// before it (system → all prior turns) is stored as one reusable prefix, so a loop's next call — which
// re-sends the grown transcript with its own breakpoint further down — reads the earlier turns from
// cache. Marking only the newest message keeps exactly one breakpoint per request by construction.
function markLastBlock(content: unknown): unknown {
  const blocks: unknown[] = typeof content === "string" ? [{ type: "text", text: content }] : [...(content as unknown[])];
  const i = blocks.length - 1;
  if (i >= 0) blocks[i] = { ...(blocks[i] as object), cache_control: { type: "ephemeral" } };
  return blocks;
}

const toMessages = (messages: Message[], cacheConversation?: boolean) =>
  messages.map((m, i) => {
    const content = toContent(m);
    const mark = cacheConversation && i === messages.length - 1 && content !== "";
    return { role: m.role, content: mark ? markLastBlock(content) : content };
  });

// A cached system prompt is sent as a content block carrying cache_control; Anthropic then reuses the
// prefix across calls that share it (a big saving on a fan-out with a stable system prompt).
function systemParam(system: string | undefined, cache: boolean | undefined): Record<string, unknown> {
  if (!system) return {};
  if (!cache) return { system };
  return { system: [{ type: "text", text: system, cache_control: { type: "ephemeral" } }] };
}

const isBlock = (b: unknown, type: string): boolean => (b as { type?: string }).type === type;

/**
 * Flat merge, call wins over endpoint wins over coax's own body — same rule as `headers`. This CAN
 * override coax's own fields (`max_tokens`, `tools`, …); that is the deliberate escape hatch described
 * on `BaseRequest.extraBody`, not a bug to guard against here.
 */
function withExtraBody(body: Record<string, unknown>, endpointExtraBody: Record<string, unknown> | undefined, callExtraBody: Record<string, unknown> | undefined): Record<string, unknown> {
  return { ...body, ...endpointExtraBody, ...callExtraBody };
}

// Anthropic's own budget, before the max_tokens cap below. Chosen to roughly track OpenAI's cheap/expensive
// split at "none"/"low" vs "high" without pretending the two vendors' reasoning tokens are equivalent.
const THINKING_BUDGETS: Record<"low" | "medium" | "high", number> = { low: 2048, medium: 8192, high: 16384 };

// `"none"` sends no `thinking` field at all — Anthropic just answers directly, same as an endpoint that
// never heard of reasoning effort. Anthropic REQUIRES budget_tokens < max_tokens (the model needs room
// left to answer after it stops thinking) AND budget_tokens >= 1024, so the budget is capped relative
// to whatever max_tokens this call actually resolved to — and below 2048 the two constraints can't
// both hold, so that's a clear error here rather than a cryptic 400 from the endpoint.
function thinkingParam(effort: ReasoningEffort | undefined, maxTokens: number): Record<string, unknown> {
  if (!effort || effort === "none") return {};
  if (maxTokens < 2048) {
    throw new Error(
      `coax: reasoningEffort "${effort}" needs maxTokens >= 2048 on Anthropic (got ${maxTokens}) — ` +
        "the thinking budget must be at least 1024 and still leave room to answer. Raise maxTokens or drop the effort.",
    );
  }
  const budget = Math.min(THINKING_BUDGETS[effort], maxTokens - 1024);
  return { thinking: { type: "enabled", budget_tokens: budget } };
}

// Anthropic's tool_choice spells "required" as "any"; "auto"/"none" already match its own vocabulary.
function toolChoiceParam(choice: ToolChoice | undefined): Record<string, unknown> {
  if (!choice) return {};
  return { tool_choice: { type: choice === "required" ? "any" : choice } };
}

export function anthropic(opts: AnthropicOptions): Provider {
  let client: AnyClient | undefined = opts.client as AnyClient | undefined;

  async function getClient(): Promise<AnyClient> {
    if (client) return client;
    const mod = await import("@anthropic-ai/sdk");
    const Ctor = (mod as unknown as { default: new (o: { apiKey?: string; baseURL?: string }) => AnyClient }).default;
    client = new Ctor({ apiKey: opts.apiKey, baseURL: opts.baseURL });
    return client;
  }

  const requestOptions = (headers: Record<string, string> | undefined, signal: AbortSignal | undefined): RequestOptions | undefined => {
    const merged = { ...opts.headers, ...headers };
    const out: RequestOptions = {
      ...(Object.keys(merged).length ? { headers: merged } : {}),
      // The SDK aborts the underlying HTTP request — the stream ends, the endpoint sees the disconnect.
      ...(signal ? { signal } : {}),
    };
    return Object.keys(out).length ? out : undefined;
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
      const maxTokens = req.maxTokens ?? opts.maxTokens ?? 8192;
      const resp = await createMessage(
        c,
        withExtraBody(
          {
            model: opts.model,
            max_tokens: maxTokens,
            ...systemParam(req.system, req.cacheSystem),
            tools: [{ name: req.schemaName, description: `Return a ${req.schemaName} object.`, input_schema: req.jsonSchema }],
            tool_choice: { type: "tool", name: req.schemaName },
            messages: toMessages(req.messages, req.cacheConversation),
            ...thinkingParam(req.reasoningEffort, maxTokens),
          },
          opts.extraBody,
          req.extraBody,
        ),
        requestOptions(req.headers, req.signal),
      );
      const tool = resp.content.find((b): b is { type: "tool_use"; input: unknown } => isBlock(b, "tool_use"));
      const raw = tool?.input;
      return { raw, text: raw === undefined ? "" : JSON.stringify(raw), usage: mapUsage(resp.usage), model: opts.model };
    },

    async text(req: TextRequest): Promise<ProviderResponse> {
      const c = await getClient();
      const maxTokens = req.maxTokens ?? opts.maxTokens ?? 8192;
      const resp = await createMessage(
        c,
        withExtraBody(
          {
            model: opts.model,
            max_tokens: maxTokens,
            ...systemParam(req.system, req.cacheSystem),
            messages: toMessages(req.messages, req.cacheConversation),
            ...thinkingParam(req.reasoningEffort, maxTokens),
          },
          opts.extraBody,
          req.extraBody,
        ),
        requestOptions(req.headers, req.signal),
      );
      const text = textOf(resp.content);
      return { raw: text, text, usage: mapUsage(resp.usage), model: opts.model };
    },

    async *structuredStream(req: StructuredRequest): AsyncGenerator<string, ProviderResponse, void> {
      const c = await getClient();
      const maxTokens = req.maxTokens ?? opts.maxTokens ?? 8192;
      const s = c.messages.stream(
        withExtraBody(
          {
            model: opts.model,
            max_tokens: maxTokens,
            ...systemParam(req.system, req.cacheSystem),
            tools: [{ name: req.schemaName, description: `Return a ${req.schemaName} object.`, input_schema: req.jsonSchema }],
            tool_choice: { type: "tool", name: req.schemaName },
            messages: toMessages(req.messages, req.cacheConversation),
            ...thinkingParam(req.reasoningEffort, maxTokens),
          },
          opts.extraBody,
          req.extraBody,
        ),
        requestOptions(req.headers, req.signal),
      );
      // Forced tool use arrives as input_json_delta fragments — the raw JSON text of the output.
      for await (const event of s as unknown as AsyncIterable<{ type: string; delta?: { type?: string; partial_json?: string } }>) {
        if (event.type === "content_block_delta" && event.delta?.type === "input_json_delta" && event.delta.partial_json) yield event.delta.partial_json;
      }
      const final = await s.finalMessage();
      const tool = final.content.find((b): b is { type: "tool_use"; input: unknown } => isBlock(b, "tool_use"));
      const raw = tool?.input;
      return { raw, text: raw === undefined ? "" : JSON.stringify(raw), usage: mapUsage(final.usage), model: opts.model };
    },

    async *textStream(req: TextRequest): AsyncGenerator<string, ProviderResponse, void> {
      const c = await getClient();
      const maxTokens = req.maxTokens ?? opts.maxTokens ?? 8192;
      const s = c.messages.stream(
        withExtraBody(
          {
            model: opts.model,
            max_tokens: maxTokens,
            ...systemParam(req.system, req.cacheSystem),
            messages: toMessages(req.messages, req.cacheConversation),
            ...thinkingParam(req.reasoningEffort, maxTokens),
          },
          opts.extraBody,
          req.extraBody,
        ),
        requestOptions(req.headers, req.signal),
      );
      // The SDK's MessageStream is itself async-iterable over raw events; text deltas are the ones we
      // surface (thinking deltas stay internal — they are process, not answer).
      for await (const event of s as unknown as AsyncIterable<{ type: string; delta?: { type?: string; text?: string } }>) {
        if (event.type === "content_block_delta" && event.delta?.type === "text_delta" && event.delta.text) yield event.delta.text;
      }
      const final = await s.finalMessage();
      const text = textOf(final.content);
      return { raw: text, text, usage: mapUsage(final.usage), model: opts.model };
    },

    async tools(req: ToolsRequest): Promise<ToolsResponse> {
      const thinking = Boolean(req.reasoningEffort && req.reasoningEffort !== "none");
      // Anthropic only allows tool_choice "auto"/"none" while thinking — forcing a tool ("required" →
      // "any") is a 400 from the endpoint; make it a clear error here instead.
      if (thinking && req.toolChoice === "required") {
        throw new Error(
          'coax: toolChoice "required" cannot be combined with a reasoningEffort on Anthropic — ' +
            'thinking only permits tool_choice "auto"/"none". Drop one of the two for this call (a step-indexed ' +
            'toolChoice can force step 0 without thinking and re-enable it later).',
        );
      }
      const c = await getClient();
      const maxTokens = req.maxTokens ?? opts.maxTokens ?? 8192;
      const resp = await createMessage(
        c,
        withExtraBody(
          {
            model: opts.model,
            max_tokens: maxTokens,
            ...systemParam(req.system, req.cacheSystem),
            tools: req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.jsonSchema })),
            messages: toMessages(req.messages, req.cacheConversation),
            ...toolChoiceParam(req.toolChoice),
            ...thinkingParam(req.reasoningEffort, maxTokens),
          },
          opts.extraBody,
          req.extraBody,
        ),
        requestOptions(req.headers, req.signal),
      );
      const calls: ToolCall[] = resp.content
        .filter((b): b is { type: "tool_use"; id: string; name: string; input: unknown } => isBlock(b, "tool_use"))
        .map((b) => ({ id: b.id, name: b.name, input: b.input }));
      // Thinking blocks (incl. redacted ones) must be replayed verbatim — signature and all — at the
      // START of this turn's assistant message on the next request. Hand them back opaquely; runTools
      // stores them on the message, toContent puts them back in front.
      const carried = resp.content.filter((b) => isBlock(b, "thinking") || isBlock(b, "redacted_thinking"));
      return {
        text: textOf(resp.content),
        calls,
        usage: mapUsage(resp.usage),
        model: opts.model,
        ...(carried.length ? { providerData: carried } : {}),
      };
    },
  };
}
