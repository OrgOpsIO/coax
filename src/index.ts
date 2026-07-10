// High-level, config-driven interface (recommended entry point).
export { createAI } from "./ai";
export type { AI, ObjectCall, TextCall } from "./ai";
export type { AIConfig, ProviderConfig, ModelConfig, RetryConfig, CallDefaults, CallMeta } from "./config";
export { parsePrompt, renderTemplate } from "./prompt-file";
export type { ParsedPrompt, PromptMeta } from "./prompt-file";

// Low-level primitives (single provider, no config layer).
export { createClient, CoaxSchemaError } from "./client";
export type { Client, ClientOptions, ObjectRequest, ObjectResult, TextResult } from "./client";
export { createRegistry, retrying } from "./registry";
export type { ResolvedModel } from "./registry";
export { withRetry, isTransient } from "./retry";

// Providers.
export { anthropic } from "./providers/anthropic";
export type { AnthropicOptions } from "./providers/anthropic";
export { openai } from "./providers/openai";
export type { OpenAiOptions } from "./providers/openai";

// Building blocks / types.
export { extractJson } from "./parse";
export { addUsage, emptyUsage } from "./types";
export type { Media, Message, Provider, ProviderResponse, StructuredRequest, TextRequest, Usage } from "./types";
