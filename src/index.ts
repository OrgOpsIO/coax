// Ambient app-wide client: configure() once at startup, then use `ai` anywhere (recommended for apps).
export { configure, ai, isConfigured, reset } from "./runtime";
// Explicit instance (recommended for libraries, tests, multiple configs).
export { createAI } from "./ai";
export type { AI, ObjectCall, TextCall, TextStream, JudgeCall, Judgement, RunCall, TranscribeCall, SpeakCall } from "./ai";
export type { AIConfig, ProviderConfig, ProviderEndpoint, ModelConfig, RetryConfig, CallDefaults, CallMeta } from "./config";
export { parsePrompt, renderTemplate } from "./prompt-file";
export type { ParsedPrompt, PromptMeta } from "./prompt-file";

// Tools — definitions the model may call, and the driver behind ai.run().
export { tool, runTools, CoaxToolError, FINAL_TOOL } from "./tools";
export type { Tool, ToolContext, ToolInvocation, RunOptions, RunResult } from "./tools";

// Low-level primitives (single provider, no config layer).
export { createClient, CoaxAbortError, CoaxSchemaError, CoaxUnsupportedError } from "./client";
export type { Client, ClientOptions, ObjectRequest, ObjectResult, TextResult, TranscribeResult, SpeakResult } from "./client";
export { createRegistry, retrying } from "./registry";
export type { CallSettings, ResolvedModel } from "./registry";
export { withRetry, isTransient } from "./retry";
export { runLoop, CoaxLoopError } from "./loop";
export type { LoopOptions, LoopControl } from "./loop";
export { createBudget } from "./budget";
export type { Budget } from "./budget";

// Providers. Both double as adapters for any endpoint speaking their wire protocol.
export { anthropic } from "./providers/anthropic";
export type { AnthropicOptions } from "./providers/anthropic";
export { openai } from "./providers/openai";
export type { OpenAiOptions } from "./providers/openai";

// Building blocks / types.
export { extractJson } from "./parse";
export { toProviderSchema, formatIssues, safeParse } from "./schema";
export { addUsage, emptyUsage } from "./types";
export type {
  AudioFormat,
  AudioInput,
  Media,
  Message,
  Provider,
  ProviderResponse,
  ReasoningEffort,
  SpeakRequest,
  SpeakResponse,
  StructuredRequest,
  TextRequest,
  ToolCall,
  ToolChoice,
  ToolDefinition,
  ToolResult,
  ToolsRequest,
  ToolsResponse,
  TranscribeRequest,
  TranscribeResponse,
  Usage,
} from "./types";
