import { describe, expect, it } from "vitest";
import { z } from "zod";
import { createAI } from "../src/ai";
import { anthropic } from "../src/providers/anthropic";
import { google } from "../src/providers/google";
import { openai } from "../src/providers/openai";
import { tool } from "../src/tools";
import { CoaxRefusalError, type Provider, type Usage } from "../src/types";

// Same contract on every vendor: a caller's code does not change when the model reference changes. The
// same calls run against a fake anthropic, openai and google client (each answering in its own wire
// shape) and must produce results of identical shape. (T18 — the measurer extends this.)

type Script = { text: string; object: string; call: { name: string; input: Record<string, unknown> }; answer: string };
const SCRIPT: Script = { text: "hi", object: '{"answer":"42"}', call: { name: "lookup", input: { q: "x" } }, answer: "done" };

function anthropicFake(replies: Record<string, unknown>[]) {
  let i = 0;
  const next = () => replies[Math.min(i++, replies.length - 1)]!;
  const usage = { input_tokens: 3, output_tokens: 1 };
  const client = {
    messages: {
      create: async () => ({ usage, ...next() }),
      stream: () => {
        const final = { usage, ...next() };
        return {
          finalMessage: async () => final,
          async *[Symbol.asyncIterator]() {
            for (const b of final.content as { type: string; text?: string }[]) {
              if (b.type === "text") yield { type: "content_block_delta", delta: { type: "text_delta", text: b.text } };
            }
          },
        };
      },
    },
  };
  return client as never;
}

function openaiFake(replies: unknown[]) {
  let i = 0;
  const client = {
    chat: {
      completions: {
        create: async (body: Record<string, unknown>) => {
          const reply = replies[Math.min(i++, replies.length - 1)];
          if (!body.stream) return reply;
          return (async function* () {
            yield* reply as unknown[];
          })();
        },
      },
    },
  };
  return client as never;
}

function googleFake(replies: unknown[]) {
  let i = 0;
  const next = () => JSON.parse(JSON.stringify(replies[Math.min(i++, replies.length - 1)]));
  const client = {
    models: {
      generateContent: async () => next(),
      generateContentStream: async () =>
        (async function* () {
          yield* next() as unknown[];
        })(),
    },
  };
  return client as never;
}

const googleUsage = { promptTokenCount: 3, candidatesTokenCount: 1, totalTokenCount: 4 };
const googleText = (text: string) => ({ candidates: [{ content: { role: "model", parts: [{ text }] }, finishReason: "STOP" }], usageMetadata: googleUsage });
const openaiUsage = { prompt_tokens: 3, completion_tokens: 1 };

/** One fake provider per vendor and scenario, answering in that vendor's wire shape. */
const VENDORS: Record<string, (scenario: "text" | "object" | "stream" | "run" | "refusal") => Provider> = {
  anthropic: (scenario) =>
    anthropic({
      model: "m",
      client: anthropicFake(
        {
          text: [{ content: [{ type: "text", text: SCRIPT.text }] }],
          stream: [{ content: [{ type: "text", text: SCRIPT.text }] }],
          object: [{ content: [{ type: "tool_use", id: "t", name: "output", input: JSON.parse(SCRIPT.object) }] }],
          run: [
            { content: [{ type: "tool_use", id: "tu_1", name: SCRIPT.call.name, input: SCRIPT.call.input }] },
            { content: [{ type: "text", text: SCRIPT.answer }] },
          ],
          refusal: [{ content: [], usage: { input_tokens: 9, output_tokens: 0 }, stop_reason: "refusal", stop_details: null }],
        }[scenario],
      ),
    }),
  openai: (scenario) =>
    openai({
      model: "m",
      client: openaiFake(
        {
          text: [{ choices: [{ message: { content: SCRIPT.text } }], usage: openaiUsage }],
          stream: [[{ choices: [{ delta: { content: SCRIPT.text } }] }, { choices: [], usage: openaiUsage }]],
          object: [{ choices: [{ message: { tool_calls: [{ id: "c", function: { name: "output", arguments: SCRIPT.object } }] } }], usage: openaiUsage }],
          run: [
            { choices: [{ message: { tool_calls: [{ id: "c1", function: { name: SCRIPT.call.name, arguments: JSON.stringify(SCRIPT.call.input) } }] } }], usage: openaiUsage },
            { choices: [{ message: { content: SCRIPT.answer } }], usage: openaiUsage },
          ],
          refusal: [],
        }[scenario],
      ),
    }),
  google: (scenario) =>
    google({
      model: "m",
      project: "p",
      client: googleFake(
        {
          text: [googleText(SCRIPT.text)],
          stream: [[googleText(SCRIPT.text)]],
          object: [googleText(SCRIPT.object)],
          run: [
            {
              candidates: [{ content: { role: "model", parts: [{ functionCall: { id: "g1", name: SCRIPT.call.name, args: SCRIPT.call.input }, thoughtSignature: "c2ln" }] }, finishReason: "STOP" }],
              usageMetadata: googleUsage,
            },
            googleText(SCRIPT.answer),
          ],
          refusal: [{ promptFeedback: { blockReason: "SAFETY" }, usageMetadata: { promptTokenCount: 9, totalTokenCount: 9 } }],
        }[scenario],
      ),
    }),
};

const aiFor = (vendor: string, scenario: Parameters<(typeof VENDORS)[string]>[0], onUsage?: (u: Usage) => void) =>
  createAI({ providers: { [vendor]: () => VENDORS[vendor]!(scenario) }, onUsage: onUsage && ((u: Usage) => onUsage(u)) });

const keys = (o: object) => Object.keys(o).sort();
const USAGE_KEYS = ["cacheReadTokens", "cacheWriteTokens", "inputTokens", "outputTokens"];

const lookup = tool({ name: SCRIPT.call.name, description: "Look something up.", input: z.object({ q: z.string() }), run: () => 42 });

describe("contract parity: anthropic, openai, google (T18)", () => {
  for (const vendor of Object.keys(VENDORS)) {
    const model = `${vendor}:m`;

    it(`${vendor}: ai.text`, async () => {
      const res = await aiFor(vendor, "text").text({ model, prompt: "?" });
      expect(keys(res)).toEqual(["model", "text", "usage"]);
      expect(keys(res.usage)).toEqual(USAGE_KEYS);
      expect(res.text).toBe(SCRIPT.text);
    });

    it(`${vendor}: ai.object`, async () => {
      const res = await aiFor(vendor, "object").object({ model, schema: z.object({ answer: z.string() }), prompt: "?" });
      expect(keys(res)).toEqual(["data", "model", "repairs", "usage"]);
      expect(keys(res.usage)).toEqual(USAGE_KEYS);
      expect(res.data).toEqual({ answer: "42" });
    });

    it(`${vendor}: ai.stream`, async () => {
      const { stream, result } = await aiFor(vendor, "stream").stream({ model, prompt: "?" });
      const deltas: string[] = [];
      for await (const d of stream) deltas.push(d);
      const res = await result;
      expect(deltas.join("")).toBe(SCRIPT.text);
      expect(keys(res)).toEqual(["model", "text", "usage"]);
      expect(keys(res.usage)).toEqual(USAGE_KEYS);
    });

    it(`${vendor}: ai.run (one tool turn, then the answer)`, async () => {
      const res = await aiFor(vendor, "run").run({ model, prompt: "?", tools: [lookup] });
      expect(keys(res)).toEqual(["calls", "messages", "model", "steps", "text", "usage"]);
      expect(keys(res.usage)).toEqual(USAGE_KEYS);
      expect(res.text).toBe(SCRIPT.answer);
      expect(res.steps).toBe(2);
      expect(res.calls.map((c) => [c.name, c.input, c.output])).toEqual([[SCRIPT.call.name, SCRIPT.call.input, 42]]);
    });
  }

  for (const vendor of ["anthropic", "google"]) {
    it(`${vendor}: a safety refusal rejects with CoaxRefusalError carrying the billed usage, reported through onUsage`, async () => {
      const seen: Usage[] = [];
      const err = await aiFor(vendor, "refusal", (u) => seen.push(u))
        .text({ model: `${vendor}:m`, prompt: "?" })
        .catch((e: unknown) => e);
      expect(err).toBeInstanceOf(CoaxRefusalError);
      expect((err as CoaxRefusalError).usage.inputTokens).toBe(9);
      expect(seen).toEqual([(err as CoaxRefusalError).usage]);
    });
  }
});
