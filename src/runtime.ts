import type { ZodType } from "zod";
import type { AIConfig } from "./config";
import { createAI, type AI } from "./ai";
import type { ObjectResult, TextResult } from "./client";

/**
 * The ambient (app-wide) instance. Configure it ONCE at startup — a Nuxt/Nitro server plugin, a Vite
 * entry, your `main.ts` — then `import { ai } from "@orgops/coax"` anywhere and just call it. No threading
 * an instance through your app; no `createAI` at every call site.
 *
 *   // server/plugins/coax.ts  (once)
 *   import { configure } from "@orgops/coax";
 *   configure({ providers: { anthropic: process.env.ANTHROPIC_API_KEY! }, models: { smart: "anthropic:claude-opus-4-8" }, onUsage });
 *
 *   // anywhere
 *   import { ai } from "@orgops/coax";
 *   await ai.object({ model: "smart", schema, prompt });
 *
 * For libraries, tests, or multiple isolated configs, use `createAI()` directly instead — no global state.
 */
let _default: AI | undefined;

export function configure(config: AIConfig): AI {
  _default = createAI(config);
  return _default;
}

export function isConfigured(): boolean {
  return _default !== undefined;
}

/** Reset the ambient instance (mainly for tests). */
export function reset(): void {
  _default = undefined;
}

function current(): AI {
  if (!_default) {
    throw new Error(
      "coax: not configured. Call configure({ providers, models, … }) once at startup " +
        "(e.g. a Nuxt server plugin) before using the ambient `ai`. Or use createAI() for an explicit instance.",
    );
  }
  return _default;
}

/** The ambient client — delegates to whatever `configure()` set. Rejects with a clear error if unconfigured. */
export const ai: AI = {
  object: async (call) => current().object(call),
  text: async (call) => current().text(call),
  judge: async (call) => current().judge(call),
  loop: async (opts) => current().loop(opts),
  // Bind to the configured instance lazily on first call (config may be set after this module loads),
  // and cache the underlying prompt fn so the file is parsed once.
  prompt<T = string>(path: string, opts?: { schema?: ZodType<T>; model?: string }) {
    type Fn = (vars?: Record<string, unknown>) => Promise<T extends string ? TextResult : ObjectResult<T>>;
    let fn: Fn | undefined;
    return ((vars?: Record<string, unknown>) => {
      const f = (fn ??= current().prompt<T>(path, opts));
      return f(vars);
    }) as Fn;
  },
};
