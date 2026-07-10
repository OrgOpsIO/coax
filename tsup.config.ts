import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm", "cjs"],
  dts: true,
  clean: true,
  sourcemap: true,
  treeshake: true,
  // The SDKs are optional peer deps, loaded lazily at runtime — never bundle them into coax.
  external: ["@anthropic-ai/sdk", "openai", "zod", "zod-to-json-schema", "jsonrepair"],
});
