import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";
import { describe, expect, it } from "vitest";

// What a TypeScript consumer of coax compiles against: the declarations of src/index.ts. They are emitted
// in memory (no dist/, no temp dir) and a consumer is compiled against them with `skipLibCheck: false`,
// the compiler's default. Every vendor SDK whose types those declarations import is then type-checked
// too — SDK 2.71.0 of ElevenLabs has 9 errors of its own (review R2.1, measurement D1), so its types
// must never reach the public surface.
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DECL = join(ROOT, ".coax-decl"); // virtual directory: exists only inside the compiler hosts below

function emitDeclarations(): Map<string, string> {
  const options: ts.CompilerOptions = {
    target: ts.ScriptTarget.ES2022,
    module: ts.ModuleKind.ESNext,
    moduleResolution: ts.ModuleResolutionKind.Bundler,
    lib: ["lib.es2022.d.ts"],
    types: ["node"],
    strict: true,
    esModuleInterop: true,
    skipLibCheck: true,
    declaration: true,
    emitDeclarationOnly: true,
    rootDir: ROOT,
    outDir: DECL,
  };
  const out = new Map<string, string>();
  const program = ts.createProgram([join(ROOT, "src/index.ts")], options);
  const result = program.emit(undefined, (file, text) => out.set(resolve(file), text));
  expect(result.emitSkipped).toBe(false);
  return out;
}

function compileConsumer(decls: Map<string, string>, source: string): string[] {
  const consumer = join(ROOT, "coax-consumer.ts");
  const files = new Map(decls);
  files.set(consumer, source);
  const options: ts.CompilerOptions = {
    target: ts.ScriptTarget.ES2022,
    module: ts.ModuleKind.ESNext,
    moduleResolution: ts.ModuleResolutionKind.Bundler,
    lib: ["lib.es2022.d.ts", "lib.dom.d.ts"],
    types: ["node"],
    strict: true,
    esModuleInterop: true,
    skipLibCheck: false,
    noEmit: true,
  };
  const host = ts.createCompilerHost(options);
  const { fileExists, readFile, getSourceFile, directoryExists } = host;
  host.fileExists = (f) => files.has(resolve(f)) || fileExists.call(host, f);
  host.readFile = (f) => files.get(resolve(f)) ?? readFile.call(host, f);
  host.directoryExists = (d) => resolve(d).startsWith(DECL) || (directoryExists ? directoryExists.call(host, d) : true);
  host.getSourceFile = (f, lang, onError, create) => {
    const text = files.get(resolve(f));
    return text != null ? ts.createSourceFile(f, text, lang) : getSourceFile.call(host, f, lang, onError, create);
  };
  const program = ts.createProgram([consumer], options, host);
  return ts.getPreEmitDiagnostics(program).map((d) => {
    const where = d.file ? `${d.file.fileName.replace(ROOT, "")}: ` : "";
    return `${where}TS${d.code} ${ts.flattenDiagnosticMessageText(d.messageText, "\n").slice(0, 200)}`;
  });
}

describe("the public declarations", () => {
  const decls = emitDeclarations();

  it("never import the ElevenLabs SDK's types (they do not compile with skipLibCheck: false)", () => {
    expect(decls.size).toBeGreaterThan(0);
    const leaking = [...decls].filter(([, text]) => text.includes("@elevenlabs/elevenlabs-js")).map(([f]) => f.replace(ROOT, ""));
    expect(leaking).toEqual([]);
  });

  it("compile in a consumer with skipLibCheck: false — one that never uses ElevenLabs, and one that does", () => {
    const diagnostics = compileConsumer(
      decls,
      [
        `import { createAI, elevenlabs, type AIConfig, type ElevenLabsOptions } from "./.coax-decl/src/index";`,
        `const config: AIConfig = { providers: { openai: "k", elevenlabs: { apiKey: "k", voice: "v" } } };`,
        `export const ai = createAI(config);`,
        `const opts: ElevenLabsOptions = { model: "eleven_flash_v2_5", apiKey: "k" };`,
        `export const p = elevenlabs(opts);`,
      ].join("\n"),
    );
    expect(diagnostics).toEqual([]);
  }, 60_000);
});
