/**
 * A `.prompt.md` file: optional YAML-ish frontmatter for call config, then a body split into a SYSTEM
 * and a USER section by `# SYSTEM` / `# USER` headings (case-insensitive, at line start). No headings =
 * the whole body is the user prompt. `{{ var }}` (and `{{ a.b }}`) are filled from the vars object.
 *
 *   ---
 *   model: smart
 *   maxRepairs: 2
 *   ---
 *   # SYSTEM
 *   You are an expert at {{ domain }}.
 *
 *   # USER
 *   {{ input }}
 */
export interface PromptMeta {
  model?: string;
  maxRepairs?: number;
  maxTokens?: number;
  purpose?: string;
}

export interface ParsedPrompt {
  meta: PromptMeta;
  system?: string;
  user: string;
}

export function parsePrompt(source: string): ParsedPrompt {
  const meta: PromptMeta = {};
  let body = source;

  const fm = source.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  if (fm) {
    body = source.slice(fm[0].length);
    for (const line of fm[1]!.split(/\r?\n/)) {
      const m = line.match(/^(\w+)\s*:\s*(.+)$/);
      if (!m) continue;
      const key = m[1]!;
      const value = m[2]!.trim();
      if (key === "model") meta.model = value;
      else if (key === "purpose") meta.purpose = value;
      else if (key === "maxRepairs") meta.maxRepairs = Number(value);
      else if (key === "maxTokens") meta.maxTokens = Number(value);
    }
  }

  const sys = body.match(/^#\s*system\s*$/im);
  const usr = body.match(/^#\s*user\s*$/im);
  let system: string | undefined;
  let user: string;
  if (sys && usr) {
    const sysStart = sys.index! + sys[0].length;
    const usrStart = usr.index! + usr[0].length;
    if (sys.index! < usr.index!) {
      system = body.slice(sysStart, usr.index!).trim();
      user = body.slice(usrStart).trim();
    } else {
      user = body.slice(usrStart, sys.index!).trim();
      system = body.slice(sysStart).trim();
    }
  } else if (usr) {
    user = body.slice(usr.index! + usr[0].length).trim();
  } else {
    user = body.trim();
  }

  return { meta, system, user };
}

/** Fill `{{ var }}` / `{{ a.b }}` placeholders. Missing values render as empty. */
export function renderTemplate(template: string, vars: Record<string, unknown>): string {
  return template.replace(/\{\{\s*([\w.]+)\s*\}\}/g, (_, path: string) => {
    const value = path.split(".").reduce<unknown>((o, k) => (o == null ? undefined : (o as Record<string, unknown>)[k]), vars);
    return value == null ? "" : String(value);
  });
}
