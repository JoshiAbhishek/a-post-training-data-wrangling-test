// Versioned prompt files (prompts/*.md) and message building for the harness.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { basename } from "node:path";
import { fileURLToPath } from "node:url";
import type { Message } from "../llm/provider.ts";
import type { Example } from "./task.ts";

export type PromptTemplate = {
  /** File name without extension, e.g. "program-synthesis-v1". Recorded in results. */
  version: string;
  /** SHA-256 of the file, so results pin the exact prompt text. */
  sha256: string;
  sections: Record<string, string>;
};

export function parsePrompt(version: string, text: string): PromptTemplate {
  const sections: Record<string, string> = {};
  for (const m of text.matchAll(/<!-- (\w+) -->\n([\s\S]*?)\n<!-- \/\1 -->/g)) {
    sections[m[1]!] = m[2]!;
  }
  return { version, sha256: createHash("sha256").update(text).digest("hex"), sections };
}

export function loadPrompt(path: string): PromptTemplate {
  return parsePrompt(basename(path, ".md"), readFileSync(path, "utf8"));
}

export function promptPath(name: string): string {
  return fileURLToPath(new URL(`../../prompts/${name}.md`, import.meta.url));
}

export function fill(template: PromptTemplate, section: string, vars: Record<string, string>): string {
  const body = template.sections[section];
  if (body === undefined) throw new Error(`prompt ${template.version} has no "${section}" section`);
  return body.replace(/\{\{(\w+)\}\}/g, (_, key: string) => {
    const v = vars[key];
    if (v === undefined) throw new Error(`prompt ${template.version}: no value for {{${key}}}`);
    return v;
  });
}

/** Strings are shown as JSON so empty lines, tabs and edge spaces are unambiguous. */
export const quote = (s: string) => JSON.stringify(s);

export function formatExamples(examples: readonly Example[]): string {
  return examples.map((e, i) => `${i + 1}. input: ${quote(e.input)}\n   output: ${quote(e.output)}`).join("\n");
}

/**
 * What the model may see of a task. Built only from the description and the shown examples, so code
 * that receives a PublicTask has no way to put held-out data into a prompt.
 */
export type PublicTask = { id: string; description: string; examples: readonly Example[] };

export function programMessages(t: PromptTemplate, task: PublicTask): Message[] {
  return [
    { role: "system", content: fill(t, "system", {}) },
    {
      role: "user",
      content: fill(t, "user", { description: task.description, examples: formatExamples(task.examples) }),
    },
  ];
}

export function retryMessage(t: PromptTemplate, feedback: string): Message {
  return { role: "user", content: fill(t, "retry", { feedback }) };
}

/**
 * The worked examples in a direct-rewrite prompt's system section: for each, the line it reformats and
 * its answer, from its `Line to reformat:` and `Answer:` lines. Empty if it has none.
 */
export function workedExamples(t: PromptTemplate): { input: string; answer: string }[] {
  const pairs = (t.sections["system"] ?? "").matchAll(
    /^Line to reformat: ("(?:[^"\\\r\n]|\\.)*")\r?\nAnswer: ("(?:[^"\\\r\n]|\\.)*")$/gm,
  );
  return [...pairs].map((m) => ({ input: JSON.parse(m[1]!) as string, answer: JSON.parse(m[2]!) as string }));
}

/** Direct rewrite: the description, the shown examples and exactly one held-out input (never its output). */
export function rewriteMessages(t: PromptTemplate, task: PublicTask, input: string): Message[] {
  return [
    { role: "system", content: fill(t, "system", {}) },
    {
      role: "user",
      content: fill(t, "user", {
        description: task.description,
        examples: formatExamples(task.examples),
        input: quote(input),
      }),
    },
  ];
}
