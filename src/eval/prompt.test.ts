import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { runLine } from "../text/interpret.ts";
import { parseProgram } from "../text/schema.ts";
import {
  formatExamples,
  loadPrompt,
  parsePrompt,
  programMessages,
  promptPath,
  retryMessage,
  rewriteMessages,
  workedExamples,
} from "./prompt.ts";
import { publicView } from "./harness.ts";
import { CANARY, loadTasks } from "./task.ts";

const program = loadPrompt(promptPath("program-synthesis-v1"));
const rewrite = loadPrompt(promptPath("direct-rewrite-v1"));
const task = { id: "t", description: "Do the thing", examples: [{ input: "a\tb", output: "" }] };

describe("prompt files", () => {
  it("have the sections the harness uses", () => {
    expect(Object.keys(program.sections).sort()).toEqual(["retry", "system", "user"]);
    expect(Object.keys(rewrite.sections).sort()).toEqual(["system", "user"]);
    expect(program.version).toBe("program-synthesis-v1");
    expect(program.sha256).toMatch(/^[0-9a-f]{64}$/);
  });

  it("fill every placeholder", () => {
    const all = [
      ...programMessages(program, task),
      retryMessage(program, "feedback"),
      ...rewriteMessages(rewrite, task, "x"),
    ];
    for (const m of all) expect(m.content).not.toMatch(/\{\{\w+\}\}/);
  });

  it("show strings as JSON so tabs and empty lines are visible", () => {
    expect(formatExamples(task.examples)).toBe('1. input: "a\\tb"\n   output: ""');
  });

  it("contain worked examples whose programs are valid and correct", () => {
    const system = program.sections["system"]!;
    const blocks = [...system.matchAll(/Examples:\n([\s\S]*?)Program:\n(\{.*\})/g)];
    expect(blocks).toHaveLength(2);
    for (const [, examples, json] of blocks) {
      const parsed = parseProgram(json!);
      if (!parsed.ok) throw new Error(parsed.errors.join("\n"));
      const pairs = [...examples!.matchAll(/input: (".*")\n\s+output: (".*")/g)];
      expect(pairs.length).toBeGreaterThan(1);
      for (const [, input, output] of pairs) {
        expect(runLine(parsed.program, JSON.parse(input!))).toBe(JSON.parse(output!));
      }
    }
  });

  it("use worked examples that aren't tasks in the task set", () => {
    const tasks = loadTasks(fileURLToPath(new URL("../../tasks/", import.meta.url))).map((t) => t.task);
    const promptText = program.sections["system"]! + rewrite.sections["system"]!;
    const overlaps = tasks.flatMap((t) => [
      ...(promptText.includes(t.description) ? [`${t.id}: description`] : []),
      ...[...t.examples, ...t.heldOut]
        .flatMap((e) => [e.input, e.output])
        .filter((s) => s.length > 2 && promptText.includes(JSON.stringify(s)))
        .map((s) => `${t.id}: ${JSON.stringify(s)}`),
    ]);
    expect(overlaps).toEqual([]);
  });
});

describe("direct-rewrite worked examples", () => {
  it("are read from the system section", () => {
    expect(workedExamples(rewrite)).toEqual([{ input: "hello world", answer: "[hello world]" }]);
    expect(workedExamples(program)).toEqual([]);
    const two = parsePrompt(
      "t",
      '<!-- system -->\nLine to reformat: "a"\nAnswer: "[a]"\n\nLine to reformat: "b \\"c\\""\nAnswer: "B"\n<!-- /system -->',
    );
    expect(workedExamples(two)).toEqual([
      { input: "a", answer: "[a]" },
      { input: 'b "c"', answer: "B" },
    ]);
  });

  // The grader passes over a restated worked answer, so one that was a task line's right answer could
  // cost a correct reply. Every string counts here, however short (the overlap check above skips the
  // shortest).
  it("never answer a task line", () => {
    const tasks = loadTasks(fileURLToPath(new URL("../../tasks/", import.meta.url))).map((t) => t.task);
    const answers = new Set(workedExamples(rewrite).map((w) => w.answer));
    const clashes = tasks.flatMap((t) =>
      [...t.examples, ...t.heldOut].filter((e) => answers.has(e.output)).map((e) => `${t.id}: ${e.input}`),
    );
    expect(clashes).toEqual([]);
  });
});

describe("the task canary", () => {
  it("never reaches a prompt, in either mode", () => {
    const guid = CANARY.split(" ").at(-1)!;
    const tasks = loadTasks(fileURLToPath(new URL("../../tasks/", import.meta.url))).map((t) => t.task);
    expect(tasks.every((t) => t.canary === CANARY)).toBe(true);
    for (const t of tasks) {
      const messages = [
        ...programMessages(program, publicView(t)),
        ...t.heldOut.flatMap((e) => rewriteMessages(rewrite, publicView(t), e.input)),
      ];
      for (const m of messages) expect(m.content, t.id).not.toContain(guid);
    }
  });
});
