// Verifies the hand-written task set: every reference program
// must reproduce every shown and held-out output, so a wrong answer key never makes a model look wrong.
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { evalCond, runLine } from "../text/interpret.ts";
import { validateProgram } from "../text/schema.ts";
import { CANARY, CATEGORIES, coverage, loadTasks, TaskSchema, taskSetSha256 } from "./task.ts";

const TASKS_URL = new URL("../../tasks/", import.meta.url);
const TASKS_DIR = fileURLToPath(TASKS_URL);
const tasks = loadTasks(TASKS_DIR);

describe("task set", () => {
  it("has 20–40 tasks with unique ids", () => {
    expect(tasks.length).toBeGreaterThanOrEqual(20);
    expect(tasks.length).toBeLessThanOrEqual(40);
    const ids = tasks.map((t) => t.task.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("covers every category at least 4 times", () => {
    const counts = coverage(tasks.map((t) => t.task));
    for (const c of CATEGORIES) expect(counts[c], `category ${c}`).toBeGreaterThanOrEqual(4);
  });

  it("marks every task file with the canary", () => {
    for (const t of tasks) expect(t.task.canary, t.file).toBe(CANARY);
  });

  it("includes empty lines in some held-out sets", () => {
    expect(tasks.some((t) => t.task.heldOut.some((e) => e.input === ""))).toBe(true);
  });
});

describe.each(tasks.map((t) => [t.task.id, t.task] as const))("task %s", (_id, task) => {
  it("reference passes the program schema", () => {
    expect(validateProgram(task.reference)).toEqual({ ok: true, program: task.reference });
  });

  it.each(task.examples.map((e) => [e.input, e.output] as const))("shown: %j → %j", (input, output) => {
    expect(runLine(task.reference, input)).toBe(output);
  });

  it.each(task.heldOut.map((e) => [e.input, e.output] as const))("held-out: %j → %j", (input, output) => {
    expect(runLine(task.reference, input)).toBe(output);
  });

  // Every task, not only multi-branch ones: a reference may need several rules for one edit (one per
  // place a word can stand), and a program missing one of them must not get full credit.
  it("held-out covers every rule plus a line matching no rule", () => {
    const branchOf = (line: string) => task.reference.rules.findIndex((r) => evalCond(r.when, Array.from(line)));
    const covered = new Set(task.heldOut.map((e) => branchOf(e.input)));
    // The empty line is the one input an isEmpty rule matches, and held-out inputs can't repeat shown
    // ones, so a shown empty line covers that branch.
    const shownEmpty = task.examples.some((e) => e.input === "");
    for (let i = 0; i < task.reference.rules.length; i++) {
      const onlyEmpty = task.reference.rules[i]!.when.op === "isEmpty";
      expect(covered.has(i) || (onlyEmpty && shownEmpty), `rule ${i}`).toBe(true);
    }
    expect(covered.has(-1), "a line matching no rule").toBe(true);
  });

  it.runIf(task.category.includes("unicode"))("uses non-ASCII text", () => {
    const all = [...task.examples, ...task.heldOut].flatMap((e) => [e.input, e.output]).join("");
    expect([...all].some((ch) => ch.codePointAt(0)! > 0x7f)).toBe(true);
  });
});

describe("taskSetSha256", () => {
  const all = tasks.map((t) => t.task);

  it("is a full SHA-256 that depends only on the tasks' content", () => {
    expect(taskSetSha256(all)).toMatch(/^[0-9a-f]{64}$/);
    expect(taskSetSha256(structuredClone(all))).toBe(taskSetSha256(all));
    // Key order in a file doesn't count: tasks are hashed as loaded.
    const reordered = TaskSchema.parse(Object.fromEntries(Object.entries(all[0]!).reverse()));
    expect(taskSetSha256([reordered, ...all.slice(1)])).toBe(taskSetSha256(all));
  });

  it("changes with a held-out line, the reference or which tasks run", () => {
    const first = all[0]!;
    const moreHeldOut = { ...first, heldOut: [...first.heldOut, { input: "new line", output: "new line" }] };
    const otherReference = { ...first, reference: { unit: "line" as const, rules: [] } };
    const digests = [
      all,
      [moreHeldOut, ...all.slice(1)],
      [otherReference, ...all.slice(1)],
      all.slice(0, -1), // --limit
      [...all].reverse(),
    ].map(taskSetSha256);
    expect(new Set(digests).size).toBe(digests.length);
  });
});

describe("task schema", () => {
  const base = {
    id: "x",
    description: "d",
    category: ["insert"],
    examples: [
      { input: "a", output: "a" },
      { input: "b", output: "b" },
    ],
    heldOut: [
      { input: "c", output: "c" },
      { input: "d", output: "d" },
      { input: "e", output: "e" },
    ],
    reference: { unit: "line", rules: [] },
  };

  it("accepts a minimal valid task", () => {
    expect(TaskSchema.safeParse(base).success).toBe(true);
  });

  it("accepts the canary, and nothing else in its place", () => {
    expect(TaskSchema.safeParse({ ...base, canary: CANARY }).success).toBe(true);
    expect(TaskSchema.safeParse({ ...base, canary: CANARY.replace(/.$/, "x") }).success).toBe(false);
  });

  it("rejects bad ids, categories, example counts and multi-line examples", () => {
    const bad = [
      { ...base, id: "Not_Kebab" },
      { ...base, category: ["regex"] },
      { ...base, examples: base.examples.slice(0, 1) },
      { ...base, heldOut: base.heldOut.slice(0, 2) },
      { ...base, examples: [...base.examples, ...base.examples], heldOut: base.heldOut }, // 4 shown, 3 held-out
      { ...base, examples: [{ input: "a\nb", output: "a" }, base.examples[1]] },
      { ...base, reference: { unit: "line", rules: [{ when: { op: "nope" }, then: [] }] } },
      { ...base, heldOut: [...base.heldOut.slice(0, 2), { input: "a", output: "a" }] }, // repeats a shown input
      { ...base, heldOut: [...base.heldOut, base.heldOut[0]] }, // repeats a held-out input
    ];
    for (const t of bad) expect(TaskSchema.safeParse(t).success, JSON.stringify(t)).toBe(false);
  });
});
