import { describe, expect, it } from "vitest";
import { MAX_CONDITION_DEPTH, MAX_LIST_ITEMS, parseProgram, validateProgram } from "./schema.ts";

const tooDeep = `conditions can be nested at most ${MAX_CONDITION_DEPTH} levels deep`;

const insertEnd = (text: string) => ({ op: "insert", text, at: { from: "end", offset: 0 } });
/** A program with one rule; override its condition or actions to test one field at a time. */
const prog = (when: unknown = { op: "isEmpty" }, then: unknown[] = [insertEnd("x")], extra: object = {}) => ({
  unit: "line",
  rules: [{ when, then }],
  ...extra,
});
const errorsOf = (input: unknown) => {
  const r = validateProgram(input);
  return r.ok ? [] : r.errors;
};
const rejects = (input: unknown, pathPrefix: string) => {
  const errors = errorsOf(input);
  expect(errors.length, `expected rejection at ${pathPrefix}`).toBeGreaterThan(0);
  expect(
    errors.some((e) => e.startsWith(pathPrefix)),
    errors.join("\n"),
  ).toBe(true);
};

describe("accepts valid programs", () => {
  it("every condition", () => {
    const conds = [
      { op: "startsWith", text: "a" },
      { op: "endsWith", text: "a", ignoreCase: true },
      { op: "contains", text: "a" },
      { op: "equals", text: "a", ignoreCase: false },
      { op: "startsWithClass", class: "digit" },
      { op: "endsWithClass", class: "punct" },
      { op: "lengthGt", n: 0 },
      { op: "lengthLt", n: 10 },
      { op: "isEmpty" },
      { op: "and", args: [{ op: "isEmpty" }, { op: "not", arg: { op: "contains", text: "x" } }] },
      { op: "or", args: [{ op: "isEmpty" }] },
    ];
    for (const c of conds) expect(errorsOf(prog(c)), JSON.stringify(c)).toEqual([]);
  });

  it("every action and position", () => {
    const actions = [
      { op: "insert", text: "", at: { from: "start", offset: 0 } },
      { op: "delete", at: { from: "end", offset: 3 }, count: 0 },
      { op: "deleteText", text: "a", ignoreCase: true, all: true },
      { op: "replaceSpan", at: { anchor: "after", text: ":", occurrence: -1 }, count: 2, with: "" },
      { op: "replaceText", find: "a", with: "b", all: false },
      { op: "insert", text: "x", at: { anchor: "before", text: "a", ignoreCase: true, occurrence: 3 } },
    ];
    for (const a of actions) expect(errorsOf(prog(undefined, [a])), JSON.stringify(a)).toEqual([]);
  });

  it("no rules with an otherwise, and empty action lists", () => {
    expect(errorsOf({ unit: "line", rules: [], otherwise: [insertEnd("x")] })).toEqual([]);
    expect(errorsOf({ unit: "line", rules: [{ when: { op: "isEmpty" }, then: [] }] })).toEqual([]);
  });
});

describe("rejects malformed programs with a readable path", () => {
  it("unknown op", () => {
    rejects(prog({ op: "matches", pattern: "a+" }), "rules.0.when.op");
    rejects(prog(undefined, [{ op: "uppercase" }]), "rules.0.then.0.op");
  });

  it("wrong field types and missing fields", () => {
    rejects(prog({ op: "startsWith", text: 5 }), "rules.0.when.text");
    rejects(prog({ op: "startsWith", text: "a", ignoreCase: "yes" }), "rules.0.when.ignoreCase");
    rejects(prog({ op: "startsWithClass", class: "emoji" }), "rules.0.when.class");
    rejects(prog(undefined, [{ op: "delete", at: { from: "start", offset: 0 } }]), "rules.0.then.0.count");
    rejects({ unit: "document", rules: [] }, "unit");
    rejects({ unit: "line" }, "rules");
  });

  it("unknown extra fields", () => {
    rejects(prog({ op: "isEmpty", text: "a" }), "rules.0.when");
    rejects(prog(undefined, undefined, { comment: "hi" }), "(root)");
  });

  it("negative or fractional offset, count and n", () => {
    rejects(prog(undefined, [{ op: "insert", text: "x", at: { from: "start", offset: -1 } }]), "rules.0.then.0.at");
    rejects(prog(undefined, [{ op: "insert", text: "x", at: { from: "end", offset: 1.5 } }]), "rules.0.then.0.at");
    rejects(prog(undefined, [{ op: "delete", at: { from: "start", offset: 0 }, count: -2 }]), "rules.0.then.0.count");
    rejects(
      prog(undefined, [{ op: "replaceSpan", at: { from: "start", offset: 0 }, count: 0.5, with: "" }]),
      "rules.0.then.0.count",
    );
    rejects(prog({ op: "lengthGt", n: -1 }), "rules.0.when.n");
    rejects(prog({ op: "lengthLt", n: 2.5 }), "rules.0.when.n");
  });

  it("occurrence of 0, -2 or 1.5", () => {
    for (const occurrence of [0, -2, 1.5]) {
      rejects(
        prog(undefined, [{ op: "insert", text: "x", at: { anchor: "after", text: ":", occurrence } }]),
        "rules.0.then.0.at",
      );
    }
  });

  it("empty search text everywhere it is searched for", () => {
    for (const op of ["startsWith", "endsWith", "contains", "equals"])
      rejects(prog({ op, text: "" }), "rules.0.when.text");
    rejects(prog(undefined, [{ op: "insert", text: "x", at: { anchor: "before", text: "" } }]), "rules.0.then.0.at");
    rejects(prog(undefined, [{ op: "deleteText", text: "" }]), "rules.0.then.0.text");
    rejects(prog(undefined, [{ op: "replaceText", find: "", with: "x" }]), "rules.0.then.0.find");
  });

  it("empty and/or argument lists", () => {
    rejects(prog({ op: "and", args: [] }), "rules.0.when.args");
  });

  it("errors in nested conditions point at the nested field", () => {
    rejects(
      prog({ op: "and", args: [{ op: "isEmpty" }, { op: "not", arg: { op: "contains", text: "" } }] }),
      "rules.0.when.args.1.arg.text",
    );
  });

  it("line breaks in added text, but not in search text", () => {
    for (const text of ["a\nb", "\r", "\r\n"]) {
      rejects(prog(undefined, [insertEnd(text)]), "rules.0.then.0.text");
      rejects(prog(undefined, [{ op: "replaceText", find: "x", with: text }]), "rules.0.then.0.with");
      rejects(
        prog(undefined, [{ op: "replaceSpan", at: { from: "start", offset: 0 }, count: 1, with: text }]),
        "rules.0.then.0.with",
      );
    }
    expect(errorsOf(prog(undefined, [insertEnd("x")])), "no line break").toEqual([]);
    expect(errorsOf(prog({ op: "contains", text: "\r" }, [{ op: "deleteText", text: "\r" }]))).toEqual([]);
  });
});

describe("limits", () => {
  /** `levels` conditions deep: `levels - 1` nots around an isEmpty. */
  const nested = (levels: number) => {
    let c: unknown = { op: "isEmpty" };
    for (let i = 1; i < levels; i++) c = { op: "not", arg: c };
    return c;
  };

  it(`conditions nest at most ${MAX_CONDITION_DEPTH} levels`, () => {
    expect(errorsOf(prog(nested(MAX_CONDITION_DEPTH)))).toEqual([]);
    // The error is on the deepest allowed condition, the one that tries to hold another.
    const errors = errorsOf(prog(nested(MAX_CONDITION_DEPTH + 1)));
    expect(errors).toEqual([`rules.0.when${".arg".repeat(MAX_CONDITION_DEPTH - 1)}: ${tooDeep}`]);
    // and/or count as a level too.
    let c: unknown = { op: "isEmpty" };
    for (let i = 1; i <= MAX_CONDITION_DEPTH; i++) c = { op: i % 2 ? "and" : "or", args: [{ op: "isEmpty" }, c] };
    rejects(prog(c), "rules.0.when.args.1");
  });

  it("a condition nested thousands of levels deep is rejected, not a stack overflow", () => {
    for (const levels of [2_000, 100_000]) {
      const json = '{"op":"not","arg":'.repeat(levels) + '{"op":"isEmpty"}' + "}".repeat(levels);
      const r = parseProgram(`{"unit":"line","rules":[{"when":${json},"then":[]}]}`);
      expect(r.ok ? [] : r.errors.map((e) => e.split(": ")[1])).toEqual([tooDeep]);
    }
  });

  it(`rules, action lists and and/or arguments hold at most ${MAX_LIST_ITEMS} items`, () => {
    const many = (n: number, item: unknown) => Array.from({ length: n }, () => item);
    const rule = { when: { op: "isEmpty" }, then: [] };
    expect(errorsOf({ unit: "line", rules: many(MAX_LIST_ITEMS, rule) })).toEqual([]);
    expect(errorsOf({ unit: "line", rules: [], otherwise: many(MAX_LIST_ITEMS, insertEnd("x")) })).toEqual([]);
    expect(errorsOf(prog({ op: "or", args: many(MAX_LIST_ITEMS, { op: "isEmpty" }) }))).toEqual([]);
    const tooMany = `must have at most ${MAX_LIST_ITEMS} items`;
    expect(errorsOf({ unit: "line", rules: many(MAX_LIST_ITEMS + 1, rule) })).toEqual([`rules: ${tooMany}`]);
    expect(errorsOf(prog(undefined, many(MAX_LIST_ITEMS + 1, insertEnd("x"))))).toEqual([`rules.0.then: ${tooMany}`]);
    expect(errorsOf({ unit: "line", rules: [], otherwise: many(MAX_LIST_ITEMS + 1, insertEnd("x")) })).toEqual([
      `otherwise: ${tooMany}`,
    ]);
    expect(errorsOf(prog({ op: "and", args: many(MAX_LIST_ITEMS + 1, { op: "isEmpty" }) }))).toEqual([
      `rules.0.when.args: ${tooMany}`,
    ]);
  });
});

describe("parseProgram", () => {
  it("parses and validates JSON text", () => {
    expect(parseProgram(JSON.stringify(prog())).ok).toBe(true);
  });

  it("reports invalid JSON as a validation error", () => {
    expect(parseProgram("{ unit: line ")).toMatchObject({
      ok: false,
      errors: [expect.stringMatching(/^\(root\): invalid JSON/)],
    });
  });
});
