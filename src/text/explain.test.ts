import { describe, expect, it } from "vitest";
import { explain, explainAction, explainCond, explainPos } from "./explain.ts";
import type { Action, Cond, Pos, Program } from "./types.ts";

describe("explain", () => {
  it("every condition", () => {
    const cases: [Cond, string][] = [
      [{ op: "startsWith", text: "A", ignoreCase: true }, 'the line starts with "A" (ignoring case)'],
      [{ op: "endsWith", text: ";" }, 'the line ends with ";"'],
      [{ op: "contains", text: "TODO" }, 'the line contains "TODO"'],
      [{ op: "equals", text: "---" }, 'the line is exactly "---"'],
      [{ op: "startsWithClass", class: "digit" }, "the line starts with a digit"],
      [{ op: "endsWithClass", class: "punct" }, "the line ends with punctuation"],
      [{ op: "lengthGt", n: 1 }, "the line is longer than 1 character"],
      [{ op: "lengthLt", n: 80 }, "the line is shorter than 80 characters"],
      [{ op: "isEmpty" }, "the line is empty"],
      [
        { op: "and", args: [{ op: "isEmpty" }, { op: "not", arg: { op: "contains", text: "x" } }] },
        '(the line is empty) and (not (the line contains "x"))',
      ],
      [
        { op: "or", args: [{ op: "isEmpty" }, { op: "lengthLt", n: 2 }] },
        "(the line is empty) or (the line is shorter than 2 characters)",
      ],
    ];
    for (const [c, text] of cases) expect(explainCond(c)).toBe(text);
  });

  it("every position", () => {
    const cases: [Pos, string][] = [
      [{ from: "start", offset: 0 }, "at the start"],
      [{ from: "end", offset: 0 }, "at the end"],
      [{ from: "start", offset: 3 }, "3 characters from the start"],
      [{ from: "end", offset: 1 }, "1 character from the end"],
      [{ anchor: "before", text: ":" }, 'before the first ":"'],
      [{ anchor: "after", text: ":", occurrence: 2 }, 'after the second ":"'],
      [{ anchor: "after", text: "x", occurrence: -1, ignoreCase: true }, 'after the last "x" (ignoring case)'],
      [{ anchor: "before", text: "x", occurrence: 12 }, 'before the 12th "x"'],
    ];
    for (const [p, text] of cases) expect(explainPos(p)).toBe(text);
  });

  it("every action", () => {
    const cases: [Action, string][] = [
      [{ op: "insert", text: "1", at: { from: "end", offset: 0 } }, 'insert "1" at the end'],
      [{ op: "delete", at: { from: "start", offset: 0 }, count: 2 }, "delete 2 characters at the start"],
      [
        { op: "replaceSpan", at: { from: "start", offset: 4 }, count: 1, with: "/" },
        'replace 1 character 4 characters from the start with "/"',
      ],
      [{ op: "deleteText", text: " " }, 'delete the first " "'],
      [{ op: "deleteText", text: "x", all: true, ignoreCase: true }, 'delete every "x" (ignoring case)'],
      [{ op: "replaceText", find: "-", with: "_", all: true }, 'replace every "-" with "_"'],
    ];
    for (const [a, text] of cases) expect(explainAction(a)).toBe(text);
  });

  it("a whole program as numbered steps", () => {
    const program: Program = {
      unit: "line",
      rules: [
        {
          when: { op: "startsWith", text: "a", ignoreCase: true },
          then: [{ op: "insert", text: "1", at: { from: "end", offset: 0 } }],
        },
        {
          when: { op: "startsWith", text: "b", ignoreCase: true },
          then: [{ op: "insert", text: "2", at: { from: "end", offset: 0 } }],
        },
      ],
    };
    expect(explain(program)).toBe(
      [
        "For each line:",
        '1. If the line starts with "a" (ignoring case): insert "1" at the end.',
        '2. Otherwise, if the line starts with "b" (ignoring case): insert "2" at the end.',
        "Otherwise: leave it unchanged.",
      ].join("\n"),
    );
  });

  it("a program with only an otherwise, and chained actions", () => {
    const program: Program = {
      unit: "line",
      rules: [],
      otherwise: [
        { op: "deleteText", text: " ", all: true },
        { op: "insert", text: "> ", at: { from: "start", offset: 0 } },
      ],
    };
    expect(explain(program)).toBe('For each line:\nFor every line: delete every " ", then insert "> " at the start.');
  });
});
