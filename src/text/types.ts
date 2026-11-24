// Text edit DSL v0: the program's types. interpret.ts defines what each one does.
// Programs are plain JSON so a model can emit them directly; validate with schema.ts before running.

/** Applied to each line independently. The first rule whose condition matches wins. */
export type Program = { unit: "line"; rules: Rule[]; otherwise?: Action[] };

/** Actions run left to right, each on the result of the previous one. */
export type Rule = { when: Cond; then: Action[] };

export type CharClass = "digit" | "letter" | "upper" | "lower" | "space" | "punct";

export type Cond =
  | { op: "startsWith" | "endsWith" | "contains" | "equals"; text: string; ignoreCase?: boolean }
  | { op: "startsWithClass" | "endsWithClass"; class: CharClass }
  | { op: "lengthGt" | "lengthLt"; n: number }
  | { op: "isEmpty" }
  | { op: "and" | "or"; args: Cond[] }
  | { op: "not"; arg: Cond };

/** A position in the line, in Unicode code points. Out-of-range offsets are clamped. */
export type Pos =
  | { from: "start" | "end"; offset: number }
  | { anchor: "before" | "after"; text: string; ignoreCase?: boolean; occurrence?: number };

export type Action =
  | { op: "insert"; text: string; at: Pos }
  | { op: "delete"; at: Pos; count: number }
  | { op: "deleteText"; text: string; ignoreCase?: boolean; all?: boolean }
  | { op: "replaceSpan"; at: Pos; count: number; with: string }
  | { op: "replaceText"; find: string; with: string; ignoreCase?: boolean; all?: boolean };
