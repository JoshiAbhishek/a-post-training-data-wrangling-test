// Render a text DSL v0 program as readable steps, to make model output easy to inspect.
import type { Action, Cond, Pos, Program } from "./types.ts";

const q = (s: string) => JSON.stringify(s);
const ci = (ignoreCase?: boolean) => (ignoreCase ? " (ignoring case)" : "");
const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

const ORDINAL = ["", "first", "second", "third", "fourth", "fifth"];
const nth = (occurrence = 1) => (occurrence === -1 ? "last" : (ORDINAL[occurrence] ?? `${occurrence}th`));

const CLASS_NAME = {
  digit: "a digit",
  letter: "a letter",
  upper: "an uppercase letter",
  lower: "a lowercase letter",
  space: "whitespace",
  punct: "punctuation",
} as const;

export function explainCond(c: Cond): string {
  switch (c.op) {
    case "startsWith":
      return `the line starts with ${q(c.text)}${ci(c.ignoreCase)}`;
    case "endsWith":
      return `the line ends with ${q(c.text)}${ci(c.ignoreCase)}`;
    case "contains":
      return `the line contains ${q(c.text)}${ci(c.ignoreCase)}`;
    case "equals":
      return `the line is exactly ${q(c.text)}${ci(c.ignoreCase)}`;
    case "startsWithClass":
      return `the line starts with ${CLASS_NAME[c.class]}`;
    case "endsWithClass":
      return `the line ends with ${CLASS_NAME[c.class]}`;
    case "lengthGt":
      return `the line is longer than ${plural(c.n, "character")}`;
    case "lengthLt":
      return `the line is shorter than ${plural(c.n, "character")}`;
    case "isEmpty":
      return "the line is empty";
    case "and":
      return c.args.map((a) => `(${explainCond(a)})`).join(" and ");
    case "or":
      return c.args.map((a) => `(${explainCond(a)})`).join(" or ");
    case "not":
      return `not (${explainCond(c.arg)})`;
  }
}

export function explainPos(p: Pos): string {
  if ("from" in p) {
    if (p.offset === 0) return p.from === "start" ? "at the start" : "at the end";
    return `${plural(p.offset, "character")} from the ${p.from}`;
  }
  return `${p.anchor} the ${nth(p.occurrence)} ${q(p.text)}${ci(p.ignoreCase)}`;
}

export function explainAction(a: Action): string {
  const every = (all?: boolean) => (all ? "every" : "the first");
  switch (a.op) {
    case "insert":
      return `insert ${q(a.text)} ${explainPos(a.at)}`;
    case "delete":
      return `delete ${plural(a.count, "character")} ${explainPos(a.at)}`;
    case "replaceSpan":
      return `replace ${plural(a.count, "character")} ${explainPos(a.at)} with ${q(a.with)}`;
    case "deleteText":
      return `delete ${every(a.all)} ${q(a.text)}${ci(a.ignoreCase)}`;
    case "replaceText":
      return `replace ${every(a.all)} ${q(a.find)}${ci(a.ignoreCase)} with ${q(a.with)}`;
  }
}

const actions = (as: readonly Action[]) => (as.length ? as.map(explainAction).join(", then ") : "leave it unchanged");

/** One numbered step per rule, then the fallback. */
export function explain(program: Program): string {
  const steps = program.rules.map(
    (r, i) => `${i + 1}. ${i === 0 ? "If" : "Otherwise, if"} ${explainCond(r.when)}: ${actions(r.then)}.`,
  );
  const otherwise = `${program.rules.length ? "Otherwise" : "For every line"}: ${actions(program.otherwise ?? [])}.`;
  return ["For each line:", ...steps, otherwise].join("\n");
}
