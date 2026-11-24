// Reference interpreter for text DSL v0. This file is the source of truth for semantics, and its tests
// pin them down. Only run programs that passed validateProgram().
import type { Action, CharClass, Cond, Pos, Program } from "./types.ts";

// ---------- Characters and case folding ----------
// A "character" is a Unicode code point; lines are handled as code-point arrays throughout.
const chars = (s: string): string[] => Array.from(s);
const isOneCodePoint = (s: string) => chars(s).length === 1;

/**
 * Simple case folding pairs the round trip in foldChar misses: each of these has a multi-character
 * uppercase, so the round trip leaves it alone, yet Unicode folds it to another single character
 * (CaseFolding.txt, status S).
 */
const FOLD_EXCEPTIONS: ReadonlyMap<string, string> = new Map([
  ["\u1fd3", "\u0390"], // ΐ (with oxia) folds to ΐ (with tonos)
  ["\u1fe3", "\u03b0"], // ΰ (with oxia) folds to ΰ (with tonos)
  ["\ufb05", "\ufb06"], // the long s t ligature ﬅ folds to the s t ligature ﬆ
]);

/**
 * Per-code-point simple case folding that never changes length, for ignoreCase: two characters match
 * when Unicode's simple case folding (CaseFolding.txt, statuses C and S) maps them to the same one.
 */
export function foldChar(c: string): string {
  const exception = FOLD_EXCEPTIONS.get(c);
  if (exception !== undefined) return exception;
  // Dotless ı uppercases to I, so the round trip below would fold it into i. Unicode folds it to itself
  // (only Turkic folding pairs it with I), so it stays a letter of its own, as İ does.
  if (c === "\u0131") return c;
  const viaUpper = c.toUpperCase().toLowerCase();
  if (isOneCodePoint(viaUpper)) return viaUpper;
  const lower = c.toLowerCase();
  return isOneCodePoint(lower) ? lower : c;
}

const prepare = (cs: string[], ignoreCase: boolean | undefined) => (ignoreCase ? cs.map(foldChar) : cs);

function matchesAt(hay: string[], needle: string[], at: number): boolean {
  if (at < 0 || at + needle.length > hay.length) return false;
  for (let i = 0; i < needle.length; i++) if (hay[at + i] !== needle[i]) return false;
  return true;
}

/**
 * Start indices of non-overlapping matches, scanning left to right. Used for `all`, `occurrence`
 * and `contains`, so every op counts matches the same way.
 */
export function findMatches(line: string[], text: string, ignoreCase?: boolean): number[] {
  const hay = prepare(line, ignoreCase);
  const needle = prepare(chars(text), ignoreCase);
  const out: number[] = [];
  if (needle.length === 0) return out; // rejected by the schema; never matches here
  for (let i = 0; i + needle.length <= hay.length;) {
    if (matchesAt(hay, needle, i)) {
      out.push(i);
      i += needle.length;
    } else i++;
  }
  return out;
}

// ---------- Conditions ----------
const CLASS_RE: Record<CharClass, RegExp> = {
  letter: /^\p{L}$/u,
  upper: /^\p{Lu}$/u,
  lower: /^\p{Ll}$/u,
  digit: /^\p{Nd}$/u,
  space: /^\p{White_Space}$/u,
  punct: /^\p{P}$/u,
};

export function evalCond(c: Cond, line: string[]): boolean {
  switch (c.op) {
    case "startsWith": {
      const t = chars(c.text);
      return matchesAt(prepare(line, c.ignoreCase), prepare(t, c.ignoreCase), 0);
    }
    case "endsWith": {
      const t = chars(c.text);
      return matchesAt(prepare(line, c.ignoreCase), prepare(t, c.ignoreCase), line.length - t.length);
    }
    case "contains":
      return findMatches(line, c.text, c.ignoreCase).length > 0;
    case "equals": {
      const t = chars(c.text);
      return t.length === line.length && matchesAt(prepare(line, c.ignoreCase), prepare(t, c.ignoreCase), 0);
    }
    case "startsWithClass": {
      const first = line[0];
      return first !== undefined && CLASS_RE[c.class].test(first);
    }
    case "endsWithClass": {
      const last = line[line.length - 1];
      return last !== undefined && CLASS_RE[c.class].test(last);
    }
    case "lengthGt":
      return line.length > c.n;
    case "lengthLt":
      return line.length < c.n;
    case "isEmpty":
      return line.length === 0;
    case "and":
      return c.args.every((a) => evalCond(a, line));
    case "or":
      return c.args.some((a) => evalCond(a, line));
    case "not":
      return !evalCond(c.arg, line);
  }
}

// ---------- Positions ----------
const clamp = (n: number, lo: number, hi: number) => Math.min(Math.max(n, lo), hi);

/** Resolve a position to an index in [0, line.length], or null if an anchor isn't found. */
export function resolvePos(p: Pos, line: string[]): number | null {
  if ("from" in p)
    return p.from === "start" ? clamp(p.offset, 0, line.length) : clamp(line.length - p.offset, 0, line.length);
  const matches = findMatches(line, p.text, p.ignoreCase);
  const occ = p.occurrence ?? 1;
  const start = occ === -1 ? matches[matches.length - 1] : matches[occ - 1];
  if (start === undefined) return null;
  return p.anchor === "before" ? start : start + chars(p.text).length;
}

// ---------- Line length limit ----------
/** The longest a line may grow while a program runs: 10,000 characters, or 10 times its input if longer. */
export const maxLineLength = (inputLength: number) => Math.max(10_000, 10 * inputLength);

/** An action would have made the line longer than its limit. Thrown before the text is built. */
export class LineTooLongError extends Error {
  readonly limit: number;
  constructor(limit: number) {
    super(`an action would make the line longer than ${limit.toLocaleString("en-US")} characters, its limit`);
    this.name = "LineTooLongError";
    this.limit = limit;
  }
}

function checkLength(length: number, limit: number) {
  if (length > limit) throw new LineTooLongError(limit);
}

// ---------- Actions ----------
function splice(line: string[], limit: number, at: number, count: number, insert: string): string[] {
  const start = clamp(at, 0, line.length);
  const end = clamp(start + count, start, line.length);
  const added = chars(insert);
  checkLength(line.length - (end - start) + added.length, limit);
  return [...line.slice(0, start), ...added, ...line.slice(end)];
}

/** Replace the first (or every, with `all`) non-overlapping match; replaced text is never rescanned. */
function replaceMatches(
  line: string[],
  limit: number,
  find: string,
  replacement: string,
  ignoreCase?: boolean,
  all?: boolean,
): string[] {
  const matches = findMatches(line, find, ignoreCase);
  const targets = all ? matches : matches.slice(0, 1);
  if (targets.length === 0) return line;
  const len = chars(find).length;
  const added = chars(replacement);
  checkLength(line.length + targets.length * (added.length - len), limit);
  // One pass, left to right: the text before each match, then its replacement. Copying the whole line
  // once per match instead takes quadratic time.
  const parts: string[][] = [];
  let from = 0;
  for (const start of targets) {
    parts.push(line.slice(from, start), added);
    from = start + len;
  }
  parts.push(line.slice(from));
  return parts.flat();
}

/** Apply one action. Throws LineTooLongError if it would make the line longer than `limit`. */
export function applyAction(a: Action, line: string[], limit: number): string[] {
  switch (a.op) {
    case "insert": {
      const at = resolvePos(a.at, line);
      return at === null ? line : splice(line, limit, at, 0, a.text);
    }
    case "delete": {
      const at = resolvePos(a.at, line);
      return at === null ? line : splice(line, limit, at, a.count, "");
    }
    case "replaceSpan": {
      const at = resolvePos(a.at, line);
      return at === null ? line : splice(line, limit, at, a.count, a.with);
    }
    case "deleteText":
      return replaceMatches(line, limit, a.text, "", a.ignoreCase, a.all);
    case "replaceText":
      return replaceMatches(line, limit, a.find, a.with, a.ignoreCase, a.all);
  }
}

// ---------- Lines and documents ----------
/**
 * Apply a program to one line (no line ending). Throws LineTooLongError if an action would make the
 * line longer than maxLineLength() of the input.
 */
export function runLine(program: Program, line: string): string {
  const cs = chars(line);
  const rule = program.rules.find((r) => evalCond(r.when, cs));
  const actions = rule ? rule.then : (program.otherwise ?? []);
  const limit = maxLineLength(cs.length);
  return actions.reduce((acc, a) => applyAction(a, acc, limit), cs).join("");
}

type Line = { text: string; eol: "" | "\n" | "\r\n" };

/**
 * Split on `\r\n` or `\n`, keeping each line's ending. A trailing newline does not start an extra
 * line, so "" has zero lines and "a\n" has one.
 */
export function splitLines(doc: string): Line[] {
  const out: Line[] = [];
  const re = /\r\n|\n/g;
  let last = 0;
  for (let m = re.exec(doc); m; m = re.exec(doc)) {
    out.push({ text: doc.slice(last, m.index), eol: m[0] as Line["eol"] });
    last = m.index + m[0].length;
  }
  if (last < doc.length) out.push({ text: doc.slice(last), eol: "" });
  return out;
}

/** Run a validated program over a whole document, preserving each line's ending. Throws as runLine does. */
export function run(program: Program, doc: string): string {
  return splitLines(doc)
    .map((l) => runLine(program, l.text) + l.eol)
    .join("");
}
