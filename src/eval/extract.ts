// Pull a JSON value out of a model response. Models often wrap answers in code fences or
// add a sentence of prose; both modes tolerate that equally so the comparison stays fair.

export type Extracted = { ok: true; value: unknown } | { ok: false; error: string };

const tryParse = (s: string): Extracted => {
  try {
    return { ok: true, value: JSON.parse(s) };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
};

/**
 * The balanced {...} or [...] starting at `start` (whichever bracket is there), respecting JSON strings;
 * undefined if it never closes.
 */
function balancedAt(text: string, start: number): string | undefined {
  const open = text[start];
  const close = open === "{" ? "}" : "]";
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const c = text[i];
    if (inString) {
      if (c === "\\") i++;
      else if (c === '"') inString = false;
    } else if (c === '"') inString = true;
    else if (c === open) depth++;
    else if (c === close && --depth === 0) return text.slice(start, i + 1);
  }
  return undefined;
}

const OPENER = /[[{]/g;

/** The index of the next `{` or `[` at or after `from`, or -1. */
function nextOpener(text: string, from: number): number {
  OPENER.lastIndex = from;
  return OPENER.exec(text)?.index ?? -1;
}

// Only horizontal whitespace may follow the language tag; `\s*` there would also swallow payload
// lines made of whitespace (a fenced "\t" came back as ""). Accepts \n or \r\n line endings.
const FENCE = /```[a-zA-Z]*[ \t]*\r?\n([\s\S]*?)\r?\n?```/;

const FENCES = new RegExp(FENCE.source, "g");

/**
 * A program is the only object with a "unit" or "rules" key, the two every program has; its rules,
 * conditions, actions and positions have neither. Either one marks a program, so a final program that
 * leaves out "unit" is still the answer, and fails validation, rather than being passed over for a draft
 * before it. "otherwise" alone doesn't: it's optional, so an object with nothing else is more likely a
 * piece of the program quoted in an explanation (`the {"otherwise": []} part`) than an answer.
 */
const isProgramObject = (v: unknown): boolean =>
  typeof v === "object" && v !== null && !Array.isArray(v) && ("unit" in v || "rules" in v);

/**
 * A program the model gave as its answer: a program object, or an array holding one. The array is graded
 * as it is, and fails validation, the same whether it's the whole response, fenced or after prose: the
 * program inside it isn't picked out.
 */
export const looksLikeProgram = (v: unknown): boolean =>
  isProgramObject(v) || (Array.isArray(v) && v.some(isProgramObject));

/**
 * Whether text that doesn't parse is a program the model wrote, not one it describes: it has a program's
 * key and no `...` or `…` outside strings, so shorthand in a sentence (`the outer
 * {"unit": "line", "rules": [...]} object`) isn't taken for an answer. A string still open at the end
 * (an answer cut off inside one) counts as a string.
 */
function writesProgram(text: string): boolean {
  const outsideStrings = text.replace(/"(?:[^"\\]|\\.)*(?:"|$)/g, '""');
  return /"(?:unit|rules)"\s*:/.test(text) && !/\.\.\.|…/.test(outsideStrings);
}

/** A JSON string, `null`, `true`, `false` or a number, where the scan stands (sticky). */
const SCALAR = /"(?:[^"\\]|\\.)*"|null|true|false|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/y;

/**
 * What the `[` at `start` opens, read as a JSON array of any values, commas and whitespace: "closed" if
 * it reaches its `]`, "open" if the text ends first (inside a value or between them), undefined if it
 * holds anything else (prose in brackets, `[Program: {...}]`). Commas aren't checked, so an array that
 * isn't valid JSON (a trailing or missing comma) still counts.
 */
function arrayShape(text: string, start: number): "closed" | "open" | undefined {
  for (let i = start + 1; i < text.length;) {
    const c = text[i]!;
    if (c === "]") return "closed";
    if (c === "," || /\s/.test(c)) i++;
    else if (c === "{" || c === "[") {
      const value = balancedAt(text, i);
      if (value === undefined) return "open";
      i += value.length;
    } else {
      SCALAR.lastIndex = i;
      const scalar = SCALAR.exec(text)?.[0];
      if (scalar !== undefined) i += scalar.length;
      else if (c === '"')
        return "open"; // a string that never closes
      else return undefined;
    }
  }
  return "open";
}

/**
 * `unfinished`: the response opens an object after the returned one and never closes it, so its last
 * answer was cut off part way (meaningful only when the response hit the token limit). `malformed`: the
 * error is a complete last program that isn't valid JSON, so it wasn't cut off, whatever the finish
 * reason.
 */
export type ExtractedProgram =
  { ok: true; value: unknown; unfinished?: true } | { ok: false; error: string; malformed?: true };

/**
 * The program's JSON: the whole response if it parses, else the last program candidate (see
 * looksLikeProgram), else the first candidate that parses (so a lone condition is reported as a schema
 * error). Candidates are every fenced block, every balanced {...} and every balanced [...] holding a
 * program, in the order they appear. The last program is the answer: one before it is a worked example
 * restated from the prompt, or an earlier program the model then corrected. Conditions or examples quoted
 * around the program have no program key, so they don't hide it. Braces in prose ("lines like {a}") don't
 * hide it either: a balanced {...} that doesn't parse is skipped as a whole, so the scan never returns an
 * object nested inside a malformed program (a trailing comma is reported as invalid JSON, not as a
 * program missing "unit"), and an array holding a program is taken whole, valid JSON or not, so neither
 * is the program inside it.
 *
 * A program that isn't valid JSON is still a program: malformed (balanced but not JSON, like a trailing
 * comma) or never closed, with a program key either way and no placeholder (see writesProgram). When it's
 * the last program, its parse error is the result: a valid draft before it isn't graded in its place, so
 * the model is told its final answer was invalid and gets a retry.
 */
export function extractJsonObject(response: string): ExtractedProgram {
  const text = response.trim();
  const whole = tryParse(text);
  if (whole.ok) return whole;
  const candidates: { value: unknown; start: number; end: number }[] = [];
  for (const m of text.matchAll(FENCES)) {
    const r = tryParse(m[1]!.trim());
    if (r.ok) candidates.push({ value: r.value, start: m.index, end: m.index + m[0].length });
  }
  let firstError: string | undefined;
  let lastInvalidProgram: { start: number; error: string; malformed?: true } | undefined;
  let lastUnclosed = -1;
  let start = nextOpener(text, 0);
  while (start >= 0) {
    const candidate = balancedAt(text, start);
    let next = start + 1; // an unclosed "{" (e.g. a stray brace in prose): try the braces inside it
    if (text[start] === "[") {
      // Only an array holding a program is a candidate, taken whole. An array that isn't valid JSON but
      // holds a program is a program that isn't valid JSON, as a {...} would be: malformed if it closes
      // (a trailing comma), unfinished if the response ends inside it. Either way the program inside isn't
      // graded. Any other [...] (a list or a note in prose, a cut-off program's rules) is looked inside.
      const shape = arrayShape(text, start);
      const span = candidate ?? text.slice(start);
      const r = tryParse(span);
      if (r.ok && looksLikeProgram(r.value)) {
        candidates.push({ value: r.value, start, end: start + span.length });
        next = start + span.length;
      } else if (!r.ok && shape !== undefined && writesProgram(span)) {
        lastInvalidProgram =
          shape === "closed" ? { start, error: r.error, malformed: true } : { start, error: r.error };
        if (shape === "open") lastUnclosed = start;
        next = start + span.length;
      }
    } else if (candidate === undefined) {
      lastUnclosed = start;
      const rest = text.slice(start);
      const r = tryParse(rest);
      if (!r.ok && writesProgram(rest)) lastInvalidProgram = { start, error: r.error };
    } else {
      const r = tryParse(candidate);
      if (r.ok) candidates.push({ value: r.value, start, end: start + candidate.length });
      else {
        firstError ??= r.error;
        if (writesProgram(candidate)) lastInvalidProgram = { start, error: r.error, malformed: true };
      }
      next = start + candidate.length; // skip everything inside it: its inner objects are part of it
    }
    start = nextOpener(text, next);
  }
  candidates.sort((a, b) => a.start - b.start);
  const program = candidates.findLast((c) => looksLikeProgram(c.value));
  const invalid = lastInvalidProgram;
  if (invalid && (program === undefined || invalid.start > program.start))
    return invalid.malformed
      ? { ok: false, error: invalid.error, malformed: true }
      : { ok: false, error: invalid.error };
  if (candidates.length === 0) return { ok: false, error: firstError ?? "no JSON object found in the response" };
  const chosen = program ?? candidates[0]!;
  return lastUnclosed > chosen.end
    ? { ok: true, value: chosen.value, unfinished: true }
    : { ok: true, value: chosen.value };
}

/**
 * A JSON string literal at the start of a line, optionally after a list bullet (`-`, `*`, `+`, `1.`,
 * `1)`) and a short label such as "Output:", with markdown emphasis or a backtick allowed around either
 * (`- **Output:** "x"`). Group 1 is the bullet, group 2 the label, group 3 the literal. JSON strings
 * can't contain raw line breaks, so a literal never spans lines.
 */
const LEADING_STRING =
  /^[ \t]*(?:([-*+]|\d+[.)])[ \t]+)?(?:[*_]{0,2}([A-Za-z][A-Za-z ]{0,30}?)[ \t]*[*_]{0,2}[ \t]*:[*_]{0,2}[ \t]*)?[*_`]{0,2}("(?:[^"\\\r\n]|\\.)*")/gm;

/**
 * The prompt's own answer label at the start of a line, with the bullets and emphasis LEADING_STRING
 * allows (`**Answer:**`, `- Answer:`), and the spaces after it.
 */
const ANSWER_LABEL = /^[ \t]*(?:(?:[-*+]|\d+[.)])[ \t]+)?[*_]{0,2}answer[ \t]*[*_]{0,2}[ \t]*:[*_]{0,2}[ \t]*/gim;

/**
 * Labels for an input rather than an answer: the prompt's own ("Line to reformat:", "Input:") and the
 * model's variations on them ("Input line:", "Line:"). A string after one is the model restating it.
 */
const INPUT_LABEL = /^(?:input|line)\b/;

/** No strings to pass over. */
const NONE: ReadonlySet<string> = new Set();

/**
 * The first JSON string that starts a line and isn't a restated input or a restated shown example: an
 * `output:` string right after a numbered `input:` one (`1. input: "a"` then `output: "b"`), the way the
 * prompt lists the shown examples. Strings in `skip` (restated worked answers) are passed over too.
 */
function firstLeadingString(text: string, skip: ReadonlySet<string> = NONE): string | undefined {
  let afterShownInput = false;
  for (const m of text.matchAll(LEADING_STRING)) {
    const r = tryParse(m[3]!);
    if (!r.ok || typeof r.value !== "string") continue;
    const label = m[2]?.trim().toLowerCase();
    const shownOutput = afterShownInput && label === "output";
    afterShownInput = /^\d/.test(m[1] ?? "") && label === "input";
    if (!shownOutput && (label === undefined || !INPUT_LABEL.test(label)) && !skip.has(r.value)) return r.value;
  }
  return undefined;
}

/** A direct-rewrite answer. `quoted`: given as a JSON string; otherwise the raw text taken as is. */
export type LineAnswer = { line: string; quoted: boolean };

/**
 * What follows the last `Answer:` label, less the line break after a label alone on its line; undefined
 * with no label. A label whose own answer is a string in `skip` (a restated worked answer) doesn't count.
 * Its own answer is the first JSON string that starts a line between it and the next label, so one on a
 * line of its own, after a blank line or inside a code fence, is seen as the answer extractor would see it.
 */
function answerSection(response: string, skip: ReadonlySet<string> = NONE): string | undefined {
  const labels = [...response.matchAll(ANSWER_LABEL)];
  let afterLabel: string | undefined;
  for (const [i, m] of labels.entries()) {
    const from = m.index + m[0].length;
    const own = skip.size === 0 ? undefined : firstLeadingString(response.slice(from, labels[i + 1]?.index));
    if (own === undefined || !skip.has(own)) afterLabel = response.slice(from).replace(/^\r?\n/, "");
  }
  return afterLabel;
}

/**
 * A direct-rewrite answer, a JSON string as the prompt asks, with prose around it ignored. The answer is
 * what follows the last `Answer:` label that starts a line (the prompt ends with that label, and models
 * often repeat it): anything before it is the model restating the prompt or a worked example, so a reply
 * cut off at `Answer: "` has no answer even if a restated example held one. After the label: the first
 * JSON string that starts a line (on the label's line or later), else the raw text up to the end.
 *
 * With no label: the first string in the first fenced block, else the first in the whole response (an
 * unrelated fence after the answer doesn't hide it, as in program mode), skipping restated inputs and
 * shown examples; else the raw response (or fenced block). Quotes inside a raw line ('She said "hi"')
 * aren't taken for the answer, and raw text loses one trailing newline. An empty answer is no answer,
 * whether blank or a code block holding nothing but line breaks: an empty line is answered `""`, and
 * neither can be told apart from a model that produced nothing. (A code block holding only spaces or a
 * tab answers with them.)
 *
 * `workedAnswers` are the prompt's worked-example answers, each passed when the line asked about isn't its
 * example's own. A model may restate a worked example, label and all (`Answer: "[hello world]"`), and give
 * its own answer with no label, or with one placed before the restatement. When the answer found is a
 * worked one, the response is read again with every restatement of a worked answer passed over (the
 * labels answered with one and the strings equal to one), and the JSON-string answer that reading finds,
 * before or after them, is the answer. With none, the worked answer stays the answer, and is graded like
 * any other.
 */
export function extractLine(response: string, workedAnswers: readonly string[] = []): LineAnswer | undefined {
  const found = lineAnswer(response);
  if (found?.quoted !== true || !workedAnswers.includes(found.line)) return found;
  const without = lineAnswer(response, new Set(workedAnswers));
  return without?.quoted ? without : found;
}

/** extractLine's answer, passing over labels answered with a string in `skip` and strings in it. */
function lineAnswer(response: string, skip: ReadonlySet<string> = NONE): LineAnswer | undefined {
  const section = answerSection(response, skip);
  if (section !== undefined) {
    if (section.trim() === "") return undefined;
    const quoted = firstLeadingString(section, skip);
    if (quoted !== undefined) return { line: quoted, quoted: true };
    return raw(FENCE.exec(section)?.[1], section);
  }
  if (response.trim() === "") return undefined;
  const fenced = FENCE.exec(response.trim())?.[1];
  const quoted =
    (fenced === undefined ? undefined : firstLeadingString(fenced, skip)) ?? firstLeadingString(response, skip);
  if (quoted !== undefined) return { line: quoted, quoted: true };
  return raw(fenced, response);
}

/**
 * The unquoted answer: the fenced block if there is one, else the text, less one trailing newline. A code
 * block of nothing but line breaks is empty, however many: the fence pattern and the trailing-newline
 * rule each drop one, so two blank lines would otherwise come out as an empty line.
 */
function raw(fenced: string | undefined, text: string): LineAnswer | undefined {
  if (fenced !== undefined && /^[\r\n]*$/.test(fenced)) return undefined; // an empty code block
  return { line: (fenced ?? text).replace(/\r?\n$/, ""), quoted: false };
}

/** Why extractLine found no answer in `response` (call it only when it found none). */
export function noLineReason(response: string): string {
  if (response.trim() === "") return "the response was empty";
  if (answerSection(response)?.trim() === "") return "nothing followed the Answer: label";
  return 'the answer was an empty code block (an empty line is answered "")';
}
