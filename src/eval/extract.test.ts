import { describe, expect, it } from "vitest";
import { extractJsonObject, extractLine, noLineReason } from "./extract.ts";

describe("extractJsonObject", () => {
  const program = { unit: "line", rules: [] };

  it("parses a bare object, with surrounding whitespace", () => {
    expect(extractJsonObject('  {"unit":"line","rules":[]}\n')).toEqual({ ok: true, value: program });
  });

  it("parses a fenced block, with or without a language tag", () => {
    expect(extractJsonObject('Here you go:\n```json\n{"unit":"line","rules":[]}\n```\nDone.')).toEqual({
      ok: true,
      value: program,
    });
    expect(extractJsonObject('```\n{"unit":"line","rules":[]}\n```')).toEqual({ ok: true, value: program });
  });

  it("finds the first balanced object after prose, ignoring braces inside strings", () => {
    const r = extractJsonObject(
      'Program: {"unit":"line","rules":[],"otherwise":[{"op":"insert","text":"}{","at":{"from":"start","offset":0}}]} hope it helps',
    );
    expect(r.ok && (r.value as { otherwise: { text: string }[] }).otherwise[0]?.text).toBe("}{");
  });

  it("skips braces in leading prose that aren't the program", () => {
    expect(extractJsonObject('For lines like {a}, use:\n{"unit":"line","rules":[]}')).toEqual({
      ok: true,
      value: program,
    });
    expect(extractJsonObject('Use {"x": } then {"unit":"line","rules":[]}')).toEqual({ ok: true, value: program });
  });

  it("prefers the program over a condition or example quoted before it", () => {
    const inline = 'Lines matching {"op":"endsWith","text":"."} get a suffix:\n{"unit":"line","rules":[]}';
    expect(extractJsonObject(inline)).toEqual({ ok: true, value: program });
    const exampleFence =
      'A condition:\n```json\n{"op":"isEmpty"}\n```\nProgram:\n```json\n{"unit":"line","rules":[]}\n```';
    expect(extractJsonObject(exampleFence)).toEqual({ ok: true, value: program });
    // A program that fails validation is still the program: its schema error is the useful feedback.
    const badProgram = '{"op":"isEmpty"} then {"unit":"line","rules":"none"}';
    expect(extractJsonObject(badProgram)).toEqual({ ok: true, value: { unit: "line", rules: "none" } });
  });

  it("takes the last program, so a restated worked example or a corrected draft isn't graded", () => {
    const worked =
      '{"unit":"line","rules":[],"otherwise":[{"op":"insert","text":"[","at":{"from":"start","offset":0}}]}';
    expect(extractJsonObject(`Like Example 1:\n${worked}\nFor this task:\n{"unit":"line","rules":[]}`)).toEqual({
      ok: true,
      value: program,
    });
    const fenced = "```json\n" + worked + "\n```\nProgram:\n```json\n" + JSON.stringify(program) + "\n```";
    expect(extractJsonObject(fenced)).toEqual({ ok: true, value: program });
    // A condition quoted after the program has no "unit" key, so it doesn't replace it.
    expect(extractJsonObject('{"unit":"line","rules":[]}\nIt never needs {"op":"isEmpty"}.')).toEqual({
      ok: true,
      value: program,
    });
  });

  it("reports a program opened after the last one that never closes, and flags any other unclosed object", () => {
    const worked = '{"unit":"line","rules":[],"otherwise":[]}';
    // The unclosed program is the answer, cut off or never finished: invalid, not the program before it.
    const unclosed = '{"unit":"line","rules":[{"when":';
    expect(extractJsonObject(`${worked}\nFor this task: ${unclosed}`)).toEqual({
      ok: false,
      error: tryParseError(unclosed),
    });
    // So is one that has only "rules" so far: any program key marks a program.
    const rulesOnly = '{"rules":[{"when":';
    expect(extractJsonObject(`${worked}\nFor this task: ${rulesOnly}`)).toEqual({
      ok: false,
      error: tryParseError(rulesOnly),
    });
    // An unclosed object with no program key yet may still be the cut-off answer, so it's flagged.
    expect(extractJsonObject(`${worked}\nFor this task: {"un`)).toEqual({
      ok: true,
      value: { unit: "line", rules: [], otherwise: [] },
      unfinished: true,
    });
    expect(extractJsonObject(`${worked}\nThis keeps every line`)).toEqual({
      ok: true,
      value: { unit: "line", rules: [], otherwise: [] },
    });
  });

  it("falls back to the first object that parses when none is a program", () => {
    expect(extractJsonObject('Use {"op":"isEmpty"} or {"op":"contains","text":"x"}')).toEqual({
      ok: true,
      value: { op: "isEmpty" },
    });
  });

  it("reports responses with no JSON object", () => {
    expect(extractJsonObject("I can't do that.")).toMatchObject({ ok: false });
    expect(extractJsonObject('{"unit": "line", "rules": [')).toMatchObject({ ok: false });
  });

  it("never returns an object nested inside a malformed program", () => {
    // Trailing comma: the program is balanced but isn't JSON. Its inner action must not be returned.
    const trailingComma =
      '{"unit":"line","rules":[],"otherwise":[{"op":"insert","text":"1","at":{"from":"end","offset":0}},]}';
    const r = extractJsonObject(trailingComma);
    expect(r).toMatchObject({ ok: false });
    expect(r.ok ? "" : r.error).toBe(tryParseError(trailingComma));
    // A malformed object in prose is skipped as a whole; the program after it is still found.
    expect(extractJsonObject('Not {"x": {"y": 1},} but {"unit":"line","rules":[]}')).toEqual({
      ok: true,
      value: program,
    });
  });

  it("reports a malformed final program instead of grading a valid draft before it", () => {
    const draft = '{"unit":"line","rules":[]}';
    const final = '{"unit":"line","rules":[],"otherwise":[{"op":"insert","text":"!","at":{"from":"end","offset":0}},]}';
    const error = tryParseError(final);
    const malformed = { ok: false, error, malformed: true };
    expect(extractJsonObject(`Draft:\n${draft}\nFinal answer:\n${final}`)).toEqual(malformed);
    const fenced = "```json\n" + draft + "\n```\nFinal answer:\n```json\n" + final + "\n```";
    expect(extractJsonObject(fenced)).toEqual(malformed);
    // A malformed program the model then corrected isn't the answer; the corrected one is.
    expect(extractJsonObject(`${final}\nFixed:\n${draft}`)).toEqual({ ok: true, value: program });
    // A malformed program beats a quoted condition as the thing to report: its error is the useful one.
    expect(extractJsonObject(`It needs {"op":"isEmpty"}:\n${final}`)).toEqual(malformed);
    // A final program missing its closing brace is invalid too, though the reply ended normally.
    const unclosed = final.slice(0, -1);
    expect(extractJsonObject(`Draft:\n${draft}\nFinal answer:\n${unclosed}`)).toEqual({
      ok: false,
      error: tryParseError(unclosed),
    });
  });

  it('takes a final program missing its "unit" key as the answer, not a valid draft before it', () => {
    const draft = '{"unit":"line","rules":[]}';
    const noUnit = { rules: [], otherwise: [{ op: "delete", at: { from: "end", offset: 1 }, count: 1 }] };
    expect(extractJsonObject(`Draft:\n${draft}\n\nFinal program:\n${JSON.stringify(noUnit)}`)).toEqual({
      ok: true,
      value: noUnit,
    });
    expect(extractJsonObject(`${draft}\nFinal: {"rules":[]}`)).toEqual({ ok: true, value: { rules: [] } });
    // A malformed one is reported, as one with "unit" is.
    expect(extractJsonObject(`${draft}\nFinal: {"rules":[],}`)).toMatchObject({ ok: false, malformed: true });
  });

  it("doesn't take a piece of the program quoted after it for the answer", () => {
    // "otherwise" is optional, so an object with only that key is a quoted piece, not a program.
    for (const after of [
      'The {"otherwise": []} part is left out, so no line changes.',
      'Lines that match no rule get {"otherwise":[{"op":"insert","text":"!","at":{"from":"end","offset":0}}]}.',
      'An unfinished {"otherwise": [',
    ])
      expect(extractJsonObject(`{"unit":"line","rules":[]}\n${after}`)).toMatchObject({ ok: true, value: program });
  });

  it("grades an array holding a program as it is, fenced, after prose or bare", () => {
    const arr = [program];
    for (const response of [
      JSON.stringify(arr),
      "```json\n" + JSON.stringify(arr) + "\n```",
      "Here it is:\n" + JSON.stringify(arr),
      "Here it is:\n```json\n" + JSON.stringify(arr, null, 2) + "\n```\nDone.",
      // The final answer, after a valid draft.
      `Draft: ${JSON.stringify(program)}\nFinal: ${JSON.stringify(arr)}`,
    ])
      expect(extractJsonObject(response)).toEqual({ ok: true, value: arr });
    // Lists in prose, and arrays without a program, still don't hide the program.
    expect(extractJsonObject('Steps [1, 2] and ["a"]:\n{"unit":"line","rules":[]}')).toEqual({
      ok: true,
      value: program,
    });
    expect(extractJsonObject('[Program: {"unit":"line","rules":[]}]')).toEqual({ ok: true, value: program });
  });

  it("reports an array of objects holding a program that isn't valid JSON, not the program inside it", () => {
    const p = JSON.stringify(program);
    for (const broken of [`[${p},]`, `[${p} ${p}]`, `[\n  ${p},\n]`]) {
      const error = tryParseError(broken);
      expect(extractJsonObject(broken)).toEqual({ ok: false, error, malformed: true });
      expect(extractJsonObject(`Draft: ${p}\nFinal: ${broken}`)).toEqual({ ok: false, error, malformed: true });
    }
    // Any JSON values may sit beside the program.
    for (const broken of [`[${p}, null,]`, `["note", ${p} 3]`, `[[1], ${p},]`]) {
      const error = tryParseError(broken);
      expect(extractJsonObject(`Draft: ${p}\nFinal: ${broken}`)).toEqual({ ok: false, error, malformed: true });
    }
    // An array the response ends inside is unfinished, not malformed, and its program isn't graded.
    for (const open of [`[${p}`, `[${p},`, `[${p}, {"unit":`, `[${p}, "a`]) {
      const error = tryParseError(open);
      expect(extractJsonObject(`Draft: ${p}\nFinal: ${open}`)).toEqual({ ok: false, error });
    }
    // A corrected program after it is the answer.
    expect(extractJsonObject(`[${p},]\nFixed: ${p}`)).toEqual({ ok: true, value: program });
    // Numbered notes and lists in prose are still looked past.
    expect(extractJsonObject(`[1] Keep every line:\n${p}`)).toEqual({ ok: true, value: program });
    expect(extractJsonObject(`Unclosed [note: ${p}`)).toEqual({ ok: true, value: program });
    // Prose in brackets isn't an array, so the program in it is still found.
    expect(extractJsonObject(`[See: ${p}, which keeps every line]`)).toEqual({ ok: true, value: program });
  });

  it("doesn't take a program described in prose, with placeholders, for the answer", () => {
    for (const prose of [
      'The outer {"unit": "line", "rules": [...]} object holds the rules.',
      'It has the form {"unit": "line", rules: [ … ]}.',
      'Every answer starts {"unit": "line", ...', // unclosed: only flagged, which counts at the token limit
    ])
      expect(extractJsonObject(`{"unit":"line","rules":[]}\n\n${prose}`)).toMatchObject({ ok: true, value: program });
    // An ellipsis inside a string is text, not a placeholder.
    const dots =
      '{"unit":"line","rules":[],"otherwise":[{"op":"insert","text":"...","at":{"from":"end","offset":0}},]}';
    expect(extractJsonObject(`{"unit":"line","rules":[]}\nFinal: ${dots}`)).toMatchObject({
      ok: false,
      malformed: true,
    });
  });
});

function tryParseError(s: string): string {
  try {
    JSON.parse(s);
    return "";
  } catch (e) {
    return (e as Error).message;
  }
}

describe("extractLine", () => {
  const line = (response: string) => extractLine(response)?.line;

  it("decodes a JSON string answer, including empty lines, tabs and edge spaces", () => {
    expect(line('"Apple1"')).toBe("Apple1");
    expect(line('""')).toBe("");
    expect(line('"a\\tb"')).toBe("a\tb");
    expect(line('" padded "\n')).toBe(" padded ");
  });

  it("ignores prose around the JSON string, as program mode does around the program", () => {
    expect(line('"[x]"\n\nThis wraps the line in brackets.')).toBe("[x]");
    expect(line('"[x]" is the reformatted line.')).toBe("[x]");
    expect(line('Answer: "[x]"')).toBe("[x]");
    expect(line('Sure, here it is:\n  "[x]"\n')).toBe("[x]");
    expect(line('"say \\"hi\\"" (quotes escaped)')).toBe('say "hi"');
  });

  it("falls back to the raw line minus one trailing newline", () => {
    expect(line("Apple1\n")).toBe("Apple1");
    expect(line("Apple1\n\n")).toBe("Apple1\n");
    expect(line('"unterminated')).toBe('"unterminated');
    // Quotes inside a raw answer aren't mistaken for a JSON answer.
    expect(line('She said "hi" today')).toBe('She said "hi" today');
  });

  it("unwraps a code fence", () => {
    expect(line('```\n"[x]"\n```')).toBe("[x]");
    expect(line('```text \r\n"[x]"\r\n```')).toBe("[x]");
  });

  it("keeps whitespace-only payloads inside a fence", () => {
    expect(line("```\n\t\n```")).toBe("\t");
    expect(line("```\n\t\t\n```")).toBe("\t\t");
    expect(line("```\n  padded\n```")).toBe("  padded");
  });

  it("finds the answer before an unrelated code fence, as program mode does", () => {
    expect(line('"[x]"\n\nFor example:\n```\nwrap(line)\n```')).toBe("[x]");
    // A JSON string inside the fence still wins, like a fenced program does.
    expect(line('Draft: "[y]"\n```\n"[x]"\n```')).toBe("[x]");
  });

  it("accepts markdown labels, bullets and backticks around the answer", () => {
    expect(line('**Answer:** "[x]"')).toBe("[x]");
    expect(line('**Answer**: "[x]"')).toBe("[x]");
    expect(line('_Output_: `"[x]"`')).toBe("[x]");
    expect(line('- "[x]"')).toBe("[x]");
    expect(line('* **Answer:** "[x]"')).toBe("[x]");
    expect(line('1. "[x]"')).toBe("[x]");
    expect(line('2) **Answer:** "[x]"')).toBe("[x]");
  });

  it("doesn't take a restated input for the answer", () => {
    expect(line('Line to reformat: "abc"\nAnswer: "[abc]"')).toBe("[abc]");
    expect(line('Line to reformat: "abc"\n"[abc]"')).toBe("[abc]");
    expect(line('**Input:** "abc"\n**Output:** "[abc]"')).toBe("[abc]");
    // Restated examples come before the answer; the Answer label picks it out.
    expect(line('1. input: "a"\n   output: "[a]"\nLine to reformat: "abc"\nAnswer: "[abc]"')).toBe("[abc]");
    // Only a restated input and nothing else: no answer, so the raw text is scored (and fails).
    expect(line('Line to reformat: "abc"')).toBe('Line to reformat: "abc"');
  });

  it("treats variations on the input labels as restated inputs", () => {
    expect(line('Input line: "abc"\n"[abc]"')).toBe("[abc]");
    expect(line('Line: "abc"\nOutput: "[abc]"')).toBe("[abc]");
    expect(line('1. Input text: "abc"\n2. "[abc]"')).toBe("[abc]");
    // A label that only contains "line" later on isn't an input label.
    expect(line('Reformatted line: "[abc]"')).toBe("[abc]");
  });

  it("takes the last Answer, after a restated worked example", () => {
    const restated = 'Line to reformat: "hello world"\nAnswer: "[hello world]"\n\n';
    expect(line(restated + 'Line to reformat: "abc"\nAnswer: "[abc]"')).toBe("[abc]");
    // An Answer label still beats an earlier unlabelled string.
    expect(line('"draft"\nAnswer: "[abc]"')).toBe("[abc]");
    // ...and beats a worked example restated in a fence before it.
    const fenced = '```\nLine to reformat: "hello world"\nAnswer: "[hello world]"\n```\n';
    expect(line(fenced + 'Answer: "[abc]"')).toBe("[abc]");
    expect(line('For example:\n```\n"[hello world]"\n```\nAnswer: "[abc]"')).toBe("[abc]");
  });

  it("passes over a restated worked answer for the model's own, before or after it", () => {
    const worked = "[hello world]";
    const echo = 'Line to reformat: "hello world"\nAnswer: "[hello world]"\n\n';
    for (const [response, answer] of [
      [echo + '""', ""],
      [echo + 'Line to reformat: "xyz"\n"x"', "x"],
      ['Answer: "[hello world]"\n""', ""],
      [echo + '```\n"x"\n```', "x"],
      // The model's own labelled answer before a restatement at the end, however that's laid out.
      ['Answer: "x"\n\nAs in the example:\nAnswer: "[hello world]"', "x"],
      ['Answer: "x"\n\nAnswer:\n\n"[hello world]"', "x"],
      ['Answer: "x"\n\nAnswer:\n```\n"[hello world]"\n```', "x"],
      ['Answer: "x"\n\n**Answer:** `"[hello world]"`', "x"],
    ])
      expect(extractLine(response!, [worked])).toEqual({ line: answer, quoted: true });
    // Without the worked answer, the restated one is taken (the grader passes it only for other lines).
    expect(extractLine(echo + '""')).toEqual({ line: worked, quoted: true });
    // A worked answer with no other answer stays the answer, and is graded like any other.
    expect(extractLine('Answer: "[hello world]"', [worked])).toEqual({ line: worked, quoted: true });
    expect(extractLine(echo + "x", [worked])).toEqual({ line: worked, quoted: true });
    // Other answers are unaffected.
    expect(extractLine('Answer: "[abc]"\n"[hello world]"', [worked])).toEqual({ line: "[abc]", quoted: true });
    // With several worked examples, a restatement of any of them is passed over.
    const two = 'Answer: "[hello world]"\nAnswer: "HELLO"\n"x"';
    expect(extractLine(two, [worked, "HELLO"])).toEqual({ line: "x", quoted: true });
    expect(extractLine(two, [worked])).toEqual({ line: "HELLO", quoted: true });
  });

  it("reads only what follows the last Answer label", () => {
    // Cut off at the answer's opening quote: a restated example's "" before it isn't the answer.
    const restated = '1. input: "x - y"\n   output: ""\nLine to reformat: "abc"\nAnswer: "';
    expect(extractLine(restated)).toEqual({ line: '"', quoted: false });
    expect(extractLine('output: ""\nAnswer: "')).toEqual({ line: '"', quoted: false });
    // The answer on the line below the label, quoted, fenced or raw.
    expect(line('Answer:\n"[abc]"')).toBe("[abc]");
    expect(line('Answer:\n```\n"[abc]"\n```')).toBe("[abc]");
    expect(extractLine("Answer: [abc]\n")).toEqual({ line: "[abc]", quoted: false });
    // Strings after the answer are explanation, not the answer.
    expect(line('Answer: "[abc]"\n\nFor example, "a" becomes "[a]":\n"[a]"')).toBe("[abc]");
    // Nothing after the label is no answer.
    expect(extractLine('1. input: "a"\n   output: ""\nAnswer:')).toBeUndefined();
    expect(extractLine("Answer:  \n\n")).toBeUndefined();
  });

  it("skips shown examples restated in the prompt's numbered format when there's no label", () => {
    expect(line('1. input: "a"\n   output: ""\n2. input: "b"\n   output: "[b]"\n"[abc]"')).toBe("[abc]");
    // Restated examples and nothing else: no answer, so the raw text is scored (and fails).
    expect(extractLine('1. input: "a"\n   output: ""')).toEqual({
      line: '1. input: "a"\n   output: ""',
      quoted: false,
    });
    // An unnumbered input then output is the model's own answer, restating the line it was given.
    expect(line('Input: "abc"\nOutput: ""')).toBe("");
  });

  it("says whether the answer was a JSON string or raw text", () => {
    expect(extractLine('Answer: "[x]"')).toEqual({ line: "[x]", quoted: true });
    expect(extractLine('""')).toEqual({ line: "", quoted: true });
    expect(extractLine("[x]\n")).toEqual({ line: "[x]", quoted: false });
  });

  it("finds no answer in an empty or blank response", () => {
    expect(extractLine("")).toBeUndefined();
    expect(extractLine(" \n\t\n")).toBeUndefined();
  });

  it("finds no answer in an empty code block: an empty line is answered with a JSON string", () => {
    for (const empty of [
      "Answer:\n```\n```",
      "Answer: ```\n```",
      "```\n```",
      "```text\n\n```",
      "Answer:\nHere:\n```\n```",
      // Nothing but blank lines, however many, is empty too.
      "```\n\n\n```",
      "Answer:\n```\n\n\n```",
      "```\r\n\r\n\r\n\r\n```",
    ])
      expect(extractLine(empty)).toBeUndefined();
    expect(extractLine('Answer:\n```\n""\n```')).toEqual({ line: "", quoted: true });
    // A code block holding a tab or spaces isn't empty: the fence shows exactly what's in it.
    expect(extractLine("Answer:\n```\n\t\n```")).toEqual({ line: "\t", quoted: false });
  });

  it("says why there was no answer", () => {
    expect(noLineReason(" \n")).toBe("the response was empty");
    expect(noLineReason('1. input: "a"\n   output: ""\nAnswer:')).toBe("nothing followed the Answer: label");
    for (const empty of ["Answer:\n```\n```", "```\n```"])
      expect(noLineReason(empty)).toBe('the answer was an empty code block (an empty line is answered "")');
  });
});
