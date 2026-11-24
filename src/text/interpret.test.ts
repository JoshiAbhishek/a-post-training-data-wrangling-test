import { describe, expect, it } from "vitest";
import { LineTooLongError, foldChar, maxLineLength, run, runLine, splitLines } from "./interpret.ts";
import { parseProgram } from "./schema.ts";
import type { Action, CharClass, Cond, Pos, Program } from "./types.ts";

// ---------- helpers ----------
const ADD_X: Action = { op: "insert", text: "X", at: { from: "end", offset: 0 } };
/** Does the condition match the line? (appends "X" when it does) */
const matches = (when: Cond, line: string) =>
  runLine({ unit: "line", rules: [{ when, then: [ADD_X] }] }, line) !== line;
/** Run actions unconditionally on one line. */
const apply = (line: string, ...then: Action[]) => runLine({ unit: "line", rules: [], otherwise: then }, line);
const insertAt = (line: string, at: Pos, text = "|") => apply(line, { op: "insert", text, at });

describe("conditions", () => {
  it("startsWith / endsWith / contains / equals, case-sensitive by default", () => {
    expect(matches({ op: "startsWith", text: "Ab" }, "Abc")).toBe(true);
    expect(matches({ op: "startsWith", text: "ab" }, "Abc")).toBe(false);
    expect(matches({ op: "startsWith", text: "Abcd" }, "Abc")).toBe(false);
    expect(matches({ op: "endsWith", text: "bc" }, "Abc")).toBe(true);
    expect(matches({ op: "endsWith", text: "BC" }, "Abc")).toBe(false);
    expect(matches({ op: "endsWith", text: "xAbc" }, "Abc")).toBe(false);
    expect(matches({ op: "contains", text: "b" }, "Abc")).toBe(true);
    expect(matches({ op: "contains", text: "B" }, "Abc")).toBe(false);
    expect(matches({ op: "equals", text: "Abc" }, "Abc")).toBe(true);
    expect(matches({ op: "equals", text: "Ab" }, "Abc")).toBe(false);
  });

  it("ignoreCase on every text condition", () => {
    expect(matches({ op: "startsWith", text: "ab", ignoreCase: true }, "ABC")).toBe(true);
    expect(matches({ op: "endsWith", text: "BC", ignoreCase: true }, "abc")).toBe(true);
    expect(matches({ op: "contains", text: "B", ignoreCase: true }, "abc")).toBe(true);
    expect(matches({ op: "equals", text: "abc", ignoreCase: true }, "AbC")).toBe(true);
  });

  it("character classes are Unicode-aware", () => {
    const sw = (c: CharClass, line: string) => matches({ op: "startsWithClass", class: c }, line);
    expect(["a", "é", "中", "ß", "Ω"].map((s) => sw("letter", s))).toEqual([true, true, true, true, true]);
    expect(["1", "_", " "].map((s) => sw("letter", s))).toEqual([false, false, false]);
    expect(["A", "É", "Ω"].map((s) => sw("upper", s))).toEqual([true, true, true]);
    expect(["a", "ß", "中"].map((s) => sw("upper", s))).toEqual([false, false, false]);
    expect(["a", "é", "ß"].map((s) => sw("lower", s))).toEqual([true, true, true]);
    expect(["A", "中"].map((s) => sw("lower", s))).toEqual([false, false]);
    expect(["7", "٣", "३"].map((s) => sw("digit", s))).toEqual([true, true, true]); // ASCII, Arabic-Indic, Devanagari
    expect(["½", "a"].map((s) => sw("digit", s))).toEqual([false, false]);
    expect([" ", "\t", " "].map((s) => sw("space", s))).toEqual([true, true, true]);
    expect(["!", "¿", "«", "-"].map((s) => sw("punct", s))).toEqual([true, true, true, true]);
    expect(["$", "+", "a"].map((s) => sw("punct", s))).toEqual([false, false, false]);
    expect(matches({ op: "endsWithClass", class: "digit" }, "item ٣")).toBe(true);
    expect(matches({ op: "endsWithClass", class: "digit" }, "item 3a")).toBe(false);
  });

  it("class conditions are false on an empty line", () => {
    expect(matches({ op: "startsWithClass", class: "letter" }, "")).toBe(false);
    expect(matches({ op: "endsWithClass", class: "space" }, "")).toBe(false);
  });

  it("lengthGt / lengthLt count code points", () => {
    expect(matches({ op: "lengthGt", n: 2 }, "abc")).toBe(true);
    expect(matches({ op: "lengthGt", n: 3 }, "abc")).toBe(false);
    expect(matches({ op: "lengthLt", n: 3 }, "ab")).toBe(true);
    expect(matches({ op: "lengthLt", n: 3 }, "é中😀")).toBe(false); // 3 code points, 4 UTF-16 units
    expect(matches({ op: "lengthGt", n: 2 }, "é中😀")).toBe(true);
  });

  it("isEmpty, and, or, not", () => {
    expect(matches({ op: "isEmpty" }, "")).toBe(true);
    expect(matches({ op: "isEmpty" }, " ")).toBe(false);
    const a: Cond = { op: "startsWith", text: "a" };
    const z: Cond = { op: "endsWith", text: "z" };
    expect(matches({ op: "and", args: [a, z] }, "abz")).toBe(true);
    expect(matches({ op: "and", args: [a, z] }, "abc")).toBe(false);
    expect(matches({ op: "or", args: [a, z] }, "xbz")).toBe(true);
    expect(matches({ op: "or", args: [a, z] }, "xbc")).toBe(false);
    expect(matches({ op: "not", arg: a }, "xbc")).toBe(true);
    expect(matches({ op: "not", arg: { op: "isEmpty" } }, "")).toBe(false);
  });
});

describe("positions", () => {
  it("from start / end, clamped to the line", () => {
    expect(insertAt("abc", { from: "start", offset: 0 })).toBe("|abc");
    expect(insertAt("abc", { from: "start", offset: 2 })).toBe("ab|c");
    expect(insertAt("abc", { from: "start", offset: 10 })).toBe("abc|");
    expect(insertAt("abc", { from: "end", offset: 0 })).toBe("abc|");
    expect(insertAt("abc", { from: "end", offset: 1 })).toBe("ab|c");
    expect(insertAt("abc", { from: "end", offset: 10 })).toBe("|abc");
    expect(insertAt("", { from: "end", offset: 3 })).toBe("|");
  });

  it("anchors before / after the first match by default", () => {
    expect(insertAt("a:b:c", { anchor: "before", text: ":" })).toBe("a|:b:c");
    expect(insertAt("a:b:c", { anchor: "after", text: ":" })).toBe("a:|b:c");
    expect(insertAt("a::b", { anchor: "after", text: "::" })).toBe("a::|b");
  });

  it("occurrence 1, N and -1; out of range does nothing", () => {
    const at = (occurrence: number) => insertAt("a:b:c:d", { anchor: "after", text: ":", occurrence });
    expect(at(1)).toBe("a:|b:c:d");
    expect(at(2)).toBe("a:b:|c:d");
    expect(at(3)).toBe("a:b:c:|d");
    expect(at(-1)).toBe("a:b:c:|d");
    expect(at(4)).toBe("a:b:c:d");
  });

  it("occurrences are non-overlapping", () => {
    expect(insertAt("aaaa", { anchor: "before", text: "aa", occurrence: 2 })).toBe("aa|aa");
    expect(insertAt("aaa", { anchor: "before", text: "aa", occurrence: 2 })).toBe("aaa");
  });

  it("a missing anchor makes the action do nothing", () => {
    expect(insertAt("abc", { anchor: "before", text: "x" })).toBe("abc");
    expect(apply("abc", { op: "delete", at: { anchor: "after", text: "x" }, count: 1 })).toBe("abc");
    expect(apply("abc", { op: "replaceSpan", at: { anchor: "after", text: "x" }, count: 1, with: "Z" })).toBe("abc");
  });

  it("anchors can ignore case", () => {
    expect(insertAt("Key=Value", { anchor: "after", text: "KEY", ignoreCase: true })).toBe("Key|=Value");
    expect(insertAt("Key=Value", { anchor: "after", text: "KEY" })).toBe("Key=Value");
  });
});

describe("actions", () => {
  it("insert", () => {
    expect(apply("world", { op: "insert", text: "hello ", at: { from: "start", offset: 0 } })).toBe("hello world");
    expect(apply("abc", { op: "insert", text: "", at: { from: "start", offset: 1 } })).toBe("abc");
  });

  it("delete N characters, clamped", () => {
    expect(apply("abcdef", { op: "delete", at: { from: "start", offset: 1 }, count: 2 })).toBe("adef");
    expect(apply("abcdef", { op: "delete", at: { from: "end", offset: 2 }, count: 2 })).toBe("abcd");
    expect(apply("abc", { op: "delete", at: { from: "start", offset: 0 }, count: 5 })).toBe("");
    expect(apply("abc", { op: "delete", at: { from: "start", offset: 2 }, count: 5 })).toBe("ab");
    expect(apply("abc", { op: "delete", at: { from: "end", offset: 0 }, count: 2 })).toBe("abc");
    expect(apply("abc", { op: "delete", at: { from: "start", offset: 1 }, count: 0 })).toBe("abc");
  });

  it("replaceSpan, clamped; an empty replacement deletes", () => {
    expect(apply("2024-01-05", { op: "replaceSpan", at: { from: "start", offset: 4 }, count: 1, with: "/" })).toBe(
      "2024/01-05",
    );
    expect(apply("abc", { op: "replaceSpan", at: { from: "start", offset: 1 }, count: 9, with: "Z" })).toBe("aZ");
    expect(apply("abc", { op: "replaceSpan", at: { from: "start", offset: 0 }, count: 1, with: "" })).toBe("bc");
    expect(apply("abc", { op: "replaceSpan", at: { from: "end", offset: 0 }, count: 1, with: "!" })).toBe("abc!");
  });

  it("deleteText: all omitted, false and true", () => {
    expect(apply("a-b-c", { op: "deleteText", text: "-" })).toBe("ab-c");
    expect(apply("a-b-c", { op: "deleteText", text: "-", all: false })).toBe("ab-c");
    expect(apply("a-b-c", { op: "deleteText", text: "-", all: true })).toBe("abc");
    expect(apply("abc", { op: "deleteText", text: "x", all: true })).toBe("abc");
    expect(apply("A-a-A", { op: "deleteText", text: "a", ignoreCase: true, all: true })).toBe("--");
  });

  it("replaceText: all omitted, false and true", () => {
    expect(apply("a.b.c", { op: "replaceText", find: ".", with: "/" })).toBe("a/b.c");
    expect(apply("a.b.c", { op: "replaceText", find: ".", with: "/", all: false })).toBe("a/b.c");
    expect(apply("a.b.c", { op: "replaceText", find: ".", with: "/", all: true })).toBe("a/b/c");
  });

  it("matches are non-overlapping and replaced text is never rescanned", () => {
    expect(apply("aaaa", { op: "replaceText", find: "aa", with: "a", all: true })).toBe("aa");
    expect(apply("aaa", { op: "replaceText", find: "aa", with: "b", all: true })).toBe("ba");
    expect(apply("aa", { op: "replaceText", find: "a", with: "aa", all: true })).toBe("aaaa");
    expect(apply("ab", { op: "replaceText", find: "a", with: "ab", all: true })).toBe("abb");
  });

  it("ignoreCase never changes the case of untouched text", () => {
    expect(apply("Hello HELLO hello", { op: "replaceText", find: "hello", with: "hi", ignoreCase: true })).toBe(
      "hi HELLO hello",
    );
    expect(
      apply("Hello HELLO", {
        op: "insert",
        text: "!",
        at: { anchor: "after", text: "hello", occurrence: 2, ignoreCase: true },
      }),
    ).toBe("Hello HELLO!");
  });
});

describe("case folding", () => {
  it("folds per code point without changing length", () => {
    for (const c of ["A", "é", "É", "σ", "ς", "Σ", "ß", "ẞ", "İ", "ı", "ǅ", "😀"])
      expect(Array.from(foldChar(c))).toHaveLength(1);
  });

  it("é ≈ É", () => {
    expect(apply("ÉCOLE école", { op: "replaceText", find: "école", with: "x", ignoreCase: true, all: true })).toBe(
      "x x",
    );
  });

  it("σ ≈ ς ≈ Σ", () => {
    expect(matches({ op: "equals", text: "σοφος", ignoreCase: true }, "ΣΟΦΟΣ")).toBe(true);
    expect(matches({ op: "contains", text: "ς", ignoreCase: true }, "Σ")).toBe(true);
    expect(matches({ op: "contains", text: "σ", ignoreCase: false }, "ς")).toBe(false);
  });

  it("ß ≈ ẞ, but ß does not match ss", () => {
    expect(matches({ op: "contains", text: "ß", ignoreCase: true }, "STRAẞE")).toBe(true);
    expect(matches({ op: "contains", text: "ss", ignoreCase: true }, "Straße")).toBe(false);
    expect(apply("Straße", { op: "deleteText", text: "SS", ignoreCase: true })).toBe("Straße");
  });

  it("İ matches only İ", () => {
    expect(matches({ op: "startsWith", text: "i", ignoreCase: true }, "İzmir")).toBe(false);
    expect(matches({ op: "startsWith", text: "I", ignoreCase: true }, "İzmir")).toBe(false);
    expect(matches({ op: "startsWith", text: "İ", ignoreCase: true }, "İzmir")).toBe(true);
    // Still a plain i/I match elsewhere in the same line.
    expect(apply("İzmir Istanbul", { op: "replaceText", find: "i", with: "_", ignoreCase: true, all: true })).toBe(
      "İzm_r _stanbul",
    );
  });

  it("ı matches only ı, and I still matches i", () => {
    expect(matches({ op: "contains", text: "i", ignoreCase: true }, "kırmızı")).toBe(false);
    expect(matches({ op: "contains", text: "I", ignoreCase: true }, "kırmızı")).toBe(false);
    expect(matches({ op: "contains", text: "ı", ignoreCase: true }, "KIRMIZI")).toBe(false);
    expect(matches({ op: "contains", text: "ı", ignoreCase: true }, "kırmızı")).toBe(true);
    expect(apply("kırmızı Ip", { op: "replaceText", find: "i", with: "_", ignoreCase: true, all: true })).toBe(
      "kırmızı _p",
    );
  });

  it("ΐ ≈ ΐ, ΰ ≈ ΰ and ﬅ ≈ ﬆ, which have no single-character uppercase", () => {
    // U+1FD3 and U+0390, U+1FE3 and U+03B0, U+FB05 and U+FB06.
    expect(matches({ op: "contains", text: "ΐ", ignoreCase: true }, "ΐ")).toBe(true);
    expect(matches({ op: "contains", text: "ΐ", ignoreCase: true }, "ΐ")).toBe(true);
    expect(matches({ op: "contains", text: "ΰ", ignoreCase: true }, "ΰ")).toBe(true);
    expect(matches({ op: "equals", text: "ﬅ", ignoreCase: true }, "ﬆ")).toBe(true);
    expect(matches({ op: "equals", text: "ﬅ", ignoreCase: false }, "ﬆ")).toBe(false);
    // Still single letters: neither ligature matches "st".
    expect(matches({ op: "contains", text: "st", ignoreCase: true }, "ﬅﬆ")).toBe(false);
  });

  // JavaScript's /iu matching compares characters by Unicode simple case folding, so two characters
  // match each other under it exactly when simple folding maps them to the same one. Checked against
  // the runtime's own Unicode data, from the version foldChar was last checked against on: a later
  // version that adds a folding pair fails here, which is the signal to add it.
  it.skipIf(Number.parseFloat(process.versions.unicode ?? "0") < 17)(
    "agrees with Unicode simple case folding for every character with case",
    () => {
      const cased = /[\p{Cased}\p{CWCM}\p{CWCF}\p{CWL}\p{CWU}\p{CWT}]/u;
      const all: string[] = [];
      for (let cp = 0; cp <= 0x10ffff; cp++) {
        if (cp >= 0xd800 && cp <= 0xdfff) continue;
        const c = String.fromCodePoint(cp);
        if (cased.test(c) || foldChar(c) !== c) all.push(c);
      }
      const hex = (c: string) => `U+${c.codePointAt(0)!.toString(16).toUpperCase().padStart(4, "0")}`;
      const disagreements: string[] = [];
      for (const [i, c] of all.entries()) {
        const sameAs = new RegExp(`^\\u{${c.codePointAt(0)!.toString(16)}}$`, "iu");
        for (const d of all.slice(i + 1))
          if (sameAs.test(d) !== (foldChar(c) === foldChar(d)))
            disagreements.push(`${hex(c)} ${hex(d)}: simple folding says ${sameAs.test(d) ? "same" : "different"}`);
      }
      expect(all.length).toBeGreaterThan(4000);
      expect(disagreements).toEqual([]);
    },
    // About ten million comparisons: well under two seconds on recent hardware, but far over the default
    // five seconds is possible on a slower Mac running the other test files at the same time.
    60_000,
  );

  it("edits land on the original code points", () => {
    expect(apply("xÉy", { op: "insert", text: "|", at: { anchor: "after", text: "é", ignoreCase: true } })).toBe(
      "xÉ|y",
    );
    expect(apply("STRAẞE", { op: "replaceText", find: "ß", with: "SS", ignoreCase: true })).toBe("STRASSE");
  });
});

describe("characters are code points", () => {
  it("é, 中 and an emoji each count as one", () => {
    expect(insertAt("é中😀x", { from: "start", offset: 3 })).toBe("é中😀|x");
    expect(apply("é中😀x", { op: "delete", at: { from: "start", offset: 2 }, count: 1 })).toBe("é中x");
    expect(apply("😀😀", { op: "delete", at: { from: "end", offset: 1 }, count: 1 })).toBe("😀");
  });

  it("never splits a surrogate pair", () => {
    const out = apply("a😀b", { op: "replaceSpan", at: { from: "start", offset: 1 }, count: 1, with: "_" });
    expect(out).toBe("a_b");
    expect(out.isWellFormed()).toBe(true);
  });

  it("multi-code-point emoji count as several", () => {
    // 🇫🇷 is two regional-indicator code points.
    expect(apply("🇫🇷", { op: "delete", at: { from: "start", offset: 0 }, count: 1 })).toBe("🇷");
  });
});

describe("rules", () => {
  const program: Program = {
    unit: "line",
    rules: [
      { when: { op: "startsWith", text: "a" }, then: [{ op: "insert", text: "1", at: { from: "end", offset: 0 } }] },
      { when: { op: "contains", text: "a" }, then: [{ op: "insert", text: "2", at: { from: "end", offset: 0 } }] },
    ],
    otherwise: [{ op: "insert", text: "-", at: { from: "start", offset: 0 } }],
  };

  it("first matching rule wins", () => {
    expect(runLine(program, "apple")).toBe("apple1"); // matches both rules, first wins
    expect(runLine(program, "banana")).toBe("banana2");
  });

  it("unmatched lines go to otherwise, or stay unchanged without it", () => {
    expect(runLine(program, "kiwi")).toBe("-kiwi");
    expect(runLine({ ...program, otherwise: undefined }, "kiwi")).toBe("kiwi");
  });

  it("actions run left to right on the evolving line", () => {
    expect(
      apply(
        "abc",
        { op: "insert", text: "X", at: { from: "start", offset: 0 } },
        { op: "insert", text: "Y", at: { anchor: "after", text: "X" } },
        { op: "replaceText", find: "XY", with: "!" },
      ),
    ).toBe("!abc");
  });
});

describe("documents and line endings", () => {
  const identity: Program = { unit: "line", rules: [] };
  const addX: Program = { unit: "line", rules: [], otherwise: [ADD_X] };

  it("round-trips \\n, \\r\\n and mixed endings unchanged when nothing is edited", () => {
    for (const doc of ["a\nb\nc\n", "a\r\nb\r\nc\r\n", "a\r\nb\nc\r\n\n", "a\nb", "\n\n", "\r\n", "x"]) {
      expect(run(identity, doc)).toBe(doc);
    }
  });

  it("rules never see \\r and each line keeps its ending", () => {
    expect(run(addX, "a\r\nb\nc")).toBe("aX\r\nbX\ncX");
    expect(run({ unit: "line", rules: [{ when: { op: "endsWith", text: "a" }, then: [ADD_X] }] }, "a\r\n")).toBe(
      "aX\r\n",
    );
  });

  it("a missing final newline stays missing; a trailing newline adds no line", () => {
    expect(run(addX, "a\nb")).toBe("aX\nbX");
    expect(run(addX, "a\nb\n")).toBe("aX\nbX\n");
    expect(splitLines("a\n")).toEqual([{ text: "a", eol: "\n" }]);
  });

  it("empty document and empty lines", () => {
    expect(run(addX, "")).toBe("");
    expect(splitLines("")).toEqual([]);
    expect(run(addX, "\n")).toBe("X\n");
    const markEmpty: Program = {
      unit: "line",
      rules: [{ when: { op: "isEmpty" }, then: [{ op: "insert", text: "<empty>", at: { from: "start", offset: 0 } }] }],
    };
    expect(run(markEmpty, "a\n\nb\r\n\r\n")).toBe("a\n<empty>\nb\r\n<empty>\r\n");
  });
});

describe("line length limit", () => {
  const doubleA: Action = { op: "replaceText", find: "a", with: "aa", all: true };

  it("is 10,000 characters, or 10 times the input line if that is longer", () => {
    expect(maxLineLength(0)).toBe(10_000);
    expect(maxLineLength(1_000)).toBe(10_000);
    expect(maxLineLength(5_000)).toBe(50_000);
  });

  it("an action that would grow the line past it stops with LineTooLongError, quickly", () => {
    const started = performance.now();
    expect(() => apply("a", ...Array<Action>(16).fill(doubleA))).toThrow(LineTooLongError);
    expect(performance.now() - started).toBeLessThan(500);
    expect(apply("a", ...Array<Action>(13).fill(doubleA))).toHaveLength(8_192); // 2^13, under the limit
    expect(() => apply("a", ...Array<Action>(14).fill(doubleA))).toThrow(
      "an action would make the line longer than 10,000 characters, its limit",
    );
  });

  it("applies to every action that adds text", () => {
    const big = "x".repeat(10_000);
    expect(apply("", { op: "insert", text: big, at: { from: "start", offset: 0 } })).toHaveLength(10_000);
    expect(() => apply("", { op: "insert", text: big + "x", at: { from: "start", offset: 0 } })).toThrow(
      LineTooLongError,
    );
    expect(() => apply("ab", { op: "replaceSpan", at: { from: "start", offset: 0 }, count: 1, with: big })).toThrow(
      LineTooLongError,
    );
    expect(
      apply("ab", { op: "replaceSpan", at: { from: "start", offset: 0 }, count: 1, with: big.slice(1) }),
    ).toHaveLength(10_000);
  });

  it("scales with a long input line, and run() lets the error through", () => {
    const long = "a".repeat(2_000);
    expect(apply(long, doubleA, doubleA, doubleA)).toHaveLength(16_000);
    expect(() => apply(long, doubleA, doubleA, doubleA, doubleA)).toThrow(LineTooLongError); // 32,000 > 20,000
    // "ok" is untouched; doubling "a" 14 times passes its 10,000 limit.
    const doubler: Program = { unit: "line", rules: [], otherwise: Array<Action>(14).fill(doubleA) };
    expect(() => run(doubler, "ok\na\n")).toThrow(LineTooLongError);
  });

  it("replacing every match takes time in proportion to the line", () => {
    const line = "ab".repeat(50_000); // 50,000 matches; copying the line per match took seconds
    const started = performance.now();
    const out = apply(line, { op: "replaceText", find: "a", with: "c", all: true });
    expect(performance.now() - started).toBeLessThan(1_000);
    expect(out).toBe("cb".repeat(50_000));
  });
});

describe("end to end", () => {
  it("A→1 / B→2 example: JSON → schema → run", () => {
    const json = `{ "unit": "line", "rules": [
      { "when": { "op": "startsWith", "text": "a", "ignoreCase": true }, "then": [{ "op": "insert", "text": "1", "at": { "from": "end", "offset": 0 } }] },
      { "when": { "op": "startsWith", "text": "b", "ignoreCase": true }, "then": [{ "op": "insert", "text": "2", "at": { "from": "end", "offset": 0 } }] }
    ]}`;
    const parsed = parseProgram(json);
    if (!parsed.ok) throw new Error(parsed.errors.join("\n"));
    expect(run(parsed.program, "Apple\nbanana\ncherry\nAvocado\nBerry\n")).toBe(
      "Apple1\nbanana2\ncherry\nAvocado1\nBerry2\n",
    );
  });
});
