import { describe, expect, it } from "vitest";
import type { RewriteCall, RewriteTaskResult } from "./harness.ts";
import { fmtCounts, incompleteSummary, rewriteMetrics, rewriteSummary } from "./metrics.ts";

describe("incompleteSummary", () => {
  it("records the error and the run settings in the same table format as a complete summary", () => {
    const meta = { model: "m", prompt: "a|b", quantization: undefined };
    const md = incompleteSummary("program run for m", meta, { written: 4, error: "server went away" });
    expect(md.split("\n")).toEqual([
      "# INCOMPLETE: program run for m",
      "",
      "The run stopped after 4 task result(s), saved in attempts.jsonl.",
      "",
      "Error: server went away",
      "",
      "| Setting | Value |",
      "| --- | --- |",
      "| model | m |",
      "| prompt | a\\|b |",
      "",
    ]);
  });

  it("without an error, is the placeholder that marks a run in progress or interrupted", () => {
    const md = incompleteSummary("program run for m", { model: "m" });
    expect(md).toMatch(/^# INCOMPLETE: program run for m\n\nThe run is in progress, or was interrupted/);
    expect(md).toContain("| model | m |");
  });
});

describe("fmtCounts (one count per seed)", () => {
  it("same denominator: mean count, with a range when seeds disagree", () => {
    expect(fmtCounts([{ n: 14, of: 27 }])).toBe("14/27 (51.9%)");
    expect(fmtCounts([14, 14, 14].map((n) => ({ n, of: 27 })))).toBe("14/27 (51.9%)");
    expect(fmtCounts([12, 14, 16].map((n) => ({ n, of: 27 })))).toBe("14.0/27 (51.9%), range 12–16");
  });

  it("different denominators: mean of each seed's rate, never numerator over seed 1's denominator", () => {
    // Both seeds are 100% overfit; the old formula reported (1+3)/2 / 1 = 200%.
    expect(
      fmtCounts([
        { n: 1, of: 1 },
        { n: 3, of: 3 },
      ]),
    ).toBe("100.0% mean of seed rates (per seed: 1/1, 3/3)");
    expect(
      fmtCounts([
        { n: 1, of: 2 },
        { n: 3, of: 4 },
      ]),
    ).toBe("62.5% mean of seed rates, range 50.0%–75.0% (per seed: 1/2, 3/4)");
  });

  it("leaves seeds with a zero denominator out of the average", () => {
    expect(
      fmtCounts([
        { n: 0, of: 0 },
        { n: 1, of: 4 },
      ]),
    ).toBe("25.0% mean of seed rates (per seed: 0/0, 1/4)");
    expect(fmtCounts([{ n: 0, of: 0 }])).toBe("0/0 (–)");
    expect(fmtCounts([])).toBe("–");
  });
});

describe("rewrite summary", () => {
  const call = (over: Partial<RewriteCall>): RewriteCall => ({
    input: "a",
    expected: "a!",
    got: "a!",
    correct: true,
    userMessage: "",
    response: '"a!"',
    finishReason: "stop",
    latencyMs: 1000,
    promptTokens: 40,
    completionTokens: 5,
    ...over,
  });
  const result = (calls: RewriteCall[]): RewriteTaskResult => ({
    mode: "rewrite",
    taskId: "t",
    categories: ["insert"],
    calls,
    pass: calls.every((c) => c.correct),
  });

  it("counts answers cut off at the max-token limit", () => {
    const error = "the answer hit the max-token limit before it was complete";
    const cut = call({ correct: false, got: "", error, response: "Let me think", finishReason: "length" });
    // A JSON-string answer, then text that ran on to the limit: complete and graded, not cut off.
    const ranOn = call({ response: '"a!"\n\nI added an exclamation mark to the', finishReason: "length" });
    const m = rewriteMetrics([result([call({}), cut]), result([call({}), ranOn])]);
    expect(m.heldOutExamples).toEqual({ n: 3, of: 4 });
    expect(m.truncated).toEqual({ n: 1, of: 4 });
    expect(rewriteSummary("t", {}, m, [])).toContain(
      "| answers cut off at the max-token limit / calls | 1/4 (25.0%) |",
    );
  });

  it("shows token figures as unknown, not 0, when the server sent no counts", () => {
    const md = (calls: RewriteCall[]) => rewriteSummary("t", {}, rewriteMetrics([result(calls)]), []);
    expect(md([call({}), call({})])).toContain("| output tokens/sec, end to end (includes prompt processing) | 5.0 |");
    const missing = md([call({}), call({ promptTokens: undefined, completionTokens: undefined })]);
    expect(missing).toContain("| output tokens/sec, end to end (includes prompt processing) | – |");
    expect(missing).toContain("| largest prompt | – |");
    expect(missing).toContain("| calls the server sent no token counts for | 1 |");
    expect(md([call({})])).not.toContain("no token counts");
  });
});
