// Metrics and the summary.md report. Counts are always shown next to percentages.
import { INVALID_VERDICTS, type ProgramTaskResult, type RewriteTaskResult } from "./harness.ts";
import { CATEGORIES, type Category } from "./task.ts";

export type Count = { n: number; of: number };
export type Spread = { mean: number; min: number; max: number }; // of a count across seeds

export type ProgramSeedMetrics = {
  seed: number;
  pass1: Count;
  passK: Count;
  invalidAttempts: Count; // truncated, JSON or schema failures ÷ all attempts
  overfit: Count; // accepted programs failing held-out ÷ accepted programs
  heldOutExamples: Count; // held-out examples matched by accepted programs ÷ all held-out examples
  outcomes: Record<string, number>;
  byCategory: Record<Category, Count>; // passK per tag
  speed: Speed;
};

export type RewriteMetrics = {
  pass: Count;
  heldOutExamples: Count;
  truncated: Count; // calls that hit the max-token limit with no answer accepted ÷ calls

  byCategory: Record<Category, Count>;
  speed: Speed;
};

/**
 * `tokensPerSec` is output tokens over each call's whole latency (request sent to response body read):
 * prompt processing and HTTP included, not generation speed alone. Responses aren't streamed, so the
 * two can't be told apart, and mlx_lm.server's prompt cache makes the prompt-processing share vary.
 * Token figures are undefined if any call came back without that count: a partial sum would understate
 * them, and a missing count isn't 0 tokens.
 */
export type Speed = {
  calls: number;
  tokensPerSec: number | undefined;
  meanLatencyMs: number;
  maxPromptTokens: number | undefined;
  callsWithoutTokenCounts: number;
};

/** The summary row for Speed.tokensPerSec, named for what it measures. */
export const TOKENS_PER_SEC_LABEL = "output tokens/sec, end to end (includes prompt processing)";

const count = (n: number, of: number): Count => ({ n, of });
export const pct = (c: Count) => (c.of === 0 ? "–" : `${((100 * c.n) / c.of).toFixed(1)}%`);
export const fmt = (c: Count) => `${c.n}/${c.of} (${pct(c)})`;

type Call = { latencyMs: number; promptTokens?: number; completionTokens?: number };

function speed(calls: Call[]): Speed {
  const ms = calls.reduce((s, c) => s + c.latencyMs, 0);
  const output = calls.map((c) => c.completionTokens);
  const prompts = calls.map((c) => c.promptTokens);
  const known = (xs: (number | undefined)[]): xs is number[] => xs.every((x) => x !== undefined);
  return {
    calls: calls.length,
    tokensPerSec: known(output) ? (ms > 0 ? output.reduce((s, t) => s + t, 0) / (ms / 1000) : 0) : undefined,
    meanLatencyMs: calls.length ? ms / calls.length : 0,
    maxPromptTokens: known(prompts) ? Math.max(0, ...prompts) : undefined,
    callsWithoutTokenCounts: calls.filter((c) => c.promptTokens === undefined || c.completionTokens === undefined)
      .length,
  };
}

/** The speed rows of a summary's results table. */
function speedRows(s: Speed): string[] {
  return [
    `| model calls | ${s.calls} |`,
    `| ${TOKENS_PER_SEC_LABEL} | ${s.tokensPerSec === undefined ? "–" : s.tokensPerSec.toFixed(1)} |`,
    `| mean latency per call | ${(s.meanLatencyMs / 1000).toFixed(2)} s |`,
    `| largest prompt | ${s.maxPromptTokens === undefined ? "–" : `${s.maxPromptTokens} tokens`} |`,
    ...(s.callsWithoutTokenCounts
      ? [`| calls the server sent no token counts for | ${s.callsWithoutTokenCounts} |`]
      : []),
  ];
}

function byCategory<T extends { categories: Category[] }>(results: T[], passed: (r: T) => boolean) {
  return Object.fromEntries(
    CATEGORIES.map((c) => {
      const tagged = results.filter((r) => r.categories.includes(c));
      return [c, count(tagged.filter(passed).length, tagged.length)];
    }),
  ) as Record<Category, Count>;
}

/** Attempts that called the model in this run: an attempt 1 reused from an earlier seed made no call. */
export const modelCalls = (results: readonly ProgramTaskResult[]) =>
  results.flatMap((r) => (r.firstAttemptReused ? r.attempts.slice(1) : r.attempts));

export function programSeedMetrics(results: ProgramTaskResult[]): ProgramSeedMetrics {
  const attempts = results.flatMap((r) => r.attempts); // each seed's full trajectory, reused attempt 1 included
  const accepted = results.filter((r) => r.heldOut);
  // Every task's held-out examples count; tasks with no accepted program match none of theirs.
  const heldOutTotal = results.reduce((s, r) => s + r.heldOutTotal, 0);
  const outcomes: Record<string, number> = {};
  for (const r of results) outcomes[r.outcome] = (outcomes[r.outcome] ?? 0) + 1;
  return {
    seed: results[0]?.seed ?? 0,
    pass1: count(results.filter((r) => r.pass1).length, results.length),
    passK: count(results.filter((r) => r.passK).length, results.length),
    invalidAttempts: count(attempts.filter((a) => INVALID_VERDICTS.includes(a.verdict)).length, attempts.length),
    overfit: count(accepted.filter((r) => !r.passK).length, accepted.length),
    heldOutExamples: count(
      results.reduce((s, r) => s + (r.heldOut?.passed ?? 0), 0),
      heldOutTotal,
    ),
    outcomes,
    byCategory: byCategory(results, (r) => r.passK),
    speed: speed(modelCalls(results)),
  };
}

export function rewriteMetrics(results: RewriteTaskResult[]): RewriteMetrics {
  const calls = results.flatMap((r) => r.calls);
  return {
    pass: count(results.filter((r) => r.pass).length, results.length),
    heldOutExamples: count(calls.filter((c) => c.correct).length, calls.length),
    // As in program mode, a call counts as cut off only if the limit cost it its answer: a JSON-string
    // answer followed by text that ran on to the limit was complete, and is graded.
    truncated: count(calls.filter((c) => c.finishReason === "length" && c.error !== undefined).length, calls.length),
    byCategory: byCategory(results, (r) => r.pass),
    speed: speed(calls),
  };
}

export function spread(values: number[]): Spread {
  return {
    mean: values.reduce((s, v) => s + v, 0) / Math.max(values.length, 1),
    min: Math.min(...values),
    max: Math.max(...values),
  };
}

const rate = (x: number) => `${(100 * x).toFixed(1)}%`;

/**
 * One count per seed → one table cell. With the same denominator in every seed, show the mean count
 * (and its range). When denominators differ (attempts, accepted programs), averaging numerators would
 * be meaningless, so show the mean and range of each seed's own rate plus the raw per-seed counts;
 * seeds with a zero denominator have no rate and are left out of the average.
 */
export function fmtCounts(counts: readonly Count[]): string {
  if (counts.length === 0) return "–";
  const of = counts[0]!.of;
  if (counts.every((c) => c.of === of)) {
    const s = spread(counts.map((c) => c.n));
    return s.min === s.max
      ? `${s.mean}/${of} (${pct({ n: s.mean, of })})`
      : `${s.mean.toFixed(1)}/${of} (${pct({ n: s.mean, of })}), range ${s.min}–${s.max}`;
  }
  const perSeed = counts.map((c) => `${c.n}/${c.of}`).join(", ");
  const rates = counts.filter((c) => c.of > 0).map((c) => c.n / c.of);
  if (rates.length === 0) return `– (per seed: ${perSeed})`;
  const r = spread(rates);
  const range = r.min === r.max ? "" : `, range ${rate(r.min)}–${rate(r.max)}`;
  return `${rate(r.mean)} mean of seed rates${range} (per seed: ${perSeed})`;
}

export type RunMetadata = Record<string, string | number | boolean | undefined>;

function metadataTable(meta: RunMetadata): string[] {
  return [
    "| Setting | Value |",
    "| --- | --- |",
    ...Object.entries(meta)
      .filter(([, v]) => v !== undefined)
      .map(([k, v]) => `| ${k} | ${String(v).replace(/\|/g, "\\|")} |`),
  ];
}

/**
 * summary.md for a run that hasn't finished; its finished task results are already in attempts.jsonl.
 * Without `stopped` it is the placeholder written when the run starts, which stays if the run is
 * interrupted (Ctrl-C, a kill) before it can write anything else.
 */
export function incompleteSummary(
  title: string,
  meta: RunMetadata,
  stopped?: { written: number; error: string },
): string {
  const status = stopped
    ? [
        `The run stopped after ${stopped.written} task result(s), saved in attempts.jsonl.`,
        "",
        `Error: ${stopped.error}`,
      ]
    : [
        "The run is in progress, or was interrupted before it finished (for example with Ctrl-C).",
        "attempts.jsonl holds one line per task result completed so far.",
      ];
  return [`# INCOMPLETE: ${title}`, "", ...status, "", ...metadataTable(meta), ""].join("\n");
}

/** Markdown report for program-synthesis runs (one per seed). */
export function programSummary(
  title: string,
  meta: RunMetadata,
  seeds: { metrics: ProgramSeedMetrics; results: ProgramTaskResult[] }[],
): string {
  const tasks = seeds[0]?.metrics.pass1.of ?? 0;
  const row = (label: string, f: (m: ProgramSeedMetrics) => Count) =>
    `| ${label} | ${fmtCounts(seeds.map((s) => f(s.metrics)))} |`;
  const allOutcomes = [...new Set(seeds.flatMap((s) => Object.keys(s.metrics.outcomes)))].sort();
  // Speed over every model call in the run, once each (later seeds reuse attempt 1 without a call).
  const runSpeed = speed(seeds.flatMap((s) => modelCalls(s.results)));
  const lines = [
    `# ${title}`,
    "",
    "## Run",
    "",
    ...metadataTable(meta),
    "",
    `## Results (${tasks} tasks, ${seeds.length} seed${seeds.length === 1 ? "" : "s"})`,
    "",
    "Counts are the mean across seeds, with the range when seeds disagree. Rows whose denominator differs",
    "between seeds (attempts, accepted programs) show the mean of each seed's own rate instead.",
    "",
    "| Metric | Value |",
    "| --- | --- |",
    row("pass@1 (first attempt passes shown + held-out)", (m) => m.pass1),
    row(`pass@k with verify (k = ${meta["k"] ?? "?"})`, (m) => m.passK),
    row("invalid programs (truncated, JSON or schema) / attempts", (m) => m.invalidAttempts),
    row("overfit: accepted but fails held-out / accepted", (m) => m.overfit),
    row("held-out examples matched (secondary)", (m) => m.heldOutExamples),
    ...speedRows(runSpeed),
    "",
    "## pass@k by category",
    "",
    "| Category | pass@k |",
    "| --- | --- |",
    ...CATEGORIES.map((c) => row(c, (m) => m.byCategory[c])),
    "",
    "## Outcomes",
    "",
    "| Outcome | Tasks |",
    "| --- | --- |",
    ...allOutcomes.map(
      (o) => `| ${o} | ${fmtCounts(seeds.map((x) => ({ n: x.metrics.outcomes[o] ?? 0, of: tasks })))} |`,
    ),
    "",
    "## Per task",
    "",
    `| Task | pass@1 | pass@k (seeds passing) | outcomes |`,
    "| --- | --- | --- | --- |",
    ...(seeds[0]?.results ?? []).map((r) => {
      const runs = seeds.map((s) => s.results.find((x) => x.taskId === r.taskId)!);
      const outcomes = [...new Set(runs.map((x) => x.outcome))].join(", ");
      return `| \`${r.taskId}\` | ${r.pass1 ? "✓" : "✗"} | ${runs.filter((x) => x.passK).length}/${runs.length} | ${outcomes} |`;
    }),
    "",
  ];
  return lines.join("\n");
}

/** Markdown report for a direct-rewrite run. */
export function rewriteSummary(
  title: string,
  meta: RunMetadata,
  m: RewriteMetrics,
  results: RewriteTaskResult[],
): string {
  return [
    `# ${title}`,
    "",
    "## Run",
    "",
    ...metadataTable(meta),
    "",
    `## Results (${m.pass.of} tasks)`,
    "",
    "| Metric | Value |",
    "| --- | --- |",
    `| tasks with every held-out line correct | ${fmt(m.pass)} |`,
    `| held-out lines correct (secondary) | ${fmt(m.heldOutExamples)} |`,
    // A cut-off answer is usually wrong; this row says how many failures were length, not reformatting.
    `| answers cut off at the max-token limit / calls | ${fmt(m.truncated)} |`,
    ...speedRows(m.speed),
    "",
    "## By category",
    "",
    "| Category | tasks passing |",
    "| --- | --- |",
    ...CATEGORIES.map((c) => `| ${c} | ${fmt(m.byCategory[c])} |`),
    "",
    "## Per task",
    "",
    "| Task | pass | lines correct |",
    "| --- | --- | --- |",
    ...results.map(
      (r) =>
        `| \`${r.taskId}\` | ${r.pass ? "✓" : "✗"} | ${r.calls.filter((c) => c.correct).length}/${r.calls.length} |`,
    ),
    "",
  ].join("\n");
}
