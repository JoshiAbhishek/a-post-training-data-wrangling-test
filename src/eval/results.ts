// Results folders for eval runs: where a run's attempts.jsonl and summary.md go, and writing
// them so a failed or interrupted run is never mistaken for a finished one. cli.ts calls these with the
// real results folder; tests use a temporary one.
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { incompleteSummary, type RunMetadata } from "./metrics.ts";

/** Runs against a model that couldn't be verified go under <out>/unverified/, which git ignores. */
export function resultsRoot(out: string, verified: boolean): string {
  return verified ? out : join(out, "unverified");
}

/**
 * Create and return a new, empty folder <root>/<name>, or <name>-run2, -run3, ... if that's taken.
 * mkdir without `recursive` fails if the folder already exists, so two runs started at the same time
 * never get the same folder (an existsSync check followed by mkdir could give both the same one).
 */
export function newRunDir(root: string, name: string): string {
  mkdirSync(root, { recursive: true });
  for (let i = 1; ; i++) {
    const dir = join(root, i === 1 ? name : `${name}-run${i}`);
    try {
      mkdirSync(dir);
      return dir;
    } catch (e) {
      if ((e as { code?: string }).code !== "EEXIST") throw e;
    }
  }
}

export type RecordOptions = { log?: (msg: string) => void; show?: (path: string) => string };

/**
 * Run one mode in `dir`, appending each task result to attempts.jsonl as it completes. summary.md says
 * the run is INCOMPLETE until it finishes: written up front, so it stays that way if the process is
 * killed or interrupted (no catch block runs on Ctrl-C), and rewritten with the error if the run
 * fails. `run` returns the finished summary. `show` formats paths in log messages. `meta` may be a
 * function, read each time the summary is written, for metadata learned during the run (the server
 * runtime). The INCOMPLETE summary is rewritten as each result is saved, so a run that is killed or
 * interrupted, which no catch block sees, still records what was known when its last result was saved.
 */
export async function recordRun<T>(
  dir: string,
  title: string,
  meta: RunMetadata | (() => RunMetadata),
  run: (onResult: (r: T) => void) => Promise<string>,
  { log = () => {}, show = (p) => p }: RecordOptions = {},
): Promise<void> {
  const metaNow = typeof meta === "function" ? meta : () => meta;
  const attempts = join(dir, "attempts.jsonl");
  const summaryPath = join(dir, "summary.md");
  writeFileSync(attempts, "");
  writeFileSync(summaryPath, incompleteSummary(title, metaNow()));
  let written = 0;
  try {
    const summary = await run((r) => {
      appendFileSync(attempts, JSON.stringify(r) + "\n");
      written++;
      writeFileSync(summaryPath, incompleteSummary(title, metaNow()));
    });
    writeFileSync(summaryPath, summary);
    log(`wrote ${show(summaryPath)}`);
  } catch (e) {
    writeFileSync(summaryPath, incompleteSummary(title, metaNow(), { written, error: (e as Error).message }));
    log(`run failed; ${written} result(s) kept in ${show(dir)}`);
    throw e;
  }
}
