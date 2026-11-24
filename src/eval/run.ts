// Run a whole evaluation: every task, every seed, either mode. No file or network I/O here,
// so it can be tested end to end with a fake provider; cli.ts does the I/O. `onResult` fires as each
// task finishes, so the CLI can save results as it goes.
import { type Provider, ProviderError } from "../llm/provider.ts";
import {
  type Attempt,
  evaluateProgramTask,
  evaluateRewriteTask,
  type ProgramTaskResult,
  type RewriteTaskResult,
  type SamplingConfig,
} from "./harness.ts";
import {
  type ProgramSeedMetrics,
  programSeedMetrics,
  type RewriteMetrics,
  rewriteMetrics,
  type RunMetadata,
} from "./metrics.ts";
import type { PromptTemplate } from "./prompt.ts";
import type { Task } from "./task.ts";

export type Progress = (msg: string) => void;

/** YYYY-MM-DD in the local time zone (toISOString would give the UTC date). */
export function localDate(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

/** Show paths under the home directory as ~/..., so results don't carry the local username. */
export function displayPath(path: string, home: string): string {
  return home && (path === home || path.startsWith(home + "/")) ? "~" + path.slice(home.length) : path;
}

export function expandHome(path: string, home: string): string {
  return path === "~" || path.startsWith("~/") ? home + path.slice(1) : path;
}

export type ServedCheck = { verified: true; servedPath: string } | { verified: false; reason: string };

/** What the server reported as its runtime: its system_fingerprint, undefined if it sent none. */
export type Runtime = { fingerprint: string | undefined };

/** A provider that also says which server runtime answered it. */
export type PinnedProvider = Provider & {
  /** The runtime every completion so far came from; undefined before the first. */
  runtime(): Runtime | undefined;
};

/**
 * A provider that stops the run if the server's runtime changes part way. mlx_lm.server reports its
 * mlx-lm and MLX versions, OS and GPU as `system_fingerprint` with every completion; a different value
 * means the server was restarted with different software, so later answers wouldn't come from the setup
 * the run's results describe. The error names both values, and the results so far are kept, marked
 * incomplete. (A restart with the same software and model can't be told apart, and isn't caught.)
 */
export function pinRuntime(provider: Provider): PinnedProvider {
  let first: Runtime | undefined;
  const show = (f: string | undefined) => (f === undefined ? "none" : JSON.stringify(f));
  return {
    name: provider.name,
    async complete(messages, opts) {
      const c = await provider.complete(messages, opts);
      if (first === undefined) first = { fingerprint: c.systemFingerprint };
      else if (c.systemFingerprint !== first.fingerprint)
        throw new ProviderError(
          `the server's system_fingerprint changed during the run, from ${show(first.fingerprint)} to ` +
            `${show(c.systemFingerprint)}, so it was restarted with different software. Rerun with one server.`,
        );
      return c;
    },
    runtime: () => first,
  };
}

/**
 * The run's metadata with the server runtime it ran against, after "served model": the system_fingerprint
 * pinRuntime saw, which the conversion's own record (the versions that converted it) doesn't give.
 * Unchanged before the first completion, when there's nothing to record yet.
 */
export function withRuntime(meta: RunMetadata, seen: Runtime | undefined): RunMetadata {
  if (seen === undefined) return meta;
  const runtime = seen.fingerprint ?? "not reported by the server";
  const out: RunMetadata = {};
  for (const [k, v] of Object.entries(meta)) {
    out[k] = v;
    if (k === "served model") out["server runtime (system_fingerprint)"] = runtime;
  }
  if (!("served model" in meta)) out["server runtime (system_fingerprint)"] = runtime;
  return out;
}

/**
 * Is the server serving the converted model recorded in models.lock.json? mlx_lm.server lists the
 * model it was started with by its resolved path, alongside any models in the Hugging Face cache.
 * `show` formats each path in the reason (e.g. displayPath, so results don't carry the username).
 */
export function matchServedModel(
  servedIds: readonly string[],
  expectedPath: string | undefined,
  show: (path: string) => string = (p) => p,
): ServedCheck {
  if (expectedPath === undefined) return { verified: false, reason: "model has no mlx_path in models.lock.json" };
  if (servedIds.includes(expectedPath)) return { verified: true, servedPath: expectedPath };
  return {
    verified: false,
    reason: `server isn't serving ${show(expectedPath)}; it reports: ${servedIds.map(show).join(", ") || "nothing"}`,
  };
}

/**
 * Run metadata from a verified conversion's .conversion.json (written by scripts/fetch-models.sh): the
 * source it was built from, the converter versions that built it, and the digest of the files it wrote,
 * which models.lock.json pins. Results from a reconverted model therefore never look like the same model.
 */
export function conversionMetadata(marker: Record<string, unknown> | undefined): RunMetadata {
  const text = (key: string) => (typeof marker?.[key] === "string" ? (marker[key] as string) : undefined);
  const converter = marker?.["converted_with"];
  return {
    "source repo": text("repo"),
    "source commit": text("revision"),
    "source files sha256": text("source_files_sha256"),
    quantization: text("quantization"),
    "converted with":
      converter !== null && typeof converter === "object"
        ? Object.entries(converter)
            .map(([pkg, version]) => `${pkg} ${String(version)}`)
            .join(", ")
        : undefined,
    "converted files sha256": text("converted_files_sha256"),
  };
}

/** The options scripts/serve-model.sh starts mlx_lm.server with. */
const SERVE_OPTIONS = new Set(["--model", "--host", "--port"]);

/**
 * Why the server process's command line (as `ps` shows it) isn't one scripts/serve-model.sh would run,
 * or undefined if it is: mlx_lm.server with --model, --host and --port only. Other options can change
 * what the model answers without changing what /v1/models lists (--adapter-path, --draft-model,
 * --chat-template, ...), so a server started with them isn't the model models.lock.json describes.
 * Unknown options fail too, including argparse abbreviations such as --adapter.
 */
export function serverCommandMismatch(command: string | undefined): string | undefined {
  if (command === undefined) return "couldn't read the command line of the process serving the port";
  const words = command.trim().split(/\s+/);
  const at = words.findIndex((w) => w.split("/").at(-1) === "mlx_lm.server");
  if (at < 0) return `the process serving the port isn't mlx_lm.server`;
  const extra = words.slice(at + 1).filter((w) => w.startsWith("-") && !SERVE_OPTIONS.has(w.split("=")[0]!));
  return extra.length
    ? `mlx_lm.server was started with options serve-model.sh doesn't use: ${extra.join(", ")}`
    : undefined;
}

/** The process serving the port: its command line, and when it started (epoch ms) if `ps` said. */
export type ServerProcess = { command: string; startedAtMs?: number };

/** Seconds in a `ps -o etime` value, `[[dd-]hh:]mm:ss`; undefined if it isn't one. */
export function elapsedSeconds(etime: string): number | undefined {
  const m = /^(?:(?:(\d+)-)?(\d+):)?(\d+):(\d+)$/.exec(etime.trim());
  if (!m) return undefined;
  const n = (i: number) => Number(m[i] ?? 0);
  return ((n(1) * 24 + n(2)) * 60 + n(3)) * 60 + n(4);
}

/**
 * A server process from a `ps -ww -o etime=,command=` line, read at `nowMs`. `etime` counts whole
 * seconds, so the start it gives is never before the real one (by under a second).
 */
export function parseServerProcess(psLine: string, nowMs: number): ServerProcess | undefined {
  const m = /^\s*(\S+)\s+(\S.*)$/.exec(psLine);
  if (!m) return undefined;
  const elapsed = elapsedSeconds(m[1]!);
  return elapsed === undefined ? { command: psLine.trim() } : { command: m[2]!, startedAtMs: nowMs - elapsed * 1000 };
}

/**
 * Why the server may not hold the conversion now in `dir`, or undefined if it does. mlx_lm.server loads
 * its model once, when it starts, and keeps it while the path stays the same, so a conversion rebuilt
 * at that path afterwards (moved away and fetched again) isn't the model it answers with, though the
 * folder passes every other check. `convertedAtMs` is when the folder's conversion was written.
 */
export function staleServerReason(
  startedAtMs: number | undefined,
  convertedAtMs: number | undefined,
  dir: string,
): string | undefined {
  if (startedAtMs === undefined) return "couldn't read when the process serving the port started";
  if (convertedAtMs === undefined) return `couldn't read when the conversion in ${dir} was written`;
  return convertedAtMs > startedAtMs
    ? `the conversion in ${dir} was written after the server started, and the server keeps the model it ` +
        "loaded then; restart the server"
    : undefined;
}

/**
 * The whole served-model check, in order: the server lists the converted folder recorded in
 * models.lock.json, that folder is the unchanged conversion models.lock.json pins, the server process
 * was started the way scripts/serve-model.sh starts it, and it started after that conversion was
 * written, so it loaded it. `checkConversion` gives the reason a folder fails scripts/verify_model.py
 * check-conversion (the check serve-model.sh runs, repeated in case anything changed since), or
 * undefined if it passes; `serverProcess` gives the server process; `convertedAt` gives when a folder's
 * conversion was written (epoch ms). All three are I/O, supplied by the CLI.
 */
export function verifyServedModel(check: {
  servedIds: readonly string[];
  expectedPath: string | undefined;
  checkConversion: (dir: string) => string | undefined;
  serverProcess: () => ServerProcess | undefined;
  convertedAt: (dir: string) => number | undefined;
  show?: (path: string) => string;
}): ServedCheck {
  const show = check.show ?? ((p: string) => p);
  const served = matchServedModel(check.servedIds, check.expectedPath, show);
  if (!served.verified) return served;
  // The right folder isn't enough: it must hold the converted files the lockfile pins,
  const conversion = check.checkConversion(served.servedPath);
  if (conversion !== undefined) return { verified: false, reason: conversion };
  const server = check.serverProcess();
  const reason =
    // be served as is: no adapter, draft model or chat template on top,
    serverCommandMismatch(server?.command) ??
    // and be what the server loaded: not a conversion rebuilt at the same path while it ran.
    staleServerReason(server?.startedAtMs, check.convertedAt(served.servedPath), show(served.servedPath));
  return reason === undefined ? served : { verified: false, reason };
}

/** CLI number parsing: whole numbers >= 1 only (rejects "", "0", "-2", "1.5", "three"). */
export function positiveInt(flag: string, value: string): number {
  const n = Number(value);
  if (value.trim() === "" || !Number.isInteger(n) || n < 1)
    throw new Error(`${flag} must be a whole number >= 1, got "${value}"`);
  return n;
}

/**
 * Comma-separated whole-number seeds; empty items (e.g. "1,,2") are rejected, not read as 0. A repeated
 * seed is rejected too: retries are seeded deterministically, so it would rerun the same trajectory
 * and count it as another, independent seed.
 */
export function parseSeeds(value: string): number[] {
  const parts = value.split(",").map((s) => s.trim());
  const seeds = parts.map(Number);
  if (parts.some((p) => p === "") || seeds.some((s) => !Number.isInteger(s) || s < 0)) {
    throw new Error(`--seeds must be comma-separated whole numbers, got "${value}"`);
  }
  if (new Set(seeds).size !== seeds.length) throw new Error(`--seeds must not repeat a seed, got "${value}"`);
  return seeds;
}

export type ProgramRun = { seed: number; results: ProgramTaskResult[]; metrics: ProgramSeedMetrics };

/**
 * Program synthesis over all seeds. Attempt 1 is greedy (temperature 0), so it runs once per task and
 * is reused for later seeds; only the retries differ between seeds.
 */
export async function runProgramMode(
  tasks: readonly Task[],
  provider: Provider,
  prompt: PromptTemplate,
  base: SamplingConfig,
  seeds: readonly number[],
  progress: Progress = () => {},
  onResult: (r: ProgramTaskResult) => void = () => {},
): Promise<ProgramRun[]> {
  const firstAttempts = new Map<string, Attempt>();
  const runs: ProgramRun[] = [];
  for (const seed of seeds) {
    const cfg = { ...base, seed };
    const results: ProgramTaskResult[] = [];
    for (const [i, task] of tasks.entries()) {
      const reuse = base.firstTemperature === 0 ? firstAttempts.get(task.id) : undefined;
      const r = await evaluateProgramTask(task, provider, prompt, cfg, reuse);
      if (r.attempts[0]) firstAttempts.set(task.id, r.attempts[0]);
      results.push(r);
      onResult(r);
      progress(`seed ${seed} [${i + 1}/${tasks.length}] ${task.id}: ${r.outcome} (${r.attempts.length} attempt(s))`);
    }
    runs.push({ seed, results, metrics: programSeedMetrics(results) });
  }
  return runs;
}

export type RewriteRun = { results: RewriteTaskResult[]; metrics: RewriteMetrics };

/** Direct rewrite runs once: every call is greedy, so seeds wouldn't change it. */
export async function runRewriteMode(
  tasks: readonly Task[],
  provider: Provider,
  prompt: PromptTemplate,
  base: SamplingConfig,
  progress: Progress = () => {},
  onResult: (r: RewriteTaskResult) => void = () => {},
): Promise<RewriteRun> {
  const results: RewriteTaskResult[] = [];
  for (const [i, task] of tasks.entries()) {
    const r = await evaluateRewriteTask(task, provider, prompt, base);
    results.push(r);
    onResult(r);
    const correct = r.calls.filter((c) => c.correct).length;
    progress(`[${i + 1}/${tasks.length}] ${task.id}: ${r.pass ? "pass" : "fail"} (${correct}/${r.calls.length} lines)`);
  }
  return { results, metrics: rewriteMetrics(results) };
}
