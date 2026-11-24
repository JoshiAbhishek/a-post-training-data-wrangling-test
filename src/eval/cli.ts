// Evaluate a local model on the task set.
//
//   scripts/serve-model.sh phi-4-mini-instruct     # in another terminal
//   npm run eval -- --model phi-4-mini-instruct     # both modes, k = 3, seeds 1,2,3
//
// Writes results/<local date>-<model>-<mode>/{attempts.jsonl,summary.md}; runs against a model that
// couldn't be verified (--allow-unverified-model) go under results/unverified/, which git ignores.
// attempts.jsonl grows as each task finishes, and summary.md says INCOMPLETE from the start until the
// run finishes, so a failed or interrupted (Ctrl-C) run keeps what it completed and is marked as such.
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, realpathSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { healthCheck, openAICompatible } from "../llm/openai-compatible.ts";
import { DEFAULT_SAMPLING, type SamplingConfig } from "./harness.ts";
import { programSummary, type RunMetadata, rewriteSummary } from "./metrics.ts";
import { loadPrompt, promptPath } from "./prompt.ts";
import { newRunDir, recordRun, resultsRoot } from "./results.ts";
import {
  conversionMetadata,
  displayPath,
  expandHome,
  localDate,
  parseServerProcess,
  parseSeeds,
  pinRuntime,
  positiveInt,
  runProgramMode,
  runRewriteMode,
  type ServerProcess,
  verifyServedModel,
  withRuntime,
} from "./run.ts";
import { loadTasks, taskSetSha256 } from "./task.ts";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const HOME = homedir();

const { values: args } = parseArgs({
  options: {
    mode: { type: "string", default: "both" }, // program | rewrite | both
    model: { type: "string", default: "phi-4-mini-instruct" },
    "base-url": { type: "string", default: "http://127.0.0.1:8080" },
    seeds: { type: "string", default: "1,2,3" },
    k: { type: "string", default: String(DEFAULT_SAMPLING.k) },
    "max-tokens": { type: "string", default: String(DEFAULT_SAMPLING.maxTokens) },
    limit: { type: "string" }, // first N tasks, for smoke tests
    out: { type: "string", default: join(ROOT, "results") },
    // Run even if the served model can't be matched to models.lock.json (e.g. a fake server for smoke
    // tests). The summary then says the model is unverified, and results go under results/unverified/.
    "allow-unverified-model": { type: "boolean", default: false },
  },
});

// Validate every option before contacting the server.
const mode = args.mode!;
if (!["program", "rewrite", "both"].includes(mode)) throw new Error(`--mode must be program, rewrite or both`);
const seeds = parseSeeds(args.seeds!);
const sampling: SamplingConfig = {
  ...DEFAULT_SAMPLING,
  k: positiveInt("--k", args.k!),
  maxTokens: positiveInt("--max-tokens", args["max-tokens"]!),
};
const limit = args.limit === undefined ? undefined : positiveInt("--limit", args.limit);

const log = (msg: string) => process.stderr.write(`${msg}\n`);

// What decides a run's behaviour, including whether it counts as verified: scripts/ holds the
// check-conversion verifier and the models.json registry it reads, and pyproject.toml, .python-version and
// uv.lock set the Python environment it and the server run in (uv.lock pins mlx and mlx-lm). Untracked
// results/ folders and a freshly written models.lock.json (recorded separately, from the conversion
// marker) don't make the code "uncommitted".
const EVAL_INPUTS = [
  "src",
  "prompts",
  "tasks",
  "scripts",
  "package.json",
  "package-lock.json",
  "pyproject.toml",
  ".python-version",
  "uv.lock",
];

function gitCommit(): string {
  try {
    const sha = execFileSync("git", ["rev-parse", "--short", "HEAD"], { cwd: ROOT }).toString().trim();
    const status = execFileSync("git", ["status", "--porcelain", "--", ...EVAL_INPUTS], { cwd: ROOT });
    const dirty = status.toString().trim() !== "";
    return dirty ? `${sha} (uncommitted changes in ${EVAL_INPUTS.join(", ")})` : sha;
  } catch {
    return "unknown";
  }
}

function lockEntry(model: string): Record<string, unknown> | undefined {
  const lock = join(ROOT, "models.lock.json");
  if (!existsSync(lock)) return undefined;
  return (JSON.parse(readFileSync(lock, "utf8")) as Record<string, Record<string, unknown>>)[model];
}

/** The converted model's real path, as mlx_lm.server reports it (resolved, no ~). */
function expectedServedPath(lock: Record<string, unknown> | undefined): string | undefined {
  const p = lock?.["mlx_path"];
  if (typeof p !== "string") return undefined;
  const expanded = expandHome(p, HOME);
  return existsSync(expanded) ? realpathSync(expanded) : expanded;
}

/** Run one mode into a new results/<local date>-<model>-<mode>[-runN]/ folder (see results.ts). */
async function recorded<T>(
  root: string,
  kind: string,
  meta: RunMetadata | (() => RunMetadata),
  run: (onResult: (r: T) => void) => Promise<string>,
): Promise<void> {
  const dir = newRunDir(root, `${localDate(new Date())}-${args.model}-${kind}`);
  await recordRun(dir, `${kind} run for ${args.model}`, meta, run, { log, show: (p) => displayPath(p, HOME) });
}

/**
 * Why the served folder fails scripts/verify_model.py check-conversion (the one conversion check, shared
 * with fetch-models.sh and serve-model.sh), or undefined if it passes. Runs in the project's Python
 * environment the way serve-model.sh does: offline and frozen to uv.lock, so it never syncs packages.
 */
function checkConversion(dir: string): string | undefined {
  try {
    execFileSync("uv", ["run", "--quiet", "python", "scripts/verify_model.py", "check-conversion", args.model!, dir], {
      cwd: ROOT,
      env: { ...process.env, UV_OFFLINE: "1", UV_FROZEN: "1" },
      stdio: ["ignore", "ignore", "pipe"],
    });
    return undefined;
  } catch (e) {
    const stderr = String((e as { stderr?: unknown }).stderr ?? "").trim();
    const why = stderr.split("\n").at(-1) || `couldn't run it: ${(e as Error).message}`;
    return `scripts/verify_model.py check-conversion failed: ${why.replaceAll(HOME + "/", "~/")}`;
  }
}

/** The served folder's conversion marker (written by scripts/fetch-models.sh), if it can be read. */
function readMarker(dir: string): Record<string, unknown> | undefined {
  try {
    return JSON.parse(readFileSync(join(dir, ".conversion.json"), "utf8")) as Record<string, unknown>;
  } catch {
    return undefined;
  }
}

/**
 * The one process listening on the server's local port (macOS lsof and ps): its command line and when
 * it started, or undefined if it can't be read. /v1/models doesn't show options like --adapter-path;
 * this does.
 */
function serverProcess(baseUrl: string): ServerProcess | undefined {
  try {
    const url = new URL(baseUrl);
    if (!["127.0.0.1", "localhost", "[::1]"].includes(url.hostname)) return undefined;
    const port = url.port || (url.protocol === "https:" ? "443" : "80");
    const run = (cmd: string, cmdArgs: string[]) =>
      execFileSync(cmd, cmdArgs, { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
    const pids = run("lsof", ["-nP", `-iTCP:${port}`, "-sTCP:LISTEN", "-t"]).split("\n");
    if (pids.length !== 1 || !pids[0]) return undefined;
    const line = run("ps", ["-ww", "-o", "etime=,command=", "-p", pids[0]]);
    return line ? parseServerProcess(line, Date.now()) : undefined;
  } catch {
    return undefined;
  }
}

/**
 * When the conversion in `dir` was written: the later creation time of the folder and of its marker.
 * fetch-models.sh builds each conversion in a new folder and moves it into place, which keeps the
 * creation time, so a conversion rebuilt at the same path is newer than a server that loaded the old one.
 */
function convertedAt(dir: string): number | undefined {
  try {
    const born = (p: string) => {
      const s = statSync(p);
      return s.birthtimeMs > 0 ? s.birthtimeMs : s.ctimeMs; // no creation time on this file system: last change
    };
    return Math.max(born(dir), born(join(dir, ".conversion.json")));
  } catch {
    return undefined;
  }
}

const served = await healthCheck(args["base-url"]!);
const lock = lockEntry(args.model!);
const servedCheck = verifyServedModel({
  servedIds: served,
  expectedPath: expectedServedPath(lock),
  checkConversion,
  serverProcess: () => serverProcess(args["base-url"]!),
  convertedAt,
  show: (p) => displayPath(p, HOME),
});
if (!servedCheck.verified && !args["allow-unverified-model"]) {
  throw new Error(
    `Can't confirm the server is serving ${args.model}: ${servedCheck.reason}. ` +
      `Start it with scripts/serve-model.sh ${args.model}, or pass --allow-unverified-model for a smoke test.`,
  );
}
// Smoke-test output stays apart from real results, so it can't be committed or read as a baseline.
const outRoot = resultsRoot(args.out!, servedCheck.verified);
if (!servedCheck.verified) log(`model unverified (${servedCheck.reason}); writing under ${displayPath(outRoot, HOME)}`);

// One provider for both modes, so a server restarted with different software between them is caught too.
const provider = pinRuntime(openAICompatible({ baseUrl: args["base-url"]!, name: args.model }));
let tasks = loadTasks(join(ROOT, "tasks")).map((t) => t.task);
if (limit !== undefined) tasks = tasks.slice(0, limit);

const baseMeta: RunMetadata = {
  date: new Date().toISOString(),
  model: args.model,
  "served model": servedCheck.verified
    ? `${displayPath(servedCheck.servedPath, HOME)} (verified against models.lock.json; no adapter or other server options)`
    : `UNVERIFIED: ${servedCheck.reason}`,
  // From the marker check-conversion just verified: the source files, converter and converted files.
  ...conversionMetadata(servedCheck.verified ? readMarker(servedCheck.servedPath) : undefined),
  "repo commit": gitCommit(),
  tasks: `${tasks.length} (sha256 ${taskSetSha256(tasks)})`,
  "max output tokens": sampling.maxTokens,
};

if (mode === "program" || mode === "both") {
  const prompt = loadPrompt(promptPath("program-synthesis-v1"));
  const meta: RunMetadata = {
    ...baseMeta,
    mode: "program synthesis",
    prompt: `${prompt.version} (sha256 ${prompt.sha256})`,
    k: sampling.k,
    "attempt 1 temperature": sampling.firstTemperature,
    "retry temperature": sampling.retryTemperature,
    seeds: seeds.join(", "),
  };
  log(`program synthesis: ${tasks.length} tasks × seeds ${seeds.join(",")}, k = ${sampling.k}`);
  // The summary, finished or not, records the server runtime the run saw.
  await recorded(
    outRoot,
    "program",
    () => withRuntime(meta, provider.runtime()),
    async (onResult) => {
      const runs = await runProgramMode(tasks, provider, prompt, sampling, seeds, log, onResult);
      return programSummary(`Program synthesis: ${args.model}`, withRuntime(meta, provider.runtime()), runs);
    },
  );
}

if (mode === "rewrite" || mode === "both") {
  const prompt = loadPrompt(promptPath("direct-rewrite-v1"));
  const meta: RunMetadata = {
    ...baseMeta,
    mode: "direct rewrite",
    prompt: `${prompt.version} (sha256 ${prompt.sha256})`,
    temperature: sampling.firstTemperature,
  };
  log(`direct rewrite: ${tasks.length} tasks, one call per held-out line`);
  await recorded(
    outRoot,
    "rewrite",
    () => withRuntime(meta, provider.runtime()),
    async (onResult) => {
      const run = await runRewriteMode(tasks, provider, prompt, sampling, log, onResult);
      return rewriteSummary(
        `Direct rewrite: ${args.model}`,
        withRuntime(meta, provider.runtime()),
        run.metrics,
        run.results,
      );
    },
  );
}
