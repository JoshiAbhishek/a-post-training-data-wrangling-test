// End to end over the real task set with fake models: no server, no weights.
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import type { Provider } from "../llm/provider.ts";
import { fakeProvider } from "./fake-provider.ts";
import { DEFAULT_SAMPLING } from "./harness.ts";
import { programSummary, rewriteSummary } from "./metrics.ts";
import { loadPrompt, promptPath } from "./prompt.ts";
import { resultsRoot } from "./results.ts";
import {
  conversionMetadata,
  displayPath,
  elapsedSeconds,
  expandHome,
  localDate,
  matchServedModel,
  parseServerProcess,
  parseSeeds,
  pinRuntime,
  positiveInt,
  runProgramMode,
  runRewriteMode,
  serverCommandMismatch,
  staleServerReason,
  verifyServedModel,
  withRuntime,
} from "./run.ts";
import { loadTasks } from "./task.ts";

const tasks = loadTasks(fileURLToPath(new URL("../../tasks/", import.meta.url))).map((t) => t.task);
const programPrompt = loadPrompt(promptPath("program-synthesis-v1"));
const rewritePrompt = loadPrompt(promptPath("direct-rewrite-v1"));
const byDescription = new Map(tasks.map((t) => [t.description, t]));
const taskFor = (content: string) => byDescription.get(/Task: (.*)\n/.exec(content)![1]!)!;

describe("program mode, end to end", () => {
  it("an oracle that returns each reference program scores 100%, reusing attempt 1 across seeds", async () => {
    const { provider, calls } = fakeProvider((messages) => JSON.stringify(taskFor(messages.at(-1)!.content).reference));
    const runs = await runProgramMode(tasks, provider, programPrompt, DEFAULT_SAMPLING, [1, 2, 3]);
    expect(calls).toHaveLength(tasks.length); // attempt 1 runs once per task, not once per seed
    for (const run of runs) {
      expect(run.metrics.pass1).toEqual({ n: tasks.length, of: tasks.length });
      expect(run.metrics.passK).toEqual({ n: tasks.length, of: tasks.length });
      expect(run.metrics.invalidAttempts).toEqual({ n: 0, of: tasks.length });
      expect(run.metrics.overfit).toEqual({ n: 0, of: tasks.length });
    }
    const md = programSummary("Program synthesis: oracle", { k: 3, seeds: "1, 2, 3" }, runs);
    expect(md).toContain(
      `| pass@1 (first attempt passes shown + held-out) | ${tasks.length}/${tasks.length} (100.0%) |`,
    );
    expect(md).toContain("| `suffix-by-initial` | ✓ | 3/3 | pass |");
    // Speed counts each model call once: seeds 2 and 3 reused attempt 1 and made no calls.
    expect(runs.map((r) => r.metrics.speed.calls)).toEqual([tasks.length, 0, 0]);
    expect(md).toContain(`| model calls | ${tasks.length} |`);
    expect(md).toContain("| mean latency per call | 0.10 s |");
    // Latency covers the whole call, so the rate is labelled as end to end, not generation speed.
    expect(md).toContain("| output tokens/sec, end to end (includes prompt processing) |");
  });

  it("a model that always answers garbage scores 0 with every attempt invalid", async () => {
    const { provider, calls } = fakeProvider(() => "Sorry, I can't help with that.");
    const [run] = await runProgramMode(tasks, provider, programPrompt, DEFAULT_SAMPLING, [1]);
    expect(calls).toHaveLength(tasks.length * 3);
    expect(run!.metrics.passK.n).toBe(0);
    expect(run!.metrics.invalidAttempts).toEqual({ n: tasks.length * 3, of: tasks.length * 3 });
    // Held-out accuracy counts every task's held-out examples, not only tasks with an accepted program.
    const heldOut = tasks.reduce((s, t) => s + t.heldOut.length, 0);
    expect(run!.metrics.heldOutExamples).toEqual({ n: 0, of: heldOut });
    expect(run!.metrics.outcomes).toEqual({ "invalid-json": tasks.length });
  });

  it("no prompt in a full run contains any held-out output", async () => {
    const { provider, calls } = fakeProvider(() => '{"unit":"line","rules":[]}'); // identity: fails most tasks, so retries happen
    await runProgramMode(tasks, provider, programPrompt, DEFAULT_SAMPLING, [1]);
    expect(calls.length).toBeGreaterThan(tasks.length); // retries happened
    // Held-out strings that also appear among a task's shown examples are legitimately visible.
    const leaks = calls.flatMap((c) => {
      const task = taskFor(c.messages[1]!.content);
      const text = c.messages.map((m) => m.content).join("\n");
      const shown = new Set(task.examples.flatMap((s) => [s.input, s.output]));
      return task.heldOut
        .flatMap((ex) => [ex.input, ex.output])
        .filter((s) => s.length > 2 && !shown.has(s) && text.includes(JSON.stringify(s)))
        .map((s) => `${task.id}: ${JSON.stringify(s)}`);
    });
    expect(leaks).toEqual([]);
  });
});

describe("results are saved as each task finishes", () => {
  it("calls onResult once per task (per seed) before the run returns", async () => {
    const seen: string[] = [];
    const { provider } = fakeProvider((messages) => JSON.stringify(taskFor(messages.at(-1)!.content).reference));
    await runProgramMode(
      tasks.slice(0, 3),
      provider,
      programPrompt,
      DEFAULT_SAMPLING,
      [1, 2],
      () => {},
      (r) => seen.push(`${r.seed}:${r.taskId}`),
    );
    expect(seen).toHaveLength(6);
    expect(seen[0]).toBe(`1:${tasks[0]!.id}`);
  });

  it("keeps the results finished before a provider failure", async () => {
    const seen: string[] = [];
    // Fail on the first call of the second task: the first task's result must already be saved.
    const failAt = tasks[0]!.heldOut.length + 1;
    const { provider } = fakeProvider((_, call) => {
      if (call === failAt) throw new Error("server went away");
      return '"x"';
    });
    await expect(
      runRewriteMode(
        tasks.slice(0, 5),
        provider,
        rewritePrompt,
        DEFAULT_SAMPLING,
        () => {},
        (r) => seen.push(r.taskId),
      ),
    ).rejects.toThrow("server went away");
    expect(seen).toEqual([tasks[0]!.id]);
  });
});

describe("run helpers", () => {
  it("dates results by the local calendar day, not UTC", () => {
    expect(localDate(new Date(2001, 1, 3, 23, 30))).toBe("2001-02-03"); // 11:30 pm local
  });

  it("shows home paths as ~ and expands them back", () => {
    expect(displayPath("/Users/me/models/mlx/x", "/Users/me")).toBe("~/models/mlx/x");
    expect(displayPath("/Users/meow/x", "/Users/me")).toBe("/Users/meow/x");
    expect(expandHome("~/models/mlx/x", "/Users/me")).toBe("/Users/me/models/mlx/x");
    expect(expandHome("/opt/x", "/Users/me")).toBe("/opt/x");
  });

  it("verifies the served model against the lockfile path", () => {
    const path = "/Users/me/models/mlx/phi-4-mini-instruct-4bit";
    // mlx_lm.server lists Hugging Face cache models too; only the exact converted path counts.
    expect(matchServedModel(["mlx-community/other-4bit", path], path)).toEqual({ verified: true, servedPath: path });
    expect(matchServedModel(["/Users/me/models/mlx/gemma-4-e4b-it-4bit"], path)).toMatchObject({
      verified: false,
      reason: expect.stringContaining("isn't serving"),
    });
    expect(matchServedModel([path], undefined)).toMatchObject({ verified: false });
  });

  it("records where a verified conversion came from, how it was converted and what it wrote", () => {
    const marker = {
      repo: "microsoft/Phi-4-mini-instruct",
      revision: "a".repeat(40),
      quantization: "4bit",
      source_files_sha256: "1".repeat(64),
      converted_with: { mlx: "0.32.4", "mlx-lm": "0.32.2", transformers: "5.22.2" },
      converted_files_sha256: "2".repeat(64),
    };
    expect(conversionMetadata(marker)).toEqual({
      "source repo": "microsoft/Phi-4-mini-instruct",
      "source commit": "a".repeat(40),
      "source files sha256": "1".repeat(64),
      quantization: "4bit",
      "converted with": "mlx 0.32.4, mlx-lm 0.32.2, transformers 5.22.2",
      "converted files sha256": "2".repeat(64),
    });
    // An unverified run has no marker to go on, so the summary leaves these rows out.
    expect(Object.values(conversionMetadata(undefined)).every((v) => v === undefined)).toBe(true);
  });

  // What `ps -ww -o command=` shows for the server serve-model.sh starts (uv runs the venv's script).
  const mlx = "/Users/me/models/mlx/phi-4-mini-instruct-4bit";
  const serveCommand = `/repo/.venv/bin/python3 /repo/.venv/bin/mlx_lm.server --model ${mlx} --host 127.0.0.1 --port 8080`;

  it("accepts only the server options serve-model.sh uses", () => {
    expect(serverCommandMismatch(serveCommand)).toBeUndefined();
    expect(serverCommandMismatch(`python -m mlx_lm.server --model=${mlx} --port=8081`)).toBeUndefined();
    // Options that change the answers without changing /v1/models, abbreviations included.
    for (const extra of ["--adapter-path adapters/x", "--adapter adapters/x", "--draft-model d", "--chat-template t"]) {
      expect(serverCommandMismatch(`${serveCommand} ${extra}`)).toMatch(
        new RegExp(`options serve-model.sh doesn't use: ${extra.split(" ")[0]}$`),
      );
    }
    expect(serverCommandMismatch("python3 fake_server.py --port 8080")).toMatch(/isn't mlx_lm\.server/);
    expect(serverCommandMismatch(undefined)).toMatch(/couldn't read/);
  });

  it("verifies path, conversion, server options, then that the server loaded that conversion", () => {
    const checked: string[] = [];
    const started = Date.UTC(2001, 1, 3, 9);
    const check = (over: { conversion?: string; command?: string; ids?: string[]; convertedAt?: number } = {}) =>
      verifyServedModel({
        servedIds: over.ids ?? [mlx],
        expectedPath: mlx,
        checkConversion: (dir) => {
          checked.push(dir);
          return over.conversion;
        },
        serverProcess: () => ({ command: over.command ?? serveCommand, startedAtMs: started }),
        convertedAt: () => over.convertedAt ?? started - 60_000,
      });
    expect(check()).toEqual({ verified: true, servedPath: mlx });
    expect(checked).toEqual([mlx]); // the folder the server lists, as the lockfile records it
    expect(resultsRoot("/r/results", check().verified)).toBe("/r/results");
    const changed = `the files in ${mlx} changed after the conversion wrote them`;
    const adapter = `${serveCommand} --adapter-path adapters/x`;
    const failures = [
      check({ ids: ["/other"] }),
      check({ conversion: changed, command: adapter, convertedAt: started + 1 }),
      check({ command: adapter, convertedAt: started + 1 }),
      // Rebuilt at the same path after the server loaded the old one: every other check still passes.
      check({ convertedAt: started + 60_000 }),
    ];
    expect(failures.map((c) => (c.verified ? "verified" : c.reason))).toEqual([
      expect.stringContaining("isn't serving"),
      changed, // reported before the server options
      expect.stringContaining("--adapter-path"), // reported before the start time
      `the conversion in ${mlx} was written after the server started, and the server keeps the model it loaded then; restart the server`,
    ]);
    for (const c of failures) expect(resultsRoot("/r/results", c.verified)).toBe("/r/results/unverified");
  });

  it("can't verify a server whose start time or conversion time can't be read", () => {
    expect(staleServerReason(undefined, 1, mlx)).toMatch(/couldn't read when the process serving the port started/);
    expect(staleServerReason(1, undefined, mlx)).toMatch(/couldn't read when the conversion in .* was written/);
    expect(staleServerReason(2, 1, mlx)).toBeUndefined();
    expect(staleServerReason(2, 2, mlx)).toBeUndefined();
  });

  it("reads the server's start time from ps's elapsed time", () => {
    expect(["00:07", "1:02", "01:02:03", "2-01:02:03"].map(elapsedSeconds)).toEqual([7, 62, 3723, 176_523]);
    for (const bad of ["", "7", "1:2:3:4", "a:b", "-1:00"]) expect(elapsedSeconds(bad)).toBeUndefined();
    const now = Date.UTC(2001, 1, 3, 12);
    expect(parseServerProcess(`   01:02:03 ${serveCommand}`, now)).toEqual({
      command: serveCommand,
      startedAtMs: now - 3723_000,
    });
    // No elapsed time: the command is still checked, and the start time is reported as unreadable.
    expect(parseServerProcess(serveCommand, now)).toEqual({ command: serveCommand });
    expect(parseServerProcess("   ", now)).toBeUndefined();
  });

  it("formats each path in the mismatch reason, leaving look-alike paths intact", () => {
    const show = (p: string) => displayPath(p, "/Users/me");
    const r = matchServedModel(["/Users/meow/models/x", "/Users/me/models/mlx/y"], "/Users/me/models/mlx/z", show);
    expect(r).toEqual({
      verified: false,
      reason: "server isn't serving ~/models/mlx/z; it reports: /Users/meow/models/x, ~/models/mlx/y",
    });
  });
});

describe("the server runtime", () => {
  const replying = (...fingerprints: (string | undefined)[]): Provider => {
    let n = 0;
    return {
      name: "fake",
      complete: async () => {
        const f = fingerprints[Math.min(n++, fingerprints.length - 1)];
        return { text: "x", latencyMs: 1, usage: {}, ...(f === undefined ? {} : { systemFingerprint: f }) };
      },
    };
  };
  const opts = { temperature: 0, maxTokens: 8 };

  it("is the system_fingerprint the server sent, the same on every call", async () => {
    const p = pinRuntime(replying("0.32.2-0.32.4-a"));
    expect(p.runtime()).toBeUndefined();
    await p.complete([], opts);
    await p.complete([], opts);
    expect(p.runtime()).toEqual({ fingerprint: "0.32.2-0.32.4-a" });
    const silent = pinRuntime(replying(undefined));
    await silent.complete([], opts);
    await silent.complete([], opts);
    expect(silent.runtime()).toEqual({ fingerprint: undefined });
  });

  it("stops the run when it changes part way, or appears or disappears", async () => {
    for (const [first, then] of [
      ["0.32.2-0.32.4-a", "0.33.0-0.33.0-a"],
      [undefined, "0.32.2-0.32.4-a"],
      ["0.32.2-0.32.4-a", undefined],
    ]) {
      const p = pinRuntime(replying(first, then));
      await p.complete([], opts);
      await expect(p.complete([], opts)).rejects.toThrow(/system_fingerprint changed during the run/);
    }
    const p = pinRuntime(replying("0.32.2-0.32.4-a", "0.33.0-0.33.0-a"));
    await p.complete([], opts);
    await expect(p.complete([], opts)).rejects.toThrow(
      'from "0.32.2-0.32.4-a" to "0.33.0-0.33.0-a", so it was restarted with different software',
    );
  });

  it("goes in the run's metadata after the served model", () => {
    const meta = { date: "d", "served model": "~/m", "repo commit": "abc" };
    expect(Object.entries(withRuntime(meta, { fingerprint: "f" }))).toEqual([
      ["date", "d"],
      ["served model", "~/m"],
      ["server runtime (system_fingerprint)", "f"],
      ["repo commit", "abc"],
    ]);
    expect(withRuntime({ date: "d" }, { fingerprint: undefined })).toEqual({
      date: "d",
      "server runtime (system_fingerprint)": "not reported by the server",
    });
    // Before the first completion there's nothing to record.
    expect(withRuntime(meta, undefined)).toBe(meta);
  });
});

describe("CLI option parsing", () => {
  it("accepts positive whole numbers only", () => {
    expect(positiveInt("--k", "3")).toBe(3);
    for (const bad of ["0", "-2", "1.5", "three", "", " "]) expect(() => positiveInt("--k", bad)).toThrow(/--k/);
  });

  it("parses seeds and rejects empty or non-integer items", () => {
    expect(parseSeeds("1,2, 3")).toEqual([1, 2, 3]);
    expect(parseSeeds("0")).toEqual([0]);
    for (const bad of ["1,,2", "", "1,a", "1.5", "-1"]) expect(() => parseSeeds(bad)).toThrow(/--seeds/);
  });

  it("rejects a repeated seed, which would rerun the same trajectory as if it were independent", () => {
    for (const bad of ["1,1", "1,2,1", "3, 3"]) expect(() => parseSeeds(bad)).toThrow(/must not repeat/);
  });

  it("the harness rejects k < 1 instead of crashing on an empty attempt list", async () => {
    const { provider } = fakeProvider(() => "{}");
    await expect(
      runProgramMode(tasks.slice(0, 1), provider, programPrompt, { ...DEFAULT_SAMPLING, k: 0 }, [1]),
    ).rejects.toThrow(/k must be/);
  });
});

describe("rewrite mode, end to end", () => {
  it("an oracle that applies each reference program scores 100%", async () => {
    const { runLine } = await import("../text/interpret.ts");
    const { provider } = fakeProvider((messages) => {
      const content = messages.at(-1)!.content;
      const input = JSON.parse(/Line to reformat: (".*")/.exec(content)![1]!) as string;
      return JSON.stringify(runLine(taskFor(content).reference, input));
    });
    const run = await runRewriteMode(tasks, provider, rewritePrompt, DEFAULT_SAMPLING);
    expect(run.metrics.pass).toEqual({ n: tasks.length, of: tasks.length });
    const md = rewriteSummary("Direct rewrite: oracle", { temperature: 0 }, run.metrics, run.results);
    expect(md).toContain(`| tasks with every held-out line correct | ${tasks.length}/${tasks.length} (100.0%) |`);
  });
});
