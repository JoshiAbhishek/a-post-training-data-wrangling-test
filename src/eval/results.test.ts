import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { newRunDir, recordRun, resultsRoot } from "./results.ts";

let root: string;
beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "eval-results-"));
});
afterEach(() => rmSync(root, { recursive: true, force: true }));

const read = (dir: string, file: string) => readFileSync(join(dir, file), "utf8");

describe("results folders", () => {
  it("keeps unverified runs under unverified/, apart from real results", () => {
    expect(resultsRoot("/r/results", true)).toBe("/r/results");
    expect(resultsRoot("/r/results", false)).toBe("/r/results/unverified");
  });

  it("creates the root and numbers later runs -run2, -run3", () => {
    const out = join(root, "results", "unverified");
    expect(newRunDir(out, "2001-02-03-m-program")).toBe(join(out, "2001-02-03-m-program"));
    expect(newRunDir(out, "2001-02-03-m-program")).toBe(join(out, "2001-02-03-m-program-run2"));
    expect(newRunDir(out, "2001-02-03-m-program")).toBe(join(out, "2001-02-03-m-program-run3"));
    expect(newRunDir(out, "2001-02-03-m-rewrite")).toBe(join(out, "2001-02-03-m-rewrite"));
  });

  it("never hands out a folder that already exists, even one made by another run", () => {
    mkdirSync(join(root, "x")); // e.g. a concurrent run created it after this one looked
    mkdirSync(join(root, "x-run2"));
    const dir = newRunDir(root, "x");
    expect(dir).toBe(join(root, "x-run3"));
    expect(readdirSync(dir)).toEqual([]);
  });
});

describe("recordRun", () => {
  const meta = { model: "m", seeds: "1" };

  it("says INCOMPLETE while running, then writes the summary and every result", async () => {
    const seenDuringRun: string[] = [];
    const logs: string[] = [];
    await recordRun(
      root,
      "program run for m",
      meta,
      async (onResult: (r: { id: string }) => void) => {
        seenDuringRun.push(read(root, "summary.md"));
        onResult({ id: "a" });
        onResult({ id: "b" });
        seenDuringRun.push(read(root, "attempts.jsonl"));
        return "# done\n";
      },
      { log: (m) => logs.push(m), show: (p) => p.replace(root, "<root>") },
    );
    // An interrupted run (Ctrl-C, kill) would be left with this placeholder.
    expect(seenDuringRun[0]).toMatch(/^# INCOMPLETE: program run for m\n/);
    expect(seenDuringRun[0]).toContain("| seeds | 1 |");
    expect(seenDuringRun[1]).toBe('{"id":"a"}\n{"id":"b"}\n'); // appended as each result finishes
    expect(read(root, "summary.md")).toBe("# done\n");
    expect(logs).toEqual(["wrote <root>/summary.md"]);
  });

  it("keeps the finished results and records the error when the run fails", async () => {
    const logs: string[] = [];
    await expect(
      recordRun(
        root,
        "rewrite run for m",
        meta,
        async (onResult: (r: { id: string }) => void) => {
          onResult({ id: "a" });
          throw new Error("server went away");
        },
        { log: (m) => logs.push(m) },
      ),
    ).rejects.toThrow("server went away");
    expect(read(root, "attempts.jsonl")).toBe('{"id":"a"}\n');
    const summary = read(root, "summary.md");
    expect(summary).toMatch(/^# INCOMPLETE: rewrite run for m\n/);
    expect(summary).toContain("stopped after 1 task result(s)");
    expect(summary).toContain("Error: server went away");
    expect(logs).toEqual([`run failed; 1 result(s) kept in ${root}`]);
  });

  it("reads metadata given as a function each time it writes the summary", async () => {
    let learned: string | undefined;
    const metaNow = () => ({ ...meta, ...(learned === undefined ? {} : { "learned during the run": learned }) });
    await expect(
      recordRun(root, "program run for m", metaNow, async () => {
        expect(read(root, "summary.md")).not.toContain("learned during the run");
        learned = "yes";
        throw new Error("stopped");
      }),
    ).rejects.toThrow("stopped");
    expect(read(root, "summary.md")).toContain("| learned during the run | yes |");
  });

  it("keeps the INCOMPLETE summary current as results are saved, for a run that's killed", async () => {
    let learned: string | undefined;
    const metaNow = () => ({ ...meta, ...(learned === undefined ? {} : { "learned during the run": learned }) });
    let midRun = "";
    await recordRun(root, "program run for m", metaNow, async (onResult: (r: { id: string }) => void) => {
      learned = "yes";
      onResult({ id: "a" });
      // What a kill at this point would leave: no catch block or finished summary runs.
      midRun = read(root, "summary.md");
      return "# done\n";
    });
    expect(midRun).toMatch(/^# INCOMPLETE: program run for m\n/);
    expect(midRun).toContain("| learned during the run | yes |");
  });
});
