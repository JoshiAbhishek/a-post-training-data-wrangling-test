import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { ordersRows, reward, run, tablesEqual, type Program, type Table } from "./dsl.ts";

// Regression guard for the parked table prototype: the seeded generator reproduces the committed samples
// exactly, and every one re-verifies.
describe("table DSL samples", () => {
  const text = readFileSync(new URL("../../samples/train.sample.jsonl", import.meta.url), "utf8");
  const lines = text.trim().split("\n");
  const samples = lines.map((line) => JSON.parse(line) as { program: Program; input: Table; output: Table });

  it("has the expected number of samples", () => {
    expect(lines).toHaveLength(200);
  });

  const synth = (n: number) =>
    execFileSync(process.execPath, [fileURLToPath(new URL("./synth.ts", import.meta.url)), String(n)], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
      maxBuffer: 64 * 1024 * 1024,
    });

  it("is exactly what the seeded generator makes", () => {
    expect(synth(200) === text).toBe(true);
  });

  it("doesn't repeat input tables in a large run", () => {
    // The generator's RNG once lost precision and cycled, so 10,000 records held only 188 distinct inputs.
    const inputs = synth(10_000)
      .trim()
      .split("\n")
      .map((line) => JSON.stringify((JSON.parse(line) as { input: Table }).input));
    expect(inputs).toHaveLength(10_000);
    expect(new Set(inputs).size).toBe(10_000);
  });

  it("re-verifies every sample with reward 1", () => {
    for (const ex of samples) expect(reward(ex.program, ex.input, ex.output, ordersRows(ex.program))).toBe(1);
  });

  it("never gives full reward to an empty program that leaves the rows out of order", () => {
    // Unordered, an empty program earned 1 on the samples whose reference only reorders rows (and maybe
    // keeps them all), though it returns them in the wrong order.
    const free = samples.filter((ex) => reward([], ex.input, ex.output, false) === 1 && !tablesEqual(ex.input, ex.output, true));
    expect(free.length).toBeGreaterThan(0);
    for (const ex of free) expect(reward([], ex.input, ex.output, ordersRows(ex.program))).toBeLessThan(1);
  });

  it("doesn't make a program reproduce the group order a sort before a groupBy left", () => {
    // Without its sort, the reference still makes the right groups, only in another order.
    const sortThenGroup = samples.filter((ex) => {
      const ops = ex.program.map((o) => o.op);
      return ops.includes("sort") && ops.lastIndexOf("groupBy") > ops.lastIndexOf("sort");
    });
    let sameGroups = 0;
    for (const ex of sortThenGroup) {
      const unsorted = ex.program.filter((o) => o.op !== "sort");
      if (!tablesEqual(run(unsorted, ex.input), ex.output)) continue; // the sort changed what's grouped
      sameGroups++;
      expect(reward(unsorted, ex.input, ex.output, ordersRows(ex.program))).toBe(1);
    }
    expect(sameGroups).toBeGreaterThan(0);
  });

  it("never gives full reward to a sort nobody asked for that reorders the rows", () => {
    let reordered = 0;
    for (const ex of samples.filter((s) => !s.program.some((o) => o.op === "sort" || o.op === "groupBy"))) {
      const first = Object.keys(ex.input[0]!)[0]!;
      const spurious: Program = [{ op: "sort", by: [{ col: first, desc: true }] }, ...ex.program];
      if (tablesEqual(run(spurious, ex.input), ex.output, true)) continue; // the sort changed nothing
      reordered++;
      expect(reward(spurious, ex.input, ex.output, ordersRows(ex.program))).toBeLessThan(1);
    }
    expect(reordered).toBeGreaterThan(0);
  });
});

describe("reward", () => {
  const input: Table = [{ n: 3 }, { n: 1 }, { n: 2 }];
  const sortAsc: Program = [{ op: "sort", by: [{ col: "n" }] }];
  const sorted: Table = [{ n: 1 }, { n: 2 }, { n: 3 }];

  it("counts row order unless a groupBy or pivot set it after the last sort", () => {
    const group: Program = [{ op: "groupBy", keys: ["n"], aggs: [{ fn: "count", as: "c" }] }];
    const pivot: Program = [{ op: "pivot", index: ["n"], column: "n", value: "n" }];
    expect(ordersRows(sortAsc)).toBe(true);
    expect(ordersRows([])).toBe(true); // the input's order
    expect(ordersRows([{ op: "filter", where: { op: "lit", value: true } }])).toBe(true);
    expect(ordersRows(group)).toBe(false);
    expect(ordersRows(pivot)).toBe(false);
    expect(ordersRows([...sortAsc, ...group])).toBe(false);
    expect(ordersRows([...group, ...sortAsc])).toBe(true);
    expect(ordersRows([...sortAsc, ...pivot])).toBe(false);
    expect(reward(sortAsc, input, sorted, true)).toBe(1);
    // The right rows in the wrong order: full reward unordered, only the rows matched in order otherwise.
    expect(reward([], input, sorted, false)).toBe(1);
    expect(reward([], input, sorted, true)).toBeCloseTo(0.8 * (2 / 3)); // [1, 2] is in order in [3, 1, 2]
    expect(reward([{ op: "sort", by: [{ col: "n", desc: true }] }], input, sorted, true)).toBeCloseTo(0.8 * (1 / 3));
  });

  it("gives partial credit for rows matched, in order when it counts", () => {
    const keepSmall: Program = [{ op: "filter", where: { op: "lt", left: { op: "col", name: "n" }, right: { op: "lit", value: 3 } } }];
    // Two of three expected rows, in order: 0.8 × 2/3 either way.
    expect(reward([...sortAsc, ...keepSmall], input, sorted, true)).toBeCloseTo(0.8 * (2 / 3));
    expect(reward(keepSmall, input, sorted, false)).toBeCloseTo(0.8 * (2 / 3));
    // Extra rows count against it: the larger table sets the denominator.
    expect(reward([], [...input, { n: 4 }], sorted, false)).toBeCloseTo(0.8 * (3 / 4));
  });

  it("rounds numbers in partial credit as tablesEqual does", () => {
    const expected: Table = [{ x: 0.3 }, { x: 9 }];
    const got: Table = [{ x: 0.1 + 0.2 }]; // 0.30000000000000004
    expect(reward([], got, expected, false)).toBeCloseTo(0.8 * (1 / 2));
  });

  it("never produces a number JSON can't store, and tells non-finite numbers apart in tables built in code", () => {
    const col = (name: string) => ({ op: "col", name }) as const;
    const big: Table = [{ a: 1e200, b: 0, t: 1e-200 }];
    // Overflow is no number, like a division by zero: a reference's output survives being stored as JSON.
    const ops: Program = [
      { op: "derive", as: "mul", expr: { op: "mul", left: col("a"), right: col("a") } },
      { op: "derive", as: "add", expr: { op: "add", left: col("mul"), right: col("a") } },
      { op: "derive", as: "div", expr: { op: "div", left: col("a"), right: col("t") } },
      { op: "derive", as: "byZero", expr: { op: "div", left: col("a"), right: col("b") } },
      { op: "derive", as: "round", expr: { op: "round", arg: col("a"), digits: 200 } },
    ];
    const out = run(ops, big);
    expect(out).toEqual([{ ...big[0], mul: null, add: null, div: null, byZero: null, round: null }]);
    expect(JSON.parse(JSON.stringify(out))).toEqual(out);
    const sums = run([{ op: "groupBy", keys: ["b"], aggs: [{ fn: "sum", col: "a", as: "s" }, { fn: "avg", col: "a", as: "m" }] }], [
      { a: 1.7e308, b: 0 },
      { a: 1.7e308, b: 0 },
    ]);
    expect(sums).toEqual([{ b: 0, s: null, m: null }]);
    // The verifier still keeps them apart, for tables built in code.
    for (const [a, b] of [[Infinity, null], [-Infinity, null], [NaN, null], [Infinity, -Infinity], [Infinity, "Infinity"]] as const)
      expect(tablesEqual([{ r: a }], [{ r: b }])).toBe(false);
    for (const v of [Infinity, -Infinity, NaN]) expect(tablesEqual([{ r: v }], [{ r: v }])).toBe(true);
  });

  it("gives an invalid program 0", () => {
    expect(reward([{ op: "derive", as: "y", expr: { op: "col", name: "missing" } }], input, sorted, false)).toBe(0);
  });
});
