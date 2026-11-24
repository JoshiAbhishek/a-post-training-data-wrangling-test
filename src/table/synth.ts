// synth.ts — generate (input, output, program) training triples that are
// correct by construction. Run:  node src/table/synth.ts 1000 > train.jsonl
import { run, type Expr, type Program, type Schema, type Table, type TableOp, type ColType } from "./dsl.ts";

// ---------- Seeded RNG for reproducibility ----------
// A linear congruential generator mod 2^31 (full period with these constants). Math.imul keeps the
// product exact: seed * 1103515245 as a float passes 2^53, loses its low bits and falls into a cycle of
// about 10,000 draws, which made a 10,000-record run repeat the same 188 input tables.
let seed = 42;
const rand = () => (seed = (Math.imul(seed, 1103515245) + 12345) & 0x7fffffff) / 2 ** 31;
const pick = <T>(xs: readonly T[]): T => xs[Math.floor(rand() * xs.length)];
const int = (lo: number, hi: number) => lo + Math.floor(rand() * (hi - lo + 1));

// ---------- Random tables ----------
const FIRST = ["Ana", "Ben", "Chen", "Dara", "Eli", "Fatima", "Goro", "Hana"];
const LAST = ["Smith", "Okafor", "Liu", "Patel", "Garcia", "Kim"];
const CITY = ["Austin", "Berlin", "Lagos", "Osaka", "Lima"];
const DEPT = ["eng", "sales", "ops"];

type ColGen = { type: ColType; gen: () => string | number | boolean | null };
const COLS: Record<string, ColGen> = {
  full_name: { type: "string", gen: () => `${pick(FIRST)} ${pick(LAST)}` },
  username: { type: "string", gen: () => `${pick(FIRST).toLowerCase()}.${pick(LAST).toLowerCase()}` },
  city: { type: "string", gen: () => (rand() < 0.1 ? null : pick(CITY)) },
  dept: { type: "string", gen: () => pick(DEPT) },
  amount: { type: "number", gen: () => int(1, 500) * (rand() < 0.5 ? 1 : 1.5) },
  qty: { type: "number", gen: () => int(1, 20) },
  date: { type: "date", gen: () => `${int(2020, 2025)}-${String(int(1, 12)).padStart(2, "0")}-${String(int(1, 28)).padStart(2, "0")}` },
  active: { type: "boolean", gen: () => rand() < 0.6 },
};

function randomTable(): { table: Table; schema: Schema } {
  const names = Object.keys(COLS).filter(() => rand() < 0.6);
  for (const must of ["dept", "amount"]) if (!names.includes(must)) names.push(must);
  const schema: Schema = Object.fromEntries(names.map((n) => [n, COLS[n].type]));
  const table = Array.from({ length: int(4, 10) }, () => Object.fromEntries(names.map((n) => [n, COLS[n].gen()])));
  return { table, schema };
}

// ---------- Type-directed random programs ----------
const colsOf = (s: Schema, t: ColType) => Object.keys(s).filter((k) => s[k] === t);
const col = (name: string): Expr => ({ op: "col", name });

function randomExpr(s: Schema): { expr: Expr; type: ColType } | null {
  const strs = colsOf(s, "string"), nums = colsOf(s, "number"), dates = colsOf(s, "date");
  const options: (() => { expr: Expr; type: ColType } | null)[] = [
    () => strs.length ? { expr: { op: pick(["upper", "lower"] as const), arg: col(pick(strs)) }, type: "string" } : null,
    () => strs.includes("full_name") ? { expr: { op: "splitPart", arg: col("full_name"), sep: " ", index: int(0, 1) }, type: "string" } : null,
    () => strs.includes("username") ? { expr: { op: "regexExtract", arg: col("username"), pattern: "^([^.]+)", group: 1 }, type: "string" } : null,
    () => nums.length >= 2 ? { expr: { op: pick(["add", "mul"] as const), left: col(nums[0]), right: col(nums[1]) }, type: "number" } : null,
    () => nums.length ? { expr: { op: "round", arg: { op: "mul", left: col(pick(nums)), right: { op: "lit", value: pick([1.1, 0.9, 1.07]) } }, digits: 2 }, type: "number" } : null,
    () => dates.length ? { expr: { op: "datePart", arg: col(dates[0]), part: pick(["year", "month"] as const) }, type: "number" } : null,
    () => dates.length ? { expr: { op: "formatDate", arg: col(dates[0]), format: pick(["MM/DD/YYYY", "DD.MM.YYYY"] as const) }, type: "string" } : null,
    () => strs.includes("city") ? { expr: { op: "coalesce", args: [col("city"), { op: "lit", value: "Unknown" }] }, type: "string" } : null,
    () => nums.length ? { expr: { op: "if", cond: { op: "gt", left: col(pick(nums)), right: { op: "lit", value: 100 } }, then: { op: "lit", value: "high" }, else: { op: "lit", value: "low" } }, type: "string" } : null,
  ];
  for (let i = 0; i < 10; i++) { const r = pick(options)(); if (r) return r; }
  return null;
}

let fresh = 0;
function randomOp(s: Schema): { op: TableOp; schema: Schema } | null {
  const keys = Object.keys(s);
  const nums = colsOf(s, "number"), strs = colsOf(s, "string");
  const choices: (() => { op: TableOp; schema: Schema } | null)[] = [
    () => { const r = randomExpr(s); if (!r) return null; const as = `col_${++fresh}`; return { op: { op: "derive", as, expr: r.expr }, schema: { ...s, [as]: r.type } }; },
    () => nums.length ? { op: { op: "filter", where: { op: "gt", left: col(pick(nums)), right: { op: "lit", value: int(5, 200) } } }, schema: s } : null,
    () => { const k = pick(keys); return { op: { op: "sort", by: [{ col: k, desc: rand() < 0.5 }] }, schema: s }; },
    () => { const keep = keys.filter(() => rand() < 0.6); return keep.length ? { op: { op: "select", cols: keep }, schema: Object.fromEntries(keep.map((k) => [k, s[k]])) } : null; },
    () => { const k = pick(keys); const to = `${k}_renamed`; const { [k]: t, ...rest } = s; return { op: { op: "rename", map: { [k]: to } }, schema: { ...rest, [to]: t } }; },
    () => strs.includes("full_name") ? { op: { op: "splitColumn", col: "full_name", sep: " ", into: ["first", "last"] }, schema: (({ full_name, ...r }) => ({ ...r, first: "string", last: "string" }))(s) as Schema } : null,
    () => {
      const key = strs.find((k) => k === "dept" || k === "city"); if (!key || !nums.length) return null;
      const fn = pick(["sum", "avg", "max", "count"] as const), c = pick(nums);
      return { op: { op: "groupBy", keys: [key], aggs: [{ fn, col: c, as: `${fn}_${c}` }] }, schema: { [key]: "string", [`${fn}_${c}`]: "number" } };
    },
    () => nums.length >= 2 && keys.includes("dept") ? {
      op: { op: "unpivot", keep: ["dept"], cols: nums.slice(0, 2), keyAs: "metric", valueAs: "value" },
      schema: { dept: "string", metric: "string", value: "number" },
    } : null,
  ];
  for (let i = 0; i < 10; i++) { const r = pick(choices)(); if (r) return r; }
  return null;
}

function randomProgram(schema: Schema, steps: number): Program {
  const prog: Program = []; let s = schema;
  for (let i = 0; i < steps; i++) {
    const r = randomOp(s); if (!r) break;
    prog.push(r.op); s = r.schema;
    if (r.op.op === "groupBy" || r.op.op === "unpivot") break; // terminal shape-changers
  }
  return prog;
}

// ---------- Emit verified examples ----------
const N = Number(process.argv[2] ?? 100);
let emitted = 0, attempts = 0;
while (emitted < N && attempts++ < N * 20) {
  const { table, schema } = randomTable();
  const program = randomProgram(schema, int(1, 4));
  if (!program.length) continue;
  let output: Table;
  try { output = run(program, table); } catch { continue; }
  // Reject degenerate cases: empty output or no-op programs teach nothing.
  if (!output.length || JSON.stringify(output) === JSON.stringify(table)) continue;

  // PBE split: model sees a few input rows + their outputs; held-out rows test generalization.
  console.log(JSON.stringify({
    task: "synthesize_program",
    schema,
    input: table,
    output,
    program, // target for SFT; for RL, keep it from the model and score with dsl.reward(…, ordersRows(program))
  }));
  emitted++;
}
console.error(`emitted ${emitted} examples in ${attempts} attempts`);
