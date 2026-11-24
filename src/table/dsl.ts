// dsl.ts — typed table-transformation DSL + reference interpreter.
// Programs are JSON-serializable so an LLM can emit them directly
// and you can validate them with a schema before execution.

export type Scalar = string | number | boolean | null;
export type Row = Record<string, Scalar>;
export type Table = Row[];
export type ColType = "string" | "number" | "boolean" | "date";
export type Schema = Record<string, ColType>;

// ---------- Value-level expressions (foundational transformations) ----------
export type Expr =
  | { op: "col"; name: string }
  | { op: "lit"; value: Scalar }
  // string
  | { op: "upper" | "lower" | "trim"; arg: Expr }
  | { op: "concat"; args: Expr[]; sep?: string }
  | { op: "substr"; arg: Expr; start: number; length?: number }
  | { op: "splitPart"; arg: Expr; sep: string; index: number }
  | { op: "regexExtract"; arg: Expr; pattern: string; group?: number }
  | { op: "replace"; arg: Expr; find: string; with: string }
  // number
  | { op: "add" | "sub" | "mul" | "div"; left: Expr; right: Expr }
  | { op: "round"; arg: Expr; digits?: number }
  | { op: "toNumber"; arg: Expr }
  // date (ISO strings in, ISO strings out)
  | { op: "datePart"; arg: Expr; part: "year" | "month" | "day" }
  | { op: "formatDate"; arg: Expr; format: "YYYY-MM-DD" | "MM/DD/YYYY" | "DD.MM.YYYY" }
  | { op: "parseDate"; arg: Expr; format: "YYYY-MM-DD" | "MM/DD/YYYY" | "DD.MM.YYYY" }
  // logic
  | { op: "eq" | "gt" | "lt"; left: Expr; right: Expr }
  | { op: "and" | "or"; args: Expr[] }
  | { op: "not" | "isNull"; arg: Expr }
  | { op: "if"; cond: Expr; then: Expr; else: Expr }
  | { op: "coalesce"; args: Expr[] };

// ---------- Table-level operations ----------
export type Agg = { fn: "sum" | "avg" | "min" | "max" | "count" | "first"; col?: string; as: string };

export type TableOp =
  | { op: "select"; cols: string[] }
  | { op: "drop"; cols: string[] }
  | { op: "rename"; map: Record<string, string> }
  | { op: "derive"; as: string; expr: Expr }
  | { op: "filter"; where: Expr }
  | { op: "sort"; by: { col: string; desc?: boolean }[] }
  | { op: "dedupe"; cols?: string[] }
  | { op: "limit"; n: number }
  | { op: "splitColumn"; col: string; sep: string; into: string[] }
  | { op: "groupBy"; keys: string[]; aggs: Agg[] }
  | { op: "pivot"; index: string[]; column: string; value: string; agg?: Agg["fn"] }
  | { op: "unpivot"; keep: string[]; cols: string[]; keyAs: string; valueAs: string }
  | { op: "join"; right: Table; on: string[]; how?: "inner" | "left" };

export type Program = TableOp[];

export class DslError extends Error {}

// ---------- Interpreter ----------
const pad = (n: number) => String(n).padStart(2, "0");

function parseDate(s: string, fmt: string): Date | null {
  let y: number, m: number, d: number;
  const p = s.split(/[-/.]/).map(Number);
  if (p.length !== 3 || p.some(isNaN)) return null;
  if (fmt === "YYYY-MM-DD") [y, m, d] = p;
  else if (fmt === "MM/DD/YYYY") [m, d, y] = p;
  else [d, m, y] = p;
  const dt = new Date(Date.UTC(y, m - 1, d));
  return dt.getUTCMonth() === m - 1 ? dt : null;
}

function fmtDate(dt: Date, fmt: string): string {
  const y = dt.getUTCFullYear(), m = pad(dt.getUTCMonth() + 1), d = pad(dt.getUTCDate());
  return fmt === "YYYY-MM-DD" ? `${y}-${m}-${d}` : fmt === "MM/DD/YYYY" ? `${m}/${d}/${y}` : `${d}.${m}.${y}`;
}

const num = (v: Scalar): number | null => (v === null || v === "" ? null : Number.isFinite(Number(v)) ? Number(v) : null);
/**
 * A computed number, or null if it isn't finite: an overflow (1e200 * 1e200) is no number, like a division
 * by zero. Tables are stored as JSON, which can't hold Infinity or NaN, so no op produces them.
 */
const finite = (n: number): number | null => (Number.isFinite(n) ? n : null);
const str = (v: Scalar): string | null => (v === null ? null : String(v));

export function evalExpr(e: Expr, row: Row): Scalar {
  const ev = (x: Expr) => evalExpr(x, row);
  switch (e.op) {
    case "col":
      if (!(e.name in row)) throw new DslError(`Unknown column: ${e.name}`);
      return row[e.name];
    case "lit": return e.value;
    case "upper": { const s = str(ev(e.arg)); return s?.toUpperCase() ?? null; }
    case "lower": { const s = str(ev(e.arg)); return s?.toLowerCase() ?? null; }
    case "trim": { const s = str(ev(e.arg)); return s?.trim() ?? null; }
    case "concat": return e.args.map((a) => str(ev(a)) ?? "").join(e.sep ?? "");
    case "substr": { const s = str(ev(e.arg)); return s === null ? null : s.substr(e.start, e.length); }
    case "splitPart": { const s = str(ev(e.arg)); return s?.split(e.sep)[e.index] ?? null; }
    case "regexExtract": {
      const s = str(ev(e.arg)); if (s === null) return null;
      const m = new RegExp(e.pattern).exec(s); return m?.[e.group ?? 0] ?? null;
    }
    case "replace": { const s = str(ev(e.arg)); return s?.split(e.find).join(e.with) ?? null; }
    case "add": case "sub": case "mul": case "div": {
      const a = num(ev(e.left)), b = num(ev(e.right));
      if (a === null || b === null) return null;
      if (e.op === "div") return b === 0 ? null : finite(a / b);
      return finite(e.op === "add" ? a + b : e.op === "sub" ? a - b : a * b);
    }
    case "round": { const a = num(ev(e.arg)); const f = 10 ** (e.digits ?? 0); return a === null ? null : finite(Math.round(a * f) / f); }
    case "toNumber": return num(ev(e.arg));
    case "parseDate": { const s = str(ev(e.arg)); const d = s && parseDate(s, e.format); return d ? fmtDate(d, "YYYY-MM-DD") : null; }
    case "formatDate": { const s = str(ev(e.arg)); const d = s && parseDate(s, "YYYY-MM-DD"); return d ? fmtDate(d, e.format) : null; }
    case "datePart": {
      const s = str(ev(e.arg)); const d = s && parseDate(s, "YYYY-MM-DD"); if (!d) return null;
      return e.part === "year" ? d.getUTCFullYear() : e.part === "month" ? d.getUTCMonth() + 1 : d.getUTCDate();
    }
    case "eq": return ev(e.left) === ev(e.right);
    case "gt": case "lt": {
      const a = ev(e.left), b = ev(e.right); if (a === null || b === null) return false;
      return e.op === "gt" ? a > b : a < b;
    }
    case "and": return e.args.every((a) => ev(a) === true);
    case "or": return e.args.some((a) => ev(a) === true);
    case "not": return ev(e.arg) !== true;
    case "isNull": return ev(e.arg) === null;
    case "if": return ev(e.cond) === true ? ev(e.then) : ev(e.else);
    case "coalesce": { for (const a of e.args) { const v = ev(a); if (v !== null) return v; } return null; }
  }
}

function aggregate(rows: Table, a: Agg): Scalar {
  if (a.fn === "count") return rows.length;
  const vals = rows.map((r) => r[a.col!]).filter((v) => v !== null);
  if (a.fn === "first") return vals[0] ?? null;
  const ns = vals.map(Number).filter(Number.isFinite);
  if (!ns.length) return null;
  if (a.fn === "sum") return finite(ns.reduce((x, y) => x + y, 0));
  if (a.fn === "avg") return finite(ns.reduce((x, y) => x + y, 0) / ns.length);
  return a.fn === "min" ? Math.min(...ns) : Math.max(...ns);
}

const keyOf = (r: Row, cols: string[]) => JSON.stringify(cols.map((c) => r[c]));

function groupRows(t: Table, keys: string[]): Map<string, Table> {
  const g = new Map<string, Table>();
  for (const r of t) { const k = keyOf(r, keys); (g.get(k) ?? g.set(k, []).get(k)!).push(r); }
  return g;
}

export function applyOp(t: Table, o: TableOp): Table {
  switch (o.op) {
    case "select": return t.map((r) => Object.fromEntries(o.cols.map((c) => [c, r[c] ?? null])));
    case "drop": return t.map((r) => Object.fromEntries(Object.entries(r).filter(([k]) => !o.cols.includes(k))));
    case "rename": return t.map((r) => Object.fromEntries(Object.entries(r).map(([k, v]) => [o.map[k] ?? k, v])));
    case "derive": return t.map((r) => ({ ...r, [o.as]: evalExpr(o.expr, r) }));
    case "filter": return t.filter((r) => evalExpr(o.where, r) === true);
    case "sort": return [...t].sort((a, b) => {
      for (const { col, desc } of o.by) {
        const x = a[col], y = b[col]; if (x === y) continue;
        const c = x === null ? 1 : y === null ? -1 : x < y ? -1 : 1;
        return desc ? -c : c;
      }
      return 0;
    });
    case "dedupe": {
      const seen = new Set<string>();
      return t.filter((r) => { const k = keyOf(r, o.cols ?? Object.keys(r)); return !seen.has(k) && !!seen.add(k); });
    }
    case "limit": return t.slice(0, o.n);
    case "splitColumn": return t.map((r) => {
      const parts = str(r[o.col])?.split(o.sep) ?? [];
      const out: Row = { ...r }; delete out[o.col];
      o.into.forEach((name, i) => (out[name] = parts[i] ?? null));
      return out;
    });
    case "groupBy": return [...groupRows(t, o.keys).values()].map((rows) => ({
      ...Object.fromEntries(o.keys.map((k) => [k, rows[0][k]])),
      ...Object.fromEntries(o.aggs.map((a) => [a.as, aggregate(rows, a)])),
    }));
    case "pivot": {
      const cols = [...new Set(t.map((r) => String(r[o.column])))].sort();
      return [...groupRows(t, o.index).values()].map((rows) => {
        const out: Row = Object.fromEntries(o.index.map((k) => [k, rows[0][k]]));
        for (const c of cols)
          out[c] = aggregate(rows.filter((r) => String(r[o.column]) === c), { fn: o.agg ?? "first", col: o.value, as: c });
        return out;
      });
    }
    case "unpivot": return t.flatMap((r) => o.cols.map((c) => ({
      ...Object.fromEntries(o.keep.map((k) => [k, r[k]])), [o.keyAs]: c, [o.valueAs]: r[c] ?? null,
    })));
    case "join": {
      const idx = groupRows(o.right, o.on);
      return t.flatMap((l) => {
        const m = idx.get(keyOf(l, o.on));
        if (m) return m.map((r) => ({ ...r, ...l }));
        return o.how === "left" ? [{ ...l }] : [];
      });
    }
  }
}

export function run(program: Program, input: Table): Table {
  return program.reduce(applyOp, input);
}

/**
 * A row as a comparable string: keys sorted, numbers rounded to 6 decimal places. No op produces Infinity
 * or NaN (see `finite`), but a table built in code can hold them, and JSON would write them as null; they're
 * tagged instead, so none of them equals null or another. A cell is never an array, so a tag can't collide.
 */
const cellKey = (v: Scalar) => (typeof v !== "number" ? v : Number.isFinite(v) ? +v.toFixed(6) : ["number", String(v)]);
const rowKey = (r: Row) => JSON.stringify(Object.keys(r).sort().map((k) => [k, cellKey(r[k] ?? null)]));

/** Table equality, ignoring row order unless `ordered` — the verifier for PBE and RL rewards. */
export function tablesEqual(a: Table, b: Table, ordered = false): boolean {
  if (a.length !== b.length) return false;
  const A = a.map(rowKey), B = b.map(rowKey);
  if (!ordered) { A.sort(); B.sort(); }
  return A.every((x, i) => x === B[i]);
}

/**
 * Whether row order is part of a task's answer, from the last op in its reference program that sets
 * the order. A sort fixes an order every correct program must reproduce, and with no sort, groupBy or
 * pivot the rows keep the input's order, so either way order counts. A groupBy or pivot after the last
 * sort orders its groups by where each first appears, which depends on how the program is written
 * (filter before or after grouping, a sort that no longer shows), so then it doesn't.
 */
export function ordersRows(reference: Program): boolean {
  const last = reference.findLast((o) => o.op === "sort" || o.op === "groupBy" || o.op === "pivot");
  return last === undefined || last.op === "sort";
}

/** How many `want` rows `got` matches: in order (their longest common subsequence), or as a multiset. */
function matchedRows(got: string[], want: string[], ordered: boolean): number {
  if (ordered) {
    let prev = new Array<number>(want.length + 1).fill(0);
    for (const g of got) {
      const cur = [0];
      for (let j = 0; j < want.length; j++) cur.push(g === want[j] ? prev[j] + 1 : Math.max(prev[j + 1], cur[j]));
      prev = cur;
    }
    return prev[want.length];
  }
  const left = new Map<string, number>();
  for (const w of want) left.set(w, (left.get(w) ?? 0) + 1);
  let hit = 0;
  for (const g of got) { const c = left.get(g) ?? 0; if (c) { hit++; left.set(g, c - 1); } }
  return hit;
}

/**
 * Partial-credit reward in [0,1] for RL: 1 for the expected table, else 0.8 × the fraction of expected
 * rows matched. Pass `ordered = ordersRows(reference)`: rows then count only in order, so a program that
 * skips a required sort, or reorders rows nobody asked it to, doesn't earn full credit.
 */
export function reward(program: Program, input: Table, expected: Table, ordered: boolean): number {
  try {
    const got = run(program, input);
    if (tablesEqual(got, expected, ordered)) return 1;
    const hit = matchedRows(got.map(rowKey), expected.map(rowKey), ordered);
    return 0.8 * (hit / Math.max(expected.length, got.length));
  } catch {
    return 0; // invalid program
  }
}
