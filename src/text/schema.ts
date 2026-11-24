// zod schema for text DSL v0 programs. Model output must pass validateProgram() before it is run.
import { z } from "zod";
import type { Action, Cond, Pos, Program } from "./types.ts";

/** Size limits for programs. The interpreter limits how long a line can grow. */
export const MAX_CONDITION_DEPTH = 8;
export const MAX_LIST_ITEMS = 32;

const searchText = z.string().min(1, "must not be empty");
/** Text a program adds to a line. A line break would split the line, which a line program can't do. */
const addedText = z.string().refine((s) => !/[\r\n]/.test(s), "must not contain a line break (\\n or \\r)");
const nonNegInt = z.number().int().nonnegative();
const occurrence = z
  .number()
  .int()
  .refine((n) => n >= 1 || n === -1, "must be an integer >= 1, or -1 for the last match");
const list = <T extends z.ZodType>(item: T) =>
  z.array(item).max(MAX_LIST_ITEMS, `must have at most ${MAX_LIST_ITEMS} items`);

const CharClassSchema = z.enum(["digit", "letter", "upper", "lower", "space", "punct"]);

const leafConds = [
  z.strictObject({
    op: z.enum(["startsWith", "endsWith", "contains", "equals"]),
    text: searchText,
    ignoreCase: z.boolean().optional(),
  }),
  z.strictObject({ op: z.enum(["startsWithClass", "endsWithClass"]), class: CharClassSchema }),
  z.strictObject({ op: z.enum(["lengthGt", "lengthLt"]), n: nonNegInt }),
  z.strictObject({ op: z.literal("isEmpty") }),
] as const;

/**
 * Conditions nested at most `levels` deep. Built one level at a time rather than recursively, so zod
 * never descends past the limit: a condition nested thousands of levels deep gets a validation error
 * instead of overflowing the stack. At the last level, `and`/`or`/`not` are refused without looking
 * inside them.
 */
function condSchema(levels: number): z.ZodType<Cond> {
  const inner = levels > 1 ? condSchema(levels - 1) : undefined;
  const combinators = inner
    ? [
        z.strictObject({ op: z.enum(["and", "or"]), args: list(inner).min(1) }),
        z.strictObject({ op: z.literal("not"), arg: inner }),
      ]
    : [
        z
          .object({ op: z.enum(["and", "or", "not"]) })
          .refine(() => false, `conditions can be nested at most ${MAX_CONDITION_DEPTH} levels deep`),
      ];
  return z.discriminatedUnion("op", [...leafConds, ...combinators]) as z.ZodType<Cond>;
}

export const CondSchema = condSchema(MAX_CONDITION_DEPTH);

export const PosSchema: z.ZodType<Pos> = z.union([
  z.strictObject({ from: z.enum(["start", "end"]), offset: nonNegInt }),
  z.strictObject({
    anchor: z.enum(["before", "after"]),
    text: searchText,
    ignoreCase: z.boolean().optional(),
    occurrence: occurrence.optional(),
  }),
]);

export const ActionSchema: z.ZodType<Action> = z.discriminatedUnion("op", [
  z.strictObject({ op: z.literal("insert"), text: addedText, at: PosSchema }),
  z.strictObject({ op: z.literal("delete"), at: PosSchema, count: nonNegInt }),
  z.strictObject({
    op: z.literal("deleteText"),
    text: searchText,
    ignoreCase: z.boolean().optional(),
    all: z.boolean().optional(),
  }),
  z.strictObject({ op: z.literal("replaceSpan"), at: PosSchema, count: nonNegInt, with: addedText }),
  z.strictObject({
    op: z.literal("replaceText"),
    find: searchText,
    with: addedText,
    ignoreCase: z.boolean().optional(),
    all: z.boolean().optional(),
  }),
]);

export const ProgramSchema: z.ZodType<Program> = z.strictObject({
  unit: z.literal("line"),
  rules: list(z.strictObject({ when: CondSchema, then: list(ActionSchema) })),
  otherwise: list(ActionSchema).optional(),
});

export type ValidationResult = { ok: true; program: Program } | { ok: false; errors: string[] };

/** Validate untrusted input (e.g. parsed model output). Errors read like `rules.0.then.1.count: ...`. */
export function validateProgram(input: unknown): ValidationResult {
  const r = ProgramSchema.safeParse(input);
  if (r.success) return { ok: true, program: r.data };
  return {
    ok: false,
    errors: r.error.issues.map((i) => `${i.path.length ? i.path.join(".") : "(root)"}: ${i.message}`),
  };
}

/** Parse a JSON string and validate it. Invalid JSON is reported like any other validation error. */
export function parseProgram(json: string): ValidationResult {
  let data: unknown;
  try {
    data = JSON.parse(json);
  } catch (e) {
    return { ok: false, errors: [`(root): invalid JSON: ${(e as Error).message}`] };
  }
  return validateProgram(data);
}
