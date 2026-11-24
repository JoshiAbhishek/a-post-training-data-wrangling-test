// The evaluation loop. Synthesis only ever sees a PublicTask (description + shown examples);
// held-out examples are used afterwards, for scoring only, and are never fed back.
import type { Completion, Message, Provider } from "../llm/provider.ts";
import { LineTooLongError, runLine } from "../text/interpret.ts";
import { validateProgram } from "../text/schema.ts";
import type { Program } from "../text/types.ts";
import { extractJsonObject, extractLine, looksLikeProgram, noLineReason } from "./extract.ts";
import {
  type PromptTemplate,
  type PublicTask,
  programMessages,
  quote,
  retryMessage,
  rewriteMessages,
  workedExamples,
} from "./prompt.ts";
import type { Category, Example, Task } from "./task.ts";

export type SamplingConfig = {
  k: number; // attempts per task
  firstTemperature: number; // attempt 1 (0 → deterministic pass@1)
  retryTemperature: number;
  seed: number; // used for retries; attempt 1 is greedy
  maxTokens: number;
};

export const DEFAULT_SAMPLING: SamplingConfig = {
  k: 3,
  firstTemperature: 0,
  retryTemperature: 0.7,
  seed: 1,
  maxTokens: 1024,
};

/** "truncated": hit the max-token limit before a complete program; counted as an invalid program. */
export type AttemptVerdict = "passes-shown" | "truncated" | "invalid-json" | "schema-error" | "wrong-on-shown";
export type FailureCategory = Exclude<AttemptVerdict, "passes-shown"> | "fits-shown-fails-held-out";
export const INVALID_VERDICTS: readonly AttemptVerdict[] = ["truncated", "invalid-json", "schema-error"];

/** `error` is set when the program stopped on this input (it hit a size or line-length limit); `got` is then "". */
export type Mismatch = { input: string; expected: string; got: string; error?: string };

export type Attempt = {
  attempt: number;
  temperature: number;
  seed?: number;
  /**
   * Messages added to the conversation for this call. The system prompt is left out (it is fixed per
   * prompt version, recorded in the run metadata), so attempt 1 holds the task's user message and each
   * retry holds the previous answer plus the feedback.
   */
  newMessages: Message[];
  response: string;
  finishReason?: string;
  latencyMs: number;
  promptTokens?: number; // undefined if the server reported none
  completionTokens?: number;
  verdict: AttemptVerdict;
  errors?: string[];
  program?: Program;
  shownMismatches?: Mismatch[];
};

export type HeldOutScore = { passed: number; total: number; mismatches: Mismatch[] };

export type ProgramTaskResult = {
  mode: "program";
  taskId: string;
  categories: Category[];
  seed: number;
  attempts: Attempt[];
  /** Attempt 1 was reused from an earlier seed (temperature 0), so it made no model call in this run. */
  firstAttemptReused: boolean;
  acceptedAttempt?: number;
  heldOut?: HeldOutScore; // only for an accepted program
  heldOutTotal: number; // held-out examples in the task, accepted or not
  pass1: boolean; // first attempt valid, passes shown AND held-out
  passK: boolean; // an accepted program (within k) passes held-out
  outcome: "pass" | FailureCategory;
};

/** `error` is set when the response held no answer to score; `got` is then "" and the call is wrong. */
export type RewriteCall = {
  input: string;
  expected: string;
  got: string;
  correct: boolean;
  error?: string;
  /** The user message sent (the system prompt is fixed per prompt version). */
  userMessage: string;
  response: string;
  finishReason?: string;
  latencyMs: number;
  promptTokens?: number; // undefined if the server reported none
  completionTokens?: number;
};

/** Direct rewrite is greedy (temperature 0) with no seed, so results carry no seed. */
export type RewriteTaskResult = {
  mode: "rewrite";
  taskId: string;
  categories: Category[];
  calls: RewriteCall[];
  pass: boolean; // every held-out line correct
};

export type TaskResult = ProgramTaskResult | RewriteTaskResult;

/** The only view of a task that is allowed into prompts. */
export const publicView = (t: Task): PublicTask => ({ id: t.id, description: t.description, examples: t.examples });

function runExample(program: Program, e: Example): Mismatch {
  try {
    return { input: e.input, expected: e.output, got: runLine(program, e.input) };
  } catch (err) {
    if (err instanceof LineTooLongError) return { input: e.input, expected: e.output, got: "", error: err.message };
    throw err;
  }
}

/** The examples the program gets wrong. A line that hits the length limit counts as wrong. */
export function runOn(program: Program, examples: readonly Example[]): Mismatch[] {
  return examples.map((e) => runExample(program, e)).filter((m) => m.error !== undefined || m.got !== m.expected);
}

function feedbackFor(a: Attempt): string {
  if (a.verdict === "truncated")
    return "Your answer was cut off at the length limit before the program was complete. Answer with ONLY the JSON program, with no explanation.";
  if (a.verdict === "invalid-json") return `Your answer was not valid JSON: ${a.errors?.join("; ")}`;
  if (a.verdict === "schema-error")
    return `Your program failed validation:\n${a.errors?.map((e) => `- ${e}`).join("\n")}`;
  const lines = (a.shownMismatches ?? []).map(
    (m, i) =>
      `${i + 1}. input: ${quote(m.input)}\n   expected: ${quote(m.expected)}\n   got: ${m.error ? `no output (${m.error})` : quote(m.got)}`,
  );
  return `It produced the wrong output for these examples:\n${lines.join("\n")}`;
}

/** Judge one response against the shown examples only. */
export function judgeAttempt(
  response: string,
  task: PublicTask,
  finishReason?: string,
): Pick<Attempt, "verdict" | "errors" | "program" | "shownMismatches"> {
  const json = extractJsonObject(response);
  const valid = json.ok ? validateProgram(json.value) : undefined;
  // Cut off by the token limit before a complete program: a different problem from malformed JSON or a
  // bad program. A program cut off part way still holds complete inner objects (conditions, positions),
  // and one of those may parse, so at the limit only a complete program counts: the extractor's last
  // program (see looksLikeProgram) with no object opened after it that never closes. A program followed
  // by one doesn't count: the answer cut off is the last one, and the program before it may be a
  // restated worked example. A complete last program is the model's answer even when the reply then ran
  // on to the limit, so it's judged on its own errors: invalid JSON (a trailing comma) or the schema's
  // (a missing "unit", an array around it).
  const malformed = !json.ok && json.malformed === true;
  const complete = json.ok && json.unfinished !== true && looksLikeProgram(json.value);
  if (finishReason === "length" && !malformed && !complete) {
    return { verdict: "truncated", errors: ["the answer hit the max-token limit before the program was complete"] };
  }
  if (!json.ok) return { verdict: "invalid-json", errors: [json.error] };
  if (!valid?.ok) return { verdict: "schema-error", errors: valid?.errors ?? [] };
  const shownMismatches = runOn(valid.program, task.examples);
  return shownMismatches.length
    ? { verdict: "wrong-on-shown", program: valid.program, shownMismatches }
    : { verdict: "passes-shown", program: valid.program };
}

const record = (c: Completion) => ({
  response: c.text,
  finishReason: c.finishReason,
  latencyMs: c.latencyMs,
  promptTokens: c.usage.promptTokens,
  completionTokens: c.usage.completionTokens,
});

/**
 * Generate → validate → check on shown → retry with feedback, up to k attempts. Sees only the public
 * view of the task. `firstAttempt` lets callers reuse a deterministic (temperature 0) attempt 1 across
 * seeds instead of re-running it.
 */
export async function synthesize(
  task: PublicTask,
  provider: Provider,
  prompt: PromptTemplate,
  cfg: SamplingConfig,
  firstAttempt?: Attempt,
): Promise<Attempt[]> {
  if (!Number.isInteger(cfg.k) || cfg.k < 1) throw new Error(`k must be a whole number >= 1, got ${cfg.k}`);
  const attempts: Attempt[] = [];
  let messages = programMessages(prompt, task);
  let recorded = 1; // messages[0] is the system prompt, which isn't stored per attempt
  for (let n = 1; n <= cfg.k; n++) {
    let attempt: Attempt;
    if (n === 1 && firstAttempt) attempt = firstAttempt;
    else {
      const temperature = n === 1 ? cfg.firstTemperature : cfg.retryTemperature;
      const seed = n === 1 ? undefined : cfg.seed * 1000 + n;
      const completion = await provider.complete(messages, { temperature, maxTokens: cfg.maxTokens, seed });
      attempt = {
        attempt: n,
        temperature,
        seed,
        newMessages: messages.slice(recorded),
        ...record(completion),
        ...judgeAttempt(completion.text, task, completion.finishReason),
      };
    }
    recorded = messages.length;
    attempts.push(attempt);
    if (attempt.verdict === "passes-shown") break;
    messages = [
      ...messages,
      { role: "assistant", content: attempt.response },
      retryMessage(prompt, feedbackFor(attempt)),
    ];
  }
  return attempts;
}

/** Score a program on held-out examples. Never called with anything the model will see again. */
export function scoreHeldOut(program: Program, heldOut: readonly Example[]): HeldOutScore {
  const mismatches = runOn(program, heldOut);
  return { passed: heldOut.length - mismatches.length, total: heldOut.length, mismatches };
}

export async function evaluateProgramTask(
  task: Task,
  provider: Provider,
  prompt: PromptTemplate,
  cfg: SamplingConfig,
  firstAttempt?: Attempt,
): Promise<ProgramTaskResult> {
  const attempts = await synthesize(publicView(task), provider, prompt, cfg, firstAttempt);
  const accepted = attempts.find((a) => a.verdict === "passes-shown");
  const heldOut = accepted?.program ? scoreHeldOut(accepted.program, task.heldOut) : undefined;
  const heldOutPass = heldOut !== undefined && heldOut.passed === heldOut.total;
  const last = attempts[attempts.length - 1]!;
  return {
    mode: "program",
    taskId: task.id,
    categories: task.category,
    seed: cfg.seed,
    attempts,
    firstAttemptReused: firstAttempt !== undefined,
    acceptedAttempt: accepted?.attempt,
    heldOut,
    heldOutTotal: task.heldOut.length,
    pass1: accepted?.attempt === 1 && heldOutPass,
    passK: heldOutPass,
    outcome: heldOutPass ? "pass" : accepted ? "fits-shown-fails-held-out" : (last.verdict as FailureCategory),
  };
}

/**
 * The line a direct-rewrite response answers, or why it has none. A response cut off at the token limit
 * counts only if it already held a JSON-string answer, as a cut-off program counts only if it validates:
 * raw text cut off part way isn't an answer, even when it happens to equal the expected line.
 * `workedAnswers`: the prompt's worked-example answers, so a restatement of one isn't taken for the
 * answer (see extractLine).
 */
export function judgeLine(
  response: string,
  finishReason?: string,
  workedAnswers: readonly string[] = [],
): { got: string; error?: string } {
  const answer = extractLine(response, workedAnswers);
  if (answer === undefined) return { got: "", error: noLineReason(response) };
  if (finishReason === "length" && !answer.quoted)
    return { got: "", error: "the answer hit the max-token limit before it was complete" };
  return { got: answer.line };
}

/** Direct rewrite: one call per held-out input; no feedback, no retries. */
export async function evaluateRewriteTask(
  task: Task,
  provider: Provider,
  prompt: PromptTemplate,
  cfg: SamplingConfig,
): Promise<RewriteTaskResult> {
  const view = publicView(task);
  // A worked answer is never a task's expected output (a prompt test checks this, for every task line), so
  // passing over a restatement of one can't cost a right answer. Each is passed only for lines other than
  // its own example's, all the same.
  const worked = workedExamples(prompt);
  const calls: RewriteCall[] = [];
  for (const ex of task.heldOut) {
    const messages = rewriteMessages(prompt, view, ex.input);
    const c = await provider.complete(messages, { temperature: cfg.firstTemperature, maxTokens: cfg.maxTokens });
    const workedAnswers = worked.filter((w) => w.input !== ex.input).map((w) => w.answer);
    const { got, error } = judgeLine(c.text, c.finishReason, workedAnswers);
    calls.push({
      input: ex.input,
      expected: ex.output,
      got,
      correct: error === undefined && got === ex.output,
      ...(error === undefined ? {} : { error }),
      userMessage: messages[messages.length - 1]!.content,
      ...record(c),
    });
  }
  return {
    mode: "rewrite",
    taskId: task.id,
    categories: task.category,
    calls,
    pass: calls.every((c) => c.correct),
  };
}
