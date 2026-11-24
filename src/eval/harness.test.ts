import { describe, expect, it } from "vitest";
import type { Completion, Provider } from "../llm/provider.ts";
import type { Program } from "../text/types.ts";
import { fakeProvider, scripted } from "./fake-provider.ts";
import {
  DEFAULT_SAMPLING,
  evaluateProgramTask,
  evaluateRewriteTask,
  judgeAttempt,
  judgeLine,
  publicView,
  scoreHeldOut,
} from "./harness.ts";
import { loadPrompt, promptPath } from "./prompt.ts";
import type { Task } from "./task.ts";

const programPrompt = loadPrompt(promptPath("program-synthesis-v1"));
const rewritePrompt = loadPrompt(promptPath("direct-rewrite-v1"));
const cfg = { ...DEFAULT_SAMPLING, seed: 4 };

const END = { from: "end", offset: 0 } as const;
const appendX: Program = { unit: "line", rules: [], otherwise: [{ op: "insert", text: "X", at: END }] };
const prependX: Program = {
  unit: "line",
  rules: [],
  otherwise: [{ op: "insert", text: "X", at: { from: "start", offset: 0 } }],
};
const json = (p: unknown) => JSON.stringify(p);

/** A task whose held-out data is made of sentinels that must never reach the model. */
const task: Task = {
  id: "append-x",
  description: "Add an X to the end of each line",
  category: ["insert"],
  examples: [
    { input: "shown-a", output: "shown-aX" },
    { input: "shown-b", output: "shown-bX" },
  ],
  heldOut: [
    { input: "HELDOUT-IN-1", output: "HELDOUT-IN-1X" },
    { input: "HELDOUT-IN-2", output: "HELDOUT-IN-2X" },
    { input: "HELDOUT-IN-3", output: "HELDOUT-IN-3X" },
  ],
  reference: appendX,
};
const allText = (calls: { messages: { content: string }[] }[]) =>
  calls.flatMap((c) => c.messages.map((m) => m.content)).join("\n");

describe("judgeAttempt (shown examples only)", () => {
  const view = publicView(task);
  it("classifies invalid JSON, schema errors, wrong-on-shown and passes", () => {
    expect(judgeAttempt("no idea", view).verdict).toBe("invalid-json");
    // An empty response (the provider's null content) is a model failure to retry, not a crash.
    expect(judgeAttempt("", view).verdict).toBe("invalid-json");
    expect(judgeAttempt("", view, "length").verdict).toBe("truncated");
    expect(judgeAttempt(json({ unit: "line", rules: [{ when: { op: "nope" }, then: [] }] }), view)).toMatchObject({
      verdict: "schema-error",
      errors: [expect.stringContaining("rules.0.when.op")],
    });
    expect(judgeAttempt(json(prependX), view)).toMatchObject({
      verdict: "wrong-on-shown",
      shownMismatches: [{ input: "shown-a", expected: "shown-aX", got: "Xshown-a" }, expect.anything()],
    });
    expect(judgeAttempt("```json\n" + json(appendX) + "\n```", view).verdict).toBe("passes-shown");
  });

  it("judges the program, not a condition the model mentions before it", () => {
    const response = `Lines matching {"op":"endsWith","text":"."} are left alone.\n${json(appendX)}`;
    expect(judgeAttempt(response, view).verdict).toBe("passes-shown");
  });

  it("judges the last program, not a worked example restated before it", () => {
    const restated = `Example 1\nProgram:\n${json(prependX)}\n\nThis task\nProgram:\n`;
    expect(judgeAttempt(restated + json(appendX), view).verdict).toBe("passes-shown");
    // Cut off in the real program: truncated, not the restated example graded as wrong on shown.
    const cutOff = restated + '{"unit":"line","rules":[],"otherwise":[{"op":"insert","text":"X",';
    expect(judgeAttempt(cutOff, view, "length").verdict).toBe("truncated");
    // Never closed though the reply ended normally: an invalid answer, not the restated example graded.
    expect(judgeAttempt(cutOff, view, "stop").verdict).toBe("invalid-json");
  });

  it("judges a final program that isn't valid JSON, not the valid draft before it", () => {
    const corrected = `Draft:\n${json(appendX)}\nFinal answer:\n${json(appendX).replace(/\}$/, ",}")}`;
    expect(judgeAttempt(corrected, view, "stop")).toMatchObject({ verdict: "invalid-json" });
    // Complete but malformed, then prose up to the token limit: still invalid JSON, with its parse error
    // as the feedback, not "cut off".
    const ranOn = `${corrected}\nThis program appends an X to every line, because`;
    const judged = judgeAttempt(ranOn, view, "length");
    expect(judged.verdict).toBe("invalid-json");
    expect(judged.errors?.[0]).toMatch(/JSON/);
  });
});

describe("program synthesis", () => {
  it("passes on the first attempt: pass@1 and pass@k", async () => {
    const { provider, calls } = scripted(json(appendX));
    const r = await evaluateProgramTask(task, provider, programPrompt, cfg);
    expect(r).toMatchObject({ pass1: true, passK: true, outcome: "pass", acceptedAttempt: 1 });
    expect(r.heldOut).toEqual({ passed: 3, total: 3, mismatches: [] });
    expect(calls).toHaveLength(1);
    expect(calls[0]?.opts).toEqual({ temperature: 0, maxTokens: 1024, seed: undefined });
  });

  it("retries with feedback, up to k attempts, using sampled temperature and seeds", async () => {
    const { provider, calls } = scripted("not json", json({ unit: "line" }), json(prependX));
    const r = await evaluateProgramTask(task, provider, programPrompt, cfg);
    expect(r.attempts.map((a) => a.verdict)).toEqual(["invalid-json", "schema-error", "wrong-on-shown"]);
    expect(r).toMatchObject({ pass1: false, passK: false, outcome: "wrong-on-shown", heldOut: undefined });
    expect(calls.map((c) => c.opts.temperature)).toEqual([0, 0.7, 0.7]);
    expect(calls.map((c) => c.opts.seed)).toEqual([undefined, 4002, 4003]);

    // Each retry continues the conversation: previous answer, then feedback.
    const second = calls[1]!.messages;
    expect(second.at(-2)).toEqual({ role: "assistant", content: "not json" });
    expect(second.at(-1)?.content).toContain("not valid JSON");
    expect(calls[2]!.messages.at(-1)?.content).toContain("rules: ");
  });

  it("feeds back shown-example mismatches as expected vs got", async () => {
    const { provider, calls } = scripted(json(prependX), json(appendX));
    const r = await evaluateProgramTask(task, provider, programPrompt, cfg);
    expect(r).toMatchObject({ pass1: false, passK: true, acceptedAttempt: 2, outcome: "pass" });
    const feedback = calls[1]!.messages.at(-1)!.content;
    expect(feedback).toContain('input: "shown-a"');
    expect(feedback).toContain('expected: "shown-aX"');
    expect(feedback).toContain('got: "Xshown-a"');
  });

  it("a line past the length limit is a wrong output, and the feedback says why", async () => {
    // Doubles every "a": harmless on the shown inputs, past the limit on a long one.
    const doubleA = { op: "replaceText", find: "a", with: "aa", all: true } as const;
    const doubler: Program = { unit: "line", rules: [], otherwise: Array(14).fill(doubleA) };
    const longShown: Task = { ...task, examples: [{ input: "a", output: "aa" }, ...task.examples] };
    const { provider, calls } = scripted(json(doubler), json(appendX));
    const r = await evaluateProgramTask(longShown, provider, programPrompt, cfg);
    expect(r.attempts[0]).toMatchObject({
      verdict: "wrong-on-shown",
      shownMismatches: [
        { input: "a", expected: "aa", got: "", error: expect.stringContaining("longer than 10,000 characters") },
        expect.anything(),
        expect.anything(),
      ],
    });
    expect(calls[1]!.messages.at(-1)!.content).toContain(
      "got: no output (an action would make the line longer than 10,000 characters, its limit)",
    );

    // Held out: counted as failed, not thrown.
    expect(
      scoreHeldOut(doubler, [
        { input: "a", output: "" },
        { input: "b", output: "b" },
      ]),
    ).toMatchObject({
      passed: 1,
      total: 2,
      mismatches: [{ input: "a", got: "", error: expect.any(String) }],
    });
  });

  it("scores an accepted program on held-out; failing held-out is overfitting", async () => {
    // Fits the shown examples (they all start with "shown-") but not the held-out ones.
    const overfit: Program = {
      unit: "line",
      rules: [{ when: { op: "startsWith", text: "shown-" }, then: [{ op: "insert", text: "X", at: END }] }],
    };
    const r = await evaluateProgramTask(task, scripted(json(overfit)).provider, programPrompt, cfg);
    expect(r).toMatchObject({ pass1: false, passK: false, outcome: "fits-shown-fails-held-out", acceptedAttempt: 1 });
    expect(r.heldOut?.passed).toBe(0);
  });

  it("never scores a program that fails shown, even if it would fit held-out (pass@1 needs both)", async () => {
    // Appends X only to HELDOUT- lines: wrong on both shown examples, right on all held-out ones.
    const heldOutOnly: Program = {
      unit: "line",
      rules: [{ when: { op: "startsWith", text: "HELDOUT" }, then: [{ op: "insert", text: "X", at: END }] }],
    };
    const r = await evaluateProgramTask(task, scripted(json(heldOutOnly)).provider, programPrompt, { ...cfg, k: 1 });
    expect(r).toMatchObject({ pass1: false, passK: false, outcome: "wrong-on-shown", heldOut: undefined });
  });

  it("never puts any held-out input or output into a prompt or feedback", async () => {
    const { provider, calls } = scripted("garbage", json({ unit: "line" }), json(prependX));
    await evaluateProgramTask(task, provider, programPrompt, cfg);
    const text = allText(calls);
    expect(calls).toHaveLength(3);
    expect(text).toContain("shown-a");
    expect(text).not.toContain("HELDOUT");
  });

  it("treats a response cut off at the token limit as truncated, with its own feedback", async () => {
    const calls: { content: string }[][] = [];
    const responses = [
      { text: 'Let me think about this carefully. First, {"unit":"line","rules":[{"when":', finishReason: "length" },
      { text: json(appendX), finishReason: "stop" },
    ];
    const provider = {
      name: "fake",
      async complete(messages: { content: string }[]) {
        calls.push(structuredClone(messages));
        const r = responses[calls.length - 1]!;
        return {
          text: r.text,
          finishReason: r.finishReason,
          latencyMs: 1,
          usage: { promptTokens: 1, completionTokens: 1 },
        };
      },
    };
    const r = await evaluateProgramTask(task, provider as never, programPrompt, cfg);
    expect(r.attempts.map((a) => [a.verdict, a.finishReason])).toEqual([
      ["truncated", "length"],
      ["passes-shown", "stop"],
    ]);
    expect(calls[1]!.at(-1)!.content).toContain("cut off at the length limit");
    // The same unparseable text without a length stop is ordinary invalid JSON.
    expect(judgeAttempt(responses[0]!.text, publicView(task), "stop").verdict).toBe("invalid-json");
  });

  it("calls a cut-off program truncated even when complete inner objects parse", () => {
    const view = publicView(task);
    // Cut off in the second rule: the first rule's condition and position are complete JSON objects.
    const cutOff =
      '{"unit":"line","rules":[{"when":{"op":"startsWith","text":"A"},"then":[{"op":"insert","text":"1",' +
      '"at":{"from":"end","offset":0}}]},{"when":{"op":"startsWith","text":"B';
    expect(judgeAttempt(cutOff, view, "length").verdict).toBe("truncated");
    // A complete program followed by rambling that hit the limit is judged on the program.
    expect(judgeAttempt(json(appendX) + "\nThis program appends".repeat(3), view, "length").verdict).toBe(
      "passes-shown",
    );
  });

  it("calls a malformed program invalid JSON, not a schema error about an inner object", () => {
    const trailingComma = json(appendX).replace("}]}", "},]}");
    expect(judgeAttempt(trailingComma, publicView(task), "stop").verdict).toBe("invalid-json");
  });

  it("stores only the new messages per attempt, not the system prompt", async () => {
    const r = await evaluateProgramTask(task, scripted("nope", json(appendX)).provider, programPrompt, cfg);
    const [first, second] = r.attempts;
    expect(first!.newMessages.map((m) => m.role)).toEqual(["user"]);
    expect(first!.newMessages[0]!.content).toContain(task.description);
    expect(second!.newMessages).toEqual([
      { role: "assistant", content: "nope" },
      expect.objectContaining({ role: "user", content: expect.stringContaining("not valid JSON") }),
    ]);
    expect(JSON.stringify(r)).not.toContain("You write small programs"); // system prompt text
  });

  it("reuses a given first attempt instead of calling the model again", async () => {
    const first = await evaluateProgramTask(task, scripted(json(prependX), json(appendX)).provider, programPrompt, cfg);
    const { provider, calls } = scripted(json(appendX));
    const again = await evaluateProgramTask(task, provider, programPrompt, { ...cfg, seed: 9 }, first.attempts[0]);
    expect(calls).toHaveLength(1); // only the retry
    expect(calls[0]?.opts.seed).toBe(9002);
    expect(again.attempts[0]).toBe(first.attempts[0]);
    expect([first.firstAttemptReused, again.firstAttemptReused]).toEqual([false, true]);
  });
});

describe("a complete program at the token limit", () => {
  const ramble = "\n\nThis works because the action changes every line the task describes, which";
  it("is judged on its own errors, not called cut off", () => {
    const missingUnit = JSON.stringify({ rules: [], otherwise: appendX.otherwise });
    for (const response of [
      `${JSON.stringify(appendX)}\nFinal: ${missingUnit}${ramble}`,
      `[${JSON.stringify(appendX)}]${ramble}`,
      `{"unit":"line","rules":"none"}${ramble}`,
    ])
      expect(judgeAttempt(response, publicView(task), "length").verdict).toBe("schema-error");
    // A valid one passes, as before.
    expect(judgeAttempt(`${JSON.stringify(appendX)}${ramble}`, publicView(task), "length").verdict).toBe(
      "passes-shown",
    );
  });

  it("is still cut off when the reply ends inside a program or after a salvaged piece of one", () => {
    for (const response of [
      `${JSON.stringify(appendX)}\nFor this task: {"unit":"line","rules":[{"when":`,
      '{"unit":"line","rules":[{"when":{"op":"isEmpty"},"then":[',
      `${JSON.stringify(appendX)}\nFor this task: {"un`,
      `Final: [${JSON.stringify(appendX)}`,
    ])
      expect(judgeAttempt(response, publicView(task), "length").verdict).toBe("truncated");
    // The same array, ended without its "]" though the reply stopped normally, is invalid JSON.
    expect(judgeAttempt(`Final: [${JSON.stringify(appendX)}`, publicView(task), "stop").verdict).toBe("invalid-json");
  });
});

describe("direct rewrite", () => {
  it("makes one greedy call per held-out input and scores exact matches", async () => {
    const { provider, calls } = fakeProvider((messages) => {
      const line = /Line to reformat: (".*")/.exec(messages.at(-1)!.content)![1]!;
      return JSON.stringify(JSON.parse(line) + "X");
    });
    const r = await evaluateRewriteTask(task, provider, rewritePrompt, cfg);
    expect(r.pass).toBe(true);
    expect(r.calls.map((c) => c.correct)).toEqual([true, true, true]);
    expect(calls.map((c) => c.opts)).toEqual([0, 1, 2].map(() => ({ temperature: 0, maxTokens: 1024 })));
    expect(r).not.toHaveProperty("seed"); // no seed is sent, so none is recorded
    expect(r.calls[0]!.userMessage).toContain('Line to reformat: "HELDOUT-IN-1"');
  });

  it("grades the model's own answer after a restated worked example, on lines other than the worked one", async () => {
    const echo = 'Line to reformat: "hello world"\nAnswer: "[hello world]"\n\n';
    const { provider } = fakeProvider((messages) => {
      const line = /Line to reformat: (".*")\nAnswer:$/.exec(messages.at(-1)!.content)![1]!;
      return echo + JSON.stringify(JSON.parse(line) + "X");
    });
    const r = await evaluateRewriteTask(task, provider, rewritePrompt, cfg);
    expect(r.calls.map((c) => [c.got, c.correct])).toEqual([
      ["HELDOUT-IN-1X", true],
      ["HELDOUT-IN-2X", true],
      ["HELDOUT-IN-3X", true],
    ]);
    // On the worked example's own line, its answer may be the model's, so it's taken as it is.
    const worked: Task = { ...task, heldOut: [{ input: "hello world", output: "hello worldX" }] };
    const again = await evaluateRewriteTask(worked, provider, rewritePrompt, cfg);
    expect(again.calls[0]).toMatchObject({ got: "[hello world]", correct: false });
  });

  it("fails the task if any line is wrong", async () => {
    const r = await evaluateRewriteTask(task, scripted('"nope"').provider, rewritePrompt, cfg);
    expect(r.pass).toBe(false);
    expect(r.calls[0]).toMatchObject({ expected: "HELDOUT-IN-1X", got: "nope", correct: false });
  });

  it("scores an empty response, or raw text cut off at the token limit, as no answer", async () => {
    // Held-out lines that should become empty: a blank or cut-off response must not match them.
    const blanking: Task = {
      ...task,
      heldOut: [
        { input: "HELDOUT-IN-1", output: "" },
        { input: "HELDOUT-IN-2", output: "" },
        { input: "HELDOUT-IN-3", output: "" },
      ],
    };
    const replies: Completion[] = [
      { text: "", finishReason: "stop", latencyMs: 1, usage: {} },
      { text: "", finishReason: "length", latencyMs: 1, usage: {} },
      { text: 'Answer: ""\n\nThis empties the', finishReason: "length", latencyMs: 1, usage: {} },
    ];
    let n = 0;
    const provider: Provider = { name: "fake", complete: async () => replies[n++]! };
    const r = await evaluateRewriteTask(blanking, provider, rewritePrompt, cfg);
    expect(r.calls.map((c) => [c.correct, c.error])).toEqual([
      [false, "the response was empty"],
      [false, "the response was empty"],
      // A complete JSON-string answer before the cut-off still counts.
      [true, undefined],
    ]);
    expect(r.pass).toBe(false);
    expect(judgeLine("[x", "length")).toEqual({
      got: "",
      error: "the answer hit the max-token limit before it was complete",
    });
    expect(judgeLine("[x]", "stop")).toEqual({ got: "[x]" });
  });

  it("doesn't score a restated example as the answer when the real one was cut off", () => {
    const restated = 'Examples:\n1. input: "shown-a"\n   output: ""\nLine to reformat: "HELDOUT-IN-1"\nAnswer: "';
    expect(judgeLine(restated, "length")).toEqual({
      got: "",
      error: "the answer hit the max-token limit before it was complete",
    });
    expect(judgeLine(restated.replace(/ "$/, ""), "stop")).toEqual({
      got: "",
      error: "nothing followed the Answer: label",
    });
  });

  it("gives no credit for an empty code block where an empty line is expected", async () => {
    const empty: Task = { ...task, heldOut: [{ input: "ab", output: "" }] };
    const provider: Provider = {
      name: "fake",
      complete: async () => ({ text: "Answer:\n```\n```", finishReason: "stop", latencyMs: 1, usage: {} }),
    };
    const r = await evaluateRewriteTask(empty, provider, rewritePrompt, cfg);
    expect(r.calls.map((c) => [c.correct, c.error])).toEqual([
      [false, 'the answer was an empty code block (an empty line is answered "")'],
    ]);
  });

  it("shows each call only its own held-out input and never any held-out output", async () => {
    const { provider, calls } = scripted('"x"');
    await evaluateRewriteTask(task, provider, rewritePrompt, cfg);
    expect(calls).toHaveLength(3);
    calls.forEach((c, i) => {
      const text = c.messages.map((m) => m.content).join("\n");
      task.heldOut.forEach((ex, j) => {
        expect(text.includes(JSON.stringify(ex.input)), `call ${i} sees input ${j}`).toBe(i === j);
        expect(text).not.toContain(JSON.stringify(ex.output));
      });
    });
  });
});
