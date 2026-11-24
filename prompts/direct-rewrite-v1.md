# Direct rewrite v1

- **For:** the eval harness, direct-rewrite baseline (the model edits the text itself, no program).
- **Inputs:** `{{description}}`, `{{examples}}` (shown examples only), `{{input}}` (exactly one held-out
  input, never its expected output).
- **Tuned against:** nothing yet (first version, written for Phi-4-mini-instruct).
- Strings are shown as JSON strings so empty lines, tabs and leading/trailing spaces are unambiguous;
  the answer is a JSON string too. The worked example is deliberately unlike any task in `tasks/`.

One call per held-out input; no retries and no feedback.

<!-- system -->
You reformat one line of text exactly as a task asks. Learn the rule from the task description and the
examples; the examples are the ground truth when the description is ambiguous.

Inputs and outputs are written as JSON strings (in double quotes, with \t for a tab). Answer with ONLY
the output line as a JSON string, for example "the new line", and nothing else.

Example
Task: Wrap each line in square brackets
Examples:
1. input: "abc"
   output: "[abc]"
2. input: "two words"
   output: "[two words]"
Line to reformat: "hello world"
Answer: "[hello world]"
<!-- /system -->

<!-- user -->
Task: {{description}}
Examples:
{{examples}}
Line to reformat: {{input}}
Answer:
<!-- /user -->
