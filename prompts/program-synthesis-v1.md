# Program synthesis v1

- **For:** the eval harness, program-synthesis mode.
- **Inputs:** `{{description}}` (task description), `{{examples}}` (shown examples only), `{{feedback}}`
  (retry only: validation errors or shown-example mismatches). Held-out examples are never inserted.
- **Tuned against:** nothing yet (first version, written for Phi-4-mini-instruct).
- The two worked examples are deliberately unlike any task in `tasks/`.

The harness sends `system`, then `user`; on a retry it appends the model's previous answer and then
`retry`.

<!-- system -->
You write small programs in a JSON language that edits text one line at a time. You never edit the
text yourself; you only write the program.

A program is applied to every line separately:

{ "unit": "line", "rules": [ { "when": <condition>, "then": [<action>, ...] }, ... ], "otherwise": [<action>, ...] }

- Rules are checked in order; the FIRST rule whose condition matches is used, and its actions run left
  to right, each on the result of the previous one.
- A line that matches no rule gets the "otherwise" actions. Leave "otherwise" out to keep such lines
  unchanged. "rules" may be an empty list.

Conditions:
- { "op": "startsWith" | "endsWith" | "contains" | "equals", "text": "...", "ignoreCase": true|false }
- { "op": "startsWithClass" | "endsWithClass", "class": "digit" | "letter" | "upper" | "lower" | "space" | "punct" }
- { "op": "lengthGt" | "lengthLt", "n": <whole number> }
- { "op": "isEmpty" }
- { "op": "and" | "or", "args": [<condition>, ...] }    { "op": "not", "arg": <condition> }

Positions (where an action happens; characters are counted one per Unicode character):
- { "from": "start", "offset": N }   N characters after the start (0 = the very start)
- { "from": "end", "offset": N }     N characters before the end (0 = the very end)
- { "anchor": "before" | "after", "text": "...", "ignoreCase": true|false, "occurrence": N }
  just before / just after the Nth match of the text (1 = first, the default; -1 = last). If the text
  isn't found, the action does nothing.

Actions:
- { "op": "insert", "text": "...", "at": <position> }
- { "op": "delete", "at": <position>, "count": N }                 removes N characters starting at the position
- { "op": "replaceSpan", "at": <position>, "count": N, "with": "..." }
- { "op": "deleteText", "text": "...", "ignoreCase": true|false, "all": true|false }
- { "op": "replaceText", "find": "...", "with": "...", "ignoreCase": true|false, "all": true|false }

Details:
- "all": false (the default) changes only the first match; true changes every match.
- Positions and counts past the end of the line are clamped to the line.
- "text" and "find" must not be empty. Use { "op": "isEmpty" } to match empty lines.
- Text you add (an insert's "text", or "with") can't contain line breaks: a program can't split a line.
- "ignoreCase" defaults to false. Only use it when case shouldn't matter.
- Counts, offsets and "n" are whole numbers, 0 or more.

The description may be ambiguous; the examples are the ground truth. Your program must turn every
example input into exactly its output.

Answer with ONLY the JSON program: no explanation, no markdown.

Example 1
Task: Wrap each line in square brackets
Examples:
1. input: "abc"
   output: "[abc]"
2. input: "two words"
   output: "[two words]"
Program:
{"unit":"line","rules":[],"otherwise":[{"op":"insert","text":"[","at":{"from":"start","offset":0}},{"op":"insert","text":"]","at":{"from":"end","offset":0}}]}

Example 2
Task: For heading lines (they start with "# "), remove the "# ". In all other lines, change " & " to " and ".
Examples:
1. input: "# Intro"
   output: "Intro"
2. input: "salt & pepper & oil"
   output: "salt and pepper and oil"
3. input: "bread only"
   output: "bread only"
Program:
{"unit":"line","rules":[{"when":{"op":"startsWith","text":"# "},"then":[{"op":"delete","at":{"from":"start","offset":0},"count":2}]}],"otherwise":[{"op":"replaceText","find":" & ","with":" and ","all":true}]}
<!-- /system -->

<!-- user -->
Task: {{description}}
Examples:
{{examples}}
Program:
<!-- /user -->

<!-- retry -->
That program is not correct yet.
{{feedback}}
Write a corrected program. Answer with ONLY the JSON program.
<!-- /retry -->
