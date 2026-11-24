# post-training-data-wrangling

> Testing post-training techniques (with agents) to build LLMs that reformat text by writing small, verifiable programs instead of rewriting the text.

You describe a reformatting job in plain language and give a few input→output examples. The model writes
a program in a small JSON language. The program is schema-checked, run by our own interpreter and scored
on held-out examples the model never saw. The research question: which models, and which post-training
methods, make this work?

**Status:** early work in progress. No model results yet.

## Checklist

- [x] A small JSON language for line-level edits: add, remove or replace text on each line
- [x] 28 hand-written tasks, each with shown and held-out examples and a reference program
- [x] An evaluation harness: program synthesis with verify-and-retry, and a direct-rewrite baseline
- [x] Model tooling for Apple Silicon: download only from the model's maker, verify, convert and serve
      locally
- [x] Tasks that check a program changes only what it should
- [ ] Privacy-first result summaries
- [ ] Sandboxed model conversion and serving
- [ ] First baseline: Phi-4-mini in both modes, and a write-up of what it gets wrong
- [ ] More local models: Gemma, Llama, gpt-oss
- [ ] A final test set that is never published
- [ ] Synthetic training data
- [ ] Fine-tuning (LoRA) on local models
- [ ] Reinforcement learning, with the verifier as the reward
- [ ] Whole-document edits, then table edits

## How it fits together

A task is a JSON file with a plain-language description, a few shown examples, held-out examples and a
reference program. The harness asks a model for a program using only the shown examples, schema-checks
it, runs it on those examples and retries with the errors. It then scores the accepted program on the
held-out examples. The direct-rewrite baseline asks the model to rewrite each held-out line itself instead.

| Folder | What's in it |
| --- | --- |
| [`src/text/`](src/text/) | The JSON language: types, schema check and interpreter |
| [`src/eval/`](src/eval/) | The harness: prompts, answer extraction, verify-and-retry, metrics and run results |
| [`src/llm/`](src/llm/) | A small client for OpenAI-compatible model servers |
| [`tasks/`](tasks/) | The hand-written tasks |
| [`prompts/`](prompts/) | The versioned prompts for each mode |
| [`scripts/`](scripts/) | Download, verify, convert and serve models (shell, Python and MLX-LM) |
| [`src/table/`](src/table/), [`samples/`](samples/) | A parked prototype for table edits, not part of the benchmark yet |

Tests sit next to the code (`*.test.ts`, `scripts/test_*.py`).

## Run the benchmark on your Mac

Needs an Apple Silicon Mac, Node 22.18+, Python 3.14 from python.org, and uv.

```bash
npm install && uv sync
npm run check                                 # typecheck, lint and tests, all offline
scripts/fetch-models.sh phi-4-mini-instruct   # shows the size and asks first
```

The model server runs until you stop it with Ctrl-C, so start it in a terminal of its own:

```bash
scripts/serve-model.sh phi-4-mini-instruct
```

Then, in another terminal:

```bash
npm run eval -- --model phi-4-mini-instruct --limit 2
```

## License

MIT. See [`LICENSE`](LICENSE).
