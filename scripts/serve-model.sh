#!/usr/bin/env bash
# Serve a converted local model over mlx_lm.server's OpenAI-compatible API.
#
#   scripts/serve-model.sh phi-4-mini-instruct [port]
#
# Serves the conversion recorded in models.lock.json (written by scripts/fetch-models.sh), so the eval
# harness can confirm it is talking to that exact model. Then run the harness against
# http://127.0.0.1:<port> (default 8080). Ctrl-C to stop.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${1:?usage: scripts/serve-model.sh <model-name> [port]}"
PORT="${2:-8080}"
# Serving never reaches the network, uv included: by default `uv run` re-locks and syncs the environment
# (from PyPI if needed). Offline and frozen, it uses uv.lock as it is and fails if the environment needs
# packages it doesn't have; run `uv sync` yourself first.
export UV_OFFLINE=1 UV_FROZEN=1
MLX="$(uv run --quiet python scripts/verify_model.py mlx-path "$NAME")"
[[ -d "$MLX" ]] || { echo "No converted model at $MLX. Run scripts/fetch-models.sh $NAME first." >&2; exit 1; }
# Serve only a conversion built from the files and settings models.lock.json records, while the lockfile
# pins the repo and commit scripts/models.json names (the check fetch-models.sh uses): not, say, one left
# over after the lockfile changed on another branch, or an old download after models.json moved on.
if ! uv run --quiet python scripts/verify_model.py check-conversion "$NAME" "$MLX"; then
  echo "$MLX doesn't match models.lock.json. Rerun scripts/fetch-models.sh $NAME." >&2
  exit 1
fi

# mlx_lm.server's /v1/models lists the models in the Hugging Face cache and fails if that cache folder
# doesn't exist (our downloads go to ~/models, not the cache). A dedicated empty cache keeps the
# listing to just the served model and works whether or not ~/.cache/huggingface exists.
export HF_HUB_CACHE="$HOME/models/.serve-hf-cache"
mkdir -p "$HF_HUB_CACHE"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1  # serve local files only; never reach the network
exec uv run --quiet mlx_lm.server --model "$MLX" --host 127.0.0.1 --port "$PORT"
