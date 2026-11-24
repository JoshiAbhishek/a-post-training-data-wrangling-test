#!/usr/bin/env bash
# Download a model from its originator's Hugging Face repo at a pinned commit, verify every file,
# and convert it to MLX locally at the quantization in scripts/models.json.
#
#   scripts/fetch-models.sh phi-4-mini-instruct
#
# Raw files:  ~/models/raw/<name>        Converted: ~/models/mlx/<name>-<quantization>
# Lockfile:   models.lock.json (repo, commit, per-file published hash + local SHA-256, conversion path,
#             converter versions and a digest of the converted files)
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${1:?usage: scripts/fetch-models.sh <model-name> (see scripts/models.json)}"
export HF_HUB_DISABLE_TELEMETRY=1
# Every `uv run` uses uv.lock as committed: if pyproject.toml has changed since, uv fails instead of
# re-resolving packages (and rewriting uv.lock) before the download and conversion.
export UV_LOCKED=1
py() { uv run --quiet python scripts/verify_model.py "$@"; }

# Checked assignments, not process substitution: a failed lookup must stop the script (set -e) rather
# than leave an empty file list, which `hf download` would treat as "download the whole repo".
INFO="$(py info "$NAME")"
read -r REPO REV QUANT BITS <<<"$INFO"
MANIFEST="$(py manifest "$NAME")" # first line: size in GB; then one file per line
SIZE_GB="$(head -n 1 <<<"$MANIFEST")"
FILE_LIST="$(tail -n +2 <<<"$MANIFEST")"
RAW="$HOME/models/raw/$NAME"
MLX="$HOME/models/mlx/$NAME-$QUANT"

FILES=()
while IFS= read -r f; do
  if [[ -n "$f" ]]; then FILES+=("$f"); fi
done <<<"$FILE_LIST"
if [[ -z "$REPO" || -z "$REV" || -z "$QUANT" || -z "$BITS" || ${#FILES[@]} -eq 0 ]]; then
  echo "Refusing to download: empty repo, revision, quantization or file list for $NAME." >&2
  exit 1
fi

echo "Model:    $REPO"
echo "Commit:   $REV"
echo "Files:    ${#FILES[@]} (${FILES[*]})"
echo "Download: ${SIZE_GB} GB -> $RAW"
echo "Convert:  $QUANT MLX -> $MLX"
read -r -p "Continue? [y/N] " ok
[[ "$ok" == [yY]* ]] || { echo "Aborted."; exit 1; }

# One fetch per model at a time, from any checkout: two at once would both convert, and the second `mv`
# below would put its folder inside the first's finished conversion, which then fails every check. The
# lock is on a descriptor this script keeps open, so it lasts until the fetch exits, crash included.
mkdir -p "$HOME/models/.locks"
exec 9>"$HOME/models/.locks/$NAME.lock"
py lock "$NAME" 9

mkdir -p "$RAW"
# Exact file list (no globs), so nothing outside the allowed set is fetched. Already-complete files are skipped.
uv run --quiet hf download "$REPO" "${FILES[@]}" --revision "$REV" --local-dir "$RAW"

py verify "$NAME" "$RAW"

if [[ -d "$MLX" ]]; then
  # Reuse only a conversion whose completion marker matches these verified files and settings and, once a
  # conversion is recorded (verify keeps it while the source is unchanged), the converted files it pins.
  if py check-conversion "$NAME" "$MLX"; then
    echo "Converted model at $MLX matches the verified files; skipping conversion."
  else
    echo "$MLX exists but isn't the conversion of these verified files (see above). Move or delete it, then rerun." >&2
    exit 1
  fi
else
  # Convert into a temporary directory and publish it only after success, so an interrupted
  # conversion never leaves a half-written model at $MLX (or a temporary path in models.lock.json).
  mkdir -p "$(dirname "$MLX")"
  TMP="$(mktemp -d "$(dirname "$MLX")/.${NAME}-${QUANT}.partial.XXXXXX")"
  trap 'rm -rf "$TMP"' EXIT
  # mlx_lm.convert never runs repo code here: no trust_remote_code, and no .py files were downloaded.
  uv run --quiet mlx_lm.convert --hf-path "$RAW" -q --q-bits "$BITS" --mlx-path "$TMP/out"
  # The marker records the source, the converter versions and a digest of the converted files.
  py write-marker "$NAME" "$TMP/out"
  # Another fetch can't have published here (it would need the lock), but something else might have:
  # `mv` onto an existing folder would put the conversion inside it.
  if [[ -e "$MLX" ]]; then
    echo "$MLX appeared during the conversion; this conversion wasn't published. Move it away, then rerun." >&2
    exit 1
  fi
  mv "$TMP/out" "$MLX"
fi
# Checks the conversion again, then copies its converter versions and converted-files digest into
# models.lock.json, so serving and the eval can tell if the converted files ever change. A new pin replaces
# the old one only after a fresh conversion, i.e. once the old folder was moved or deleted.
py record-conversion "$NAME" "$MLX"
echo "Done. Serve it with: scripts/serve-model.sh $NAME"
