"""Model fetching helpers: weights only from the model's originator, converted locally.

Used by scripts/fetch-models.sh and scripts/serve-model.sh. Subcommands:
  info <name>                 print "<repo> <revision> <quantization> <bits>" from scripts/models.json (no network)
  manifest <name>             print the download size in GB, then the allowed top-level files at the pinned
                              commit, one per line (one Hugging Face metadata request)
  verify <name> <dir>         check every downloaded file against its published hash; update models.lock.json
                              (a changed source or quantization clears the recorded conversion; else it stays)
  write-marker <name> <mlx_dir>        after converting into <mlx_dir>: write its conversion marker (the
                                       source, the converter versions, a digest of the converted files)
  record-conversion <name> <mlx_dir>   check <mlx_dir>, then record it and its marker's converter versions and
                                       converted-files digest in models.lock.json
  check-conversion <name> <mlx_dir>    exit 0 only if models.lock.json pins the repo and commit models.json names
                                       and <mlx_dir> is an unchanged conversion of the verified files, with the
                                       settings in models.json and (once recorded) the converted files
                                       models.lock.json records; otherwise print why on stderr and exit 1
  mlx-path <name>             print the converted model's path recorded in models.lock.json
  lock <name> <fd>            lock the open file descriptor <fd> (exclusive, without waiting) for fetch-models.sh;
                              if another fetch of <name> holds it, say so and exit 1

Verification: LFS files are checked against the SHA-256 Hugging Face publishes; small non-LFS files
against their Git blob SHA-1, recomputed as sha1(b"blob <size>\\0" + content). Every file's locally
computed SHA-256 is recorded in models.lock.json. A conversion is pinned the same way: its marker
holds a digest of the files mlx_lm.convert wrote, which models.lock.json records and every check
recomputes.
"""

from __future__ import annotations

import fcntl
import fnmatch
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "scripts" / "models.json"
LOCKFILE = ROOT / "models.lock.json"

ALLOWED_PATTERNS = ("*.safetensors", "*.json", "tokenizer*")
BLOCKED_SUFFIXES = (".py", ".bin", ".pth", ".pt", ".pkl", ".pickle")
IGNORED_DIRS = (".cache",)  # created by `hf download --local-dir`
IGNORED_NAMES = (".DS_Store",)  # created by Finder when the folder is opened
CHUNK = 1 << 20


@dataclass(frozen=True)
class PublishedFile:
    path: str
    size: int
    sha256: str | None  # set for LFS files
    blob_id: str | None  # Git blob SHA-1, set for non-LFS files


def is_allowed(path: str) -> bool:
    """Top-level files matching the allowed patterns, never a blocked type."""
    if "/" in path or path.endswith(BLOCKED_SUFFIXES):
        return False
    return any(fnmatch.fnmatch(path, p) for p in ALLOWED_PATTERNS)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def local_files(directory: Path) -> list[str]:
    out = []
    for p in directory.rglob("*"):
        rel = p.relative_to(directory)
        if rel.parts[0] in IGNORED_DIRS or p.name in IGNORED_NAMES or not p.is_file():
            continue
        out.append(rel.as_posix())
    return sorted(out)


def verify_dir(directory: Path, published: list[PublishedFile]) -> tuple[list[dict], list[str]]:
    """Return (lock entries, errors). Any error means the download must not be used."""
    errors: list[str] = []
    entries: list[dict] = []
    expected = {f.path: f for f in published}
    present = local_files(directory)

    for path in present:
        if path not in expected:
            errors.append(f"unexpected file: {path}")
    for path, pub in sorted(expected.items()):
        p = directory / path
        if not p.is_file():
            errors.append(f"missing file: {path}")
            continue
        size = p.stat().st_size
        if size != pub.size:
            errors.append(f"size mismatch: {path} ({size} != published {pub.size})")
            continue
        local_sha256 = sha256_file(p)
        if pub.sha256 is not None:
            if local_sha256 != pub.sha256:
                errors.append(f"SHA-256 mismatch: {path}")
                continue
            published_id = {"type": "lfs-sha256", "value": pub.sha256}
        elif pub.blob_id is not None:
            if git_blob_sha1_file(p) != pub.blob_id:
                errors.append(f"Git blob SHA-1 mismatch: {path}")
                continue
            published_id = {"type": "git-blob-sha1", "value": pub.blob_id}
        else:
            errors.append(f"no published hash for: {path}")
            continue
        entries.append({"path": path, "size": size, "published": published_id, "sha256": local_sha256})
    return entries, errors


# ---------- registry, Hugging Face metadata, lockfile ----------


def registry_entry(name: str) -> dict:
    reg = json.loads(REGISTRY.read_text())
    if name not in reg:
        sys.exit(f"unknown model {name!r}; known: {', '.join(sorted(reg))}")
    return reg[name]


QUANT_BITS = (2, 3, 4, 5, 6, 8)  # what mlx_lm.convert --q-bits supports for affine quantization


def q_bits(quantization: str) -> int:
    """The --q-bits value for a registry "quantization" such as "4bit", so the setting recorded in the
    conversion marker is the one the conversion actually used."""
    m = re.fullmatch(r"(\d+)bit", quantization or "")
    if not m or int(m.group(1)) not in QUANT_BITS:
        sys.exit(f"unsupported quantization {quantization!r}; use one of: {', '.join(f'{b}bit' for b in QUANT_BITS)}")
    return int(m.group(1))


def manifest_lines(files: list[PublishedFile]) -> list[str]:
    """The download size in GB, then one allowed file per line."""
    return [f"{sum(f.size for f in files) / 1e9:.1f}", *(f.path for f in files)]


def fetch_published(repo: str, revision: str) -> list[PublishedFile]:
    from huggingface_hub import HfApi  # imported lazily so unit tests don't need network access

    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        sys.exit(f"Hugging Face returned commit {info.sha}, expected pinned {revision}")
    files = []
    for s in info.siblings or []:
        if not is_allowed(s.rfilename):
            continue
        files.append(
            PublishedFile(
                path=s.rfilename,
                size=s.size or 0,
                sha256=s.lfs.sha256 if s.lfs else None,
                blob_id=None if s.lfs else s.blob_id,
            )
        )
    if not files:
        sys.exit(f"no allowed files found in {repo}@{revision}")
    return files


MARKER = ".conversion.json"
MARKER_KEYS = ("repo", "revision", "quantization", "source_files_sha256", "converted_with", "converted_files_sha256")
# Copied from the marker into models.lock.json by record-conversion, so the lockfile pins the converted
# files, not just the source they came from, and results can name both.
RECORDED_FIELDS = ("converted_with", "converted_files_sha256")
CONVERTER_PACKAGES = ("mlx", "mlx-lm", "transformers")


def files_digest(files: list[dict]) -> str:
    """One hash for a set of files: SHA-256 of "<path> <sha256>\n" per file, sorted by path."""
    lines = "".join(f"{f['path']} {f['sha256']}\n" for f in sorted(files, key=lambda f: f["path"]))
    return hashlib.sha256(lines.encode()).hexdigest()


def converted_files_sha256(directory: Path) -> str:
    """files_digest of a converted model's folder, leaving out the marker (which holds this digest)."""
    return files_digest([{"path": p, "sha256": sha256_file(directory / p)} for p in local_files(directory) if p != MARKER])


def converter_versions() -> dict:
    """The installed versions of the packages that convert a model (the environment `uv run` uses)."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return {pkg: version(pkg) for pkg in CONVERTER_PACKAGES}
    except PackageNotFoundError as e:
        sys.exit(f"{e.name} isn't installed; run `uv sync`")


def source_binding(lock_entry: dict, quantization: str) -> dict:
    """What a converted model must have been built from: the pinned source, the exact verified files and
    the quantization."""
    files = lock_entry.get("files") or []
    if not files:
        sys.exit("no verified files recorded; run verification first")
    return {
        "repo": lock_entry["repo"],
        "revision": lock_entry["revision"],
        "quantization": quantization,
        "source_files_sha256": files_digest(files),
    }


def new_marker(lock_entry: dict, quantization: str, directory: Path) -> dict:
    """The marker for a conversion just written into `directory`: what it was built from, the converter
    versions that built it, and a digest of the files it wrote."""
    return {
        **source_binding(lock_entry, quantization),
        "converted_with": converter_versions(),
        "converted_files_sha256": converted_files_sha256(directory),
    }


def conversion_problem(directory: Path, lock_entry: dict, quantization: str, recorded: bool) -> str | None:
    """Why `directory` isn't an unchanged conversion of the verified source, or None if it is. Its marker
    must have exactly MARKER_KEYS, name the source and quantization, and hold the digest of the files now
    in the folder. With `recorded` (models.lock.json records this conversion), its converter versions
    and converted-files digest must also be the recorded ones."""
    marker = directory / MARKER
    if not marker.is_file():
        return f"no {MARKER} in {directory} (incomplete or foreign conversion)"
    try:
        found = json.loads(marker.read_text())
    except json.JSONDecodeError:
        found = None
    if not isinstance(found, dict):
        return f"unreadable {MARKER} in {directory}"
    expected = source_binding(lock_entry, quantization)
    if recorded:
        expected |= {k: lock_entry.get(k) for k in RECORDED_FIELDS}
    differs = sorted(
        k
        for k in set(MARKER_KEYS) | found.keys()
        if k not in MARKER_KEYS or k not in found or (k in expected and found[k] != expected[k])
    )
    if differs:
        return f"{MARKER} differs from models.lock.json in: {', '.join(differs)}"
    if found["converted_files_sha256"] != converted_files_sha256(directory):
        return f"the files in {directory} changed after the conversion wrote them"
    return None


def registry_problem(name: str, entry: dict, lock_entry: dict) -> str | None:
    """Why models.lock.json doesn't pin the source scripts/models.json names for `name`, or None if it does.
    The conversion checks bind a folder to the lockfile; this binds the lockfile to the registry, so a
    registry moved to a new commit (say, by a pull) isn't served from the old download until it is fetched."""
    differs = [f"{k} {lock_entry.get(k)!r} (models.json: {entry[k]!r})" for k in ("repo", "revision") if lock_entry.get(k) != entry[k]]
    if differs:
        return f"models.lock.json pins {', '.join(differs)}; run scripts/fetch-models.sh {name}"
    return None


def home_relative(path: Path) -> str:
    """Record paths under the home directory as ~/..., so the committed lockfile has no username."""
    path = path.expanduser().resolve()
    home = Path.home().resolve()
    try:
        return "~/" + path.relative_to(home).as_posix()
    except ValueError:
        return str(path)


def read_lock() -> dict:
    return json.loads(LOCKFILE.read_text()) if LOCKFILE.exists() else {}


def write_atomic(path: Path, text: str) -> None:
    """Write `text` to a temporary file next to `path`, then rename it over `path`. An interruption
    (Ctrl-C, a full disk) leaves the old file or the new one, never a truncated one."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# Fields that describe a conversion. Verifying a changed source clears them, because a conversion only
# belongs to the source it was built from; record-conversion restores them once a matching conversion exists.
CONVERSION_FIELDS = ("mlx_path", "quantization", *RECORDED_FIELDS)


def stale_conversion_fields(old: dict, verified: dict, quantization: str) -> tuple[str, ...]:
    """The conversion fields that verifying `verified` (repo, revision, files) must clear from the old lock
    entry. None if the source and the quantization are the ones the recorded conversion was built from:
    re-verifying them keeps the pin, so reusing a conversion is still checked against the converted files
    models.lock.json records, and a different conversion can't be pinned in its place. Otherwise, all."""
    old_files = old.get("files") or []
    unchanged = (
        bool(old_files)
        and old.get("repo") == verified["repo"]
        and old.get("revision") == verified["revision"]
        and files_digest(old_files) == files_digest(verified["files"])
        and old.get("quantization") == quantization
    )
    return () if unchanged else CONVERSION_FIELDS


def merge_lock_entry(old: dict, patch: dict, drop: tuple[str, ...] = ()) -> dict:
    return {**{k: v for k, v in old.items() if k not in drop}, **patch}


def update_lock(name: str, patch: dict, drop: tuple[str, ...] = ()) -> None:
    lock = read_lock()
    lock[name] = merge_lock_entry(lock.get(name, {}), patch, drop)
    write_atomic(LOCKFILE, json.dumps(lock, indent=2, sort_keys=True) + "\n")


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        sys.exit(__doc__)
    cmd, name, *rest = argv
    entry = registry_entry(name)
    repo, revision = entry["repo"], entry["revision"]
    if cmd == "info":
        print(repo, revision, entry["quantization"], q_bits(entry["quantization"]))
    elif cmd == "manifest":
        print("\n".join(manifest_lines(fetch_published(repo, revision))))
    elif cmd == "verify":
        directory = Path(rest[0]).expanduser()
        entries, errors = verify_dir(directory, fetch_published(repo, revision))
        if errors:
            sys.exit("verification FAILED:\n  " + "\n  ".join(errors))
        verified = {"repo": repo, "revision": revision, "files": entries}
        drop = stale_conversion_fields(read_lock().get(name, {}), verified, entry["quantization"])
        update_lock(name, {**verified, "license": entry.get("license"), "raw_path": home_relative(directory)}, drop=drop)
        print(f"verified {len(entries)} files; recorded in {LOCKFILE.relative_to(ROOT)}")
    elif cmd == "write-marker":
        directory = Path(rest[0]).expanduser()
        marker = new_marker(read_lock().get(name, {}), entry["quantization"], directory)
        write_atomic(directory / MARKER, json.dumps(marker, indent=2, sort_keys=True) + "\n")
    elif cmd == "record-conversion":
        # Only a conversion at its final path goes in the lockfile; a temporary one never does.
        directory = Path(rest[0]).expanduser()
        lock_entry = read_lock().get(name, {})
        problem = conversion_problem(directory, lock_entry, entry["quantization"], recorded=False)
        if problem:
            sys.exit(problem)
        marker = json.loads((directory / MARKER).read_text())
        patch = {"mlx_path": home_relative(directory), "quantization": entry["quantization"]}
        update_lock(name, {**patch, **{k: marker[k] for k in RECORDED_FIELDS}})
        print(f"recorded conversion in {LOCKFILE.relative_to(ROOT)}")
    elif cmd == "check-conversion":
        directory = Path(rest[0]).expanduser()
        lock_entry = read_lock().get(name, {})
        problem = registry_problem(name, entry, lock_entry) or conversion_problem(
            directory, lock_entry, entry["quantization"], recorded="mlx_path" in lock_entry
        )
        if problem:
            print(problem, file=sys.stderr)
            sys.exit(1)
    elif cmd == "lock":
        # The lock belongs to the open file description, which fetch-models.sh keeps open after this
        # process exits, so it holds until the fetch exits (crash included) and is never left behind.
        try:
            fcntl.flock(int(rest[0]), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(f"Another scripts/fetch-models.sh {name} is running. Let it finish, then rerun.")
    elif cmd == "mlx-path":
        mlx_path = read_lock().get(name, {}).get("mlx_path")
        if not mlx_path:
            sys.exit(f"no converted model recorded for {name} in models.lock.json; run scripts/fetch-models.sh {name}")
        print(Path(mlx_path).expanduser())
    else:
        sys.exit(f"unknown command {cmd!r}\n{__doc__}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    main(sys.argv[1:])
