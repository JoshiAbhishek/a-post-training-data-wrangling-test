"""Offline tests for fetch-models.sh and serve-model.sh: the real scripts, with stubbed tools.

`uv` is replaced by a stub: `uv run python` runs the real verify_model.py, while `hf download`,
`mlx_lm.convert` and `mlx_lm.server` are fakes, and the Hugging Face API is a fake `huggingface_hub`
module. The scripts run in a temporary copy of scripts/ with a temporary HOME, so nothing is
downloaded and neither the real models.lock.json nor ~/models is touched.

Run: npm run test:py
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
NAME = "test-model"

UV_STUB = r"""#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == run && "$2" == --quiet ]] || { echo "unexpected uv call: $*" >&2; exit 99; }
shift 2
tool="$1"; shift
echo "$tool $*" >> "$STUB_LOG"
echo "uv-env $tool locked=${UV_LOCKED:-} offline=${UV_OFFLINE:-} frozen=${UV_FROZEN:-}" >> "$STUB_LOG"
case "$tool" in
  python) exec "$REAL_PYTHON" "$@" ;;
  hf)  # hf download REPO FILE... --revision REV --local-dir DIR
    shift 2
    files=(); while [[ $# -gt 0 && "$1" != --* ]]; do files+=("$1"); shift; done
    while [[ $# -gt 0 ]]; do case "$1" in --local-dir) dir="$2"; shift 2 ;; *) shift 2 ;; esac; done
    for f in "${files[@]}"; do cp "$FAKE_REPO/$f" "$dir/$f"; done ;;
  mlx_lm.convert)
    while [[ $# -gt 0 ]]; do case "$1" in --mlx-path) out="$2"; shift 2 ;; --q-bits) bits="$2"; shift 2 ;; *) shift ;; esac; done
    mkdir -p "$out"; echo "{\"bits\": $bits}" > "$out/config.json"
    # A command to run while this conversion is in progress (once: the variable is cleared for it), with
    # FETCH_PID set to the fetch-models.sh process that ran this conversion.
    if [[ -n "${DURING_CONVERT:-}" ]]; then hook="$DURING_CONVERT"; FETCH_PID=$PPID DURING_CONVERT= bash -c "$hook"; fi ;;
  mlx_lm.server) echo "HF_HUB_CACHE=$HF_HUB_CACHE"; echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"; echo "server $*" ;;
  *) echo "unexpected tool: $tool" >&2; exit 97 ;;
esac
"""

FAKE_HF = """
import json, os
from types import SimpleNamespace as NS

class HfApi:
    def model_info(self, repo, revision=None, files_metadata=False):
        meta = json.loads(open(os.environ["FAKE_HF_META"]).read())
        assert repo == meta["repo"] and files_metadata
        siblings = [
            NS(rfilename=f["path"], size=f["size"], lfs=NS(sha256=f["sha256"]) if f.get("sha256") else None,
               blob_id=f.get("blob_id"))
            for f in meta["files"]
        ]
        return NS(sha=meta["revision"], siblings=siblings)
"""

LFS = {"model.safetensors", "tokenizer.json"}
CONVERTER = {"mlx": "0.1.0", "mlx-lm": "0.2.0", "transformers": "0.3.0"}


def fake_install(site: Path, versions: dict):
    """Package metadata importlib.metadata finds on PYTHONPATH, standing in for the converter packages."""
    for name, version in versions.items():
        for old in site.glob(f"{name.replace('-', '_')}-*.dist-info"):
            shutil.rmtree(old)
        info = site / f"{name.replace('-', '_')}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")


def published(path: Path) -> dict:
    data = path.read_bytes()
    if path.name in LFS:
        return {"path": path.name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    return {"path": path.name, "size": len(data), "blob_id": blob}


@unittest.skipUnless(shutil.which("bash"), "bash not available")
class ModelScriptsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.repo, self.home, self.bin, self.fake_repo, fake_hf = (
            t / "repo",
            t / "home",
            t / "bin",
            t / "hf-repo",
            t / "fake-hf",
        )
        for d in (self.repo / "scripts", self.home, self.bin, self.fake_repo, fake_hf / "huggingface_hub"):
            d.mkdir(parents=True)
        for f in ("fetch-models.sh", "serve-model.sh", "verify_model.py"):
            shutil.copy(SCRIPTS / f, self.repo / "scripts" / f)
        (self.bin / "uv").write_text(UV_STUB)
        (self.bin / "uv").chmod(0o755)
        (fake_hf / "huggingface_hub" / "__init__.py").write_text(FAKE_HF)
        self.site = fake_hf
        fake_install(self.site, CONVERTER)
        self.log = t / "stub.log"
        self.meta = t / "hf-meta.json"

        # The published repo: three allowed files and a code file that must never be downloaded.
        (self.fake_repo / "config.json").write_text('{"model_type": "phi3"}')
        (self.fake_repo / "model.safetensors").write_bytes(b"\x00weights" * 64)
        (self.fake_repo / "tokenizer.json").write_text('{"vocab": {}}')
        (self.fake_repo / "modeling_phi3.py").write_text("print('repo code')")
        self.set_registry(revision="a" * 40, quantization="4bit")

        self.env = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "HOME": str(self.home),
            "STUB_LOG": str(self.log),
            "REAL_PYTHON": sys.executable,
            "FAKE_REPO": str(self.fake_repo),
            "FAKE_HF_META": str(self.meta),
            "PYTHONPATH": str(fake_hf),
        }

    def tearDown(self):
        self.tmp.cleanup()

    def set_registry(self, revision: str, quantization: str):
        entry = {"repo": "example-org/test-model", "revision": revision, "license": "MIT", "quantization": quantization}
        (self.repo / "scripts" / "models.json").write_text(json.dumps({NAME: entry}))
        files = [published(p) for p in sorted(self.fake_repo.iterdir())]
        self.meta.write_text(json.dumps({"repo": entry["repo"], "revision": revision, "files": files}))

    def run_script(self, script: str, answer: str = "y\n") -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(self.repo / "scripts" / script), NAME],
            input=answer,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def calls(self, tool: str) -> list[str]:
        lines = self.log.read_text().splitlines() if self.log.exists() else []
        return [line for line in lines if line.startswith(tool + " ")]

    def lock(self) -> dict:
        path = self.repo / "models.lock.json"
        return json.loads(path.read_text()).get(NAME, {}) if path.exists() else {}

    def test_downloads_only_allowed_files_verifies_converts_and_records(self):
        r = self.run_script("fetch-models.sh")
        self.assertEqual(r.returncode, 0, r.stderr)
        raw = self.home / "models" / "raw" / NAME
        self.assertEqual(sorted(p.name for p in raw.iterdir()), ["config.json", "model.safetensors", "tokenizer.json"])

        mlx = self.home / "models" / "mlx" / f"{NAME}-4bit"
        self.assertEqual(json.loads((mlx / "config.json").read_text()), {"bits": 4})
        self.assertTrue((mlx / ".conversion.json").is_file())
        self.assertEqual([p.name for p in mlx.parent.iterdir()], [mlx.name])  # no .partial dir left behind

        lock = self.lock()
        self.assertEqual(lock["mlx_path"], f"~/models/mlx/{NAME}-4bit")
        self.assertEqual(lock["quantization"], "4bit")
        self.assertEqual(len(lock["files"]), 3)
        # The converted files are pinned too: the marker's converter versions and digest, in the lock.
        marker = json.loads((mlx / ".conversion.json").read_text())
        self.assertEqual(marker["converted_with"], CONVERTER)
        config_sha = hashlib.sha256((mlx / "config.json").read_bytes()).hexdigest()
        self.assertEqual(marker["converted_files_sha256"], hashlib.sha256(f"config.json {config_sha}\n".encode()).hexdigest())
        self.assertEqual({k: lock[k] for k in ("converted_with", "converted_files_sha256")}, {k: marker[k] for k in ("converted_with", "converted_files_sha256")})
        # uv never re-resolves packages here: every call is --locked to the committed uv.lock.
        uv_env = self.calls("uv-env")
        self.assertTrue(uv_env)
        for line in uv_env:
            self.assertIn("locked=1", line)
        # One metadata request before downloading (plus one to verify), not one per piece of information.
        self.assertEqual(len([c for c in self.calls("python") if "verify_model.py manifest" in c]), 1)
        self.assertEqual(len([c for c in self.calls("python") if "verify_model.py verify" in c]), 1)

    def test_a_second_fetch_of_the_model_stops_while_one_is_running(self):
        # The second fetch starts while the first is converting: without the lock both would convert,
        # and the first's `mv` would land inside the second's published folder.
        nested = self.home.parent / "nested"
        script = self.repo / "scripts" / "fetch-models.sh"
        self.env["DURING_CONVERT"] = f"bash {script} {NAME} <<<y >{nested}.out 2>&1; echo $? >{nested}.rc"
        r = self.run_script("fetch-models.sh")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(Path(f"{nested}.rc").read_text().strip(), "1")
        self.assertIn(f"Another scripts/fetch-models.sh {NAME} is running", Path(f"{nested}.out").read_text())
        mlx = self.home / "models" / "mlx" / f"{NAME}-4bit"
        self.assertEqual(sorted(p.name for p in mlx.iterdir()), [".conversion.json", "config.json"])
        self.assertEqual(len(self.calls("mlx_lm.convert")), 1)
        # The lock went with the first fetch: the next one runs, and serving accepts the conversion.
        del self.env["DURING_CONVERT"]
        self.assertIn("skipping conversion", self.run_script("fetch-models.sh").stdout)
        self.assertEqual(self.run_script("serve-model.sh", answer="").returncode, 0)

    def test_a_crashed_fetch_leaves_no_lock_held(self):
        # Killed mid-conversion, so no cleanup runs: a second fetch while it ran was refused, and the next
        # one after it died goes ahead. A lock that needed removing (a lock folder, a PID file) would block it.
        nested = self.home.parent / "nested"
        script = self.repo / "scripts" / "fetch-models.sh"
        self.env["DURING_CONVERT"] = (
            f"bash {script} {NAME} <<<y >{nested}.out 2>&1; echo $? >{nested}.rc; kill -9 $FETCH_PID"
        )
        r = self.run_script("fetch-models.sh")
        self.assertEqual(r.returncode, -9, r.stderr)
        self.assertEqual(Path(f"{nested}.rc").read_text().strip(), "1")
        self.assertIn(f"Another scripts/fetch-models.sh {NAME} is running", Path(f"{nested}.out").read_text())
        del self.env["DURING_CONVERT"]
        r = self.run_script("fetch-models.sh")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("is running", r.stderr)
        self.assertEqual(self.run_script("serve-model.sh", answer="").returncode, 0)

    def test_never_publishes_into_a_folder_that_appeared_during_conversion(self):
        mlx = self.home / "models" / "mlx" / f"{NAME}-4bit"
        self.env["DURING_CONVERT"] = f"mkdir -p {mlx} && echo other > {mlx}/other.txt"
        r = self.run_script("fetch-models.sh")
        self.assertEqual(r.returncode, 1)
        self.assertIn("appeared during the conversion", r.stderr)
        self.assertEqual([p.name for p in mlx.iterdir()], ["other.txt"])  # left as it was, nothing inside it
        self.assertEqual([p.name for p in mlx.parent.iterdir()], [mlx.name])  # no .partial dir left behind
        self.assertNotIn("mlx_path", self.lock())

    def test_rerun_reuses_a_matching_conversion(self):
        self.assertEqual(self.run_script("fetch-models.sh").returncode, 0)
        r = self.run_script("fetch-models.sh")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("skipping conversion", r.stdout)
        self.assertEqual(len(self.calls("mlx_lm.convert")), 1)

    def test_reused_conversion_keeps_the_versions_that_converted_it(self):
        self.assertEqual(self.run_script("fetch-models.sh").returncode, 0)
        fake_install(self.site, {"mlx-lm": "9.9.9"})  # mlx-lm upgraded since the conversion
        r = self.run_script("fetch-models.sh")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("skipping conversion", r.stdout)
        self.assertEqual(self.lock()["converted_with"], CONVERTER)

    def replace_conversion(self, versions: dict) -> Path:
        """Swap the converted folder for a different conversion that is consistent on its own: new files
        and a marker naming the same verified source, written by other converter versions."""
        mlx = self.home / "models" / "mlx" / f"{NAME}-4bit"
        (mlx / "config.json").write_text('{"bits": 4, "reconverted": true}')
        fake_install(self.site, versions)
        script = self.repo / "scripts" / "verify_model.py"
        subprocess.run([sys.executable, str(script), "write-marker", NAME, str(mlx)], env=self.env, check=True)
        return mlx

    def test_rerun_refuses_a_different_conversion_than_the_one_pinned(self):
        self.assertEqual(self.run_script("fetch-models.sh").returncode, 0)
        pinned = self.lock()
        self.replace_conversion({"mlx-lm": "9.9.9"})
        r = self.run_script("fetch-models.sh")
        self.assertEqual(r.returncode, 1)
        self.assertIn("differs from models.lock.json in: converted_files_sha256, converted_with", r.stderr)
        self.assertIn("isn't the conversion of these verified files", r.stderr)
        self.assertEqual(self.lock(), pinned)  # re-verifying the same source kept the pin; nothing re-pinned
        self.assertEqual(self.run_script("serve-model.sh", answer="").returncode, 1)
        self.assertEqual(self.calls("mlx_lm.server"), [])

    def test_moving_the_conversion_away_converts_and_pins_a_new_one(self):
        self.assertEqual(self.run_script("fetch-models.sh").returncode, 0)
        mlx = self.home / "models" / "mlx" / f"{NAME}-4bit"
        shutil.move(mlx, self.home / "old-conversion")
        fake_install(self.site, {"mlx-lm": "9.9.9"})
        r = self.run_script("fetch-models.sh")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(self.calls("mlx_lm.convert")), 2)
        self.assertEqual(self.lock()["converted_with"], {**CONVERTER, "mlx-lm": "9.9.9"})

    def test_new_quantization_converts_again_and_pins_it(self):
        self.assertEqual(self.run_script("fetch-models.sh").returncode, 0)
        self.set_registry(revision="a" * 40, quantization="8bit")
        r = self.run_script("fetch-models.sh")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.lock()["mlx_path"], f"~/models/mlx/{NAME}-8bit")
        self.assertEqual(self.lock()["quantization"], "8bit")

    def test_quantization_setting_drives_the_conversion(self):
        self.set_registry(revision="a" * 40, quantization="8bit")
        r = self.run_script("fetch-models.sh")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--q-bits 8", self.calls("mlx_lm.convert")[0])
        mlx = self.home / "models" / "mlx" / f"{NAME}-8bit"
        self.assertEqual(json.loads((mlx / "config.json").read_text()), {"bits": 8})
        self.assertEqual(self.lock()["quantization"], "8bit")

    def test_answering_no_downloads_nothing(self):
        r = self.run_script("fetch-models.sh", answer="n\n")
        self.assertEqual(r.returncode, 1)
        self.assertIn("Aborted", r.stdout)
        self.assertEqual(self.calls("hf"), [])
        self.assertFalse((self.home / "models").exists())

    def test_tampered_download_fails_before_conversion(self):
        p = self.fake_repo / "model.safetensors"
        data = bytearray(p.read_bytes())
        data[0] ^= 1  # same size, different content than published
        p.write_bytes(bytes(data))
        r = self.run_script("fetch-models.sh")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("SHA-256 mismatch: model.safetensors", r.stderr)
        self.assertEqual(self.calls("mlx_lm.convert"), [])
        self.assertNotIn("mlx_path", self.lock())

    def test_new_revision_refuses_the_old_conversion(self):
        self.assertEqual(self.run_script("fetch-models.sh").returncode, 0)
        self.set_registry(revision="b" * 40, quantization="4bit")
        r = self.run_script("fetch-models.sh")
        self.assertEqual(r.returncode, 1)
        self.assertIn("isn't the conversion of these verified files", r.stderr)
        self.assertNotIn("mlx_path", self.lock())  # the old conversion no longer counts as verified

    def test_serve_runs_the_recorded_conversion_with_an_empty_hf_cache(self):
        self.assertEqual(self.run_script("fetch-models.sh").returncode, 0)
        self.log.unlink()
        r = self.run_script("serve-model.sh", answer="")
        self.assertEqual(r.returncode, 0, r.stderr)
        # Every uv call runs offline and frozen, so serving can't sync from PyPI or rewrite uv.lock.
        uv_env = self.calls("uv-env")
        self.assertEqual(len(uv_env), 3, uv_env)  # mlx-path, check-conversion, server
        for line in uv_env:
            self.assertTrue(line.endswith("offline=1 frozen=1"), line)
        mlx = self.home / "models" / "mlx" / f"{NAME}-4bit"
        self.assertIn(f"server --model {mlx} --host 127.0.0.1 --port 8080", r.stdout)
        self.assertIn("HF_HUB_OFFLINE=1", r.stdout)
        cache = Path(r.stdout.split("HF_HUB_CACHE=")[1].splitlines()[0])
        self.assertTrue(cache.is_dir())  # mlx_lm.server's /v1/models fails without it
        self.assertEqual(list(cache.iterdir()), [])

    def test_serve_refuses_a_conversion_that_no_longer_matches_the_lock(self):
        self.assertEqual(self.run_script("fetch-models.sh").returncode, 0)
        # The lockfile and registry change (e.g. checked out from another branch) but the old conversion stays.
        self.set_registry(revision="c" * 40, quantization="4bit")
        path = self.repo / "models.lock.json"
        lock = json.loads(path.read_text())
        lock[NAME]["revision"] = "c" * 40
        path.write_text(json.dumps(lock))
        r = self.run_script("serve-model.sh", answer="")
        self.assertEqual(r.returncode, 1)
        self.assertIn("differs from models.lock.json in: revision", r.stderr)
        self.assertEqual(self.calls("mlx_lm.server"), [])

    def test_serve_refuses_a_lock_the_registry_has_moved_past(self):
        self.assertEqual(self.run_script("fetch-models.sh").returncode, 0)
        # models.json names a new commit (e.g. after a pull), but nobody fetched it.
        self.set_registry(revision="b" * 40, quantization="4bit")
        r = self.run_script("serve-model.sh", answer="")
        self.assertEqual(r.returncode, 1)
        self.assertIn(f"models.lock.json pins revision '{'a' * 40}' (models.json: '{'b' * 40}')", r.stderr)
        self.assertIn(f"run scripts/fetch-models.sh {NAME}", r.stderr)
        self.assertEqual(self.calls("mlx_lm.server"), [])

    def test_serve_refuses_a_conversion_whose_files_changed(self):
        self.assertEqual(self.run_script("fetch-models.sh").returncode, 0)
        mlx = self.home / "models" / "mlx" / f"{NAME}-4bit"
        (mlx / "config.json").write_text('{"bits": 3}')  # same size, different content
        r = self.run_script("serve-model.sh", answer="")
        self.assertEqual(r.returncode, 1)
        self.assertIn("changed after the conversion", r.stderr)
        self.assertEqual(self.calls("mlx_lm.server"), [])

    def test_serve_refuses_without_a_recorded_conversion(self):
        r = self.run_script("serve-model.sh", answer="")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn(f"run scripts/fetch-models.sh {NAME}", r.stderr)
        self.assertEqual(self.calls("mlx_lm.server"), [])


if __name__ == "__main__":
    unittest.main()
