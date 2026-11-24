"""Unit tests for verify_model.py. No network: published metadata is built from local files.

Run: npm run test:py   (python3 -m unittest discover -s scripts; stdlib only, so no uv sync needed)
"""

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from verify_model import (
    CONVERSION_FIELDS,
    MARKER,
    RECORDED_FIELDS,
    PublishedFile,
    conversion_problem,
    converted_files_sha256,
    files_digest,
    git_blob_sha1_file,
    home_relative,
    is_allowed,
    local_files,
    manifest_lines,
    merge_lock_entry,
    new_marker,
    q_bits,
    registry_problem,
    stale_conversion_fields,
    verify_dir,
    write_atomic,
)


def publish(directory: Path, lfs: set[str]) -> list[PublishedFile]:
    """Published metadata as Hugging Face would report it for the files currently in `directory`."""
    out = []
    for path in local_files(directory):
        data = (directory / path).read_bytes()
        if path in lfs:
            out.append(PublishedFile(path, len(data), hashlib.sha256(data).hexdigest(), None))
        else:
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            out.append(PublishedFile(path, len(data), None, blob))
    return out


class VerifyDirTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "model.safetensors").write_bytes(b"\x00weights" * 100)  # LFS
        (self.dir / "config.json").write_text('{"model_type": "phi3"}')  # non-LFS
        (self.dir / "tokenizer.json").write_text('{"vocab": {}}')  # LFS in the real repo
        self.published = publish(self.dir, lfs={"model.safetensors", "tokenizer.json"})

    def tearDown(self):
        self.tmp.cleanup()

    def test_clean_download_verifies_and_records_sha256_for_every_file(self):
        entries, errors = verify_dir(self.dir, self.published)
        self.assertEqual(errors, [])
        self.assertEqual([e["path"] for e in entries], ["config.json", "model.safetensors", "tokenizer.json"])
        for e in entries:
            self.assertEqual(e["sha256"], hashlib.sha256((self.dir / e["path"]).read_bytes()).hexdigest())
        kinds = {e["path"]: e["published"]["type"] for e in entries}
        self.assertEqual(kinds["config.json"], "git-blob-sha1")
        self.assertEqual(kinds["model.safetensors"], "lfs-sha256")

    def test_tampered_lfs_file_fails(self):
        p = self.dir / "model.safetensors"
        data = bytearray(p.read_bytes())
        data[0] ^= 1  # same size, different content
        p.write_bytes(bytes(data))
        _, errors = verify_dir(self.dir, self.published)
        self.assertEqual(errors, ["SHA-256 mismatch: model.safetensors"])

    def test_tampered_non_lfs_file_fails(self):
        (self.dir / "config.json").write_text('{"model_type": "phi4"}')  # same length
        _, errors = verify_dir(self.dir, self.published)
        self.assertEqual(errors, ["Git blob SHA-1 mismatch: config.json"])

    def test_size_change_fails(self):
        (self.dir / "config.json").write_text('{"model_type": "phi3", "x": 1}')
        _, errors = verify_dir(self.dir, self.published)
        self.assertTrue(errors[0].startswith("size mismatch: config.json"))

    def test_missing_and_unexpected_files_fail(self):
        (self.dir / "tokenizer.json").unlink()
        (self.dir / "modeling_phi3.py").write_text("print('hi')")
        (self.dir / "sub").mkdir()
        (self.dir / "sub" / "extra.json").write_text("{}")
        _, errors = verify_dir(self.dir, self.published)
        self.assertIn("missing file: tokenizer.json", errors)
        self.assertIn("unexpected file: modeling_phi3.py", errors)
        self.assertIn("unexpected file: sub/extra.json", errors)

    def test_hf_cache_dir_is_ignored(self):
        (self.dir / ".cache" / "huggingface").mkdir(parents=True)
        (self.dir / ".cache" / "huggingface" / "x.lock").write_text("")
        _, errors = verify_dir(self.dir, self.published)
        self.assertEqual(errors, [])

    def test_finder_metadata_is_ignored(self):
        (self.dir / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")
        (self.dir / "sub").mkdir()
        (self.dir / "sub" / ".DS_Store").write_bytes(b"x")
        _, errors = verify_dir(self.dir, self.published)
        self.assertEqual(errors, [])

    def test_git_blob_sha1_matches_git(self):
        p = self.dir / "config.json"
        try:
            out = subprocess.run(["git", "hash-object", str(p)], capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("git not available")
        self.assertEqual(git_blob_sha1_file(p), out.stdout.strip())


class ConversionTest(unittest.TestCase):
    VERSIONS = {"mlx": "1.0", "mlx-lm": "2.0", "transformers": "3.0"}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "model.safetensors").write_bytes(b"\x01quantized" * 50)
        (self.dir / "config.json").write_text('{"quantization": {"bits": 4}}')
        self.lock = {
            "repo": "microsoft/Phi-4-mini-instruct",
            "revision": "a" * 40,
            "files": [{"path": "b.safetensors", "sha256": "22"}, {"path": "a.json", "sha256": "11"}],
        }
        with mock.patch("verify_model.converter_versions", return_value=self.VERSIONS):
            self.marker = new_marker(self.lock, "4bit", self.dir)
        self.write_marker(self.marker)

    def tearDown(self):
        self.tmp.cleanup()

    def write_marker(self, marker: dict):
        (self.dir / MARKER).write_text(json.dumps(marker))

    def problem(self, lock: dict | None = None, quantization: str = "4bit", recorded: bool = False):
        return conversion_problem(self.dir, self.lock if lock is None else lock, quantization, recorded)

    def recorded_lock(self, **changes) -> dict:
        """The lock entry after record-conversion, optionally with some recorded fields changed."""
        return {**self.lock, "mlx_path": "~/m", "quantization": "4bit", **{k: self.marker[k] for k in RECORDED_FIELDS}, **changes}

    def test_marker_pins_the_source_the_converter_and_the_converted_files(self):
        self.assertEqual(self.marker["converted_with"], self.VERSIONS)
        files = [{"path": p, "sha256": hashlib.sha256((self.dir / p).read_bytes()).hexdigest()} for p in ("config.json", "model.safetensors")]
        self.assertEqual(self.marker["converted_files_sha256"], files_digest(files))
        # The marker itself isn't part of the digest, so writing it doesn't change it.
        self.assertEqual(converted_files_sha256(self.dir), self.marker["converted_files_sha256"])

    def test_unchanged_conversion_passes_before_and_after_it_is_recorded(self):
        self.assertIsNone(self.problem())
        self.assertIsNone(self.problem(self.recorded_lock(), recorded=True))

    def test_missing_marker_means_incomplete_conversion(self):
        (self.dir / MARKER).unlink()
        self.assertIn("incomplete", self.problem())

    def test_new_revision_files_or_settings_invalidate_the_conversion(self):
        self.assertIn("in: revision", self.problem({**self.lock, "revision": "b" * 40}))
        self.assertIn("in: source_files_sha256", self.problem({**self.lock, "files": [{"path": "a.json", "sha256": "99"}]}))
        self.assertIn("in: quantization", self.problem(quantization="8bit"))

    def test_extra_or_missing_marker_keys_invalidate_the_conversion(self):
        self.write_marker({**self.marker, "adapter": "x"})
        self.assertIn("in: adapter", self.problem())
        self.write_marker({k: v for k, v in self.marker.items() if k != "converted_with"})
        self.assertIn("in: converted_with", self.problem())
        (self.dir / MARKER).write_text("[]")
        self.assertIn("unreadable", self.problem())

    def test_converted_files_changed_after_conversion_fail(self):
        p = self.dir / "model.safetensors"
        original = p.read_bytes()
        p.write_bytes(bytes([original[0] ^ 1]) + original[1:])  # same size, different content
        self.assertIn("changed after the conversion", self.problem())
        p.write_bytes(original)
        self.assertIsNone(self.problem())
        (self.dir / "adapters.safetensors").write_bytes(b"lora")  # a file added to the folder
        self.assertIn("changed after the conversion", self.problem())

    def test_finder_metadata_does_not_change_the_converted_files(self):
        (self.dir / ".DS_Store").write_bytes(b"Bud1")
        self.assertIsNone(self.problem(self.recorded_lock(), recorded=True))

    def test_a_recorded_conversion_must_be_the_one_the_lock_records(self):
        # E.g. reconverted with a newer mlx-lm while models.lock.json (committed) still pins the old files.
        self.assertIn("in: converted_files_sha256", self.problem(self.recorded_lock(converted_files_sha256="0" * 64), recorded=True))
        newer = {**self.VERSIONS, "mlx-lm": "2.1"}
        self.assertIn("in: converted_with", self.problem(self.recorded_lock(converted_with=newer), recorded=True))
        # A lock that records the path but not the converted files doesn't pin them, so it's refused.
        unpinned = {k: v for k, v in self.recorded_lock().items() if k not in RECORDED_FIELDS}
        self.assertIn("converted_files_sha256, converted_with", self.problem(unpinned, recorded=True))

    def test_source_digest_format_is_stable(self):
        # Results record this digest ("source files sha256"); changing the format would orphan them.
        self.assertEqual(files_digest(self.lock["files"]), "3a037efffa3b98cb04c0c0dd34f190003685679918e08626bb36b0403dd0bf01")
        self.assertEqual(self.marker["source_files_sha256"], files_digest(self.lock["files"]))

    def test_file_order_does_not_matter(self):
        self.assertEqual(files_digest(self.lock["files"]), files_digest(list(reversed(self.lock["files"]))))


class LockEntryTest(unittest.TestCase):
    def test_verifying_a_new_source_clears_the_old_conversion(self):
        old = {
            "repo": "r",
            "revision": "old",
            "files": [],
            "mlx_path": "~/models/mlx/x-4bit",
            "quantization": "4bit",
            "converted_with": {"mlx-lm": "1"},
            "converted_files_sha256": "f",
        }
        new = merge_lock_entry(old, {"revision": "new", "files": [{"path": "a", "sha256": "1"}]}, drop=CONVERSION_FIELDS)
        self.assertEqual(new, {"repo": "r", "revision": "new", "files": [{"path": "a", "sha256": "1"}]})

    RECORDED = {
        "repo": "r",
        "revision": "a",
        "files": [{"path": "a", "sha256": "1"}],
        "mlx_path": "~/models/mlx/x-4bit",
        "quantization": "4bit",
        "converted_with": {"mlx-lm": "1"},
        "converted_files_sha256": "f",
    }
    SOURCE = {"repo": "r", "revision": "a", "files": [{"path": "a", "sha256": "1", "size": 1}]}

    def test_reverifying_an_unchanged_source_keeps_the_recorded_conversion(self):
        self.assertEqual(stale_conversion_fields(self.RECORDED, self.SOURCE, "4bit"), ())

    def test_a_changed_source_or_quantization_clears_the_recorded_conversion(self):
        for verified, quantization in [
            ({**self.SOURCE, "revision": "b"}, "4bit"),
            ({**self.SOURCE, "repo": "other"}, "4bit"),
            ({**self.SOURCE, "files": [{"path": "a", "sha256": "2"}]}, "4bit"),
            (self.SOURCE, "8bit"),
        ]:
            with self.subTest(verified=verified, quantization=quantization):
                self.assertEqual(stale_conversion_fields(self.RECORDED, verified, quantization), CONVERSION_FIELDS)
        self.assertEqual(stale_conversion_fields({}, self.SOURCE, "4bit"), CONVERSION_FIELDS)

    def test_the_lock_must_pin_the_source_the_registry_names(self):
        registry = {"repo": "r", "revision": "a", "quantization": "4bit"}
        self.assertIsNone(registry_problem("m", registry, self.RECORDED))
        moved = registry_problem("m", {**registry, "revision": "b"}, self.RECORDED)
        self.assertEqual(moved, "models.lock.json pins revision 'a' (models.json: 'b'); run scripts/fetch-models.sh m")
        both = registry_problem("m", {**registry, "repo": "o", "revision": "b"}, self.RECORDED)
        self.assertIn("repo 'r' (models.json: 'o'), revision 'a'", both)
        self.assertIn("repo None", registry_problem("m", registry, {}))

    def test_recording_a_conversion_keeps_the_source(self):
        src = {"repo": "r", "revision": "new", "files": []}
        merged = merge_lock_entry(src, {"mlx_path": "~/m", "quantization": "4bit"})
        self.assertEqual(merged, {**src, "mlx_path": "~/m", "quantization": "4bit"})


class AtomicWriteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "models.lock.json"
        self.path.write_text('{"old": 1}\n')

    def tearDown(self):
        self.tmp.cleanup()

    def test_replaces_the_file(self):
        write_atomic(self.path, '{"new": 2}\n')
        self.assertEqual(self.path.read_text(), '{"new": 2}\n')
        self.assertEqual(os.listdir(self.tmp.name), ["models.lock.json"])

    def test_interrupted_write_keeps_the_old_file_and_no_temp_file(self):
        # Interrupted after the new content was written but before it was renamed into place.
        with mock.patch("verify_model.os.fsync", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                write_atomic(self.path, '{"new": 2, "half written": ')
        self.assertEqual(self.path.read_text(), '{"old": 1}\n')
        self.assertEqual(os.listdir(self.tmp.name), ["models.lock.json"])


class HomeRelativeTest(unittest.TestCase):
    def test_paths_under_home_are_recorded_with_tilde(self):
        self.assertEqual(home_relative(Path.home() / "models" / "raw" / "x"), "~/models/raw/x")
        self.assertEqual(home_relative(Path("~/models/mlx/y")), "~/models/mlx/y")

    def test_paths_outside_home_stay_absolute(self):
        self.assertEqual(home_relative(Path("/opt/models/z")), str(Path("/opt/models/z").resolve()))


class RegistrySettingsTest(unittest.TestCase):
    def test_quantization_maps_to_convert_bits(self):
        self.assertEqual(q_bits("4bit"), 4)
        self.assertEqual(q_bits("8bit"), 8)
        for bad in ["4", "4-bit", "7bit", "", "fp16"]:
            with self.assertRaises(SystemExit, msg=bad):
                q_bits(bad)

    def test_manifest_is_size_then_files(self):
        files = [PublishedFile("a.json", 1_000, None, "x"), PublishedFile("m.safetensors", 7_699_999_000, "y", None)]
        self.assertEqual(manifest_lines(files), ["7.7", "a.json", "m.safetensors"])


class AllowedFilesTest(unittest.TestCase):
    def test_allowed_and_blocked(self):
        for path in ["model-00001-of-00002.safetensors", "config.json", "tokenizer.json", "tokenizer.model"]:
            self.assertTrue(is_allowed(path), path)
        for path in ["modeling_phi3.py", "pytorch_model.bin", "original/consolidated.00.pth", "original/params.json", "README.md"]:
            self.assertFalse(is_allowed(path), path)


if __name__ == "__main__":
    unittest.main()
