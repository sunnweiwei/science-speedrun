"""Pinned prepared releases and normalization must preserve evaluation semantics."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environment/data"))
sys.path.insert(0, str(ROOT / "environment/evaluation"))
import benchmark_score
import materialize_hf
import release_manifest


class ScoreTests(unittest.TestCase):
    def test_reference_anchors_and_saturation_preserve_raw_performance(self):
        lower, upper = benchmark_score.BOUNDS["lower"], benchmark_score.BOUNDS["upper"]
        for raw, expected in [(lower, 0), ((lower + upper) / 2, 0.5), (upper, 1), (0, 0), (1, 1)]:
            with self.subTest(raw=raw):
                value = benchmark_score.normalize(raw)
                self.assertAlmostEqual(value["score"], expected)
                self.assertEqual(value["raw_performance"], raw)
        above = benchmark_score.normalize(0.7)
        self.assertGreater(above["unclipped_score"], 1)
        self.assertEqual(above["saturation"], "upper")
        self.assertEqual(benchmark_score.normalize(0)["saturation"], "lower")
        self.assertIsNone(benchmark_score.normalize(upper)["saturation"])

    def test_invalid_performance_is_not_a_valid_reward(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                benchmark_score.normalize(value)

    def test_anchor_evidence_matches_frozen_results(self):
        formal = json.loads((ROOT / "docs" / "baseline_score.json").read_text())
        self.assertEqual(formal["metrics"]["hit_rate_at_1"], benchmark_score.BOUNDS["upper"])
        self.assertEqual(formal["reward"], 1)
        evidence = json.loads((ROOT / "docs" / "release_validation.json").read_text())
        self.assertIn(str(benchmark_score.BOUNDS["lower"]), json.dumps(evidence))


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        labels = self.source / "labels.jsonl"
        labels.write_text('{"query_id":"q","correct_candidate_id":"a","true_smiles":"C"}\n')
        (self.source / "manifest.json").write_text(json.dumps({
            "files": {"labels.jsonl": release_manifest.sha256(labels)}, "queries": 17556,
        }))
        self.manifest = release_manifest.build(self.source, "private")
        manifest_path = self.source / "release-manifest.json"
        manifest_path.write_text(json.dumps(self.manifest))
        self.pin = {"repo_id": "Corning/fixture", "revision": "a" * 40,
                    "manifest_sha256": release_manifest.sha256(manifest_path)}
        self.calls = []

    def download(self, *, filename: str, cache_dir: Path, **kwargs: str) -> str:
        self.calls.append(filename)
        output = cache_dir / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.source / filename, output)
        return str(output)

    def test_fresh_download_and_verified_offline_reuse(self):
        destination = self.root / "downloaded"
        result = materialize_hf.fetch_release(self.pin, destination, "private", self.download)
        self.assertTrue(result["verified"])
        self.assertEqual(len(self.calls), 3)
        self.assertEqual((destination / "labels.jsonl").stat().st_mode & 0o077, 0)
        with mock.patch.object(materialize_hf, "verify", wraps=release_manifest.verify) as verify:
            materialize_hf.fetch_release(self.pin, destination, "private", lambda **kwargs: self.fail("download on reuse"))
            verify.assert_called_once()
        (destination / "labels.jsonl").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "corrupt"):
            materialize_hf.fetch_release(self.pin, destination, "private", self.download)

    def test_hf_metadata_stays_outside_clean_payload_and_is_not_deleted(self):
        destination = self.root / "downloaded"
        def with_cache_metadata(**kwargs):
            cache = kwargs["cache_dir"]
            metadata = cache / ".cache/huggingface/download/held-open.metadata"
            metadata.parent.mkdir(parents=True, exist_ok=True)
            metadata.write_text("simulated NFS-held HF metadata")
            return self.download(**kwargs)
        result = materialize_hf.fetch_release(self.pin, destination, "private", with_cache_metadata)
        self.assertTrue(result["verified"])
        self.assertFalse((destination / ".cache").exists())
        external = self.root / ".massspecgym-hf-cache"
        self.assertTrue((external / ".cache/huggingface/download/held-open.metadata").is_file())
        self.assertEqual(external.stat().st_mode & 0o077, 0)
        self.assertEqual({p.name for p in destination.iterdir()}, {"manifest.json", "labels.jsonl", "release-manifest.json"})
        self.assertEqual((destination / "labels.jsonl").stat().st_ino, (external / "labels.jsonl").stat().st_ino)

    def test_undeclared_files_cannot_enter_a_reused_release(self):
        destination = self.root / "downloaded"
        materialize_hf.fetch_release(self.pin, destination, "private", self.download)
        (destination / "accidental-secret.txt").write_text("fixture")
        with self.assertRaisesRegex(ValueError, "undeclared"):
            materialize_hf.fetch_release(self.pin, destination, "private", self.download)

    def test_corrupt_payload_never_publishes_destination(self):
        (self.source / "labels.jsonl").write_text("corrupt")
        destination = self.root / "invalid"
        with self.assertRaisesRegex(ValueError, "corrupt"):
            materialize_hf.fetch_release(self.pin, destination, "private", self.download)
        self.assertFalse(destination.exists())
        self.assertFalse(list(self.root.glob(".massspecgym-private-*")))

    def test_pin_manifest_path_and_existing_destination_fail_closed(self):
        destination = self.root / "existing"
        destination.mkdir()
        with self.assertRaises(FileExistsError):
            materialize_hf.fetch_release(self.pin, destination, "private", self.download)
        for pin in [{**self.pin, "revision": "main"}, {**self.pin, "manifest_sha256": "bad"}]:
            with self.assertRaises(ValueError):
                materialize_hf.fetch_release(pin, self.root / "new", "private", self.download)
        for unsafe in ("../outside", "/tmp/outside", "nested/../../outside", "x\\y", "x//y"):
            with self.assertRaises(ValueError):
                release_manifest.safe_path(self.root, unsafe)
        link = self.root / "link"
        link.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(ValueError):
            materialize_hf.fetch_release(self.pin, link, "private", self.download)
        with self.assertRaisesRegex(ValueError, "complete, exact"):
            release_manifest.validate_manifest({**self.manifest, "files": {"../labels.jsonl": {}}}, "private")

    def test_wrong_manifest_is_rejected_before_payload_download(self):
        pin = {**self.pin, "manifest_sha256": "b" * 64}
        with self.assertRaisesRegex(ValueError, "pinned manifest"):
            materialize_hf.fetch_release(pin, self.root / "wrong", "private", self.download)
        self.assertEqual(self.calls, ["release-manifest.json"])

    def test_source_rebuild_selects_both_exact_upstream_files(self):
        source = (ROOT / "environment/data/materialize.sh").read_text()
        self.assertIn('MASSSPECGYM_MATERIALIZE_MODE:-prepared', source)
        self.assertIn('data/MassSpecGym1.5.tsv', source)
        self.assertIn('data/molecules/MassSpecGym1.5_retrieval_candidates_mass.json', source)
        self.assertNotIn('--include', source)
        self.assertLess(source.index('"$massspec_script_dir/materialize.py"'), source.index('/starter/glacier/prepare.py'))


if __name__ == "__main__":
    unittest.main()
