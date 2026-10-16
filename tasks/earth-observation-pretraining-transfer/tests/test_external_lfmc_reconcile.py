"""Cohort gate must not silently drop failures or trust stale success receipts."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/downstream"))
from reconcile_external_lfmc_cohort import digest, reconcile


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class CohortGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cohort = self.root / "cohort"
        self.directory = self.cohort / "0000"
        self.manifest = self.root / "manifest.json"
        self.launch = self.root / "launch.json"
        site = {"site_id": "site", "observations": [{"date": "2020-01-01"}]}
        write(self.manifest, {"sites": [site]})
        sources = {"source.py": "pinned"}
        write(
            self.launch,
            {
                "manifest_sha256": digest(self.manifest),
                "tasks": 1,
                "sites": 1,
                "source_sha256": sources,
            },
        )
        write(
            self.directory / "manifest.json",
            {
                "sites": [site],
                "parent_task_index": 0,
                "parent_manifest_sha256": digest(self.manifest),
            },
        )
        scope = {"target_used": False, "model_used": False}
        identity = {"site_id": "site", "date": "2020-01-01", **scope}
        acquired = {
            **identity,
            "manifest_sha256": digest(self.directory / "manifest.json"),
        }
        write(self.directory / "dataset/completion.json", acquired)
        verification = {
            **scope,
            "structural_checks_passed": True,
            "periods": [
                {"modality": m, "index": i}
                for m in ("sentinel1", "sentinel2")
                for i in range(12)
            ],
        }
        write(self.directory / "verification.json", verification)
        arrays = {
            "sentinel2_l2a": np.zeros((32, 32, 12, 12), dtype=np.float32),
            "sentinel1": np.zeros((32, 32, 12, 2), dtype=np.float32),
            "sentinel2_l2a_valid": np.ones((12, 3), dtype=bool),
            "sentinel1_valid": np.ones((12, 1), dtype=bool),
        }
        (self.directory / "input").mkdir()
        np.savez_compressed(self.directory / "input/input.npz", **arrays)
        input_hash = digest(self.directory / "input/input.npz")
        prepared = {
            **acquired,
            "input_sha256": input_hash,
            "verification_sha256": digest(self.directory / "verification.json"),
            "arrays": {
                k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                for k, v in arrays.items()
            },
        }
        write(self.directory / "input/report.json", prepared)
        launched = {
            **identity,
            "task_index": 0,
            "parent_manifest_sha256": digest(self.manifest),
            "source_sha256": sources,
        }
        write(self.directory / "launch.json", launched)
        write(
            self.directory / "completion.json", {**launched, "input_sha256": input_hash}
        )

    def audit(self):
        return reconcile(self.manifest, self.launch, self.cohort)

    def test_complete(self):
        self.assertTrue(self.audit()["all_inputs_verified"])

    def test_missing_receipt_stays_in_manifest(self):
        (self.directory / "completion.json").unlink()
        result = self.audit()
        self.assertFalse(result["all_inputs_verified"])
        self.assertEqual(result["counts"]["incomplete"], 1)
        self.assertEqual(len(result["rows"]), 1)

    def test_failure_overrides_success(self):
        write(self.directory / "failure.json", {"exit_code": 1})
        self.assertEqual(self.audit()["counts"]["failure_record_present"], 1)

    def test_modified_input_is_rejected(self):
        with (self.directory / "input/input.npz").open("ab") as handle:
            handle.write(b"changed")
        self.assertEqual(self.audit()["counts"]["invalid"], 1)

    def test_wrong_site_is_rejected(self):
        path = self.directory / "dataset/completion.json"
        report = json.loads(path.read_text())
        report["site_id"] = "wrong"
        write(path, report)
        self.assertEqual(self.audit()["counts"]["invalid"], 1)

    def test_extra_directory_is_rejected(self):
        (self.cohort / "0623").mkdir()
        self.assertFalse(self.audit()["all_inputs_verified"])

    def test_parent_change_is_rejected(self):
        with self.manifest.open("a") as handle:
            handle.write(" ")
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            self.audit()


if __name__ == "__main__":
    unittest.main()
