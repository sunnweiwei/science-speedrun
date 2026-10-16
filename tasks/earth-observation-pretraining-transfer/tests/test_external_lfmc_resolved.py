"""Recovery must preserve fixed coverage and immutable failure history."""

import copy
import json
import shutil
import unittest

import test_external_lfmc_reconcile as fixture
from test_external_lfmc_reconcile import write
from reconcile_external_lfmc_cohort import digest
from reconcile_external_lfmc_resolved import reconcile_resolved


class ResolvedTests(unittest.TestCase):
    def setUp(self):
        fixture.CohortGateTests.setUp(self)
        launch = json.loads(self.launch.read_text())
        launch["output_root"] = str(self.cohort)
        write(self.launch, launch)
        self.ledger_path = self.root / "ledger.json"
        self.ledger = {
            "manifest_sha256": digest(self.manifest),
            "tasks": 1,
            "sites": 1,
            "launches": {
                "original": {
                    "path": str(self.launch),
                    "sha256": digest(self.launch),
                    "indices": [0],
                }
            },
            "rows": [{"index": 0, "attempts": [{"launch": "original"}]}],
        }

    def resolved(self):
        write(self.ledger_path, self.ledger)
        return reconcile_resolved(self.manifest, self.ledger_path)

    def retry(self):
        retry_root = self.root / "retry"
        shutil.copytree(self.directory, retry_root / "0000")
        (self.directory / "completion.json").unlink()
        failure = self.directory / "failure.json"
        write(failure, {"exit_code": 1})
        launch = json.loads(self.launch.read_text())
        launch.update(output_root=str(retry_root), indices=[0])
        retry_launch = self.root / "retry-launch.json"
        write(retry_launch, launch)
        self.ledger["launches"]["retry"] = {
            "path": str(retry_launch),
            "sha256": digest(retry_launch),
            "indices": [0],
        }
        self.ledger["rows"][0]["attempts"] = [
            {"launch": "original", "failure_sha256": digest(failure)},
            {"launch": "retry"},
        ]
        return retry_root

    def test_resolved_original(self):
        self.assertTrue(self.resolved()["all_inputs_verified"])

    def test_recovery_retains_failure(self):
        self.retry()
        report = self.resolved()
        self.assertTrue(report["all_inputs_verified"])
        self.assertEqual(
            report["rows"][0]["attempts"][0]["status"], "preserved_failure"
        )

    def test_missing_fixed_row(self):
        self.ledger["rows"] = []
        with self.assertRaisesRegex(ValueError, "every fixed row"):
            self.resolved()

    def test_duplicate_fixed_row(self):
        self.ledger["rows"].append(copy.deepcopy(self.ledger["rows"][0]))
        with self.assertRaisesRegex(ValueError, "every fixed row"):
            self.resolved()

    def test_changed_failure(self):
        self.retry()
        write(self.directory / "failure.json", {"exit_code": 2})
        self.assertEqual(self.resolved()["counts"]["invalid"], 1)

    def test_completed_original_cannot_be_replaced(self):
        root = self.retry()
        shutil.copyfile(
            root / "0000/completion.json", self.directory / "completion.json"
        )
        self.assertEqual(self.resolved()["counts"]["invalid"], 1)

    def test_unlisted_attempt(self):
        root = self.retry()
        (root / "0001").mkdir()
        self.assertFalse(self.resolved()["all_inputs_verified"])

    def test_corrupt_recovered_input(self):
        root = self.retry()
        with (root / "0000/input/input.npz").open("ab") as handle:
            handle.write(b"corruption")
        self.assertEqual(self.resolved()["counts"]["invalid"], 1)

    def test_no_fallback_to_earlier_attempt(self):
        root = self.retry()
        (root / "0000/completion.json").unlink()
        result = self.resolved()
        self.assertEqual(result["counts"]["incomplete"], 1)
        self.assertEqual(len(result["rows"]), 1)
