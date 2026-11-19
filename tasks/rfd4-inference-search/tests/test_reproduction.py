"""Recompute the published run records from individual slots, using only the standard library."""
import csv
import hashlib
import json
from pathlib import Path
import unittest

BASE = Path(__file__).resolve().parents[1] / "environment/baseline"


class PublicRecordsTest(unittest.TestCase):
    def test_records(self):
        for policy in ("stochastic-tsr", "geometry-beam", "two-wide-beam"):
            root = BASE / policy / "reproduction"
            self.assertTrue((root / "development.json").is_file())
            for path in sorted(root.glob("*.json")):
                with self.subTest(policy=policy, run=path.stem):
                    report = json.loads(path.read_text())
                    digest = hashlib.sha256((root.parent / "policy.json").read_bytes()).hexdigest()
                    self.assertEqual(digest, report["policy_sha256"])
                    with (root / (path.stem + "-slots.tsv")).open() as f:
                        rows = list(csv.DictReader(f, delimiter="\t"))
                    rates = []
                    for length, counts in report["by_length"].items():
                        group = [r for r in rows if r["length"] == length]
                        passed = [r for r in group if r["scRMSD_A"] and float(r["scRMSD_A"]) <= 1.5]
                        distinct = {r["foldseek_cluster"] for r in passed}
                        self.assertEqual(len(group), counts["slots"])
                        self.assertEqual(len({r["slot"] for r in group}), len(group))
                        self.assertEqual(len(passed), counts["passes"])
                        self.assertEqual(len(distinct), counts["passing_clusters"])
                        self.assertEqual(counts["reserved_trunk_passes"], 180 * counts["slots"])
                        rates.append(len(distinct) / len(group))
                    self.assertEqual(len(rates), 3)
                    self.assertAlmostEqual(sum(rates) / 3, report["reward"])


if __name__ == "__main__":
    unittest.main()
