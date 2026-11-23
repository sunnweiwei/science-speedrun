import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

OVERALL = Path(__file__).resolve().parents[1] / "environment/references/overall"
sys.path.insert(0, str(OVERALL))
from benchmark_curve import read_targets, TRIAL_FIELDS
from measured_report import points, PROFILE, render, submission_measurements


class MeasuredReportTest(unittest.TestCase):
    def test_pinned_baselines_and_empty_panels(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report"
            status = render(out)
            self.assertEqual(status["submission_points"], 0)
            self.assertGreaterEqual(status["baseline_points"], 11)
            self.assertEqual(status["baseline_points"] + sum(len(s["missing"]) for s in status["tracks"].values()), 16)
            with (out / "points.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), status["baseline_points"])
            by_key = {(r["track"], r["method"]): int(r["clusters"]) for r in rows}
            for key, expected in (
                (("Protein binding", "Low"), 426),
                (("Protein binding", "Medium"), 481),
                (("Small-molecule binding", "Low"), 2009),
                (("Small-molecule binding", "Medium"), 2052),
                (("Small-molecule binding", "High"), 2288),
                (("Enzyme scaffolding", "Low"), 753),
                (("Enzyme scaffolding", "Medium"), 766),
                (("Enzyme scaffolding", "High"), 876),
                (("DNA binding", "Low"), 15),
                (("DNA binding", "Medium"), 12),
                (("DNA binding", "High"), 17),
            ):
                self.assertEqual(by_key.get(key), expected)
            self.assertTrue(all(r["source"] == "local baseline" for r in rows))
            self.assertEqual((out / "points.csv").read_bytes(), (OVERALL / "measured/points.csv").read_bytes())
            self.assertTrue(all((out / f).is_file() for f in ("generation.png", "including-af3.png")))

    def test_posthoc_candidate_needs_matched_receipt_and_complete_roster(self):
        baseline = read_targets(OVERALL / "reproduced_targets.csv")
        roster = baseline["DNA binding", "Low"]
        digest = "a" * 64
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(submission_measurements(root, digest), {})
            ledger = root / (digest + ".csv")
            with ledger.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(TRIAL_FIELDS)
                for target in roster:
                    for i in range(200):
                        writer.writerow(["dna_unfixed", "dna", "fixture", target, i, 1, 0, "", 5, 10])
            meta = {"profile": PROFILE, "submission_sha256": digest,
                    "trials_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest()}
            receipt = root / (digest + ".json")
            receipt.write_text(json.dumps(meta))
            candidate = submission_measurements(root, digest)
            added = points(baseline, candidate)[-1]
            self.assertEqual((added["attempts"], added["clusters"]), (600, 0))
            candidate["DNA binding", "fixture"].pop(next(iter(roster)))
            with self.assertRaises(ValueError):
                points(baseline, candidate)
            for changes in ({"profile": "cold"}, {"submission_sha256": "b"*64}, {"trials_sha256": "c"*64}):
                receipt.write_text(json.dumps(dict(meta, **changes)))
                with self.assertRaises(ValueError):
                    submission_measurements(root, digest)
