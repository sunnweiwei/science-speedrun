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
            self.assertEqual((status["baseline_points"], status["submission_points"]), (3, 0))
            self.assertEqual(sum(len(s["missing"]) for s in status["tracks"].values()), 13)
            with (out / "points.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([int(r["clusters"]) for r in rows], [15, 12, 17])
            self.assertTrue(all(r["track"] == "DNA binding" and r["source"] == "local baseline" for r in rows))
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
