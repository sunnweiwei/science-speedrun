"""Numerical regression cases for the frozen three-budget report."""
import csv
from decimal import Decimal
import json
from pathlib import Path
import sys
import tempfile
import unittest

OVERALL = Path(__file__).resolve().parents[1] / "environment/references/overall"
sys.path.insert(0, str(OVERALL))
from budget_scorecard import PROFILE, TIERS, TRACKS, TRIAL_FIELDS, build, publish, sha


def fixture(root, *, repeats=1):
    """Fixed artificial ledger for exercising reporting arithmetic only."""
    with (OVERALL / "targets.csv").open() as f:
        targets = {}
        for row in csv.DictReader(f):
            targets.setdefault(row["dataset"], row["target"])
    folders = {}
    for tier, divisor in TIERS:
        folder = root / tier
        folder.mkdir()
        folders[tier] = folder
        counts = {"tenth": [1, 1, 1, 1], "quarter": [3, 3, 3, 3], "full": [2, 2, 0, 2]}[tier]
        ledger = []
        repeat_reports = []
        for rep in range(repeats):
            by_track = {}
            for (track, (dataset, criterion, _)), hits in zip(TRACKS.items(), counts):
                record = {"baselines": []}
                for label, k in (("Submission", hits), ("Low", 1)):
                    # The last slot repeats the first passing cluster when k > 0.
                    for i in range(4):
                        passed = bool(k and (i < k or i == 3))
                        cluster = str(i % k) if passed and i < k else "0" if passed else ""
                        ledger.append([rep, dataset, criterion, label, targets[dataset], str(i), 1,
                                       int(passed), cluster, 2, 3])
                    point = dict(method=label, attempts=4, targets=1, clustered_pct=k * 25,
                                 generation_attempts_h=1800, total_attempts_h=720)
                    if label == "Submission":
                        record["candidate"] = dict(point, complete=True)
                    else:
                        record["baselines"].append(dict(point, reference_package="reference", complete=True))
                by_track[track] = record
            repeat_reports.append({"by_track": by_track})
        with (folder / "trials.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("repeat",) + TRIAL_FIELDS)
            writer.writerows(ledger)
        report = dict(timing_profile=PROFILE, panel="verify", repeats=repeats, attempts_per_target=4,
                      method="Submission", suite_sha256="c"*64, submission_sha256="a"*64,
                      reference_sha256={"reference": "b"*64}, paired_references=["reference"],
                      device={"uuid": "example-device", "driver": "test"}, seconds_per_slot=60/divisor,
                      generation_timing_boundary="per_request_infer_subprocess_wall_clock",
                      af3_timing_boundary="shared_persistent_af3_server_warm_fold",
                      trials_sha256=sha((folder / "trials.csv").read_bytes()), by_track=by_track)
        if repeats > 1:
            report["per_repeat"] = repeat_reports
        (folder / "summary.json").write_text(json.dumps(report))
    return folders


class ThreeBudgetTest(unittest.TestCase):
    def test_peak_is_full_tier_and_low_tiers_remain_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, result = build(fixture(root, repeats=2), Decimal(60))
            self.assertEqual({(r["tier"], r["budget_s_per_slot"]) for r in rows},
                             {("tenth", 6), ("quarter", 15), ("full", 60)})
            self.assertEqual(result["macro_quality_pct"], {"tenth": 25, "quarter": 75, "full": 37.5})
            self.assertEqual(result["primary_full_budget_quality"], .375)
            self.assertEqual(result["macro_low_budget_quality_pct"], 50)
            self.assertEqual(result["macro_delta_pp_vs_best_paired_at_tier"]["tenth"], 0)
            self.assertEqual(result["by_track"]["Protein binding"]["retention_vs_full"]["quarter"], 1.5)
            self.assertIsNone(result["by_track"]["Enzyme scaffolding"]["retention_vs_full"]["quarter"])
            self.assertTrue(all(r["total_s_per_attempt"] == 5 and r["repeat_sd_pp"] == 0 for r in rows))
            publish(root / "result", rows, result)
            self.assertGreater((root / "result/cost-quality.png").stat().st_size, 1000)
            with self.assertRaises(FileExistsError):
                publish(root / "result", rows, result)

    def test_incomplete_reference_disables_claims_even_across_repeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            folders = fixture(Path(tmp), repeats=2)
            f = folders["tenth"] / "summary.json"
            d = json.loads(f.read_text())
            d["per_repeat"][1]["by_track"]["Protein binding"]["baselines"][0]["complete"] = False
            f.write_text(json.dumps(d))
            _, s = build(folders, Decimal(60))
            self.assertEqual(s["status"], "descriptive_only")
            self.assertIsNone(s["macro_delta_pp_vs_best_paired_at_tier"]["tenth"])
            self.assertEqual(s["macro_low_budget_quality_pct"], 50)
            self.assertEqual(s["primary_full_budget_quality"], .375)
            f = folders["full"] / "summary.json"
            d = json.loads(f.read_text())
            d["per_repeat"][0]["by_track"]["Protein binding"]["candidate"]["complete"] = False
            f.write_text(json.dumps(d))
            _, s = build(folders, Decimal(60))
            self.assertIsNone(s["primary_full_budget_quality"])
            self.assertEqual(s["macro_low_budget_quality_pct"], 50)
            self.assertIsNone(s["by_track"]["Protein binding"]["retention_vs_full"]["quarter"])

    def test_incompatible_or_corrupt_measurements_do_not_form_a_curve(self):
        with tempfile.TemporaryDirectory() as tmp:
            folders = fixture(Path(tmp))
            path = folders["quarter"] / "summary.json"
            original = json.loads(path.read_text())
            for patch in ({"seconds_per_slot": 30}, {"suite_sha256": "f"*64},
                          {"device": {"uuid": "different", "driver": "test"}},
                          {"submission_sha256": "f"*64}, {"reference_sha256": {"reference": "d"*64}},
                          {"timing_profile": "warm"}, {"attempts_per_target": 20}, {"trials_sha256": "0"*64}):
                with self.subTest(patch=patch):
                    path.write_text(json.dumps(dict(original, **patch)))
                    with self.assertRaises(ValueError):
                        build(folders, Decimal(60))
            path.write_text(json.dumps(original))
            ledger = folders["quarter"] / "trials.csv"
            rows = list(csv.DictReader(ledger.read_text().splitlines()))
            target = rows[0]["target"]
            ledger.write_text(ledger.read_text().replace(f",{target},", ",unrecognized-target,"))
            path.write_text(json.dumps(dict(original, trials_sha256=sha(ledger.read_bytes()))))
            with self.assertRaisesRegex(ValueError, "roster"):
                build(folders, Decimal(60))

    def test_unknown_reference_completion_and_public_draw_attestation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folders = fixture(Path(tmp))
            for folder in folders.values():
                path = folder / "summary.json"
                d = json.loads(path.read_text())
                for tr in d["by_track"].values():
                    del tr["baselines"][0]["complete"]
                path.write_text(json.dumps(d))
            _, s = build(folders, Decimal(60))
            self.assertEqual(s["status"], "descriptive_only")
            self.assertEqual(s["primary_full_budget_quality"], .375)
            self.assertFalse(any(s["paired_complete_at_tier"].values()))
            for folder in folders.values():
                path = folder / "summary.json"
                d = json.loads(path.read_text())
                del d["suite_sha256"]
                path.write_text(json.dumps(d))
                (folder / "budget_review.json").write_text(json.dumps(
                    {"summary_sha256": sha(path.read_bytes()), "draws_sha256": "e"*64, "all_arms_complete": True}))
            _, s = build(folders, Decimal(60))
            self.assertEqual(s["status"], "complete")
            path = folders["full"] / "summary.json"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "controller review"):
                build(folders, Decimal(60))
