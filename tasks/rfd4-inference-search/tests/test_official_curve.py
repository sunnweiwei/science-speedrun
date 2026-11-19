import csv
from dataclasses import replace
import io
from pathlib import Path
import sys
import tempfile
import unittest

OVERALL = Path(__file__).resolve().parents[1] / "environment/references/overall"
sys.path.insert(0, str(OVERALL))
from benchmark_curve import (
    FIELDS, TRIAL_FIELDS, Measurement, frontier, operating_points, read_trials,
    summarize, write_points,
)
from verify_sources import S29


def csv_file(path, fields, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


class OfficialCurveTest(unittest.TestCase):
    def test_published_archive(self):
        points = operating_points(OVERALL / "targets.csv")
        out = io.StringIO()
        write_points(points, out)
        self.assertEqual(out.getvalue(), (OVERALL / "operating_points.csv").read_text())
        self.assertEqual(len(points), 16)
        self.assertEqual(sum(p["attempts"] for p in points), 112800)
        self.assertEqual(sum(p["targets"] for p in points), 564)
        self.assertEqual([(p["track"], p["method"]) for p in points if p["highest_total_yield"]],
                         [("Protein binding", "Medium"), ("Small-molecule binding", "High"),
                          ("Enzyme scaffolding", "High"), ("DNA binding", "Low")])
        for p in points:
            if p["method"] in ("High", "Search"):
                actual = tuple(round(p[k], 1) for k in ("clustered_pct", "generation_clusters_h", "total_clusters_h"))
                self.assertEqual(actual, S29[p["track"]][p["method"]])

    def test_macro_yield_and_pooled_speed(self):
        a = Measurement(200, 200, 6, 4, 1800, 1800)
        b = replace(a, passes=0, clusters=0, generation=3600, af3=3600)
        p = summarize("test", "test", {"a": a, "b": b}, "fixture")
        self.assertEqual(p["total_clusters_h"], 2)
        self.assertAlmostEqual(p["generation_attempts_h"], 400 / 1.5)
        self.assertAlmostEqual(p["total_attempts_h"], 400 / 3)
        self.assertEqual(p["clustered_pct"], 1)
        self.assertEqual(p["covered_targets"], 1)

    def test_external_cohort_integrity(self):
        with (OVERALL / "targets.csv").open() as f:
            rows = [dict(r, method="Control replay") for r in csv.DictReader(f) if r["method"] == "High"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            csv_file(path, FIELDS, rows)
            points = operating_points(OVERALL / "targets.csv", observations=path)
            for p in points:
                if p["method"] == "Control replay":
                    high = next(q for q in points if q["track"] == p["track"] and q["method"] == "High")
                    self.assertEqual(p["total_clusters_h"], high["total_clusters_h"])
                    self.assertEqual(p["total_pareto"], high["total_pareto"])
                    self.assertEqual(p["source"], "supplied observations")
            faster = [dict(r, generation_gpu_s=float(r["generation_gpu_s"]) / 100,
                           af3_gpu_s=float(r["af3_gpu_s"]) / 100) for r in rows]
            csv_file(path, FIELDS, faster)
            accelerated = [p for p in operating_points(OVERALL / "targets.csv", observations=path)
                           if p["method"] == "Control replay"]
            self.assertTrue(all(p["quality_margin_pp_at_total_attempts_h"] is None for p in accelerated))
            self.assertTrue(all(p["yield_gain_pct_vs_best_reference"] > 0 for p in accelerated))
            for invalid in (rows[:-1], rows + [rows[0]], [dict(r, af3_gpu_s="NaN") for r in rows],
                            [dict(r, criterion="disco") for r in rows], [dict(r, attempts=199) for r in rows]):
                csv_file(path, FIELDS, invalid)
                with self.assertRaises(ValueError):
                    operating_points(OVERALL / "targets.csv", observations=path)

    def test_attempt_counts_clusters_and_failed_cost(self):
        base = dict(dataset="dna_unfixed", criterion="dna", method="Fixture", target="toy",
                    scored=0, passed=0, cluster_id="", generation_gpu_s="2", af3_gpu_s="0")
        rows = [dict(base, attempt_id=i) for i in range(200)]
        for i, cluster in enumerate(["x", "x", "y"]):
            rows[i].update(scored=1, passed=1, cluster_id=cluster, af3_gpu_s="3")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trials.csv"
            csv_file(path, TRIAL_FIELDS, rows)
            m = read_trials(path)["DNA binding", "Fixture"]["toy"]
            self.assertEqual(m, Measurement(200, 3, 3, 2, 400, 9))
            csv_file(path, TRIAL_FIELDS, [dict(r, scored=1, passed=0, cluster_id="", af3_gpu_s="0") for r in rows])
            self.assertEqual(read_trials(path)["DNA binding", "Fixture"]["toy"].af3, 0)
            for bad in (rows[:-1], rows + [rows[0]], [dict(rows[0], cluster_id="")] + rows[1:]):
                csv_file(path, TRIAL_FIELDS, bad)
                with self.assertRaises(ValueError):
                    read_trials(path)

    def test_complete_trial_file_adds_one_point_per_suite(self):
        # Pure accounting fixture. Two successful attempts share one cluster per target.
        with (OVERALL / "targets.csv").open() as f:
            roster = [r for r in csv.DictReader(f) if r["method"] == "Low"]
        rows = (dict(dataset=r["dataset"], criterion=r["criterion"], method="Accounting fixture",
                     target=r["target"], attempt_id=i, scored=int(i < 3), passed=int(i < 2),
                     cluster_id="duplicate" if i < 2 else "", generation_gpu_s=1, af3_gpu_s=int(i < 3))
                for r in roster for i in range(200))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trials.csv"
            csv_file(path, TRIAL_FIELDS, rows)
            added = [p for p in operating_points(OVERALL / "targets.csv", trials=path)
                     if p["source"] == "supplied observations"]
        self.assertEqual(len(added), 4)
        for p in added:
            self.assertEqual(p["passes"], 2 * p["targets"])
            self.assertEqual(p["clusters"], p["targets"])
            self.assertEqual(p["clustered_pct"], 0.5)
            self.assertAlmostEqual(p["total_clusters_h"], 3600 / 203)

    def test_dominance_preserves_ties_and_tradeoffs(self):
        pts = [dict(x=x, clustered_pct=y, name=i) for i,(x,y) in enumerate([(3,1),(2,3),(1,2),(3,1)])]
        self.assertEqual([p["name"] for p in frontier(pts, "x")], [0,1,3])


if __name__ == "__main__":
    unittest.main()
