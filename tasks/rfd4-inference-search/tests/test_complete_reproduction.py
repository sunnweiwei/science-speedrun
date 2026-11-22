"""End-to-end integrity and script checks for the completed RFD4 reproduction artifacts."""
import csv
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

TASK = Path(__file__).resolve().parents[1]
OVERALL = TASK / "environment/references/overall"
for mod_dir in (
    OVERALL,
    TASK / "docs/s15",
    TASK / "docs/dna-binding",
    TASK / "docs/monomer-beam",
    TASK / "docs/s16",
):
    sys.path.insert(0, str(mod_dir))

from benchmark_curve import operating_points
from plot_dna_curve import render_dna_reports_and_plots
from plot_monomer_beam import render_monomer_beam_plot
from plot_s16_hbonds import render_s16_plot
from plot_schedule_and_repeats import render_plot as render_s15_schedule_plot
from run_dna_cohort import DNA_SCHEDULES, build_gen_argv
from run_monomer_beam import build_hydra_argv
from run_monomer_cohort import build_monomer_cfg
from run_s16_hbonds import build_s16_cfg


class CompleteReproductionTest(unittest.TestCase):
    def test_dna_binding_three_official_settings(self):
        dna_dir = TASK / "environment/references/dna_binding"
        pts = operating_points(OVERALL / "targets.csv", trials=dna_dir / "trials_all.csv", only_track="DNA binding")
        by_method = {p["method"]: p for p in pts}
        self.assertEqual(len(pts), 10)  # 4 author + 3 local warm + 3 local cold
        expected = {
            "Local Low (warm)": (600, 15, 15, 2.5),
            "Local Medium (warm)": (600, 13, 12, 2.0),
            "Local High (warm)": (600, 19, 17, 17 / 6),
        }
        for name, (att, passes, clus, pct) in expected.items():
            p = by_method[name]
            self.assertEqual(p["attempts"], att)
            self.assertEqual(p["passes"], passes)
            self.assertEqual(p["clusters"], clus)
            self.assertEqual(p["covered_targets"], 3)
            self.assertAlmostEqual(p["clustered_pct"], pct, places=6)
            cold = by_method[name.replace("(warm)", "(cold)")]
            self.assertEqual(cold["clusters"], clus)
            self.assertLess(cold["total_attempts_h"], p["total_attempts_h"])

    def test_s15_base_schedule_and_gemmi_audit(self):
        s15_dir = TASK / "environment/references/s15"
        with (s15_dir / "base_schedule_summary.csv").open() as f:
            summary = list(csv.DictReader(f))
        with (s15_dir / "base_schedule_by_length.csv").open() as f:
            lengths = list(csv.DictReader(f))
        with (s15_dir / "base_schedule_slots.tsv").open() as f:
            slots = list(csv.DictReader(f, delimiter="\t"))
        gemmi_check = json.loads((s15_dir / "base_schedule_coordinate_check.json").read_text())["by_sampler_and_length"]

        self.assertEqual((len(summary), len(lengths), len(slots)), (5, 15, 750))
        for row in summary:
            arm = row["sampler"]
            chunks = [c for c in lengths if c["sampler"] == arm]
            self.assertEqual(len(chunks), 3)
            for c in chunks:
                n = c["length"]
                g = gemmi_check[arm][n]
                self.assertEqual(g["refolds_checked"], int(c["refolded"]))
                self.assertEqual(g["passes"], int(c["passes"]))
                self.assertEqual(g["passing_clusters"], int(c["passing_clusters"]))
                self.assertLess(g["max_abs_scRMSD_difference_A"], 1e-6)
                cohort = [s for s in slots if (s["sampler"], s["length"]) == (arm, n)]
                self.assertEqual(len(cohort), 50)
                hits = {s["foldseek_cluster"] for s in cohort if s["scRMSD_A"] and float(s["scRMSD_A"]) <= 1.5}
                self.assertEqual(len(hits), int(c["passing_clusters"]))

    def test_multi_seed_paired_repeatability(self):
        s15_dir = TASK / "environment/references/s15"
        with (s15_dir / "multi_seed_repeats.csv").open() as f:
            rows = list(csv.DictReader(f))
        with (s15_dir / "multi_seed_slots.tsv").open() as f:
            slots = list(csv.DictReader(f, delimiter="\t"))
        summary = json.loads((s15_dir / "multi_seed_summary.json").read_text())

        self.assertEqual(len(rows), 10)  # 5 seeds x 2 samplers
        self.assertEqual(len(slots), 1500)  # 10 cohorts x 150 slots
        for r in rows:
            cohort = [s for s in slots if s["base_seed"] == r["base_seed"] and s["sampler"] == r["sampler"]]
            self.assertEqual(len(cohort), 150)
            by_len_clusters = sum(
                len({s["foldseek_cluster"] for s in cohort if int(s["length"]) == n and s["scRMSD_A"] and float(s["scRMSD_A"]) <= 1.5})
                for n in (100, 150, 250)
            )
            self.assertEqual(by_len_clusters, int(r["passing_clusters"]))
        st = [int(r["passing_clusters"]) for r in rows if r["sampler"] == "stochastic-tsr"]
        od = [int(r["passing_clusters"]) for r in rows if r["sampler"] == "ode-tsr"]
        self.assertAlmostEqual(summary["stochastic_tsr"]["mean_clusters"], statistics.fmean(st))
        self.assertAlmostEqual(summary["ode_tsr"]["mean_clusters"], statistics.fmean(od))

    def test_monomer_beam_matched_and_scaling(self):
        mb_dir = TASK / "environment/references/monomer_beam"
        with (mb_dir / "matched_30slot_summary.csv").open() as f:
            matched = {r["method"]: r for r in csv.DictReader(f)}
        with (mb_dir / "matched_30slot_slots.tsv").open() as f:
            slots = list(csv.DictReader(f, delimiter="\t"))
        self.assertEqual(set(matched), {"low", "medium", "high", "beam"})
        self.assertEqual(len(slots), 120)
        for m, row in matched.items():
            sub = [s for s in slots if s["method"] == m]
            self.assertEqual(len(sub), 30)
            c314 = sum(
                len({s["cluster_314159"] for s in sub if int(s["length"]) == n and float(s["scRMSD_314159_A"]) <= 1.5})
                for n in (100, 150, 250)
            )
            self.assertEqual(c314, int(row["clusters_seed_314159"]))
        self.assertGreater(float(matched["beam"]["clustered_pct_seed_314159"]),
                           float(matched["high"]["clustered_pct_seed_314159"]))
        self.assertGreater(float(matched["low"]["end_to_end_clusters_per_gpu_h_seed_314159"]),
                           float(matched["beam"]["end_to_end_clusters_per_gpu_h_seed_314159"]))

    def test_s16_hbond_search(self):
        s16_dir = TASK / "environment/references/s16"
        with (s16_dir / "summary.csv").open() as f:
            summary = list(csv.DictReader(f))
        with (s16_dir / "designs.csv").open() as f:
            designs = list(csv.DictReader(f))
        self.assertEqual(len(summary), 7)
        self.assertEqual(len(designs), 700)
        for r in summary:
            vals = [float(d["hbond_atom_pair_count_coi"]) for d in designs
                    if d["schedule_mode"] == r["schedule_mode"] and d["sampler"] == r["sampler"]]
            self.assertEqual(len(vals), 100)
            self.assertAlmostEqual(statistics.fmean(vals), float(r["local_mean_hbonds"]), places=6)

    def test_checked_in_reproduction_drivers_and_plotters(self):
        # 1. Monomer config builder
        c60 = build_monomer_cfg("stochastic-tsr", "base_1pass", 100, 50, 727100, Path("/assets"), Path("/out"))
        c400 = build_monomer_cfg("full-sde", "base_1pass", 150, 50, 727150, Path("/assets"), Path("/out"))
        chigh = build_monomer_cfg("stochastic-tsr", "high", 250, 50, 727250, Path("/assets"), Path("/out"))
        self.assertEqual(sum(w * d for w, d in c60["generation"]["n_recycle_schedule"]), 60)
        self.assertEqual(sum(w * d for w, d in c400["generation"]["n_recycle_schedule"]), 400)
        self.assertEqual(sum(w * d for w, d in chigh["generation"]["n_recycle_schedule"]), 300)

        # 2. DNA schedule & CLI builder (§S8.3: High uses flat 5 passes/step)
        self.assertEqual(DNA_SCHEDULES, {"Low": "low/front_loaded", "Medium": "medium/front_loaded", "High": "high/flat"})
        dna_argv = build_gen_argv("High", Path("/in/7m5w.cif.zst"), Path("/out"), Path("/assets"), 200, 5, 727001)
        self.assertIn("recycle_strategy=high/flat", dna_argv)

        # 3. Monomer AF3-guided beam Hydra overrides
        beam_argv = build_hydra_argv(Path("/in/m150.cif.zst"), Path("/out"), Path("/assets"), 10, 727150, 727000)
        self.assertIn("subset_to_examples=null", beam_argv)
        self.assertIn("search.save_rollouts=false", beam_argv)
        self.assertIn("search.save_criteria=null", beam_argv)

        # 4. S16 SAM H-bond search config builder
        s16_base = build_s16_cfg("full-sde", "base_1pass", Path("/in/SAM.cif.zst"), Path("/assets"), Path("/out"), 50, 5, 727000)
        s16_mid = build_s16_cfg("stochastic-tsr", "midpeak", Path("/in/SAM.cif.zst"), Path("/assets"), Path("/out"), 50, 5, 727000)
        self.assertEqual(set(s16_base["search"]["checkpoints"]), {"100", "200"})
        self.assertEqual(set(s16_mid["search"]["checkpoints"]), {"15", "30"})
        self.assertEqual(sum(w * d for w, d in s16_mid["generation"]["n_recycle_schedule"]), 150)

        # 5. Smoke-check all 4 checked-in plotters into a fresh temporary directory
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            render_dna_reports_and_plots(
                OVERALL / "targets.csv",
                TASK / "environment/references/dna_binding/trials_warm.csv",
                TASK / "environment/references/dna_binding/trials_all.csv",
                root / "dna",
            )
            render_s15_schedule_plot(
                TASK / "environment/references/s15/base_schedule_summary.csv",
                TASK / "environment/references/s15/multi_seed_repeats.csv",
                root / "s15.png",
            )
            render_monomer_beam_plot(
                TASK / "environment/references/monomer_beam/matched_30slot_summary.csv",
                TASK / "environment/references/monomer_beam/scaling_150slot_summary.csv",
                root / "mb.png",
            )
            render_s16_plot(TASK / "environment/references/s16/summary.csv", root / "s16.png")
            for expected_file in (
                root / "dna/quality-throughput-dna.png",
                root / "dna/quality-throughput-dna-af3.png",
                root / "dna/side_by_side_summary.csv",
                root / "s15.png",
                root / "mb.png",
                root / "s16.png",
            ):
                self.assertTrue(expected_file.is_file(), f"Missing {expected_file}")
                self.assertGreater(expected_file.stat().st_size, 500)


if __name__ == "__main__":
    unittest.main()
