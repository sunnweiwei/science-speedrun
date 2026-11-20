"""Regression checks for the local, target-free reference experiment."""
import csv
import json
from pathlib import Path
import sys
import unittest

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "docs/s15"))
from run_cohort import SAMPLERS, config, prior_schedule


class S15ReferenceTest(unittest.TestCase):
    def test_independent_fits_and_same_device_control(self):
        folder = TASK / "environment/references/s15"
        with (folder / "by_length.csv").open() as f:
            chunks = list(csv.DictReader(f))
        checked = json.loads((folder / "coordinate-check.json").read_text())["by_sampler_and_length"]
        for chunk in chunks:
            observed = checked[chunk["sampler"]][chunk["length"]]
            self.assertEqual(observed["refolds_checked"], int(chunk["refolded"]))
            self.assertEqual(observed["passes"], int(chunk["passes"]))
            self.assertEqual(observed["passing_clusters"], int(chunk["passing_clusters"]))
            self.assertLess(observed["max_abs_scRMSD_difference_A"], 1e-6)
        repeat = json.loads((folder / "same-device-repeat.json").read_text())["by_length"]["150"]
        self.assertEqual(repeat["identical_serialized_backbone_coordinates"], 50)
        self.assertEqual(repeat["pass_decisions_changed"], 0)
        self.assertTrue(repeat["identical_cluster_partition"])

    def test_published_local_points_match_per_slot_measurements(self):
        folder = TASK / "environment/references/s15"
        with (folder / "summary.csv").open() as f:
            points = list(csv.DictReader(f))
        with (folder / "by_length.csv").open() as f:
            lengths = list(csv.DictReader(f))
        with (folder / "slots.tsv").open() as f:
            slots = list(csv.DictReader(f, delimiter="\t"))
        self.assertEqual([r["sampler"] for r in points], list(SAMPLERS))
        self.assertEqual((len(lengths), len(slots)), (15, 750))
        for point in points:
            arm = point["sampler"]
            chunks = [r for r in lengths if r["sampler"] == arm]
            self.assertEqual({int(r["length"]) for r in chunks}, {100, 150, 250})
            for chunk in chunks:
                n = chunk["length"]
                cohort = [r for r in slots if (r["sampler"], r["length"]) == (arm, n)]
                self.assertEqual({r["slot"] for r in cohort}, {f"l{n}d{i:03d}" for i in range(50)})
                hits = [r["foldseek_cluster"] for r in cohort if r["scRMSD_A"] and float(r["scRMSD_A"]) <= 1.5]
                self.assertEqual(int(chunk["passes"]), len(hits))
                self.assertEqual(int(chunk["passing_clusters"]), len(set(hits)))
            for field in ("slots", "passes", "refolded", "passing_clusters"):
                self.assertEqual(int(point[field]), sum(int(r[field]) for r in chunks))
            clusters = int(point["passing_clusters"])
            self.assertAlmostEqual(float(point["local_clustered_pct"]), 100 * clusters / 150)
            hours = sum(float(r["generation_seconds"]) + float(r["af3_seconds"]) for r in chunks) / 3600
            self.assertAlmostEqual(float(point["generation_and_af3_attempts_per_gpu_h"]), 150 / hours)
            self.assertAlmostEqual(float(point["generation_and_af3_passing_clusters_per_gpu_h"]), clusters / hours)

    def test_five_profiles_and_costs(self):
        for arm in SAMPLERS:
            cfg = config(arm, 100, Path("/assets"), Path("/output"))
            generation = cfg["generation"]
            bands = generation["n_recycle_schedule"]
            self.assertEqual(cfg["num_replicates"], 50)
            self.assertEqual(cfg["batch_size"], 5)
            self.assertEqual(sum(n for n, _ in bands), 400 if arm == "full-sde" else 60)
            self.assertEqual(sum(n*d for n,d in bands), 1199 if arm == "full-sde" else 180)
        with self.assertRaises(ValueError):
            config("unlisted", 100, Path("/assets"), Path("/output"))

    def test_prior_schedule_leaves_the_tsr_grid_intact(self):
        import torch
        from rfproteina.flow_matching.product_space_flow_matcher import get_gt
        # Capture the time points returned by the actual scheduler interface.
        mesh = torch.tensor([0.0, 0.04, 0.3, 0.7, 0.9, 1.0])
        mode = {"gt": {"mode": "1/t", "p": 1.0, "clamp_val": None},
                "simulation_step_params": {"sampling_mode": "vf_tsr"}}
        args = {"bb_ca": mode}
        def original(matcher, nsteps, configs):
            self.assertIs(configs, args)
            self.assertEqual(configs["bb_ca"]["simulation_step_params"]["sampling_mode"], "vf_tsr")
            return {"bb_ca": mesh}, {"bb_ca": torch.zeros(5)}
        ts, gt = prior_schedule(original, None, 5, args)
        self.assertIs(ts["bb_ca"], mesh)
        torch.testing.assert_close(gt["bb_ca"], get_gt(mesh[:-1], "1/t", 1.0))
