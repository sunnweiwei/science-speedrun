"""Regression checks for the local, target-free reference experiment."""
from pathlib import Path
import sys
import unittest

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "docs/s15"))
from run_cohort import SAMPLERS, config, prior_schedule


class S15ReferenceTest(unittest.TestCase):
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
