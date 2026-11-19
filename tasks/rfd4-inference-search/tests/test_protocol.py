import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "environment/evaluation"))
from policy import InvalidPolicy, cost, load_policy, native_config
from score import score, scrmsd
from run import harbor_metrics


class ContractTest(unittest.TestCase):
    def test_baselines_equal_budget_and_rollouts_charge_discarded_work(self):
        vanilla, beam = (load_policy(TASK / "environment/baseline" / n)[100]
                         for n in ("stochastic-tsr", "geometry-beam"))
        self.assertEqual(cost(vanilla), 180)
        self.assertEqual(cost(beam), 180)
        two_wide = load_policy(TASK / "environment/baseline/two-wide-beam")[250]
        self.assertEqual(cost(two_wide), 180)
        rollout = copy.deepcopy(beam)
        rollout["checkpoints"][0]["score"] = "rollout"
        with self.assertRaises(InvalidPolicy):
            cost(rollout)
        rollout["bands"] = [[60, 1]]
        self.assertEqual(cost(rollout), 150)
        rollout["width"] = 5
        cfg = native_config(rollout, 150, 50, 1, Path("/assets"), Path("/run"))
        self.assertEqual(cfg["num_replicates"], 10)
        self.assertEqual(cfg["search"]["beam_width"], 5)
        self.assertFalse(cfg["search"]["save_rollouts"])

    def test_ode_uses_zero_diffusion_for_both_modalities(self):
        plan = dict(load_policy(TASK / "environment/baseline/stochastic-tsr")[100], solver="ode-tsr")
        cfg = native_config(plan, 100, 10, 123, Path("/assets"), Path("/run"))
        for key in ("bb_ca", "local_latents"):
            self.assertEqual(cfg["generation"]["model"][key], {"gt": {"mode": "ode"}})
        self.assertEqual(cost(plan), 180)

    def test_malformed_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "policy.json"
            for data in ('{"version":1,"version":1}', '{"version":NaN}', '[]', 'false', ''):
                f.write_text(data)
                with self.assertRaises(InvalidPolicy):
                    load_policy(tmp)
        beam = load_policy(TASK / "environment/baseline/geometry-beam")[100]
        beam["checkpoints"][0]["priority"] = ["system", "abc", 0]
        with self.assertRaises(InvalidPolicy):
            cost(beam)
        with self.assertRaises(InvalidPolicy):
            cost({"bands": [[19, 1]], "width": 1, "checkpoints": []})

    def test_harbor_reward_values_are_numeric(self):
        report = {"reward": 0.2, "af3_seconds": 1, "clusters_per_allocated_gpu_hour": 4.5,
                  "suite_sha256": "private-suite-digest", "protocol": "rfd4-monomer-v1",
                  "by_length": {"100": {"passes": 12, "timed_out": False}}}
        values = harbor_metrics(report)
        self.assertEqual(values["length_100_passes"], 12)
        self.assertEqual(values["reward"], 0.2)
        self.assertTrue(all(type(v) in (int, float) for v in values.values()))
        self.assertEqual(harbor_metrics({"reward": 0, "invalid_submission": "reason"}),
                         {"reward": 0, "invalid_submission": 1})

    def test_backbone_alignment(self):
        x = np.eye(4, 3)
        self.assertLess(scrmsd(x, x[:, [1, 2, 0]] + 17), 1e-9)
        self.assertGreater(scrmsd(x, -x), 0.1)

    def test_inclusive_cutoff_duplicates_and_failed_slots(self):
        # Three valid samples, one missing slot, two pass in one cluster and one fails.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for k in ("a", "b", "c"):
                (root / k).touch()
            with patch("score.backbone", return_value=("A", np.zeros((4, 3)))), \
                 patch("score.scrmsd", side_effect=[1.5, 0.3, 1.50001]), \
                 patch("score.subprocess.run"), \
                 patch("score.cluster", side_effect=lambda ids, *args: {k: "one" for k in ids}):
                result = score({100: [root / k for k in ("a", "b", "c")], 150: []},
                               4, root, root / "score", 123)
            self.assertEqual(result["by_length"][100]["passes"], 2)
            self.assertEqual(result["by_length"][100]["passing_clusters"], 1)
            self.assertEqual(result["reward"], 0.125)


if __name__ == "__main__":
    unittest.main()
