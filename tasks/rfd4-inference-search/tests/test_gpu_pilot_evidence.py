"""Audit compact records of actual GPU runs without launching a model."""
import hashlib
import json
import unittest
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]


class GPUReceipts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((TASK / "docs/gpu-pilot/results.json").read_text())

    def test_native_selection_and_repeated_scores(self):
        b = self.data["native_beam"]
        cfg = TASK / "environment/references/upstream/inference_beam_example.yaml"
        self.assertEqual(hashlib.sha256(cfg.read_bytes()).hexdigest(), b["config_sha256"])
        self.assertEqual(b["completed_worker_jobs"], [48, 48])
        first = b["first_checkpoint_scores"]
        second = b["repeat_checkpoint_scores"]
        actual = b["repeat_observed_selections"]
        self.assertEqual([row["step"] for row in first], [20, 40, 50, 60] * 2)
        errors = []
        for a, c, indices in zip(first, second, actual, strict=True):
            self.assertEqual((a["step"], c["step"]), (indices["step"], indices["step"]))
            for run in (a, c):
                self.assertEqual(len(run["quality"]), 6)
                for reward, raw in zip(run["reward"], run["quality"], strict=True):
                    self.assertAlmostEqual(reward, max(50, raw), places=4)
                order = sorted(range(6), key=run["reward"].__getitem__, reverse=True)
                self.assertEqual(order[:3], indices["original_candidate_indices"][0])
            errors.extend(abs(x-y) for x,y in zip(a["quality"], c["quality"], strict=True))
        self.assertAlmostEqual(b["max_raw_plddt_difference"], max(errors))
        for rounds, terminals in ((first, b["first_terminal_rewards"]), (second, b["repeat_terminal_rewards"])):
            finals = [row for row in rounds if row["step"] == 60]
            for row, rewards in zip(finals, terminals, strict=True):
                self.assertEqual(sorted(row["reward"], reverse=True)[:3], rewards)

    def test_independent_controls_and_release_regression(self):
        crystal = self.data["crystal_regression"]
        self.assertTrue(crystal["atom_order_matches"])
        self.assertLess(crystal["rmsd_A"], crystal["official_rmsd_bound_A"])
        self.assertEqual((crystal["unreasonable_bond_fraction"], crystal["clashing_fraction"]), (0, 0))
        runs = self.data["fixed_complex"]["attempted_runtimes"]
        self.assertEqual([r["adjudicated"] for r in runs], [False, False, True])
        self.assertEqual([float(r["all_metrics_present"]) for r in runs], [0, 0, 1])
        folded = self.data["independent_native_final_check"]
        self.assertNotEqual(folded["af3_seed"], self.data["native_beam"]["oracle_seed"])
        self.assertEqual(set(folded["groups"]), {"low", "medium", "high", "beam"})
        for row in folded["groups"].values():
            hits = [c for r,c in zip(row["scRMSD_A"], row["foldseek_cluster"], strict=True) if r <= 1.5]
            self.assertEqual((len(hits), len(set(hits))), (row["passes"], row["unique_passing_clusters"]))
            self.assertEqual((row["attempts"], len(row["scRMSD_A"])), (2, 2))
        repeat = self.data["published_monomer"]["comparisons"]["local_repeat"]
        self.assertTrue(repeat["same_serialized_coordinates"])
        self.assertEqual(repeat["matching_residues"], repeat["residues"])


if __name__ == "__main__":
    unittest.main()
