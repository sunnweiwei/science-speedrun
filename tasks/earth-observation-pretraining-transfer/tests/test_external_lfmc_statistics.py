"""Synthetic checks only: never load external labels, imagery or predictions."""

import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/analysis"))
from external_lfmc_statistics import aggregate_target, paired_statistics


class StatisticsTests(unittest.TestCase):
    def fixture(self, random=(2, 2, 2), trained=(1, 1, 1)):
        rows = [
            {"id": str(i), "site_id": site, "country": "country"}
            for i, site in enumerate(["a", "b", "b", "b"])
        ]
        ids = [r["id"] for r in rows]
        labels = {"ids": ids, "values": [10.0] * 4}
        pairs = [
            {
                "seed": seed,
                "random": {"ids": ids, "values": [10 + r] * 4},
                "trained": {"ids": ids, "values": [10 + t] * 4},
            }
            for seed, (r, t) in enumerate(zip(random, trained))
        ]
        return rows, labels, pairs

    def test_constant_paired_improvement(self):
        result = paired_statistics(*self.fixture())
        self.assertEqual(result["mean_seed_mae_gain"], 1)
        self.assertEqual(result["site_bootstrap_95_percent_interval"], [1, 1])
        self.assertEqual(result["seed_and_site_bootstrap_95_percent_interval"], [1, 1])
        self.assertTrue(result["statistical_criterion_met"])

    def test_one_regressing_seed_cannot_be_hidden_by_mean(self):
        result = paired_statistics(*self.fixture(random=(5, 5, 0)))
        self.assertGreater(result["mean_seed_mae_gain"], 0)
        self.assertFalse(result["statistical_criterion_met"])

    def test_zero_gain_does_not_pass(self):
        self.assertFalse(
            paired_statistics(*self.fixture(trained=(2, 2, 2)))[
                "statistical_criterion_met"
            ]
        )

    def test_cluster_bootstrap_against_explicit_resampled_rows(self):
        rows, labels, pairs = self.fixture()
        for i, pair in enumerate(pairs):
            pair["random"]["values"] = [10 + i, 11, 12, 13]
            pair["trained"]["values"] = [11, 10 + i, 11, 12]
        result = paired_statistics(rows, labels, pairs)
        rng = np.random.default_rng(42)
        site_draws = rng.integers(0, 2, size=(5000, 2))
        seed_draws = rng.integers(0, 3, size=(5000, 3))
        group_rows = [np.array([0]), np.array([1, 2, 3])]
        gains = np.array(
            [
                np.abs(np.array(p["random"]["values"]) - 10)
                - np.abs(np.array(p["trained"]["values"]) - 10)
                for p in pairs
            ]
        )
        site_boot, joint_boot = [], []
        for sites, seeds in zip(site_draws, seed_draws):
            indices = np.concatenate([group_rows[s] for s in sites])
            site_boot.append(gains[:, indices].mean())
            joint_boot.append(gains[np.ix_(seeds, indices)].mean())
        np.testing.assert_allclose(
            result["site_bootstrap_95_percent_interval"],
            np.quantile(site_boot, [0.025, 0.975]),
        )
        np.testing.assert_allclose(
            result["seed_and_site_bootstrap_95_percent_interval"],
            np.quantile(joint_boot, [0.025, 0.975]),
        )
        self.assertEqual(result["per_seed"][0]["random"]["mae"], 1.5)
        self.assertEqual(result["per_seed"][0]["random"]["site_macro_mae"], 1.0)

    def test_alignment_nonfinite_and_seed_failures(self):
        for variant in ("ids", "nan", "seed", "missing"):
            with self.subTest(variant=variant):
                rows, labels, pairs = copy.deepcopy(self.fixture())
                if variant == "ids":
                    pairs[0]["trained"]["ids"] = ["1", "0", "2", "3"]
                elif variant == "nan":
                    pairs[0]["trained"]["values"][0] = float("nan")
                elif variant == "seed":
                    pairs[1]["seed"] = pairs[0]["seed"]
                else:
                    labels["values"] = labels["values"][:-1]
                with self.assertRaises(ValueError):
                    paired_statistics(rows, labels, pairs)

    def test_target_rules_preserve_duplicates_without_upper_clip(self):
        result = aggregate_target(
            list(
                enumerate([-10, 20, 20, 10000, None, True, "20", float("inf")], start=2)
            )
        )
        self.assertEqual(result["value"], 2510)
        self.assertEqual(result["valid_source_rows"], 4)
        self.assertEqual(result["counts"]["lower_clipped"], 1)
        self.assertEqual(len(result["excluded"]), 4)

    def test_no_valid_target_is_unscoreable(self):
        result = aggregate_target([(2, None), (3, float("nan"))])
        self.assertFalse(result["scoreable"])
        self.assertIsNone(result["value"])
        with self.assertRaises(ValueError):
            aggregate_target([(2, 10), (2, 10)])


if __name__ == "__main__":
    unittest.main()
