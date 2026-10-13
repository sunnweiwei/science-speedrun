"""Scientific invariants for the surface-code scoring and split contract."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment" / "baseline" / "alphaqubit" / "baselines"))
from classical import discover_conditions, expected_conditions, split_indices
from metrics import FIT_ROUNDS, compare_paired_conditions, summarize_conditions, wilson_interval


def synthetic_counts(
    *, distance: int = 3, basis: str = "X", center: tuple[int, int] = (3, 5),
    epsilon: float = 0.03, intercept: float = -0.08, total: int = 1_000_000,
) -> dict[str, dict[str, int]]:
    return {
        f"surface_code_b{basis}_d{distance}_r{rounds:02d}_center_{center[0]}_{center[1]}": {
            "errors": round(total * (1 - np.exp(intercept) * (1 - 2 * epsilon)**rounds) / 2),
            "total": total,
        }
        for rounds in (1, *FIT_ROUNDS)
    }


class MetricsTests(unittest.TestCase):
    def test_free_intercept_recovers_round_error_and_round_one_is_excluded(self) -> None:
        counts = synthetic_counts()
        counts["surface_code_bX_d3_r01_center_3_5"]["errors"] = 900_000
        summary = summarize_conditions(counts, bootstrap_samples=256, seed=17)
        fit = summary["per_group"]["d3_bX_center_3_5"]
        self.assertAlmostEqual(fit["logical_error_per_round"], 0.03, places=6)
        self.assertAlmostEqual(fit["intercept"], -0.08, places=5)
        self.assertAlmostEqual(fit["zero_round_fidelity"], float(np.exp(-0.08)), places=5)
        self.assertGreater(fit["weighted_log_fidelity_r_squared"], 0.999999)
        self.assertGreater(fit["unweighted_log_fidelity_r_squared"], 0.999999)
        self.assertEqual(fit["diagnostic_flags"], [])
        lower, upper = fit["logical_error_per_round_ci95"]
        self.assertLess(lower, 0.03)
        self.assertGreater(upper, 0.03)
        self.assertEqual(fit["degrees_of_freedom"], 10)

    def test_macro_is_mean_of_separate_fits_and_suppression_is_ratio(self) -> None:
        counts = synthetic_counts(epsilon=0.02, intercept=-0.1)
        counts.update(synthetic_counts(basis="Z", epsilon=0.06, intercept=-0.7))
        counts.update(synthetic_counts(distance=5, center=(5, 5), epsilon=0.02))
        summary = summarize_conditions(counts, bootstrap_samples=0)
        self.assertAlmostEqual(summary["macro"]["d3"]["logical_error_per_round"], 0.04, places=6)
        self.assertAlmostEqual(summary["macro"]["error_suppression_d3_over_d5"]["value"], 2.0, places=4)

    def test_missing_round_and_invalid_fidelity_raise(self) -> None:
        counts = synthetic_counts()
        counts.pop("surface_code_bX_d3_r05_center_3_5")
        with self.assertRaisesRegex(ValueError, "expected each fit round"):
            summarize_conditions(counts)
        counts = synthetic_counts()
        counts["surface_code_bX_d3_r05_center_3_5"]["errors"] = 500_000
        with self.assertRaisesRegex(ValueError, "P_fail < 0.5"):
            summarize_conditions(counts)

    def test_low_intercept_and_unphysical_rate_are_reported_without_clipping(self) -> None:
        counts = synthetic_counts(epsilon=-0.002, intercept=-3.0)
        summary = summarize_conditions(counts, bootstrap_samples=0)
        fit = summary["per_group"]["d3_bX_center_3_5"]
        self.assertLess(fit["logical_error_per_round"], 0)
        self.assertAlmostEqual(fit["zero_round_fidelity"], float(np.exp(-3)), places=4)
        self.assertIn("unphysical_negative_logical_error_per_round", fit["diagnostic_flags"])
        self.assertIn("extrapolated_zero_round_fidelity_below_0.1", fit["diagnostic_flags"])
        self.assertEqual(len(summary["per_condition"]), 13)

    def test_flat_fidelity_has_undefined_r_squared_and_no_false_perfect_fit(self) -> None:
        counts = synthetic_counts(epsilon=0, intercept=-1)
        summary = summarize_conditions(counts, bootstrap_samples=0)
        fit = summary["per_group"]["d3_bX_center_3_5"]
        self.assertIsNone(fit["weighted_log_fidelity_r_squared"])
        self.assertIsNone(fit["unweighted_log_fidelity_r_squared"])
        self.assertIn("constant_log_fidelity_r_squared_undefined", fit["diagnostic_flags"])

    def test_paired_identical_decoders_have_exact_zero_difference_interval(self) -> None:
        counts = synthetic_counts(total=10_000)
        errors = {
            name: np.arange(record["total"]) < record["errors"] for name, record in counts.items()
        }
        comparison = compare_paired_conditions(errors, errors, bootstrap_samples=128, seed=42)
        for result in comparison["comparisons"].values():
            self.assertEqual(result["logical_error_per_round_a_minus_b"], 0)
            self.assertEqual(result["ci95"], [0, 0])
        with self.assertRaisesRegex(ValueError, "identical conditions"):
            compare_paired_conditions(errors, {})

    def test_wilson_zero_errors_retains_nonzero_upper_uncertainty(self) -> None:
        lower, upper = wilson_interval(0, 100)
        self.assertAlmostEqual(lower, 0)
        self.assertGreater(upper, 0.03)
        self.assertLess(upper, 0.04)

    def test_train_validation_and_test_are_disjoint_and_cover_all_shots(self) -> None:
        train, validation, test = [split_indices(split) for split in ("train", "validation", "test")]
        self.assertEqual(tuple(map(len, (train, validation, test))), (19880, 5120, 25000))
        all_indices = np.concatenate((train, validation, test))
        np.testing.assert_array_equal(np.sort(all_indices), np.arange(50000))
        self.assertTrue(np.all(validation % 2 == 0))
        self.assertTrue(np.all(test % 2 == 1))

    def test_full_suite_missing_condition_is_fatal(self) -> None:
        self.assertEqual(len(expected_conditions()), 130)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Expected all 130"):
                discover_conditions(Path(temporary))


if __name__ == "__main__":
    unittest.main()
