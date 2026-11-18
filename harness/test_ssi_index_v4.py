"""Unit tests for SSI V4 top-level scoring (harness/ssi_index_v3.py) & exporter invariants."""
import json
import math
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ssi_index_v3 as m  # noqa: E402


class TestSSIIndexV4(unittest.TestCase):
    def test_forecast_rows_partial_penalty_exact_equivalence(self):
        """Plan §2.1.2: For fraction=0.5, pre-blended 'relative', 'metrics'+'relative' (score.py output),
        unblended 'rel_mase'/'rel_wql', and raw 'metrics' MUST yield identical scores (< 1e-12)."""
        roster = m.INPUTS["timeseries_roster"]
        self.assertEqual(len(roster), 97)
        rows_relative_only = []
        rows_score_py_both = []
        rows_rel_mase = []
        rows_metrics_only = []

        for idx, (k, (_, _, _, base_m, base_q)) in enumerate(sorted(roster.items())):
            raw_m = 0.65 * base_m
            raw_q = 0.72 * base_q
            rel_m = max(0.001, min(10.0, raw_m / base_m))
            rel_q = max(0.001, min(10.0, raw_q / base_q))
            if idx % 3 == 0:
                f = 0.5
                blended_m = math.exp(f * math.log(rel_m) + (1.0 - f) * math.log(10.0))
                blended_q = math.exp(f * math.log(rel_q) + (1.0 - f) * math.log(10.0))
                cov = {"fraction": f, "instances_scored": 50, "instances": 100}
                rows_relative_only.append({
                    "leaderboard_key": k,
                    "status": "partial",
                    "coverage": cov,
                    "relative": {"mase": blended_m, "wql": blended_q},
                })
                rows_score_py_both.append({
                    "leaderboard_key": k,
                    "status": "partial",
                    "coverage": cov,
                    "metrics": {"mase": raw_m, "wql": raw_q},
                    "relative": {"mase": blended_m, "wql": blended_q},
                })
                rows_rel_mase.append({
                    "leaderboard_key": k,
                    "status": "partial",
                    "coverage": cov,
                    "rel_mase": rel_m,
                    "rel_wql": rel_q,
                })
                rows_metrics_only.append({
                    "leaderboard_key": k,
                    "status": "partial",
                    "coverage": cov,
                    "metrics": {"mase": raw_m, "wql": raw_q},
                })
            elif idx % 11 == 0:
                for lst in (rows_relative_only, rows_score_py_both, rows_rel_mase, rows_metrics_only):
                    lst.append({"leaderboard_key": k, "status": "failed"})
            else:
                rows_relative_only.append({
                    "leaderboard_key": k,
                    "status": "ok",
                    "relative": {"mase": rel_m, "wql": rel_q},
                })
                rows_score_py_both.append({
                    "leaderboard_key": k,
                    "status": "ok",
                    "metrics": {"mase": raw_m, "wql": raw_q},
                    "relative": {"mase": rel_m, "wql": rel_q},
                })
                rows_rel_mase.append({
                    "leaderboard_key": k,
                    "status": "ok",
                    "rel_mase": rel_m,
                    "rel_wql": rel_q,
                })
                rows_metrics_only.append({
                    "leaderboard_key": k,
                    "status": "ok",
                    "metrics": {"mase": raw_m, "wql": raw_q},
                })

        s_rel = m.forecast_rows(rows_relative_only)
        s_both = m.forecast_rows(rows_score_py_both)
        s_rmase = m.forecast_rows(rows_rel_mase)
        s_met = m.forecast_rows(rows_metrics_only)

        self.assertLess(abs(s_rel["score"] - s_both["score"]), 1e-12)
        self.assertLess(abs(s_rel["score"] - s_rmase["score"]), 1e-12)
        self.assertLess(abs(s_rel["score"] - s_met["score"]), 1e-12)
        for d in s_rel["components"]:
            self.assertLess(abs(s_rel["components"][d] - s_rmase["components"][d]), 1e-12)
            self.assertLess(abs(s_rel["components"][d] - s_met["components"][d]), 1e-12)
            self.assertLess(abs(s_rel["components"][d] - s_both["components"][d]), 1e-12)

    def test_earth_spatial_205_and_legacy_fallback(self):
        # V4 spatial 205-window majority class (59/205) + trivial LFMC MAE -> exact 0.0
        zero_v4 = m.earth({"awf_test_accuracy": 59 / 205, "lfmc_test_mae": 24.435180349813233})
        self.assertAlmostEqual(zero_v4["score"], 0.0, places=12)
        self.assertAlmostEqual(zero_v4["signed_score"], 0.0, places=12)
        # Perfect -> 1.0
        one_v4 = m.earth({"awf_test_accuracy": 1.0, "lfmc_test_mae": 0.0})
        self.assertAlmostEqual(one_v4["score"], 1.0, places=12)
        # Legacy V3 fallback
        zero_v3 = m.earth({"awf_validation_accuracy": 116 / 344, "lfmc_test_mae": 24.435180349813233})
        self.assertAlmostEqual(zero_v3["score"], 0.0, places=12)

    def test_crystal_weights(self):
        res = m.crystal({"expected": 2500, "sun": 250, "metasun": 500})
        # (250 + 0.5 * 500) / 2500 = 500 / 2500 = 0.2
        self.assertAlmostEqual(res["score"], 0.2, places=12)

    def test_design_target_weights(self):
        res = m.design({"unconditional": 0.5, "sm_binder": 0.2, "ppi_binder": 0.3, "enzyme": 0.4})
        expected = 0.5 / 51 + (50 / 153) * (0.2 + 0.3 + 0.4)
        self.assertAlmostEqual(res["score"], expected, places=12)

    def test_bci_four_tracks_no_faced(self):
        chance = {"tuev": 1 / 6, "ad65": 1 / 3, "openbmi": 0.5, "longitudinal": 0.5}
        self.assertAlmostEqual(m.bci(chance)["score"], 0.0, places=12)
        perfect = {k: 1.0 for k in chance}
        self.assertAlmostEqual(m.bci(perfect)["score"], 1.0, places=12)

    def test_tokamak_105_episodes_void_included(self):
        rew = {}
        cases = [f"case_{i}" for i in range(7)]
        for i, c in enumerate(cases):
            # Suppose 3 voids out of 15, and 6 successes out of 12 non-void -> success_fraction = 0.5
            rew[f"{c}/void_episodes"] = 3
            rew[f"{c}/success_fraction"] = 0.5
        res = m.tokamak(rew, expected_episodes=105)
        # 6 * 7 = 42 successes out of 105 scheduled episodes -> 0.4
        self.assertAlmostEqual(res["score"], 42 / 105, places=12)

    def test_measurement_sources_json_has_all_20_tasks(self):
        ms_path = HERE.parent / "docs" / "SSI_MEASUREMENT_SOURCES.json"
        data = json.loads(ms_path.read_text())
        self.assertEqual(len(data["tasks"]), 20)
        # Ensure stale V1/V2 descriptions are gone
        self.assertNotIn("geometric mean", data["tasks"]["atomistic-materials"]["ssi_implementation"])
        self.assertIn("0.5 * MetaSUN", data["tasks"]["crystal-structure-generation"]["ssi_implementation"])
        self.assertIn("50 / 153", data["tasks"]["protein-design"]["ssi_implementation"])
        self.assertIn("five structural modalities", data["tasks"]["protein-folding-prediction"]["ssi_implementation"])


if __name__ == "__main__":
    unittest.main()
