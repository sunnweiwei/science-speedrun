"""CPU-only tests of the metric definitions (no FreeGSNKE needed)."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "environment", Path("/opt/tokamak-task")):
    sys.path.insert(0, str(_p))
from tokagym import metrics as M  # noqa: E402


class ShapeMetrics(unittest.TestCase):
    def test_points_on_polyline_have_zero_error(self):
        theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        circle = np.stack([1 + 0.5 * np.cos(theta), 0.8 * np.sin(theta)], 1)
        targets = circle[::10]
        self.assertLess(M.shape_rms(circle, targets), 1e-12)

    def test_shifted_targets_measure_the_shift(self):
        theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        circle = np.stack([1 + 0.5 * np.cos(theta), 0.5 * np.sin(theta)], 1)
        targets = circle[::45] * np.array([1.0, 1.0]) + np.array([0.0, 0.0])
        bigger = np.stack([1 + 0.52 * np.cos(theta), 0.52 * np.sin(theta)], 1)
        self.assertAlmostEqual(M.shape_rms(bigger, targets), 0.02, places=3)

    def test_degenerate_polyline_is_infinite(self):
        self.assertTrue(math.isinf(M.shape_rms(np.zeros((1, 2)), np.zeros((3, 2)))))

    def test_xpoint_error_cap_and_missing(self):
        self.assertEqual(M.xpoint_error(np.zeros((0, 2)), np.array([[0.6, -1.1]])), M.XPOINT_CAP_M)
        self.assertEqual(M.xpoint_error(np.array([[5.0, 5.0]]), np.array([[0.6, -1.1]])), M.XPOINT_CAP_M)
        self.assertAlmostEqual(M.xpoint_error(np.array([[0.6, -1.0], [3, 3]]), np.array([[0.6, -1.1]])), 0.1)
        self.assertEqual(M.xpoint_error(np.array([[1, 1]]), np.zeros((0, 2))), 0.0)

    def test_summary_windows_and_feasibility(self):
        rec = [{"shape_rms": 0.02 if k < 5 else 0.01, "xpoint_error": 0.0, "ip_error_rel": 0.0,
                "coil_violation": 1.0 if k == 0 else 0.0} for k in range(10)]
        s = M.summarize_episode(rec, horizon=10)
        self.assertTrue(s["complete"])
        self.assertAlmostEqual(s["shape_rms_m_last_half"], 0.01)
        self.assertAlmostEqual(s["shape_rms_m_full"], 0.015)
        self.assertAlmostEqual(s["coil_violation_fraction"], 0.1)
        short = M.summarize_episode(rec[:4], horizon=10)
        self.assertFalse(short["complete"])
        inf = M.summarize_episode(rec + [{"shape_rms": math.inf, "xpoint_error": math.inf,
                                          "ip_error_rel": math.inf, "coil_violation": 1.0}], horizon=11)
        self.assertIsNotNone(inf["shape_rms_m_last_half"])  # infinite steps are excluded from means


def step(shape=0.005, gap=0.005, ip=0.01):
    return {"shape_rms": shape, "xpoint_error": gap, "ip_error_rel": ip, "rin_error": gap,
            "rout_error": gap, "zx_error": gap, "coil_violation": 0.0}


class SuccessCriterion(unittest.TestCase):
    def test_steady_window_ignores_the_transient(self):
        rec = [step(shape=0.05)] * 6 + [step()] * 4
        s = M.summarize_episode(rec, horizon=10, steady_start=6)
        self.assertAlmostEqual(s["shape_rms_steady"], 0.005)
        self.assertTrue(M.meets_success(s))
        self.assertFalse(M.meets_success(M.summarize_episode(rec, horizon=10, steady_start=4)))

    def test_each_tolerance_is_binding(self):
        for bad in (step(shape=0.011), step(gap=0.011), step(ip=0.031)):
            s = M.summarize_episode([step()] * 5 + [bad] * 5, horizon=10, steady_start=5)
            self.assertFalse(M.meets_success(s))
        self.assertTrue(M.meets_success(M.summarize_episode([step(0.01, 0.01, 0.03)] * 4, horizon=4, steady_start=0)))

    def test_incomplete_or_nonfinite_episode_never_succeeds(self):
        self.assertFalse(M.meets_success(M.summarize_episode([step()] * 4, horizon=10, steady_start=0)))
        rec = [step()] * 9 + [step(gap=math.inf)]
        s = M.summarize_episode(rec, horizon=10, steady_start=0)
        self.assertIsNone(s["rin_error_steady"])
        self.assertFalse(M.meets_success(s))

    def test_void_only_for_a_stall_within_tolerance(self):
        self.assertTrue(M.void_candidate([step()] * 12))
        self.assertFalse(M.void_candidate([step(shape=0.03)] * 12))
        self.assertFalse(M.void_candidate([step()] * 3))   # too short to judge
        self.assertFalse(M.void_candidate([]))


class Aggregation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "environment" / "evaluation"))
        sys.path.insert(0, "/opt/tokamak-task/evaluation")
        import harbor_verify
        cls.hv = harbor_verify

    def ep(self, outcome, **kw):
        e = {"outcome": outcome, "feasible": outcome in ("success", "kept"), "shape_rms_steady": 0.005,
             "xpoint_error_steady": 0.01, "rin_error_steady": 0.004, "rout_error_steady": 0.004,
             "zx_error_steady": 0.004, "ip_error_rel_steady": 0.01}
        e.update(kw)
        return e

    def case(self, name, eps, status="ok"):
        return {"case": name, "episodes": eps, "status": status}

    def test_reward_is_success_fraction_without_void(self):
        res = [self.case("a", [self.ep("success"), self.ep("kept"), self.ep("plasma_lost")]),
               self.case("b", [self.ep("success"), self.ep("solver_void"), self.ep("coil_limit")])]
        reward, grading = self.hv.aggregate(res)
        self.assertAlmostEqual(reward["reward"], 2 / 5)
        self.assertAlmostEqual(reward["b/success_fraction"], 1 / 2)
        self.assertEqual(reward["b/void_episodes"], 1)
        self.assertAlmostEqual(reward["a/shape_rms_cm"], 0.5)
        self.assertEqual(grading["b"]["n_void"], 1)

    def test_timeouts_are_classified_from_last_states(self):
        ok_states = [{"shape_rms": 0.004, "ip_error_rel": 0.01}] * 10
        bad_states = [{"shape_rms": 0.05, "ip_error_rel": 0.01}] * 10
        t1 = {"feasible": False, "termination": "solver_timeout", "last_states": ok_states}
        t2 = {"feasible": False, "termination": "solver_timeout", "last_states": bad_states}
        reward, _ = self.hv.aggregate([self.case("a", [self.ep("success")] * 8 + [t1, t2])])
        self.assertAlmostEqual(reward["reward"], 8 / 9)

    def test_too_many_void_episodes_withhold_the_score(self):
        reward, _ = self.hv.aggregate([self.case("a", [self.ep("success")] * 3 + [self.ep("solver_void")] * 2)])
        self.assertIsNone(reward)

    def test_invalid_package_scores_zero(self):
        reward, _ = self.hv.aggregate([self.case("a", [self.ep("success")] * 3),
                                       self.case("b", [], status="policy_error")])
        self.assertEqual(reward["reward"], 0.0)


if __name__ == "__main__":
    unittest.main()
