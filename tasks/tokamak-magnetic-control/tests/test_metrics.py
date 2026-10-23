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

    def test_within_tolerance_at_failure_needs_a_full_lookback(self):
        self.assertTrue(M.void_candidate([step()] * 12))
        self.assertFalse(M.void_candidate([step(shape=0.03)] * 12))
        self.assertFalse(M.void_candidate([step()] * 3))   # too short to judge
        self.assertFalse(M.void_candidate([]))

    def test_tolerance_time_fraction_counts_unreached_steps_as_outside(self):
        rec = [step(shape=0.05)] * 4 + [step()] * 3 + [step(ip=0.05)] + [step()] * 2
        self.assertAlmostEqual(M.summarize_episode(rec, horizon=10, steady_start=4)["tolerance_time_fraction"], 5 / 6)
        self.assertAlmostEqual(M.summarize_episode(rec[:6], horizon=10, steady_start=4)["tolerance_time_fraction"], 2 / 6)
        self.assertEqual(M.summarize_episode([step(gap=math.inf)] * 10, horizon=10, steady_start=0)
                         ["tolerance_time_fraction"], 0.0)


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

    def test_reward_counts_every_episode(self):
        res = [self.case("a", [self.ep("success"), self.ep("kept"), self.ep("plasma_lost")]),
               self.case("b", [self.ep("success"), self.ep("solver_failure"), self.ep("coil_limit")])]
        reward, grading = self.hv.aggregate(res)
        self.assertAlmostEqual(reward["reward"], 2 / 6)
        self.assertAlmostEqual(reward["b/success_fraction"], 1 / 3)
        self.assertEqual(reward["b/solver_failure_episodes"], 1)
        self.assertEqual(reward["solver_failure_episodes"], 1)
        self.assertEqual(reward["episodes"], 6)
        self.assertAlmostEqual(reward["a/shape_rms_cm"], 0.5)
        self.assertEqual(grading["b"]["outcomes"]["solver_failure"], 1)

    def test_tolerance_time_fraction_is_averaged_over_all_episodes(self):
        res = [self.case("a", [self.ep("success", tolerance_time_fraction=1.0),
                               self.ep("plasma_lost", tolerance_time_fraction=0.2),
                               self.ep("solver_failure", tolerance_time_fraction=None)])]
        reward, _ = self.hv.aggregate(res)
        self.assertAlmostEqual(reward["tolerance_time_fraction"], 0.4)
        self.assertAlmostEqual(reward["a/tolerance_time_fraction"], 0.4)

    def test_latency_gate_uses_the_isolated_measurement(self):
        res = [self.case("a", [self.ep("success", policy_compute_ms_under_load=0.9)] * 2)]
        reward, _ = self.hv.aggregate(res, policy_compute_ms=0.3)
        self.assertEqual(reward["reward"], 1.0)
        self.assertAlmostEqual(reward["policy_compute_ms_under_load"], 0.9)
        reward, grading = self.hv.aggregate(res, policy_compute_ms=0.6)
        self.assertEqual(reward["reward"], 0.0)
        self.assertIn("_latency", grading)

    def test_invalid_package_scores_zero(self):
        reward, _ = self.hv.aggregate([self.case("a", [self.ep("success")] * 3),
                                       self.case("b", [], status="policy_error")])
        self.assertEqual(reward["reward"], 0.0)


class Reruns(unittest.TestCase):
    """run_all's bounded rerun policy, with the supervisor replaced by a script (no FreeGSNKE)."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "environment" / "evaluation"))
        sys.path.insert(0, "/opt/tokamak-task/evaluation")
        import harbor_verify
        cls.hv = harbor_verify

    def run_with(self, script):
        """script(attempt, seed) -> one supervisor result for that episode."""
        calls = []

        def fake(target, jobs, seeds, episode_timeout, max_parallel, silence_timeout=None):
            out = {}
            for i, (job, seed) in enumerate(zip(jobs, seeds)):
                calls.append((job[5], seed[0]))
                out[i] = script(job[5], seed[0])
            return out
        real = self.hv.supervise.run_supervised
        self.hv.supervise.run_supervised = fake
        try:
            res = self.hv.run_all([("mastu_diverted_hold", "pkg", "out", "rz", 120.0)], [1, 2], 2, 1)
        finally:
            self.hv.supervise.run_supervised = real
        return res[0], calls

    @staticmethod
    def done(seed, outcome="success"):
        return {"episodes": [{"kind": "episode", "seed": seed, "outcome": outcome, "feasible": True}],
                "case": {"status": "ok", "error": None}, "errors": []}

    @staticmethod
    def killed(seed, termination="solver_timeout"):
        return {"episodes": [{"kind": "episode", "seed": seed, "feasible": False, "termination": termination}],
                "case": None, "errors": [f"episode seed={seed} silent for 2500s (> 2400s); process killed"]}

    def test_a_host_side_kill_is_rerun_once_and_then_scored(self):
        res, calls = self.run_with(lambda attempt, seed: self.killed(seed) if (seed == 2 and attempt == 0)
                                   else self.done(seed))
        self.assertEqual(calls, [(0, 1), (0, 2), (1, 2)])
        self.assertEqual([e["outcome"] for e in res["episodes"]], ["success", "success"])
        self.assertEqual(res["reruns"], 1)
        self.assertEqual(res["status"], "ok")

    def test_a_stall_that_repeats_with_doubled_budgets_is_a_solver_failure(self):
        res, calls = self.run_with(lambda attempt, seed: self.killed(seed) if seed == 2 else self.done(seed))
        self.assertEqual(len(calls), 3)   # bounded: one rerun only
        self.assertEqual([e["outcome"] for e in res["episodes"]], ["success", "solver_failure"])
        self.assertEqual(res["status"], "ok")

    def test_a_crash_that_repeats_withholds_the_score(self):
        res, _ = self.run_with(lambda attempt, seed: self.killed(seed, "worker_crash") if seed == 2 else self.done(seed))
        self.assertEqual(res["status"], "infrastructure_error")

    def test_a_policy_error_is_not_rerun(self):
        res, calls = self.run_with(lambda attempt, seed: {"episodes": [], "errors": [],
                                                          "case": {"status": "policy_error", "error": "bad"}}
                                   if seed == 2 else self.done(seed))
        self.assertEqual(len(calls), 2)
        self.assertEqual(res["status"], "policy_error")


class FallbackLadder(unittest.TestCase):
    """recover_step with a scripted deterministic environment (no FreeGSNKE)."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "environment" / "evaluation"))
        sys.path.insert(0, "/opt/tokamak-task/evaluation")
        import harbor_verify
        cls.hv = harbor_verify

    class Env:
        def __init__(self, fail_at, fail_levels, drift=False):
            self.fail_at, self.fail_levels, self.drift = fail_at, fail_levels, drift
            self.numerics_level, self.resets = 0, 0

        def reset(self, seed=None):
            self.k, self.x, self.failure_kind, self.terminated_reason = 0, float(seed), None, None
            self.resets += 1
            return np.array([self.x]), {}

        def step(self, action):
            if self.k == self.fail_at and self.numerics_level in self.fail_levels:
                self.failure_kind, self.terminated_reason = "exception", "solver_failure: scripted"
                return np.array([self.x]), -20.0, True, False, {"shape_rms": math.inf, "ip_error_rel": math.inf}
            self.failure_kind = None
            self.x += float(action[0]) + (1e-9 * self.resets if self.drift else 0.0)
            self.k += 1
            return np.array([self.x]), 1.0, False, False, {"shape_rms": 0.0, "ip_error_rel": 0.0}

    class Queue(list):
        put = list.append

    def episode(self, env, n=6, budget=100):
        obs, _ = env.reset(seed=3)
        actions, levels, records = [], [], []
        left = {"left": budget}
        for _ in range(n):
            before = obs
            actions.append(np.array([1.0])); levels.append(0)
            result = env.step(actions[-1])
            obs, _, terminated, _, _ = result
            if env.failure_kind is not None:
                (obs, _, terminated, _, _), rec = self.hv.recover_step(env, 3, actions, levels, before, self.Queue(),
                                                                       result, left)
                records.append(rec)
            if terminated:
                break
        return obs, levels, records

    def test_level_one_recovers_with_the_same_trajectory(self):
        obs, levels, records = self.episode(self.Env(fail_at=2, fail_levels={0}))
        self.assertEqual(obs[0], 9.0)
        self.assertEqual(levels, [0, 0, 1, 0, 0, 0])
        self.assertTrue(records[0]["recovered"])

    def test_linear_level_is_the_last_resort(self):
        _, levels, records = self.episode(self.Env(fail_at=2, fail_levels={0, 1}))
        self.assertEqual(levels[2], 2)
        self.assertEqual([lv["level"] for lv in records[0]["levels"]], [1, 2])

    def test_an_unrecoverable_step_ends_the_episode(self):
        obs, levels, records = self.episode(self.Env(fail_at=2, fail_levels={0, 1, 2}))
        self.assertFalse(records[0]["recovered"])
        self.assertEqual(len(levels), 3)

    def test_replays_stop_at_the_episode_replay_budget(self):
        # failure at step 2: level 1 needs 2 replayed steps, level 2 another 2; a budget of 3 allows only level 1
        obs, levels, records = self.episode(self.Env(fail_at=2, fail_levels={0, 1}), budget=3)
        self.assertFalse(records[0]["recovered"])
        self.assertEqual(records[0]["stopped"], "replay budget exhausted")
        self.assertEqual([lv["level"] for lv in records[0]["levels"]], [1])
        obs, levels, records = self.episode(self.Env(fail_at=2, fail_levels={0}), budget=1)
        self.assertEqual(records[0]["levels"], [])   # no retry fits: the failure stands, the episode ends
        self.assertEqual(len(levels), 3)

    def test_a_replay_that_diverges_is_an_infrastructure_error(self):
        with self.assertRaises(self.hv.protocol.InfrastructureError):
            self.episode(self.Env(fail_at=2, fail_levels={0}, drift=True))


if __name__ == "__main__":
    unittest.main()
