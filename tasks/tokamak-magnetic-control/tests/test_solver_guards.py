"""FreeGSNKE guards that keep a degenerate plasma from hanging an episode (needs FreeGSNKE)."""
import sys
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment"))

try:
    import freegsnke  # noqa: F401
    HAVE_FREEGSNKE = True
except ImportError:
    HAVE_FREEGSNKE = False


@unittest.skipUnless(HAVE_FREEGSNKE, "FreeGSNKE not installed")
class SolverGuards(unittest.TestCase):
    def setUp(self):
        from tokagym import env
        env.patch_freegsnke()
        env.patch_freegsnke()   # idempotent
        self.env = env

    def test_xpoint_search_refuses_non_finite_psi(self):
        from freegs4e import critical
        psi = np.zeros((5, 5)); psi[2, 2] = np.nan
        with self.assertRaises(FloatingPointError):
            critical.find_critical(np.zeros((5, 5)), np.zeros((5, 5)), psi)

    def test_failing_trial_updates_end_after_a_bounded_number_of_tries(self):
        from freegsnke import GSstaticsolver
        fake = types.SimpleNamespace()   # the original F_function fails on it every time
        psi = np.ones(7)
        for _ in range(self.env.MAX_TRIAL_FAILURES - 1):
            with self.assertRaises(BaseException):
                GSstaticsolver.NKGSsolver.F_function(fake, psi)
        out = GSstaticsolver.NKGSsolver.F_function(fake, psi)
        self.assertEqual(out.shape, psi.shape)
        self.assertTrue(np.isnan(out).all())

    def test_work_budget_is_counted_in_residual_evaluations(self):
        from freegsnke import GSstaticsolver, nonlinear_solve
        fake = types.SimpleNamespace()
        with self.env.work_budget(5) as work:   # no evaluation: no error
            pass
        self.assertEqual(work["calls"], 0)
        with self.assertRaises(self.env.WorkBudgetExceeded):
            with self.env.work_budget(0):
                for f in (GSstaticsolver.NKGSsolver.F_function, nonlinear_solve.nl_solver.F_function_curr,
                          nonlinear_solve.nl_solver.F_function_psi):
                    out = f(fake, np.ones(4), None)   # over budget: NaN, the original is never called
                    self.assertEqual(out.shape, (4,))
                    self.assertTrue(np.isnan(out).all())
        with self.assertRaises(self.env.WorkBudgetExceeded):
            with self.env.work_budget(0):   # FreeGSNKE calls F_function_psi with keywords
                out = nonlinear_solve.nl_solver.F_function_psi(
                    fake, trial_plasma_psi=np.ones(3), active_voltage_vec=None, rtol_NK=1e-8)
                self.assertTrue(np.isnan(out).all())
        # outside a budget nothing is limited
        with self.assertRaises(BaseException):
            nonlinear_solve.nl_solver.F_function_curr(fake, np.ones(4), None)


@unittest.skipUnless(HAVE_FREEGSNKE, "FreeGSNKE not installed")
class ControllabilityRedraw(unittest.TestCase):
    def test_uncontrollable_draws_are_redrawn_deterministically(self):
        from tokagym import env as E
        calls = []

        def fake_reset_once(self, seed=None, options=None):
            calls.append(seed)
            tau = 0.5e-3 if len(calls) == 1 else 0.03   # first draw uncontrollable, second fine
            return np.zeros(3), {"growth_rate": tau}
        fake = types.SimpleNamespace()
        fake._reset_once = types.MethodType(fake_reset_once, fake)
        obs, info = E.TokamakMagneticEnv.reset(fake, seed=1004)
        self.assertEqual(info["controllability_redraws"], 1)
        self.assertEqual(calls[0], 1004)
        first_redraw = calls[1]
        calls.clear()
        E.TokamakMagneticEnv.reset(fake, seed=1004)
        self.assertEqual(calls[1], first_redraw)      # same seed, same redraw
        self.assertNotEqual(first_redraw, 1004)

    def test_controllable_draw_is_kept(self):
        from tokagym import env as E
        fake = types.SimpleNamespace()
        fake._reset_once = types.MethodType(lambda self, seed=None, options=None: (None, {"growth_rate": 0.02}), fake)
        _, info = E.TokamakMagneticEnv.reset(fake, seed=1001)
        self.assertEqual(info["controllability_redraws"], 0)


if __name__ == "__main__":
    unittest.main()
