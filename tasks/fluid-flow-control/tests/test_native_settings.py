"""CPU regression tests for explicit, identical training/replay physics settings."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment/evaluation"))
from native_settings import apply_settings, validate_settings
from protocol import InfrastructureError


class NativeSettingsTests(unittest.TestCase):
    def env(self):
        return SimpleNamespace(max_episode_steps=200, num_substeps_per_iteration=200,
                               MAX_CONTROL=0.25, omega=0.0)

    def test_paper_protocol_and_original_defaults(self):
        env = self.env()
        settings = {"max_episode_steps": 199, "substeps_per_action": 170,
                    "max_control": 0.25, "omega": 1.0}
        audit = apply_settings(env, {"horizon": 200, "native_settings": settings})
        self.assertEqual(audit["effective"], settings)
        self.assertEqual(audit["native_defaults"]["omega"], 0)
        self.assertEqual(env.num_substeps_per_iteration, 170)
        self.assertEqual(env.max_episode_steps + 1, 200)

    def test_defaults_are_not_silently_changed(self):
        env = self.env()
        audit = apply_settings(env, {"horizon": 201})
        self.assertEqual(audit["native_defaults"], audit["effective"])
        with self.assertRaisesRegex(InfrastructureError, "horizon"):
            apply_settings(self.env(), {"horizon": 200})

    def test_explicit_time_limit_preserves_native_limit(self):
        env = self.env()
        case = {"horizon": 200, "episode_end": "time_limit",
                "native_settings": {"max_episode_steps": 200}}
        audit = apply_settings(env, case)
        self.assertEqual(env.max_episode_steps, 200)
        self.assertEqual(audit["episode_end"], "time_limit")
        self.assertEqual(audit["effective"]["max_episode_steps"], 200)

    def test_reject_mixed_or_unknown_boundary_conventions(self):
        for settings, convention in [({"max_episode_steps": 199}, "time_limit"),
                                     ({"max_episode_steps": 200}, "native"),
                                     ({}, "automatic"), ({}, True), ({}, None)]:
            with self.subTest(settings=settings, convention=convention), self.assertRaises(InfrastructureError):
                validate_settings(settings, 200, convention)

    def test_invalid_or_unreviewed_overrides(self):
        invalid = [{"Re": 4200}, {"reward_function": "replacement"},
                   {"substeps_per_action": 0}, {"substeps_per_action": 1.5},
                   {"omega": float("nan")}, {"omega": True}, {"omega": -1},
                   {"max_control": 0}, {"max_episode_steps": 200}, [], None]
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(InfrastructureError):
                validate_settings(settings, 200)


if __name__ == "__main__":
    unittest.main()
