"""Native wrapper precision and boundary tests using a mock, not CFD evidence."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import hydrogym.maia as maia

import native


class FakeFlow(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (98,), dtype=np.float32)
    action_space = gym.spaces.Box(-1., 1., (3,), dtype=np.float32)
    omega, num_substeps_per_iteration, max_episode_steps = 0., 200, 200
    MAX_CONTROL, maxRfnmntLvl, dX = .25, 11, .05
    observation_type = ['u', 'v']
    reset_value, step_value = 1., 1.

    def get_reward(self):
        return -1., {'forces': np.array([[1., 0.]])}

    def reset(self, **kwargs):
        self.count = 0
        return np.full(98, self.reset_value, dtype=np.float64), {}

    def step(self, action):
        self.count += 1
        score, _ = self.get_reward()
        done = self.count > self.max_episode_steps
        return np.full(98, self.step_value, dtype=np.float64), score, done, done, {}


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='native-observation-test-')
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.flow = FakeFlow()
        self.flow.env_data_path = str(root / 'source')
        args = SimpleNamespace(data=root, environment='Pinball_2D_Re100', protocol='author-paper',
            normalization='u-inf', max_control=.25, diagnostic_horizon=None, bootstrap_time_limits=True,
            seed=42, mode='zero', output=root)
        with patch.object(native, 'local_resources', return_value=root / 'source'), \
                patch.dict(maia.__dict__, {'from_hf': lambda *a, **kw: self.flow}):
            self.env, self.protocol = native.environment(args)
        self.addCleanup(self.env.close)
        self.root = root

    def test_actual_wrapper_external_limit(self):
        obs, _ = self.env.reset()
        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(self.flow.max_episode_steps, 200)
        for count in range(1, 201):
            obs, score, terminated, truncated, _ = self.env.step(np.zeros(3, dtype=np.float32))
            self.assertFalse(terminated)
            self.assertEqual(truncated, count == 200)
            self.assertEqual(score, -1.)

    def test_reject_finite_double_overflow_at_reset(self):
        self.flow.reset_value = 1e100
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite CFD reset'):
            self.env.reset()

    def test_reject_finite_double_overflow_at_transition(self):
        self.env.reset()
        self.flow.step_value = 1e100
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite CFD trajectory'):
            self.env.step(np.zeros(3, dtype=np.float32))
        failure = json.loads((self.root / 'transition-error.json').read_text())
        self.assertEqual(failure['control_in_episode'], 1)
        self.assertEqual(failure['observation_nonfinite_count'], 98)


if __name__ == '__main__':
    unittest.main(verbosity=2)
