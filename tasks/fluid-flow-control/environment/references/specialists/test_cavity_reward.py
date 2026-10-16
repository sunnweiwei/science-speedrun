import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cavity_reward import CavityReward, install


class CavityRewardTests(unittest.TestCase):
    def test_formula(self):
        reward = CavityReward([1, 2], [.5, 2])
        self.assertEqual(reward([1, 2]), 0)
        self.assertEqual(reward([1.5, 6]), -5)
        self.assertEqual(reward([.5, -2]), -5)

    def test_calibration_standardization(self):
        rng = np.random.default_rng(12)
        values = rng.normal(size=(10000, 50))
        reward = CavityReward(values.mean(axis=0), values.std(axis=0))
        self.assertAlmostEqual(np.mean([reward(v) for v in values]), -50)

    def test_reject_invalid(self):
        for mean, std in [([0], [0]), ([0], [-1]), ([0], [float('nan')]),
                          ([0], [1, 2]), ([], []), ([[0]], [[1]])]:
            with self.assertRaises(ValueError):
                CavityReward(mean, std)
        reward = CavityReward([0], [1])
        for obs in [[0, 1], [[0]], [float('nan')], [float('inf')]]:
            with self.assertRaises(ValueError):
                reward(obs)

    def test_binding_and_raw_reward(self):
        class Env:
            Re = 4200
            probe_locations = [1.25, -.5]
            observation_type = ['u', 'v']
            num_substeps_per_iteration = 400
            num_outputs = 2
            obs = np.array([1.5, 6.])
            obs_scale = np.array([.05, .05])
        binding = {'reynolds': 4200, 'probe_locations': Env.probe_locations,
                   'observation_types': Env.observation_type, 'substeps': 400}
        document = {'status': 'reconstructed-cavity-calibration-not-paper-certified',
                    'binding': binding, 'mean': [1, 2], 'std_population': [.5, 2]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calibration.json'
            path.write_text(json.dumps(document))
            env = Env()
            result = install(env, path)
            self.assertEqual(len(result.calibration_sha256), 64)
            self.assertEqual(env.get_reward(), (-5., {}))
            env.Re = 4140
            with self.assertRaisesRegex(ValueError, 'does not match'):
                install(env, path)


if __name__ == '__main__':
    unittest.main()
