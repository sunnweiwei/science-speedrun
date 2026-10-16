import copy
import math
import unittest

import numpy as np

from cavity_reward import CavityReward, evaluate_with_diagnostics
from cavity_velocity_metrics import analyze


class VelocityTests(unittest.TestCase):
    def setUp(self):
        self.protocol = dict(horizon=2, environment='Cavity_2D_Re4200',
            recorded_probe_velocity=dict(enabled=True, units='lattice_velocity', order='all_u_then_all_v'),
            probe_locations=[0., 0., 1., 0.], initial_state=dict(mach=math.sqrt(3)))
        self.rows = [dict(step=1, reward=-2., raw_probe_velocity=[1., 2., 3., 4.]),
                     dict(step=2, reward=-4., raw_probe_velocity=[3., 6., 3., 8.])]

    def test_trace_does_not_change_reward_or_input(self):
        reward = CavityReward([0., 0.], [2., 4.])
        obs = np.array([2., 8.])
        baseline = reward(obs)
        for enabled in (False, True):
            score, metrics = evaluate_with_diagnostics(reward, obs, record_probes=enabled)
            self.assertEqual(score, baseline)
            self.assertEqual('raw_probe_velocity' in metrics, enabled)
            np.testing.assert_array_equal(obs, [2., 8.])
            if enabled:
                obs[0] = 0
                self.assertEqual(metrics['raw_probe_velocity'], [2., 8.])

    def test_temporal_and_spatial_definitions(self):
        result = analyze(self.rows, self.protocol)
        self.assertEqual(result['raw_return'], -6.)
        self.assertEqual(result['components']['u']['temporal_rms_per_probe'], [1., 2.])
        self.assertEqual(result['components']['v']['temporal_rms_per_probe'], [0., 2.])
        self.assertAlmostEqual(result['components']['u']['spatial_rms_of_temporal_rms'], math.sqrt(2.5))
        self.assertEqual(result['components']['u']['spatial_mean_of_temporal_rms'], 1.5)

    def test_incomplete_or_malformed_measurements_rejected(self):
        with self.assertRaises(ValueError):
            analyze(self.rows[:1], self.protocol)
        for values in ([1.], [float('nan')] * 4):
            rows = copy.deepcopy(self.rows)
            rows[0]['raw_probe_velocity'] = values
            with self.assertRaises(ValueError):
                analyze(rows, self.protocol)
        changed = copy.deepcopy(self.protocol)
        changed['recorded_probe_velocity']['enabled'] = False
        with self.assertRaises(ValueError):
            analyze(self.rows, changed)


if __name__ == '__main__':
    unittest.main()
