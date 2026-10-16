"""CPU physical extraction/aggregation tests, not full CFD acceptance."""
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'environment/evaluation'))
import metrics
from case_spec import select_cases
from native_envs import ReplayAdapter
from protocol import InfrastructureError, replay_episode


class MetricTests(unittest.TestCase):
    def test_maia_empty_step_info_captures_existing_reward_objective_once(self):
        class Env:
            observation_space = action_space = SimpleNamespace(shape=(1,), dtype=np.float32)
            def reset(self, seed):
                self.calls = 0
                return np.zeros(1), {}
            def get_reward(self):
                self.calls += 1
                return -12., {'forces': [[2., 3.]]}
            def step(self, action):
                reward, discarded = self.get_reward()
                return np.zeros(1), reward, True, True, {}
        case = dict(id='Pinball_2D_Re100', environment='Pinball_2D_Re100', backend='maia',
                    horizon=1, primary_metric=metrics.definition('Pinball_2D_Re100'))
        with tempfile.TemporaryDirectory() as directory:
            native = Env()
            env = ReplayAdapter(native, case, Path(directory))
            env.reset(123)
            _, reward, _, _, info = env.step([0.])
            self.assertEqual(native.calls, 1)
            self.assertEqual(reward, -12.)
            self.assertEqual(info['CD'], 2.)
            self.assertEqual(env.evaluation_metrics()['value'], 2.)

    def test_exact_39_coverage_and_targets(self):
        spec, cases = select_cases(['all'])
        self.assertEqual(set(metrics.registry()), {c['id'] for c in cases})
        self.assertEqual(len(cases), 39)
        self.assertEqual(spec['metric_version'], metrics.VERSION)
        self.assertEqual(spec['metrics_sha256'], metrics.registry_hash())
        self.assertEqual(len(metrics.GROUPS['drag_cd'].split()), 19)
        for case in cases:
            self.assertEqual(case['primary_metric'], metrics.definition(case['id']))
        with self.assertRaises(InfrastructureError):
            metrics.definition('Cavity_2D_Re4200')  # Not in this frozen release.

    def test_fixed_window_including_odd_horizon(self):
        m = metrics.definition('Cylinder_2D_Re200')
        for values, expected, first in [([10., 10., 2., 4.], 3., 3), ([10., 2., 4.], 3., 2)]:
            result = metrics.summarize(m, [dict(CD=v) for v in values], len(values))
            self.assertEqual(result['value'], expected)
            self.assertEqual(result['first_control'], first)
        with self.assertRaises(InfrastructureError):
            metrics.summarize(m, [dict(CD=1.)], 2)

    def test_maia_force_units_and_sphere_projection(self):
        converter = Mock(side_effect=lambda **kw: np.asarray(kw['forces']) / 2)
        env = SimpleNamespace(Ma=math.sqrt(3), referenceLength=2., dX=.5, zLength=3.,
                              compute_nondim_coefficients=converter)
        for name, expected_cd in [('Cylinder_2D_Re200', 3.), ('SquareCylinder_2D_Re200', 3.),
                                 ('Cube_3D_Re300', 3.), ('Sphere_3D_Re300', 3.), ('Pinball_2D_Re100', 6.)]:
            converter.reset_mock()
            s = metrics.sample(env, {'environment': name}, {'forces': [[2., 1., 0.], [4., 3., 0.]]}, -99.)
            self.assertEqual(s['CD'], expected_cd)
            if name.startswith('Pinball'):
                converter.assert_not_called()
            else:
                expected_area = math.pi * 4 if name.startswith('Sphere') else 24.
                self.assertAlmostEqual(converter.call_args.kwargs['projectionLength'], expected_area)

    def test_gust_full_episode_error_and_frozen_target(self):
        name = 'NACA0012Gust_2D_Re100_AOA20'
        m = metrics.definition(name)
        t = m['target']
        env = SimpleNamespace(Ma=1., referenceLength=1., dX=1., zLength=1.,
            unperturbed_avg_lift=t['CL'], unperturbed_avg_drag=t['CD'], omega=t['omega'],
            compute_nondim_coefficients=lambda **kw: np.asarray(kw['forces']))
        s = metrics.sample(env, {'environment': name, 'primary_metric': m},
                           {'forces': [t['CD'] + .4, t['CL'] + .2]}, -999.)
        self.assertAlmostEqual(s['gust_load_error'], .3)
        result = metrics.summarize(m, [s, dict(gust_load_error=0.)], 2)
        self.assertAlmostEqual(result['value'], .15)
        env.omega = 1
        with self.assertRaises(InfrastructureError):
            metrics.sample(env, {'environment': name}, {'forces': [1., 1.]}, 0.)

    def test_lift_drag_is_mean_of_ratios_not_ratio_of_means(self):
        m = metrics.definition('NACA0012_2D_Re100_AOA20')
        rows = [dict(lift_drag=50., lift_drag_valid=1.), dict(lift_drag=50., lift_drag_valid=1.),
                dict(lift_drag=2., lift_drag_valid=1.), dict(lift_drag=4., lift_drag_valid=1.)]
        self.assertEqual(metrics.summarize(m, rows, 4)['value'], 3.)
        rows[-1]['lift_drag_valid'] = 0.
        invalid = metrics.summarize(m, rows, 4)
        self.assertIsNone(invalid['value'])
        self.assertEqual(invalid['status'], 'invalid_operating_regime')

    def test_nozzle_radians_to_degrees_full_window(self):
        name = 'Nozzle2D_coarse'
        s = metrics.sample(None, {'environment': name}, dict(target_angle=math.pi/6, thrust_angle=0.), -99.)
        self.assertAlmostEqual(s['angle_error_deg'], 30.)
        m = metrics.definition(name)
        result = metrics.summarize(m, [s, dict(angle_error_deg=0.)], 2)
        self.assertAlmostEqual(result['value'], 15.)

    def test_energy_not_action_penalized_reward_and_channel_exact_identity(self):
        s = metrics.sample(None, {'environment': 'Kolmogorov_2D'}, {'mean_tke': .2}, -999.)
        self.assertEqual(s, dict(TKE=.2))
        s = metrics.sample(None, {'environment': 'Channel_3D_Retau180'}, {}, -.4)
        self.assertEqual(s, dict(wall_shear=.4))

    def test_missing_nonfinite_or_reward_only_is_not_a_primary_metric(self):
        m = metrics.definition('Cylinder_2D_Re200')
        for episodes in ([{'return': 10.}], [{'primary_metric': dict(m, status='ok', value=float('nan'))}],
                         [{'primary_metric': dict(m, status='ok', value=1., window='full_episode')}]):
            with self.assertRaises(InfrastructureError):
                metrics.aggregate(m, episodes)
        with self.assertRaises(InfrastructureError):
            metrics.summarize(m, [dict(CD=float('nan'))], 1)

    def test_aggregate_equal_weight_episodes(self):
        m = metrics.definition('Cylinder_2D_Re200')
        es = [dict(primary_metric=dict(m, status='ok', value=x)) for x in (1., 3.)]
        self.assertEqual(metrics.aggregate(m, es)['value'], 2.)

    def test_policy_replay_reports_physical_mean_not_return_or_terminal_value(self):
        class Env:
            observation_space = action_space = SimpleNamespace(shape=(1,), dtype=np.float32,
                low=np.array([-1.]), high=np.array([1.]))
            def reset(self, seed):
                self.i = 0
                return np.zeros(1), {}
            def step(self, action):
                self.i += 1
                cd = [9., 9., 2., 4.][self.i-1]
                return np.zeros(1), -100., self.i == 4, False, {'forces': [[cd, 0.]]}
        case = dict(id='Pinball_2D_Re100', environment='Pinball_2D_Re100', family='pinball',
                    horizon=4, primary_metric=metrics.definition('Pinball_2D_Re100'))
        policy = Mock()
        policy.act.return_value = [0.]
        with tempfile.TemporaryDirectory() as directory:
            env = ReplayAdapter(Env(), case, Path(directory))
            result = replay_episode(env, policy, case, 123)
            self.assertEqual(result['return'], -400.)
            self.assertEqual(result['primary_metric']['value'], 3.)
            self.assertEqual(result['diagnostics']['CD'], 4.)


if __name__ == '__main__':
    unittest.main()
