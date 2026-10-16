"""CPU interface tests; synthetic numbers are not reported baseline results."""
import copy
from pathlib import Path
import sys
import json
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'environment/evaluation'))
from case_spec import select_cases
from harbor_verify import physical_rewards, verify
from protocol import InfrastructureError, PolicyError


class PhysicalHarborTests(unittest.TestCase):
    def fixture(self,names=('all',)):
        _,cases=select_cases(list(names))
        return {'cases':{c['id']:{'status':'ok','primary_metric':
            dict(c['primary_metric'],status='ok',value=1.25)} for c in cases}}

    def test_complete_reward_has_exactly_39_named_physical_values(self):
        rewards=physical_rewards(self.fixture(),['all'])
        self.assertEqual(rewards.pop('failed_cases'),0)
        self.assertEqual(len(rewards),39)
        self.assertNotIn('score',rewards)
        self.assertEqual(rewards['Cylinder_2D_Re200/drag_cd'],1.25)
        self.assertEqual(rewards['NACA0012_2D_Re100_AOA20/lift_drag'],1.25)

    def test_missing_case_cannot_be_silently_dropped(self):
        result=self.fixture();result['cases'].pop('Cylinder_3D_Re3900')
        with self.assertRaises(InfrastructureError):physical_rewards(result,['all'])

    def test_policy_and_infrastructure_errors_are_distinct(self):
        result=self.fixture();result['cases']['Cylinder_2D_Re200']={'status':'infrastructure_error'}
        with self.assertRaises(InfrastructureError):physical_rewards(result,['all'])
        # A policy error fails its case only; every other case keeps its physical value.
        result=self.fixture();result['cases']['Cylinder_2D_Re200']={'status':'policy_error'}
        rewards=physical_rewards(result,['all'])
        self.assertEqual(rewards['Cylinder_2D_Re200/failed'],1)
        self.assertNotIn('Cylinder_2D_Re200/drag_cd',rewards)
        self.assertEqual(rewards['failed_cases'],1)
        self.assertEqual(rewards['NACA0012_2D_Re100_AOA20/lift_drag'],1.25)

    def test_all_cases_failed_is_a_policy_error(self):
        result=self.fixture()
        for name in result['cases']:result['cases'][name]={'status':'policy_error'}
        with self.assertRaises(PolicyError):physical_rewards(result,['all'])

    def test_nonfinite_and_wrong_units_rejected(self):
        for change in [dict(value=float('nan')),dict(value=True),dict(units='wrong')]:
            result=self.fixture();result['cases']['Cylinder_2D_Re200']['primary_metric'].update(change)
            with self.assertRaises(InfrastructureError):physical_rewards(result,['all'])

    def test_invalid_physical_regime_is_not_a_zero_or_reward(self):
        result=self.fixture()
        result['cases']['NACA0012_2D_Re100_AOA20']['primary_metric'].update(status='invalid_operating_regime',value=None)
        rewards=physical_rewards(result,['all'])
        self.assertEqual(rewards['NACA0012_2D_Re100_AOA20/failed'],1)
        self.assertNotIn('NACA0012_2D_Re100_AOA20/lift_drag',rewards)

    def test_subset_is_explicit_and_keeps_original_units(self):
        result=self.fixture();result['cases']={'Nozzle2D_fine':result['cases']['Nozzle2D_fine']}
        self.assertEqual(physical_rewards(result,['Nozzle2D_fine']),{'Nozzle2D_fine/nozzle_angle_mae':1.25,'failed_cases':0})

    def test_verifier_default_is_the_scored_suite_and_writes_numeric_file(self):
        with TemporaryDirectory() as folder:
            def fake(data,submission,output,names):
                self.assertEqual(names,['verifier'])
                return self.fixture(names)
            reward=verify('/unused-data','/unused-submission',folder,evaluator=fake)
            self.assertEqual(json.loads((Path(folder)/'reward.json').read_text()),reward)
            self.assertEqual(json.loads((Path(folder)/'grading.json').read_text())['scope'],'verifier-suite')
            self.assertEqual(len(reward),26+1)

    def test_complete_suite_is_still_gradable(self):
        with TemporaryDirectory() as folder:
            reward=verify('/unused','/unused2',folder,['all'],evaluator=lambda *a:self.fixture())
            self.assertEqual(json.loads((Path(folder)/'grading.json').read_text())['scope'],'full-39')
            self.assertEqual(len(reward),39+1)

    def test_policy_error_preserves_diagnostic_and_marks_invalid(self):
        with TemporaryDirectory() as folder:
            def fake(data,submission,output,names):
                result=self.fixture(names)
                for name in result['cases']:result['cases'][name]={'status':'policy_error'}
                return result
            self.assertEqual(verify('/unused','/unused2',folder,evaluator=fake),{'invalid_submission':1})
            # Harbor needs a numeric reward.json; an invalid package has no physical metrics.
            self.assertEqual(json.loads((Path(folder)/'reward.json').read_text()),{'invalid_submission':1})
            self.assertEqual(json.loads((Path(folder)/'grading-error.json').read_text())['status'],'policy_error')



class DivergenceTests(unittest.TestCase):
    """Invalid simulation state after the policy acts fails the case, not the suite."""

    class Space:
        def __init__(self, n):
            import numpy as np
            self.shape, self.low, self.high = (n,), np.full(n, -1.0), np.full(n, 1.0)

    class Env:
        def __init__(self, fail_at):
            self.observation_space = self.action_space = DivergenceTests.Space(1)
            self.fail_at, self.steps = fail_at, 0
        def reset(self, seed=None):
            return [0.0], {}
        def step(self, action):
            self.steps += 1
            if self.steps == self.fail_at:
                raise InfrastructureError('Missing/invalid per-body force measurements')
            return [0.0], 1.0, self.steps == 3, False, {}

    class Policy:
        def __init__(self, bad_first=False):
            self.bad_first = bad_first
        def reset(self, *args):
            pass
        def act(self, observation):
            if self.bad_first:
                raise InfrastructureError('Nonfinite/invalid native observation')
            return [0.0]

    def test_divergence_after_actions_is_a_policy_error(self):
        from protocol import replay_episode
        case = {'environment': 'x', 'family': 'f', 'horizon': 3}
        with self.assertRaises(PolicyError):
            replay_episode(self.Env(fail_at=2), self.Policy(), case, 0)
        self.assertEqual(replay_episode(self.Env(fail_at=0), self.Policy(), case, 0)['steps'], 3)

    def test_invalid_state_before_the_first_action_stays_infrastructure(self):
        from protocol import replay_episode
        case = {'environment': 'x', 'family': 'f', 'horizon': 3}
        with self.assertRaises(InfrastructureError):
            replay_episode(self.Env(fail_at=0), self.Policy(bad_first=True), case, 0)


if __name__=='__main__':unittest.main()
