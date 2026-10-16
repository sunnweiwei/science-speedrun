"""Executable author StepCounter vs native horizon-override semantics audit."""

import unittest

import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Unbounded
from torchrl.envs import EnvBase, TransformedEnv
from torchrl.envs.transforms import StepCounter
from torchrl.objectives.value import GAE

from time_limits import apply_time_limit


class NativeHorizon(EnvBase):
    def __init__(self, native_limit):
        super().__init__(device='cpu', batch_size=[])
        self.native_limit = native_limit
        self.action_spec = Bounded(-1., 1., shape=(1,))
        # EnvBase requires a composite observation specification.
        from torchrl.data import Composite
        self.observation_spec = Composite(observation=Unbounded(shape=(1,)))
        self.reward_spec = Unbounded(shape=(1,))

    def _set_seed(self, seed):
        return seed

    def _reset(self, tensordict=None):
        self.iter = 0
        return TensorDict({'observation': torch.zeros(1)}, [])

    def _step(self, tensordict):
        self.iter += 1
        return TensorDict({'observation': torch.zeros(1), 'reward': -torch.ones(1),
                           'done': torch.tensor([self.iter > self.native_limit])}, [])


class TimeLimitTests(unittest.TestCase):
    def test_corrected_adapter_matches_author_all_steps(self):
        env = TransformedEnv(NativeHorizon(200), StepCounter(200))
        data = env.rollout(200)
        for index, item in enumerate(data):
            observed = apply_time_limit(False, False, index + 1, 200, True)
            expected = (item['next', 'terminated'].item(), item['next', 'truncated'].item())
            self.assertEqual(observed, expected)
        env.close()

    def test_never_masks_early_native_completion(self):
        with self.assertRaisesRegex(ValueError, 'Native completion'):
            apply_time_limit(True, True, 20, 200, True)

    def test_legacy_remains_replayable(self):
        self.assertEqual(apply_time_limit(True, True, 200, 200, False), (True, True))

    def test_author_stepcounter_truncates_before_native_done(self):
        env = TransformedEnv(NativeHorizon(200), StepCounter(200))
        data = env.rollout(200)
        self.assertEqual(data.batch_size, torch.Size([200]))
        self.assertTrue(data[-1]['next', 'done'].item())
        self.assertTrue(data[-1]['next', 'truncated'].item())
        self.assertFalse(data[-1]['next', 'terminated'].item())
        env.close()

    def test_legacy_native_199_override_marks_termination(self):
        env = TransformedEnv(NativeHorizon(199), StepCounter(200))
        data = env.rollout(200)
        self.assertTrue(data[-1]['next', 'terminated'].item())
        env.close()

    def test_truncation_keeps_value_bootstrap(self):
        gae = GAE(gamma=.99, lmbda=.95, value_network=None, average_gae=False)
        targets = []
        for terminated in (False, True):
            data = TensorDict({'state_value': torch.tensor([[-5.]]),
                'next': {'state_value': torch.tensor([[-5.]]), 'reward': torch.tensor([[-1.]]),
                         'done': torch.tensor([[True]]), 'terminated': torch.tensor([[terminated]])}}, [1])
            targets.append(gae(data)['value_target'].item())
        torch.testing.assert_close(torch.tensor(targets), torch.tensor([-5.95, -1.]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
