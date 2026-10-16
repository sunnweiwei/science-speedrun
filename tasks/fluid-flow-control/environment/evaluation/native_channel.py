"""Pinned 3D Channel with an explicit control-window clock repair.

The upstream actuator ramp is defined on 0..50 CFD substeps, but its caller
passes the global episode clock. From the second control onward the unbounded
negative ramp increases jet amplitude throughout the episode. Reset ONLY the
clock passed to the action-window integrator. Physical fields, dt, grid, forcing,
reward, action bounds and the OUTER episode clock are unchanged.

This is an organizer runtime profile, not an unmodified-paper reproduction.
"""
import gymnasium as gym
import jax
import numpy as np
from hydrogym.jax.envs.channel import ChannelFlowSpectralEnv

PROFILE = 'channel-72cube-local-actuator-clock-v1'


class LocalActuatorClock(ChannelFlowSpectralEnv):
    def _wss_with_aux(self, state, action, params):
        return super()._wss_with_aux(state.replace(time=0), action, params)


class ChannelEnv(gym.Env):
    def __init__(self, data, *, repair=True):
        cls = LocalActuatorClock if repair else ChannelFlowSpectralEnv
        self.native = cls({'initial_field_dir': str(data/'environments/Channel_3D_Retau180/initial_field')})
        self.params = self.native.default_params
        self.key = jax.random.PRNGKey(42)
        self.advance = jax.jit(lambda k, s, a: self.native.step_env(k, s, a, self.params))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(128,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=(24,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.key = jax.random.PRNGKey(seed)
        obs, self.state = self.native.reset_env(self.key, self.params)
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action):
        self.key, key = jax.random.split(self.key)
        obs, self.state, reward, done, info = self.advance(key, self.state, np.asarray(action,dtype=np.float32))
        return np.asarray(obs, dtype=np.float32), float(reward), bool(done), False, {}
