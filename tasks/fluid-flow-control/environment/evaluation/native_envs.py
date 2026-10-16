"""MAIA, JAX and JAX-Fluids bindings for one shared action-only contract."""
from pathlib import Path

from protocol import InfrastructureError


def kolmogorov():
    import gymnasium as gym
    import jax
    import numpy as np
    jax.config.update('jax_enable_x64', True)
    from hydrogym.jax.envs.kolmogorov import KolmogorovFlow

    class KolmogorovEnv(gym.Env):
        def __init__(self):
            self.native = KolmogorovFlow(env_config={}, flow_config={})
            self.params = self.native.default_params.replace(reward_alpha=1.0)
            self.key = jax.random.PRNGKey(123)
            self.advance = jax.jit(lambda k, s, a: self.native.step_env(k, s, a, self.params))
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(64,), dtype=np.float64)
            self.action_space = gym.spaces.Box(-.5, .5, shape=(4,), dtype=np.float64)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.key = jax.random.PRNGKey(123 if seed is None else seed)
            obs, self.state = self.native.reset_env(self.key, self.params)
            return np.asarray(obs, dtype=np.float64), {}

        def step(self, action):
            self.key, key = jax.random.split(self.key)
            obs, self.state, reward, done, info = self.advance(
                key, self.state, np.asarray(action, dtype=np.float64))
            return np.asarray(obs, dtype=np.float64), float(reward), bool(done), False, info
    return KolmogorovEnv()


def nozzle(case, data):
    import hydrogym.jaxfluids.envs.nozzle as module
    from native_resource_bindings import bind_nozzle_restart
    original = module.build_tvc_runtime_setup
    if case['environment'] == 'Nozzle2D_fine':
        def setup_with_source_restart(**kwargs):
            setup = original(**kwargs)
            restart = Path(setup.restart_file_path).resolve()
            if not restart.is_relative_to(data / 'environments' / case['environment']):
                raise InfrastructureError('Nozzle restart escaped the selected source')
            # The upstream named tuple contains a mutable config dict.
            updated = bind_nozzle_restart(case['environment'], setup.case_setup_dict, str(restart))
            setup.case_setup_dict.update(updated)
            return setup
        module.build_tvc_runtime_setup = setup_with_source_restart
    try:
        env = module.Nozzle2D({'environment_name': case['environment'],
            'use_clean_cache': False, 'local_fallback_dir': str(data / 'environments')})
    finally:
        module.build_tvc_runtime_setup = original
    if Path(env.env_data_path).resolve() != data / 'environments' / case['environment']:
        raise InfrastructureError('Nozzle loaded a different source')
    return env


def make_env(case, data):
    import numpy as np
    profile = case['runtime_profile']
    backend = case['backend']
    if backend == 'maia':
        from run_maia import make_env as make_maia
        env = make_maia(case, data)
        expected = {'max_episode_steps': profile['native_max_episode_steps'],
            'substeps_per_action': profile['substeps_per_control'],
            'max_control': profile['max_control'], 'omega': profile['omega']}
        if env.reviewed_protocol['native_defaults'] != expected:
            raise InfrastructureError('MAIA source dynamics differ from the frozen profile')
    elif case['environment'] == 'Channel_3D_Retau180':
        from native_channel import ChannelEnv
        env = ChannelEnv(data)
        for key, expected in profile['params'].items():
            if getattr(env.params, key) != expected:
                raise InfrastructureError(f'Channel parameter changed: {key}')
    elif case['environment'] == 'Kolmogorov_2D':
        env = kolmogorov()
        for key, expected in profile['params'].items():
            if isinstance(expected, str) and expected in ('inf', '-inf'):
                expected = float(expected)
            if getattr(env.params, key) != expected:
                raise InfrastructureError(f'Kolmogorov parameter changed: {key}')
        if env.native.flow.Re != 200 or tuple(env.native.flow.grid_size) != (64, 64):
            raise InfrastructureError('Kolmogorov grid/Re changed')
    elif backend == 'jax-fluids':
        env = nozzle(case, data)
        from omegaconf import OmegaConf
        actual = OmegaConf.to_container(env.conf, resolve=True)
        if actual != profile['source_configuration']:
            raise InfrastructureError('Nozzle source configuration changed')
    else:
        raise InfrastructureError(f'Backend not implemented: {backend}')
    if (list(env.observation_space.shape) != case['observation_shape'] or
            list(env.action_space.shape) != case['action_shape']):
        raise InfrastructureError('Native observation/action spaces changed')
    if backend != 'maia':
        import jax
        if len(jax.devices()) != case['solver_processes'] or any(device.platform != 'gpu' for device in jax.devices()):
            raise InfrastructureError('Native JAX evaluation requires the allocated GPU')
    return env


class ReplayAdapter:
    """Strict complete-episode checking, scalar diagnostics and progress receipts."""
    def __init__(self, env, case, output):
        self.env, self.case, self.output = env, case, output
        self.observation_space, self.action_space = env.observation_space, env.action_space
        self.count = None
        self.captured_physical = None
        self.capture_reward = case.get('backend') == 'maia' and 'primary_metric' in case
        if self.capture_reward:
            # Pinned MaiaFlowEnv.step discards get_reward()'s objective dict and
            # returns info={}. Capture the existing force read in that call;
            # never make a second MPI/CFD query or change the native reward.
            base = getattr(env, 'unwrapped', env)
            original_reward = base.get_reward
            def recorded_reward():
                reward, objective = original_reward()
                from metrics import sample
                self.captured_physical = sample(base, self.case, objective, reward)
                return reward, objective
            base.get_reward = recorded_reward

    def report(self, phase):
        from evaluate import write_json
        write_json(self.output / 'progress.json', {'case': self.case['id'], 'phase': phase,
            'step': self.count, 'horizon': self.case['horizon']})

    def reset(self, seed):
        self.count = 0
        self.physical_samples = []
        self.report('reset')
        result = self.env.reset(seed=seed)
        self.report('control')
        return result

    def step(self, action):
        import numpy as np
        if self.count is None or self.count >= self.case['horizon']:
            raise InfrastructureError('Reset required before another native step')
        self.captured_physical = None
        obs, reward, done, truncated, info = self.env.step(
            np.asarray(action, dtype=self.action_space.dtype).reshape(self.action_space.shape))
        self.count += 1
        if np.asarray(reward).ndim != 0:
            raise InfrastructureError('Native vector reward has no reviewed scalar adapter')
        if bool(done or truncated) != (self.count == self.case['horizon']):
            raise InfrastructureError(f'Native termination differs from frozen horizon: {self.count}')
        scalar_info = {}
        if 'primary_metric' in self.case:
            from metrics import sample
            physical = self.captured_physical if self.capture_reward else sample(self.env, self.case, info, reward)
            if physical is None:
                raise InfrastructureError('MAIA did not expose this step\'s physical objective')
            self.physical_samples.append(physical)
            scalar_info.update(physical)
        for key, value in info.items():
            if isinstance(value, (str, dict)) or value is None:
                continue
            a = np.asarray(value)
            if a.size == 1 and np.issubdtype(a.dtype, np.number):
                scalar_info[key] = float(a.reshape(-1)[0])
        if not np.all(np.isfinite(np.asarray(obs))) or not np.isfinite(reward):
            raise InfrastructureError(f'Nonfinite native observation/reward at control {self.count}')
        if self.count == 1 or self.count % 10 == 0 or done or truncated:
            self.report('complete' if done or truncated else 'control')
        return obs, float(reward), bool(done), bool(truncated), scalar_info

    def evaluation_metrics(self):
        from metrics import summarize
        return summarize(self.case['primary_metric'], self.physical_samples, self.case['horizon'])
