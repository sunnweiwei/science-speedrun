"""Run pinned public JAX PPO with an explicitly reconstructed mixing objective.

Native solver and author PPO updates are reused. This is not the undisclosed
Figure 3b configuration. Optional Eq.34 reward augmentation is distinct from
the unpublished adaptive Eq.32 GPPO gradient-combination implementation.
"""

import argparse
import copy
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from hydrogym.jax.envs.kolmogorov import KolmogorovFlow


class MixingFlow(KolmogorovFlow):
    @property
    def default_params(self):
        # Related author's published environment: nonnegative forcing controls,
        # 100 physical time units per episode / 10 units per action = 10 controls.
        # The native class increments time in actions, not physical time units.
        return super().default_params.replace(min_action=0.0, max_action=1.0,
                                              max_episode_steps=10)

    def _reward(self, action, trajectory, params):
        # Published related-author mixing reward. Bounds make L1=sum(action).
        return self._avg_tke(trajectory) - 75.0 * jnp.sum(jnp.abs(action))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path(os.environ.get("HYDROGYM_UPSTREAM_SOURCE", "/upstream")))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--rollout-steps", type=int, choices=[10, 40], default=10,
                        help="Controls per environment per PPO update:10 is the early reconstruction;40 matches the public example")
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--float64", action="store_true", help="Explicit solver-precision diagnostic")
    parser.add_argument("--solver-corrections", choices=["none", "dealias", "dealias-curl"], default="none")
    parser.add_argument("--learning-reward-scale", type=float, default=1.0,
                        help="Explicit constant optimizer-side scaling; published returns stay in physical reward units")
    parser.add_argument("--gradient-reward", action="store_true",
                        help="Explicit Supplement Eq.34 reconstruction, not original Fig.3b adaptive GPPO")
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % (4 * args.rollout_steps):
        parser.error("steps must be a positive multiple of4*rollout-steps; do not silently drop a partial public batch")
    if not np.isfinite(args.learning_reward_scale) or args.learning_reward_scale <= 0:
        parser.error("Learning reward scale must be finite and positive")
    # Atomic publication is a host callback even when the solver runs on CUDA.
    # Reject a CUDA-only backend configuration before paying for a full rollout.
    try:
        jax.devices("cpu")
    except RuntimeError as error:
        raise RuntimeError("PPO publication requires a CPU backend; use JAX_PLATFORMS=cuda,cpu for GPU runs") from error
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    import jax_source
    source_audit = jax_source.verify()
    (args.output / "jax_source.py").write_bytes(Path(jax_source.__file__).read_bytes())
    jax.config.update("jax_enable_x64", args.float64)
    solver_audit = {"mode": "none"}
    if args.solver_corrections != "none":
        import jax_corrections
        solver_audit = jax_corrections.apply_corrections(args.solver_corrections)
        (args.output / "jax_corrections.py").write_bytes(Path(jax_corrections.__file__).read_bytes())
    path = args.upstream / "examples/jax/getting_started/3_ppo/run_ppo.py"
    spec = importlib.util.spec_from_file_location("public_hydrogym_ppo", path)
    module = importlib.util.module_from_spec(spec)
    module.diagnostic_output = args.output
    module.learning_reward_scale = args.learning_reward_scale
    module.gradient_reward_enabled = args.gradient_reward

    def publish_intermediate(info, policy_parameters, optimizer_step):
        step = int(info["timestep"].max()) * config["NUM_ENVS"]
        payload = flax.serialization.to_bytes(policy_parameters)
        checkpoint = args.output / f"params-{step:07d}.msgpack"
        temporary = checkpoint.with_suffix(".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, checkpoint)
        latest = args.output / "params-latest.msgpack"
        latest.with_suffix(".tmp").write_bytes(payload)
        os.replace(latest.with_suffix(".tmp"), latest)
        mask = np.asarray(info["returned_episode"], dtype=bool)
        if args.gradient_reward:
            # Raw-return reconstruction below is valid because each batch starts
            # at reset and spans whole ten-control episodes per environment.
            expected = np.broadcast_to(((np.arange(args.rollout_steps) + 1) % 10 == 0)[:, None],
                                       (args.rollout_steps, 4))
            if not np.array_equal(mask, expected):
                raise RuntimeError("Eq.34 physical-return aggregation lost episode alignment")
        returns = np.asarray(info["returned_episode_returns"])[mask]
        progress = {"status": "training-native-ppo-not-paper-matched", "training_steps": step,
                    "gradient_updates": int(optimizer_step), "checkpoint": checkpoint.name,
                    "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
                    "last_training_returns": returns.tolist(), "wall_seconds": time.monotonic() - started}
        (args.output / "progress.tmp").write_text(json.dumps(progress, indent=2, allow_nan=False) + "\n")
        os.replace(args.output / "progress.tmp", args.output / "progress.json")

    module.publish_intermediate = publish_intermediate
    raw = path.read_text()
    instrumentation = {
        "            metric = traj_batch.info": """            metric = dict(traj_batch.info)
            # Restore native units before logging, saving or comparing returns.
            metric['returned_episode_returns'] = metric['returned_episode_returns'] / learning_reward_scale
            if gradient_reward_enabled:
                # Optimizer-side Eq.34 shaping is NEVER a benchmark score.
                metric['augmented_episode_returns'] = metric['returned_episode_returns']
                physical = metric['physical_reward']
                episodes = physical.reshape((-1, 10, config['NUM_ENVS']))
                metric['returned_episode_returns'] = jnp.cumsum(episodes, axis=1).reshape(physical.shape)
            finite_loss = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(loss_info)]))
            finite_params = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(train_state.params)]))
            metric['learner_finite'] = jnp.full_like(metric['timestep'], finite_loss & finite_params)
            metric['raw_actions'] = traj_batch.action
            metric['raw_observations'] = traj_batch.obs""",
        "                def callback(info):": """                def callback(info, policy_parameters, optimizer_step):
                    diagnostic_step = int(info['timestep'].max())
                    np.savez_compressed(diagnostic_output / f'rollout-{diagnostic_step:06d}.npz', **info)
                    finite_cfd = np.isfinite(info['mean_tke']).all() and np.isfinite(info['returned_episode_returns']).all()
                    finite_learner = info['learner_finite'].all()
                    if not finite_cfd or not finite_learner:
                        print('NONFINITE', {'cfd': bool(finite_cfd), 'learner': bool(finite_learner), 'timestep_per_env': int(info['timestep'].max())}, flush=True)
                        raise RuntimeError('Nonfinite native CFD or PPO optimization; no accepted checkpoint')
                    publish_intermediate(info, policy_parameters, optimizer_step)""",
        '                    step = int(info["timestep"].max())': '                    step = int(info["timestep"].max()) * config["NUM_ENVS"]',
        "                jax.debug.callback(callback, metric)": "                jax.debug.callback(callback, metric, train_state.params, train_state.step, ordered=True)",
    }
    for old, new in instrumentation.items():
        if raw.count(old) != 1:
            raise RuntimeError("Pinned PPO instrumentation anchor changed")
        raw = raw.replace(old, new)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    config = copy.deepcopy(module.config)
    config.update(ENV_NAME="kolmogorov", TOTAL_TIMESTEPS=args.steps, NUM_ENVS=4,
                  NUM_STEPS=args.rollout_steps, NUM_MINIBATCHES=8, DEBUG=True)

    gradient_reward_sha = None
    gradient_base = None
    if args.gradient_reward:
        import jax_gradient_reward
        payload = Path(jax_gradient_reward.__file__).read_bytes()
        (args.output / "jax_gradient_reward.py").write_bytes(payload)
        gradient_reward_sha = hashlib.sha256(payload).hexdigest()
        gradient_base = jax_gradient_reward.equation34_flow(MixingFlow)

    def make_env(_):
        # Grid follows paper Table 1 (4096 DOF), Reynolds number follows related
        # author's Re250 example. Figure-specific Reynolds was not disclosed.
        class LearningFlow(MixingFlow):
            def _reward(self, action, trajectory, params):
                return args.learning_reward_scale * super()._reward(action, trajectory, params)

        flow_type = LearningFlow
        if args.gradient_reward:
            class GradientLearningFlow(gradient_base):
                def step_env(self, key, state, action, params):
                    obs, next_state, reward, done, info = super().step_env(key, state, action, params)
                    return obs, next_state, args.learning_reward_scale * reward, done, info
            flow_type = GradientLearningFlow
        env = flow_type(env_config={}, flow_config={"Re": 250, "grid_size": (64, 64)})
        return env, env.default_params

    module.make_env = make_env
    runtime_source = os.environ.get("HYDROGYM_JAX_RUNTIME_SOURCE")
    runtime_sha = None
    if runtime_source:
        payload = Path(runtime_source).read_bytes()
        (args.output / "jax_runtime.sh").write_bytes(payload)
        runtime_sha = hashlib.sha256(payload).hexdigest()
    metadata = {"status": "public-description-reconstruction-not-paper-matched",
                "algorithm": "PPO-Eq34-reconstruction" if args.gradient_reward else "PPO",
                "seed": args.seed, "source_revision": "4ab9854dea3d84e38a59c25e0f5835a00cf8225f",
                "learner_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "instrumentation": "Save per-rollout actions/observations/metrics and finite checks; atomically publish parameters/progress after each update; display global step count; no algorithm changes",
                "float64": args.float64,
                "learning_reward_scale": args.learning_reward_scale,
                "rollout_protocol": {"controls_per_environment": args.rollout_steps,
                    "global_batch_size": 4 * args.rollout_steps,
                    "minibatch_size": 4 * args.rollout_steps // 8,
                    "public_example_controls_per_environment": 40,
                    "caveat": "The early rollout10 reconstruction uses minibatches5; the public example's rollout40 uses minibatches20"},
                "gradient_reward": {"enabled": args.gradient_reward, "source_sha256": gradient_reward_sha,
                    "formula": "optimizer reward = physical r + dot(dr/da, applied a)" if args.gradient_reward else None,
                    "coefficient": 1.0 if args.gradient_reward else None,
                    "caveat": "Supplement Eq.34 alternative; not recovered original adaptive Eq.32 GPPO"},
                "return_units": "unscaled, unaugmented physical reward, never optimizer-side scaled/shaped reward",
                "devices": str(jax.devices()), "runtime_bootstrap_sha256": runtime_sha,
                "runtime_environment": {key: os.environ.get(key) for key in
                    ("JAX_PLATFORMS", "XLA_FLAGS", "JAX_DEFAULT_MATMUL_PRECISION", "LD_LIBRARY_PATH")},
                "solver_corrections": solver_audit,
                "imported_cfd_sources": source_audit,
                "related_author_repository": "https://github.com/smokbel/Controlling-Kolmogorov-Flow",
                "related_author_revision": "45563a836b2eb3ad8ba319ec873b7b2df4802585",
                "config": config, "physics": {"reynolds": 250, "grid_size": [64, 64],
                  "dt": 0.001, "action_time": 10.0, "horizon_actions": 10,
                  "control_wavenumbers": [4,5,6,7], "action_bounds": [0,1],
                  "reward": "mean_tke - 75 * sum(abs(action))", "observation_size": 64},
                "versions": {name: version(name) for name in ("jax", "jaxlib", "flax", "gymnax", "optax")},
                "unresolved": ["exact Figure 3b configuration and reward aggregation",
                               "GPPO implementation/gradient mixing weight"]}
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    started = time.monotonic()
    try:
        out = jax.jit(module.make_train(config))(jax.random.PRNGKey(args.seed))
        jax.block_until_ready(out)
    except Exception as error:
        (args.output / "error.json").write_text(json.dumps({"status": "numerical-or-runtime-failure",
            "error": str(error), "wall_seconds": time.monotonic() - started}, indent=2) + "\n")
        raise
    state = out["runner_state"][0]
    if int(state.step) != args.steps // (4 * args.rollout_steps) * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]:
        raise RuntimeError("Unexpected PPO optimizer-update count")
    if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(state.params)):
        raise RuntimeError("Nonfinite trained PPO parameters")
    network = module.ActorCritic(4, activation=config["ACTIVATION"])
    _, initial_key = jax.random.split(jax.random.PRNGKey(args.seed))
    initial = network.init(initial_key, jnp.zeros((64,)))
    differences = jax.tree_util.tree_map(lambda a, b: jnp.sum(jnp.square(a - b)), state.params, initial)
    weight_change = float(jnp.sqrt(sum(jax.tree_util.tree_leaves(differences))))
    if not np.isfinite(weight_change) or weight_change <= 0:
        raise RuntimeError("No finite parameter change from initialization")
    (args.output / "params.msgpack").write_bytes(flax.serialization.to_bytes(state.params))
    metrics = {key: np.asarray(value) for key, value in out["metrics"].items()}
    np.savez_compressed(args.output / "metrics.npz", **metrics)
    mask = metrics["returned_episode"].astype(bool)
    returns = metrics["returned_episode_returns"][mask]
    if not len(returns) or not np.isfinite(returns).all():
        raise RuntimeError("No finite completed training episodes")
    result = {"status": "trained-native-ppo-not-paper-matched", "training_steps": args.steps,
              "gradient_updates": int(state.step), "training_episodes": int(mask.sum()),
              "parameter_delta_l2": weight_change,
              "last_4_training_returns": returns[-4:].tolist(),
              "wall_seconds": time.monotonic() - started,
              "checkpoint_sha256": hashlib.sha256((args.output / "params.msgpack").read_bytes()).hexdigest()}
    (args.output / "completed.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
