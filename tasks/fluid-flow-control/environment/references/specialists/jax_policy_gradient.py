"""Check analytic gradients through a frozen actor AND the native CFD trajectory.

This is a prerequisite diagnostic for reconstructing supplementary Eq.32, not a
GPPO trainer or paper-performance result. No policy weights are updated here.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import flax.serialization
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from jax_evaluate import load_module
from jax_source import verify


def trajectory_objective(env, network, env_params, initial_obs, initial_state,
                         weights, controls, key):
    """No stop_gradient or Gym auto-reset: all policy/flow dependencies survive."""
    def step(carry, _):
        obs, state = carry
        action = jnp.clip(network.apply(weights, obs)[0].mean(),
                          env_params.min_action, env_params.max_action)
        obs, state, reward, _, info = env.step_env(key, state, action, env_params)
        return (obs, state), (reward, info["mean_tke"], action)

    _, (rewards, tke, actions) = jax.lax.scan(jax.checkpoint(step),
        (initial_obs, initial_state), None, length=controls)
    # TKE-only diagnostic proves sensitivity of the physical dynamics, which
    # differentiating the direct action-cost term by itself cannot establish.
    return tke.sum(), (rewards, tke, actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path,
                        default=Path(os.environ.get("HYDROGYM_UPSTREAM_SOURCE", "/upstream")))
    parser.add_argument("--controls", type=int, choices=[1, 2, 10], default=2)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1e-5, 1e-6, 1e-7],
                        help="Decreasing perturbations; both finest checks must pass 2%%")
    args = parser.parse_args()
    if (len(args.epsilons) < 2 or not all(np.isfinite(x) and x > 0 for x in args.epsilons)
            or not all(a > b for a, b in zip(args.epsilons, args.epsilons[1:]))):
        parser.error("Use at least two strictly decreasing finite positive perturbations")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("jax_policy_gradient.py", "jax_evaluate.py", "jax_source.py"):
        shutil.copyfile(Path(__file__).with_name(name), args.output / name)
    config = json.loads((args.run / "config.json").read_text())
    source_audit = verify()
    if config["imported_cfd_sources"]["revision"] != source_audit["revision"]:
        raise ValueError("Training source does not match the audited solver")
    jax.config.update("jax_enable_x64", True)
    correction = config["solver_corrections"]
    if correction["mode"] != "none":
        path = args.run / "jax_corrections.py"
        if hashlib.sha256(path.read_bytes()).hexdigest() != correction["source_sha256"]:
            raise ValueError("Training correction source changed")
        load_module("frozen_correction", path).apply_corrections(correction["mode"])
    frozen = load_module("frozen_mixing_environment", args.run / "source.py")
    physics = config["physics"]
    env = frozen.MixingFlow(env_config={}, flow_config={"Re": physics["reynolds"],
        "grid_size": tuple(physics["grid_size"])})
    env_params = env.default_params
    if (env_params.dt != physics["dt"] or env_params.action_time != physics["action_time"]
            or env_params.max_episode_steps != physics["horizon_actions"]):
        raise ValueError("Frozen physics changed")
    # Pure rematerialization of identical RK arithmetic; no timestep/grid change.
    original_rk = env.integrator.RK4_CN
    env.integrator.RK4_CN = lambda control_field=None: jax.checkpoint(original_rk(control_field))
    actor_path = args.upstream / "examples/jax/getting_started/3_ppo/run_ppo.py"
    if hashlib.sha256(actor_path.read_bytes()).hexdigest() != config["learner_source_sha256"]:
        raise ValueError("Published actor source changed")
    actor = load_module("public_actor", actor_path).ActorCritic(4, activation=config["config"]["ACTIVATION"])
    template = actor.init(jax.random.PRNGKey(0), jnp.zeros(physics["observation_size"]))
    payload = args.checkpoint.read_bytes()
    weights = flax.serialization.from_bytes(template, payload)
    # FP64 is an explicit derivative-check choice, not the training precision.
    weights = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), weights)
    flat, unpack = ravel_pytree(weights)
    started = time.monotonic()
    key = jax.random.PRNGKey(config["seed"])
    obs, state = jax.jit(env.reset_env)(key, env_params)
    obs.block_until_ready()
    print("RESET", time.monotonic() - started, flush=True)

    def objective(parameters):
        return trajectory_objective(env, actor, env_params, obs, state,
                                    unpack(parameters), args.controls, key)

    (value, auxiliary), gradient = jax.jit(jax.value_and_grad(objective, has_aux=True))(flat)
    gradient.block_until_ready()
    norm = float(jnp.linalg.norm(gradient))
    if not np.isfinite(norm) or norm <= 0 or not np.isfinite(float(value)):
        raise RuntimeError("No nonzero finite full-policy physical gradient")
    print("GRADIENT", json.dumps({"tke_sum": float(value), "gradient_norm": norm,
          "seconds": time.monotonic() - started}), flush=True)
    direction = gradient / norm
    forward = jax.jit(lambda parameters: objective(parameters)[0])
    forward_value = float(forward(flat))
    np.testing.assert_allclose(forward_value, float(value), rtol=1e-8, atol=1e-10)
    checks = []
    for epsilon in args.epsilons:
        numerical = float((forward(flat + epsilon * direction) - forward(flat - epsilon * direction)) / (2 * epsilon))
        checks.append({"epsilon": epsilon, "analytical": norm, "numerical": numerical,
                       "relative_error": abs(numerical - norm) / norm})
    result = {"status": "full-policy-CFD-gradient-diagnostic-not-GPPO-performance",
        "controls": args.controls, "tke_sum": float(value), "gradient_norm": norm,
        "finite_difference": checks,
        "passed": all(c["relative_error"] <= .02 for c in checks[-2:]),
        "acceptance": "Both finest requested central differences agree within2%; all coarser diagnostics retained",
        "forward_vs_valuegrad_absolute_error": abs(forward_value - float(value)),
        "rewards": np.asarray(auxiliary[0]).tolist(), "tke": np.asarray(auxiliary[1]).tolist(),
        "actions": np.asarray(auxiliary[2]).tolist(), "wall_seconds": time.monotonic() - started,
        "physics": physics, "float64_diagnostic": True, "backend": str(jax.devices()),
        "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
        "imported_cfd_sources": source_audit, "solver_corrections": correction,
        "memory_strategy": "jax.checkpoint on each control interval and each RK step; forward arithmetic unchanged",
        "limitations": ["deterministic mean policy, not stochastic GPPO training",
                        "original GPPO adaptive gradient weighting is not public"]}
    (args.output / "diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.checkpoint.read_bytes() != payload:
        raise RuntimeError("Frozen checkpoint changed")
    if not result["passed"]:
        raise RuntimeError("Full-policy directional derivative did not pass 2% finite-difference tolerance")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
