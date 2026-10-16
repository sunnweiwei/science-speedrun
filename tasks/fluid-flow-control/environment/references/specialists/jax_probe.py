"""Native differentiable-solver diagnostic; not a GPPO reproduction or score."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from hydrogym.jax.envs.kolmogorov import KolmogorovFlow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controls", type=int, choices=[1, 2], default=1)
    parser.add_argument("--reynolds", type=float, default=200)
    parser.add_argument("--solver-corrections", choices=["none", "dealias", "dealias-curl"], default="none")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    import jax_source
    source_audit = jax_source.verify()
    (args.output / "jax_source.py").write_bytes(Path(jax_source.__file__).read_bytes())
    started = time.monotonic()
    jax.config.update("jax_enable_x64", True)
    audit = {"mode": "none"}
    if args.solver_corrections != "none":
        import jax_corrections
        audit = jax_corrections.apply_corrections(args.solver_corrections)
        (args.output / "jax_corrections.py").write_bytes(Path(jax_corrections.__file__).read_bytes())
    env = KolmogorovFlow(env_config={}, flow_config={"Re": args.reynolds})
    params = env.default_params
    key = jax.random.PRNGKey(17)
    reset = jax.jit(env.reset_env)
    obs, state = reset(key, params)
    obs.block_until_ready()
    print("RESET", json.dumps({"seconds": time.monotonic() - started,
                               "observation_shape": list(obs.shape)}), flush=True)
    action = jnp.array([[0.13, 0.07, -0.04, 0.02], [0.09, -0.02, 0.03, 0.04]])[:args.controls]

    def energy(control):
        current = state
        total = 0.0
        for index in range(args.controls):
            _, current, _, _, info = env.step_env(key, current, control[index], params)
            total = total + info["mean_tke"]
        return total

    value_grad = jax.jit(jax.value_and_grad(energy))
    value, gradient = value_grad(action)
    gradient.block_until_ready()
    print("GRADIENT", json.dumps({"tke": float(value), "gradient": gradient.tolist(),
                                  "seconds": time.monotonic() - started}), flush=True)
    direction = jnp.array([[0.3, -0.2, 0.4, 0.1], [-0.1, 0.2, 0.1, -0.3]])[:args.controls]
    objective = jax.jit(energy)
    epsilon = 1e-4
    numerical = (objective(action + epsilon * direction) - objective(action - epsilon * direction)) / (2 * epsilon)
    analytical = jnp.sum(gradient * direction)
    if not np.isfinite(np.asarray(obs)).all() or not np.isfinite(np.asarray(gradient)).all():
        raise RuntimeError("Nonfinite native differentiable CFD")
    if float(jnp.linalg.norm(gradient)) == 0:
        raise RuntimeError("Solver objective has no action gradient")
    np.testing.assert_allclose(float(analytical), float(numerical), rtol=0.02, atol=1e-6)
    result = {"status": "native-cfd-gradient-diagnostic-not-paper-performance",
              "source_revision": "4ab9854dea3d84e38a59c25e0f5835a00cf8225f",
              "backend": str(jax.devices()), "reynolds": env.flow.Re,
              "grid_size": list(env.flow.grid_size), "dt": float(params.dt),
              "controls": args.controls, "solver_corrections": audit,
              "imported_cfd_sources": source_audit,
              "action_time": float(params.action_time), "reward_alpha": float(params.reward_alpha),
              "observation_shape": list(obs.shape), "action": action.tolist(),
              "tke_sum": float(value), "tke_gradient": gradient.tolist(),
              "directional_analytical": float(analytical), "directional_numerical": float(numerical),
              "wall_seconds": time.monotonic() - started,
              "versions": {name: version(name) for name in ("jax", "jaxlib", "flax", "gymnax")},
              "unresolved": ["Figure 3b Reynolds number/grid/configuration",
                             "public default reward_alpha=0 omits the mixing objective",
                             "original GPPO analytic-gradient weighting and rollout schedule"]}
    (args.output / "diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
