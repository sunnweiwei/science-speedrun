"""Replay recorded PPO actions without learning to diagnose native CFD failures."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--env-index", type=int, choices=range(4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--solver-corrections", choices=["none", "dealias", "dealias-curl"], default="none")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import jax_source
    source_audit = jax_source.verify()
    jax.config.update("jax_enable_x64", args.float64)
    from jax_public import MixingFlow
    audit = {"mode": "none"}
    if args.solver_corrections != "none":
        import jax_corrections
        audit = jax_corrections.apply_corrections(args.solver_corrections)
        (args.output / "jax_corrections.py").write_bytes(Path(jax_corrections.__file__).read_bytes())
    for filename in ("jax_action_replay.py", "jax_public.py", "jax_source.py"):
        (args.output / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    trace = np.load(args.rollout)
    # A rollout starts at a common deterministic native reset after ten controls.
    # Match actual applied controls, not unconstrained Gaussian policy samples.
    controls = np.clip(trace["raw_actions"][:, args.env_index], 0, 1)
    dtype = jnp.float64 if args.float64 else jnp.float32
    env = MixingFlow(env_config={}, flow_config={"Re": 250, "grid_size": (64, 64)})
    params, key = env.default_params, jax.random.PRNGKey(30)
    started = time.monotonic()
    observation, state = jax.jit(env.reset_env)(key, params)
    observation.block_until_ready()
    step_fn = jax.jit(env.step_env)
    result = {"status": "recorded-actions-CFD-diagnostic-not-a-score",
        "rollout_sha256": hashlib.sha256(args.rollout.read_bytes()).hexdigest(),
        "env_index": args.env_index, "float64": args.float64, "solver_corrections": audit,
        "imported_cfd_sources": source_audit,
        "source": str(args.rollout), "records": []}
    for index, action in enumerate(controls):
        if not np.isfinite(action).all():
            result["stopped_at_nonfinite_recorded_action"] = index + 1
            break
        observation, state, reward, done, info = step_fn(key, state, jnp.asarray(action, dtype=dtype), params)
        observation.block_until_ready()
        finite = bool(np.isfinite(np.asarray(observation)).all() and np.isfinite(float(reward)))
        record = {"step": index + 1, "action": action.tolist(), "finite": finite,
                  "tke": float(info["mean_tke"]) if finite else None,
                  "reward": float(reward) if finite else None,
                  "observation_max": float(jnp.max(observation)) if finite else None,
                  "wall_seconds": time.monotonic() - started}
        result["records"].append(record)
        print(json.dumps(record, allow_nan=False), flush=True)
        if not finite:
            break
    result["wall_seconds"] = time.monotonic() - started
    (args.output / "diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
