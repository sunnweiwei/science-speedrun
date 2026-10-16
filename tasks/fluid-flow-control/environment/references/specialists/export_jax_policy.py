"""Export a trained public JAX actor to the resident policy contract."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from jax_evaluate import load_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Never replace an existing exported reference")
    cfg = json.loads((args.run / "config.json").read_text())
    if cfg["config"]["NORMALIZE_ENV"]:
        raise ValueError("This exporter does not have normalized-environment statistics")
    jax.config.update("jax_enable_x64", cfg["float64"])
    source = args.upstream / "examples/jax/getting_started/3_ppo/run_ppo.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != cfg["learner_source_sha256"]:
        raise ValueError("Author actor source changed")
    original = load_module("public_actor", source).ActorCritic(4, cfg["config"]["ACTIVATION"])
    payload = args.checkpoint.read_bytes()
    weights = flax.serialization.msgpack_restore(payload)
    pure_weights = {"params": {name: weights["params"][name] for name in ("Dense_0", "Dense_1", "Dense_2")}}
    pure = load_module("standalone_actor", args.predictor).Actor(4, cfg["config"]["ACTIVATION"])
    low, high = cfg["physics"]["action_bounds"]
    generator = np.random.default_rng(991)
    observations = generator.normal(size=(64, cfg["physics"]["observation_size"])) * 3
    observations[0] = 0
    observations[1] = 1
    observations = jnp.asarray(observations, dtype=jnp.float64 if cfg["float64"] else jnp.float32)
    original_infer = jax.jit(lambda obs: jnp.clip(original.apply(weights, obs)[0].mean(), low, high))
    pure_infer = jax.jit(lambda obs: jnp.clip(pure.apply(pure_weights, obs), low, high))
    expected = np.stack([np.asarray(original_infer(obs)) for obs in observations])
    actual = np.stack([np.asarray(pure_infer(obs)) for obs in observations])
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    if not np.isfinite(actual).all():
        raise ValueError("Nonfinite exported reference")
    model_bytes = flax.serialization.to_bytes(pure_weights)
    restored = flax.serialization.msgpack_restore(model_bytes)
    np.testing.assert_allclose(np.stack([np.asarray(jnp.clip(pure.apply(restored, obs), low, high))
                                        for obs in observations]), expected, atol=2e-6, rtol=2e-6)
    metadata = {"format": "hydrogym-reference-flax-actor/v1", "scope": "per-environment-reference-not-shared-submission",
        "paper_performance_status": "not-yet-matched", "algorithm": cfg["algorithm"],
        "seed": cfg["seed"], "source_checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "activation": cfg["config"]["ACTIVATION"], "float64": cfg["float64"],
        "observation_size": cfg["physics"]["observation_size"], "action_size": 4,
        "action_low": low, "action_high": high, "training_config": cfg,
        "export_test": {"observations": 64, "max_abs_error": float(np.max(np.abs(actual - expected)))}}
    args.output.mkdir(parents=True)
    (args.output / "model.msgpack").write_bytes(model_bytes)
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    shutil.copyfile(args.predictor, args.output / "predict.py")
    (args.output / "infer.sh").write_text('#!/usr/bin/env bash\nset -euo pipefail\nroot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\nexport JAX_PLATFORMS=cpu\nexport OMP_NUM_THREADS=1\nexec python "$root/predict.py"\n')
    (args.output / "README.md").write_text(
        "# Exported JAX reference actor\n\nA per-environment comparison policy, not the required shared-policy submission. "
        "Paper performance is not established. Only actor parameters are packaged: no optimizer, critic, CFD solver "
        "or author checkout is needed for inference. Requires the declared Python/JAX/Flax runtime. The resident "
        "policy emits actions under hydrogym-policy/v1; only the external environment computes scores.\n")
    if args.checkpoint.read_bytes() != payload:
        raise ValueError("Checkpoint changed during export; use an immutable snapshot")
    print(json.dumps({key: metadata[key] for key in ("model_sha256", "export_test")}))


if __name__ == "__main__":
    main()
