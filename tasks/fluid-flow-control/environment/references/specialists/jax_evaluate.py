"""Independent frozen-checkpoint/zero-control JAX evaluation, without training."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from jax_source import verify


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=Path(os.environ.get("HYDROGYM_UPSTREAM_SOURCE", "/upstream")))
    parser.add_argument("--output", type=Path, required=True)
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--checkpoint", type=Path)
    policy.add_argument("--zero", action="store_true")
    policy.add_argument("--package", type=Path, help="Relocate and replay a resident policy via the production JSON-lines transport")
    parser.add_argument("--evaluator-code", type=Path)
    args = parser.parse_args()
    cfg = json.loads((args.run / "config.json").read_text())
    jax.config.update("jax_enable_x64", cfg["float64"])
    source_audit = verify()
    if "imported_cfd_sources" not in cfg or source_audit["revision"] != cfg["imported_cfd_sources"]["revision"]:
        raise RuntimeError("Training source was not verified; excluded legacy diagnostics cannot be evaluated")
    correction = cfg["solver_corrections"]["mode"]
    if correction != "none":
        path = args.run / "jax_corrections.py"
        if hashlib.sha256(path.read_bytes()).hexdigest() != cfg["solver_corrections"]["source_sha256"]:
            raise RuntimeError("Training numerical-correction source changed")
        load_module("frozen_solver_corrections", path).apply_corrections(correction)
    frozen = load_module("frozen_hydrogym_reconstruction", args.run / "source.py")
    physics = cfg["physics"]
    env = frozen.MixingFlow(env_config={}, flow_config={"Re": physics["reynolds"],
        "grid_size": tuple(physics["grid_size"])})
    params = env.default_params
    if (float(params.dt) != physics["dt"] or float(params.action_time) != physics["action_time"]
            or int(params.max_episode_steps) != physics["horizon_actions"]):
        raise RuntimeError("Frozen evaluator physics differs from training")
    checkpoint_hash = None
    inference = None
    if args.checkpoint:
        learner = args.upstream / "examples/jax/getting_started/3_ppo/run_ppo.py"
        if hashlib.sha256(learner.read_bytes()).hexdigest() != cfg["learner_source_sha256"]:
            raise RuntimeError("Public actor implementation changed")
        module = load_module("frozen_public_actor", learner)
        network = module.ActorCritic(4, activation=cfg["config"]["ACTIVATION"])
        template = network.init(jax.random.PRNGKey(0), jnp.zeros((physics["observation_size"],)))
        payload = args.checkpoint.read_bytes()
        checkpoint_hash = hashlib.sha256(payload).hexdigest()
        weights = flax.serialization.from_bytes(template, payload)
        if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(weights)):
            raise RuntimeError("Nonfinite policy checkpoint")
        inference = jax.jit(lambda obs: network.apply(weights, obs)[0].mean())

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    started = time.monotonic()
    key = jax.random.PRNGKey(cfg["seed"])
    obs, state = jax.jit(env.reset_env)(key, params)
    obs.block_until_ready()
    step_fn = jax.jit(env.step_env)
    records = []
    resident = None
    package_digest = None
    try:
        if args.package:
            if args.evaluator_code is None:
                raise ValueError("Resident replay requires --evaluator-code")
            sys.path.insert(0, str(args.evaluator_code))
            from protocol import Policy
            from evaluate import package_hash
            package_digest = package_hash(args.package)
            snapshot = args.output / "relocated-policy"
            shutil.copytree(args.package, snapshot)
            if package_hash(snapshot) != package_digest:
                raise RuntimeError("Reference package changed during relocation")
            resident = Policy(snapshot, args.output / "policy.stderr", timeout=120)
            resident.reset({"environment": "Kolmogorov_2D_Re250", "family": "kolmogorov", "seed": cfg["seed"]},
                           (physics["observation_size"],), (4,), [float(params.min_action)] * 4, [float(params.max_action)] * 4)
        with (args.output / "trajectory.jsonl").open("w", buffering=1) as stream:
            for step in range(physics["horizon_actions"]):
                before = np.asarray(obs).tolist()
                if resident:
                    raw_action = jnp.asarray(resident.act(before), dtype=obs.dtype)
                else:
                    raw_action = jnp.zeros((4,)) if args.zero else inference(obs)
                action = jnp.clip(raw_action, params.min_action, params.max_action)
                if not np.isfinite(np.asarray(action)).all():
                    raise RuntimeError("Nonfinite frozen policy action")
                obs, state, reward, done, info = step_fn(key, state, action, params)
                obs.block_until_ready()
                if not np.isfinite(np.asarray(obs)).all() or not np.isfinite(float(reward)):
                    raise RuntimeError("Nonfinite frozen-policy CFD evaluation")
                if bool(done) != (step + 1 == physics["horizon_actions"]):
                    raise RuntimeError("Wrong JAX evaluation horizon")
                record = {"step": step + 1, "action": np.asarray(action).tolist(),
                          "reward": float(reward), "tke": float(info["mean_tke"]),
                          "observation": before, "next_observation": np.asarray(obs).tolist()}
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                records.append(record)
                print(json.dumps({k: record[k] for k in ("step", "action", "reward", "tke")}), flush=True)
        if resident:
            resident.close()
            if package_hash(snapshot) != package_digest:
                raise RuntimeError("Resident package mutated during evaluation")
    finally:
        if resident:
            resident.abort()
    if args.checkpoint and hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != checkpoint_hash:
        raise RuntimeError("Checkpoint changed during independent evaluation; use an immutable snapshot")
    result = {"status": "frozen-JAX-reconstruction-evaluated-not-paper-matched",
        "policy": "resident-JSON-lines" if args.package else ("zero" if args.zero else "deterministic-PPO-mean"),
        "package_hash": package_digest,
        "devices": str(jax.devices()), "float64": cfg["float64"],
        "runtime_environment": {key: os.environ.get(key) for key in
            ("JAX_PLATFORMS", "XLA_FLAGS", "JAX_DEFAULT_MATMUL_PRECISION")},
        "checkpoint_sha256": checkpoint_hash, "training_run": str(args.run),
        "steps": len(records), "return": sum(r["reward"] for r in records),
        "mean_tke": float(np.mean([r["tke"] for r in records])),
        "wall_seconds": time.monotonic() - started, "imported_cfd_sources": source_audit,
        "solver_corrections": cfg["solver_corrections"], "physics": physics,
        "comparison_caveat": "Deterministic frozen replay is not the stochastic training-return curve in Fig.3b"}
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
