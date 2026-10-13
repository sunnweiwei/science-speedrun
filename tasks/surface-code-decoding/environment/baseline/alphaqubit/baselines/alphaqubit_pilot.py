"""Bounded real-data forward/backward throughput probe for AlphaQubit."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import time

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .alphaqubit_data import AlphaQubitSampler
from .alphaqubit_model import AlphaQubitConfig, loss_terms, transformed_model
from .cache import atomic_json
from .data import group_conditions


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--overfit", action="store_true", help="Repeat one fixed synthetic batch as a training diagnostic")
    a = p.parse_args()
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
               for name in ("alphaqubit_model.py", "alphaqubit_data.py", "alphaqubit_pilot.py")}
    devices = jax.devices()
    if not a.allow_cpu and not all(d.platform == "gpu" for d in devices):
        raise RuntimeError(f"Expected GPU device, got {devices}")
    condition = next(c for c in group_conditions(a.data_root)[a.group] if c.rounds == a.rounds)
    sampler = AlphaQubitSampler(condition, seed=401)
    features, labels = sampler.sample(a.batch_size)
    features, labels = jax.device_put(features), jax.device_put(labels)
    config = AlphaQubitConfig(condition.distance, dtype=a.dtype)
    model = transformed_model(config, sampler.layout.metadata())
    started = time.monotonic()
    params = jax.jit(model.init)(jax.random.PRNGKey(42), features)
    jax.block_until_ready(params)
    init_seconds = time.monotonic() - started
    assert all(v.dtype == jnp.float32 for v in jax.tree.leaves(params)), "Master parameters must be float32"
    optimizer = optax.lamb(3e-4, b2=0.95, weight_decay=1e-5)
    state = optimizer.init(params)

    @jax.jit
    def update(parameters, optimizer_state, inputs, targets):
        def objective(weights):
            return loss_terms(model.apply(weights, inputs), inputs, targets)
        (_, metrics), grads = jax.value_and_grad(objective, has_aux=True)(parameters)
        updates, optimizer_state = optimizer.update(grads, optimizer_state, parameters)
        parameters = optax.apply_updates(parameters, updates)
        return parameters, optimizer_state, metrics, optax.global_norm(grads)

    started = time.monotonic()
    params, state, metrics, grad_norm = update(params, state, features, labels)
    jax.block_until_ready(metrics)
    compile_seconds = time.monotonic() - started
    first_metrics = {k: float(v) for k, v in metrics.items()}
    initial_grad = float(grad_norm)
    durations = []
    for _ in range(a.steps):
        if not a.overfit:
            features, labels = sampler.sample(a.batch_size)
        started = time.monotonic()
        params, state, metrics, grad_norm = update(params, state, features, labels)
        jax.block_until_ready(metrics)
        durations.append(time.monotonic() - started)
        if a.overfit and float(metrics["main_error"]) <= .02 and float(metrics["main_loss"]) < .1:
            break
    latest = {k: float(v) for k, v in metrics.items()}
    if not all(np.isfinite(v) for v in (*first_metrics.values(), *latest.values(), initial_grad, float(grad_norm))):
        raise RuntimeError("Nonfinite losses or gradients")
    memory = devices[0].memory_stats() or {}
    result = dict(config=config.to_dict(), group=a.group, rounds=a.rounds, batch_size=a.batch_size,
                  parameters=hk.data_structures.tree_size(params), master_dtype="float32",
                  first_metrics=first_metrics, latest_metrics=latest, gradient_norm=float(grad_norm),
                  initialization_seconds=init_seconds, first_update_compile_seconds=compile_seconds,
                  median_step_seconds=float(np.median(durations)), steady_steps=len(durations),
                  samples_per_second=a.batch_size / float(np.median(durations)),
                  timing_scope="one-device compute plus host feature transfer; fresh simulation generation excluded",
                  memory={str(k): int(v) for k, v in memory.items()},
                  devices=[str(d) for d in devices], jax_version=jax.__version__,
                  gpu=subprocess.getoutput("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"),
                  source_sha256=sources,
                  fixed_batch_training_diagnostic=a.overfit,
                  overfit_passed=(float(metrics["main_error"]) <= .02 and float(metrics["main_loss"]) < .1) if a.overfit else None,
                  quality_claim="throughput and finite-gradient probe only; no convergence or held-out experiment score")
    atomic_json(a.output, result)
    print(json.dumps(result), flush=True)
    if a.overfit and not result["overfit_passed"]:
        raise RuntimeError("Fixed-batch overfit gate did not pass within bounded steps")


if __name__ == "__main__":
    main()
