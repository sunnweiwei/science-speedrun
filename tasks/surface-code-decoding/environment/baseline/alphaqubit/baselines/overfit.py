"""Bounded training-only gradient/target diagnostic; not a scientific score."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import time
from typing import Any

import numpy as np
import stim

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.data import CircuitSampler, group_conditions
from baselines.neural import EXPECTED_PARAMETER_COUNTS, UPSTREAM_COMMIT, build_model, model_config
from baselines.observables import ObservableTransform
from baselines.predict import atomic_json, sha256_file


GROUP = "d3_bX_center_3_5"
NUM_SHOTS = 128
BATCH_SIZE = 64
MAXIMUM_STEPS = 1000


def training_metrics(model: Any, features: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, float]:
    """Evaluate both heads deterministically on the same 128 training shots."""
    outputs = model(features, training=False)
    named = outputs if isinstance(outputs, dict) else dict(zip(model.output_names, outputs))
    actual = labels.astype(np.float64)
    result: dict[str, float] = {}
    for head in ("main_output", "aux_output"):
        probabilities = np.asarray(named[head], dtype=np.float64)
        if probabilities.shape != actual.shape or not np.isfinite(probabilities).all():
            raise ValueError(f"Invalid diagnostic probabilities in {head}")
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError(f"Diagnostic probabilities outside [0,1] in {head}")
        clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
        prefix = head.removesuffix("_output")
        result[f"{prefix}_bce"] = float(np.mean(-actual * np.log(clipped) - (1 - actual) * np.log1p(-clipped)))
        result[f"{prefix}_train_shot_error"] = float(np.mean((probabilities >= 0.5) != actual))
    result["weighted_bce"] = result["main_bce"] + 0.5 * result["aux_bce"]
    return result


def overfit_gate(before: dict[str, float], after: dict[str, float]) -> dict[str, Any]:
    reduction = 1 - after["main_bce"] / before["main_bce"] if before["main_bce"] > 0 else 0.0
    return {
        "passed": bool(reduction >= 0.5 and after["main_train_shot_error"] <= 0.05),
        "main_bce_relative_reduction": reduction,
        "required_main_bce_relative_reduction": 0.5,
        "maximum_main_train_shot_error": 0.05,
        "interpretation": "training-only gradient and target sanity evidence; not validation or benchmark accuracy",
    }


def run_overfit(data_root: Path, output: Path, max_steps: int = 1000, allow_cpu: bool = False, seed: int = 0) -> dict[str, Any]:
    if not 1 <= max_steps <= MAXIMUM_STEPS:
        raise ValueError(f"max_steps must be between 1 and {MAXIMUM_STEPS}")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    started = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": 1, "status": "running",
        "purpose": "training_only_gradient_and_target_sanity",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "group": GROUP, "distance": 3, "rounds": 3, "shots": NUM_SHOTS,
            "batch_size": BATCH_SIZE, "max_steps": max_steps, "model_seed": seed,
            "sampler_seed": seed + 1, "noise_source": "released_noisy_circuit",
            "label_mode": "qrennd_global_data_parity", "allow_cpu": allow_cpu,
            "batch_schedule": "alternate fixed shots 0:64 and 64:128; no resampling",
            "evaluation_interval_steps": 25, "upstream_commit": UPSTREAM_COMMIT,
            "native_model": model_config(distance=3),
        },
        "history": [], "steps_completed": 0,
        "data_access": "ideal/noisy circuits only; no hardware shots or label files read",
    }
    atomic_json(output, result)
    try:
        import tensorflow as tf

        devices = tf.config.list_physical_devices("GPU")
        result["runtime"] = {
            "hostname": socket.gethostname(), "platform": platform.platform(),
            "python": platform.python_version(), "tensorflow": tf.__version__,
            "numpy": np.__version__, "stim": stim.__version__,
            "tensorflow_build": tf.sysconfig.get_build_info(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpus": [{"name": device.name, **tf.config.experimental.get_device_details(device)} for device in devices],
            "cpu_diagnostic": not bool(devices),
        }
        if not devices and not allow_cpu:
            raise RuntimeError("GPU required for this diagnostic; use --allow-cpu only for explicit CPU smoke checks")
        for device in devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except RuntimeError:
                pass  # The function can also be called by an in-process test.
        condition = next(item for item in group_conditions(data_root)[GROUP] if item.rounds == 3)
        sampler = CircuitSampler(condition, seed=seed + 1, noise_source="circuit")
        packed, path_labels = sampler.sample_packed(NUM_SHOTS)
        transform = ObservableTransform.from_condition(condition)
        labels = transform.path_to_global_packed(packed, path_labels).astype(np.float32)
        features = sampler.layout.decode_packed_features(packed)
        # Hash the exact packed inputs and targets, including dtype/shape in
        # metadata, so this diagnostic's fixed population can be reconstructed.
        digest = hashlib.sha256()
        digest.update(packed.tobytes(order="C"))
        digest.update(labels.astype(np.uint8).tobytes(order="C"))
        _, inverse = np.unique(packed, axis=0, return_inverse=True)
        positive = np.bincount(inverse, weights=labels[:, 0])
        total = np.bincount(inverse)
        result["sample"] = {
            "sha256_packed_detectors_then_uint8_global_labels": digest.hexdigest(),
            "packed_detector_shape": list(packed.shape), "global_label_shape": list(labels.shape),
            "global_positive_shots": int(labels.sum()), "path_positive_shots": int(path_labels.sum()),
            "unique_detector_patterns": len(total),
            "minimum_error_due_to_conflicting_identical_inputs": float(np.minimum(positive, total - positive).sum() / NUM_SHOTS),
            "ideal_circuit_sha256": sha256_file(condition.path / "circuit_ideal.stim"),
            "noisy_circuit_sha256": sha256_file(condition.path / "circuit_noisy.stim"),
            "observable_transform": transform.metadata(),
        }
        model = build_model(distance=3, seed=seed)
        if model.count_params() != EXPECTED_PARAMETER_COUNTS[3]:
            raise ValueError("Diagnostic did not construct the native d3 paper architecture")
        result["parameters"] = model.count_params()
        result["before"] = training_metrics(model, features, labels)
        result["history"].append({"step": 0, **result["before"]})
        training_started = time.perf_counter()
        for step in range(1, max_steps + 1):
            first = ((step - 1) % 2) * BATCH_SIZE
            batch = slice(first, first + BATCH_SIZE)
            batch_inputs = {key: values[batch] for key, values in features.items()}
            batch_labels = {"main_output": labels[batch], "aux_output": labels[batch]}
            losses = model.train_on_batch(batch_inputs, batch_labels, return_dict=True)
            if not all(np.isfinite(value) for value in losses.values()):
                raise FloatingPointError(f"Nonfinite native training loss at step {step}")
            result["steps_completed"] = step
            if step % 25 == 0 or step == max_steps:
                result["after"] = training_metrics(model, features, labels)
                result["gate"] = overfit_gate(result["before"], result["after"])
                result["history"].append({"step": step, **result["after"]})
                print(json.dumps({"stage": "overfit_diagnostic", "step": step,
                                  **result["after"], "gate_passed": result["gate"]["passed"]}), flush=True)
                atomic_json(output, result)
                if result["gate"]["passed"]:
                    break
        result["training_seconds"] = time.perf_counter() - training_started
        result["optimizer_iterations"] = int(model.optimizer.iterations.numpy())
        result["status"] = "complete"
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Machine-readable JSON file")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = run_overfit(args.data_root, args.output, args.max_steps, args.allow_cpu, args.seed)
    print(json.dumps({"status": result["status"], "gate": result["gate"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
