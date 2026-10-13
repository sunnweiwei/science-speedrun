"""Replay packaged AlphaQubit weights from detector-only Sycamore inputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.alphaqubit_artifact import (GEOMETRY_KEYS, INFERENCE_SOURCES, LABEL_MODE, atomic_bytes, atomic_json,
                                          load_model, sha256_file, verify_package)
from baselines.alphaqubit_data import AlphaQubitLayout
from baselines.data import Condition, group_conditions, hardware_split_indices


def predict_condition(apply: Any, parameters: Any, geometry: dict[str, Any], condition: Condition,
                      split: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Read only ideal-circuit metadata and detector bytes; never read labels."""
    if split not in ("validation", "test") or batch_size <= 0:
        raise ValueError("Require validation/test split and positive batch size")
    layout = AlphaQubitLayout.from_condition(condition)
    actual_geometry = layout.metadata()
    if any(geometry.get(key) != actual_geometry[key] for key in GEOMETRY_KEYS):
        raise ValueError(f"Checkpoint/input geometry mismatch: {condition.name}")
    width = (layout.num_detectors + 7) // 8
    path = condition.path / "detection_events.b8"
    if path.stat().st_size != 50_000 * width:
        raise ValueError(f"{condition.name}: expected exactly 50,000 detector shots")
    packed = np.memmap(path, dtype=np.uint8, mode="r", shape=(50_000, width))
    indices = hardware_split_indices(split).astype(np.uint64)
    result = np.empty(len(indices), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        batch = np.asarray(packed[indices[start:stop]])
        if len(batch) < batch_size:
            batch = np.pad(batch, ((0, batch_size - len(batch)), (0, 0)))
        probabilities = np.asarray(apply(parameters, layout.decode_packed_features(batch)), dtype=np.float32)
        if probabilities.shape != (batch_size,) or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("Model must return finite [batch] probabilities in [0,1]")
        result[start:stop] = probabilities[:stop - start]
    return indices, result


def _write_prediction(path: Path, indices: np.ndarray, probabilities: np.ndarray) -> None:
    payload = io.BytesIO()
    np.savez(payload, shot_indices=indices, probabilities=probabilities)
    atomic_bytes(path, payload.getvalue())


def run_prediction(package: Path, data_root: Path, split: str, output: Path,
                   group: str | None = None, stage: str | None = None,
                   batch_size: int = 256, require_gpu: bool = False) -> dict[str, Any]:
    """Verify a package and emit scorer-compatible provenance for each group."""
    manifest = verify_package(package)
    for name in INFERENCE_SOURCES:
        if sha256_file(Path(__file__).with_name(name)) != manifest["files"][f"baselines/{name}"]["sha256"]:
            raise ValueError("Executing inference source differs from the package; run from its relocated root")
    if split not in ("validation", "test") or batch_size <= 0:
        raise ValueError("Require validation/test split and positive batch size")
    if output.resolve().is_relative_to(package.resolve()):
        raise ValueError("Prediction output must be outside the immutable package")
    if stage is None:
        if len(manifest["stages"]) != 1:
            raise ValueError("Choose --stage when package contains both stages")
        stage = manifest["stages"][0]
    if stage not in manifest["stages"] or (group is not None and group not in manifest["groups"]):
        raise ValueError("Requested group/stage is not packaged")
    selected_groups = [group] if group else manifest["groups"]
    groups = group_conditions(data_root)
    for name in selected_groups:
        if name not in groups or [condition.rounds for condition in groups[name]] != list(range(1, 26, 2)):
            raise ValueError(f"Inference requires all 13 round conditions for {name}")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Prediction output must be empty to prevent mixing runs")

    import haiku as hk
    import jax
    import jax.numpy as jnp
    import stim

    devices = jax.devices()
    if require_gpu and not any(device.platform == "gpu" for device in devices):
        raise RuntimeError("--require-gpu requested but JAX has no visible GPU")
    output.mkdir(parents=True, exist_ok=True)
    completed: dict[str, Any] = {}
    started = time.perf_counter()
    for name in selected_groups:
        record = manifest["models"][f"{name}/{stage}"]
        model, parameters, checkpoint = load_model(package / record["model"], package / record["checkpoint_metadata"])
        if checkpoint["group"] != name or checkpoint["stage"] != stage:
            raise ValueError("Checkpoint group/stage identity mismatch")
        parameters = jax.device_put(parameters)
        apply = jax.jit(lambda params, features: jax.nn.sigmoid(model.apply(params, features)["logits"].astype(jnp.float32)))
        folder = output if group is not None else output / name
        folder.mkdir(parents=True, exist_ok=True)
        metadata: dict[str, Any] = {
            "schema_version": 1, "status": "running", "group": name, "stage": stage, "split": split,
            "started_utc": datetime.now(timezone.utc).isoformat(), "model_file": "model.npz",
            "model_sha256": checkpoint["checkpoint_sha256"], "package_manifest_sha256": sha256_file(package / "manifest.json"),
            "checkpoint_metadata": {"model_sha256": checkpoint["checkpoint_sha256"], "label_mode": LABEL_MODE,
                "group": name, "distance": checkpoint["distance"], "identity_sha256": checkpoint["identity_sha256"],
                "metadata_sha256": sha256_file(package / record["checkpoint_metadata"])},
            "model_label_mode": LABEL_MODE, "prediction_label_mode": LABEL_MODE,
            "input_contract": "detectors and ideal circuit only; no label files read; direct official-path probability",
            "batch_size": batch_size, "jax_version": jax.__version__, "haiku_version": hk.__version__,
            "numpy_version": np.__version__, "stim_version": stim.__version__, "python_version": platform.python_version(),
            "hardware": {"host": socket.gethostname(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
                "devices": [{"platform": device.platform, "kind": device.device_kind, "id": device.id} for device in devices]},
            "conditions": {},
        }
        atomic_json(folder / "metadata.json", metadata)
        group_started = time.perf_counter()
        try:
            for condition in groups[name]:
                condition_started = time.perf_counter()
                inputs = {"detectors_sha256": sha256_file(condition.path / "detection_events.b8"),
                          "circuit_sha256": sha256_file(condition.path / "circuit_ideal.stim")}
                indices, probabilities = predict_condition(apply, parameters, checkpoint["geometry"], condition, split, batch_size)
                if inputs != {"detectors_sha256": sha256_file(condition.path / "detection_events.b8"),
                              "circuit_sha256": sha256_file(condition.path / "circuit_ideal.stim")}:
                    raise ValueError("Inference input changed during prediction")
                destination = folder / f"{condition.name}.npz"
                _write_prediction(destination, indices, probabilities)
                metadata["conditions"][condition.name] = {"shots": len(indices), "prediction_sha256": sha256_file(destination),
                    **inputs, "seconds_including_input_and_write": time.perf_counter() - condition_started}
            metadata["status"] = "complete"
        except Exception as error:
            metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            metadata["elapsed_seconds"] = time.perf_counter() - group_started
            metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(folder / "metadata.json", metadata)
        completed[name] = metadata
    return {"status": "complete", "stage": stage, "groups": completed, "elapsed_seconds": time.perf_counter() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group")
    parser.add_argument("--stage", choices=("pretrain", "finetune"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    result = run_prediction(args.package, args.data_root, args.split, args.output, args.group, args.stage, args.batch_size, args.require_gpu)
    print(json.dumps({"status": result["status"], "groups": list(result["groups"]), "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
