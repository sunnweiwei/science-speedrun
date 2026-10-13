"""Replay a portable Keras decoder using detector files only; never read labels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import tempfile
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

if __package__:
    from .data import Condition, DetectorLayout, group_conditions, hardware_split_indices
    from .observables import ObservableTransform
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from baselines.data import Condition, DetectorLayout, group_conditions, hardware_split_indices
    from baselines.observables import ObservableTransform


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace an artifact only after a complete JSON document is on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_prediction(
    path: Path, shot_indices: NDArray[Any], probabilities: NDArray[Any]
) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle, shot_indices=shot_indices.astype(np.uint64),
                probabilities=probabilities.astype(np.float32),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def predict_condition(
    model: Any, condition: Condition, split: str, batch_size: int
) -> tuple[NDArray[np.uint64], NDArray[np.float32]]:
    """Run the named main output, without loading any observable-label file."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if split not in {"validation", "test"}:
        raise ValueError("Inference split must be validation or test")
    layout = DetectorLayout.from_condition(condition)
    observable = ObservableTransform.from_condition(condition)
    byte_width = (layout.num_detectors + 7) // 8
    detector_path = condition.path / "detection_events.b8"
    if detector_path.stat().st_size != 50_000 * byte_width:
        raise ValueError(f"{condition.name}: expected exactly 50,000 detector shots")
    detectors = np.memmap(detector_path, dtype=np.uint8, mode="r", shape=(50_000, byte_width))
    indices = hardware_split_indices(split)
    names = list(model.output_names)
    if names.count("main_output") != 1 or set(names) != {"main_output", "aux_output"}:
        raise ValueError("Model must expose named main_output and aux_output heads")
    if set(model.input_names) != {"rec_input", "eval_input"}:
        raise ValueError("Model must accept named rec_input and eval_input tensors")
    probabilities = np.empty(len(indices), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        packed_batch = np.asarray(detectors[indices[start:end]])
        features = layout.decode_packed_features(packed_batch)
        outputs = model(features, training=False)
        output = outputs["main_output"] if isinstance(outputs, dict) else outputs[names.index("main_output")]
        values = np.asarray(output)
        if values.shape != (end - start, 1):
            raise ValueError(f"main_output must have shape [batch, 1], got {values.shape}")
        if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError("main_output must contain finite probabilities in [0, 1]")
        path_probabilities = observable.global_to_path_probabilities_packed(packed_batch, values)
        probabilities[start:end] = np.asarray(path_probabilities, dtype=np.float32).reshape(-1)
    return indices.astype(np.uint64), probabilities


def validate_model_metadata(path: Path, model_path: Path, group: str, distance: int) -> dict[str, Any]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("label_mode") != "qrennd_global_data_parity":
        raise ValueError("Checkpoint metadata must declare label_mode=qrennd_global_data_parity")
    if metadata.get("group") != group:
        raise ValueError("Checkpoint metadata group does not match the requested group")
    if "distance" in metadata and metadata["distance"] != distance:
        raise ValueError("Checkpoint metadata distance does not match the requested group")
    digest = sha256_file(model_path)
    for key in ("model_sha256", "checkpoint_sha256"):
        if key in metadata and metadata[key] != digest:
            raise ValueError(f"Checkpoint does not match metadata {key}")
    return {"label_mode": metadata["label_mode"], "group": group, "distance": distance,
            "model_sha256": digest, "metadata_sha256": sha256_file(path)}


def run_prediction(
    model_path: Path, data_root: Path, group: str, split: str, output: Path,
    batch_size: int = 2048, require_gpu: bool = False,
    model_metadata: Path | None = None,
) -> dict[str, Any]:
    """Load only standard Keras layers; no original training workspace needed."""
    if not model_path.is_file() or model_path.suffix != ".keras":
        raise ValueError("--model must name a saved .keras file")
    groups = group_conditions(data_root)
    if group not in groups:
        raise ValueError(f"Unknown group: {group}")
    conditions = groups[group]
    if [item.rounds for item in conditions] != list(range(1, 26, 2)):
        raise ValueError("Inference requires all 13 round conditions in the group")
    if batch_size <= 0 or split not in {"validation", "test"}:
        raise ValueError("Require positive batch_size and validation/test split")
    metadata_path = model_metadata if model_metadata is not None else model_path.parent / "metadata.json"
    if metadata_path.resolve() == (output / "metadata.json").resolve():
        raise ValueError("Prediction output must not overwrite the checkpoint metadata")
    checkpoint_metadata = validate_model_metadata(metadata_path, model_path, group, conditions[0].distance)
    output.mkdir(parents=True, exist_ok=True)
    unexpected = {path.stem for path in output.glob("*.npz")} - {item.name for item in conditions}
    if unexpected:
        raise ValueError(f"Output directory contains unrelated predictions: {sorted(unexpected)}")

    import tensorflow as tf

    devices = tf.config.list_physical_devices("GPU")
    if require_gpu and not devices:
        raise RuntimeError("--require-gpu requested but TensorFlow has no visible GPU")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=True)
    metadata: dict[str, Any] = {
        "schema_version": 1, "status": "running", "group": group, "split": split,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "model_file": model_path.name, "model_sha256": checkpoint_metadata["model_sha256"],
        "checkpoint_metadata": checkpoint_metadata,
        "model_label_mode": "qrennd_global_data_parity",
        "prediction_label_mode": "official_path_observable_flip",
        "input_contract": "detectors only; no label files read; main_output global parity transformed to path probability using known detector parity",
        "batch_size": batch_size, "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__, "python_version": platform.python_version(),
        "hardware": {
            "host": socket.gethostname(), "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "gpus": [{"name": device.name, **tf.config.experimental.get_device_details(device)} for device in devices],
        },
        "conditions": {},
    }
    atomic_json(output / "metadata.json", metadata)
    started = time.perf_counter()
    try:
        for condition in conditions:
            condition_start = time.perf_counter()
            indices, probabilities = predict_condition(model, condition, split, batch_size)
            destination = output / f"{condition.name}.npz"
            atomic_prediction(destination, indices, probabilities)
            metadata["conditions"][condition.name] = {
                "shots": len(indices), "prediction_sha256": sha256_file(destination),
                "detectors_sha256": sha256_file(condition.path / "detection_events.b8"),
                "circuit_sha256": sha256_file(condition.path / "circuit_ideal.stim"),
                "seconds_including_input_and_write": time.perf_counter() - condition_start,
            }
        metadata["status"] = "complete"
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        metadata["elapsed_seconds"] = time.perf_counter() - started
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(output / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-metadata", type=Path, help="Defaults to metadata.json beside the model")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    result = run_prediction(args.model, args.data_root, args.group, args.split, args.output, args.batch_size, args.require_gpu, args.model_metadata)
    print(json.dumps({"status": result["status"], "output": str(args.output), "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
