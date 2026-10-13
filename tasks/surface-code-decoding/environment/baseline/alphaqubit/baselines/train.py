"""Train the native QRENND decoder on fixed simulation or allowed even shots.

Run as ``python -m baselines.train``. Full-model Keras checkpoints retain Adam
state and contain the exact data cursor needed to resume an interrupted epoch.
The published architecture is retained; data conventions, shuffling, finite
budgets, and hardware fine-tuning are explicitly recorded task adaptations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import time
from typing import Any, Callable
import uuid
import zipfile

import numpy as np
import tensorflow as tf

from .cache import verify_cache
from .data import Condition, DetectorLayout, group_conditions, hardware_split_indices
from .neural import (
    EXPECTED_PARAMETER_COUNTS,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    build_model,
    model_config,
)


CHECKPOINT_STATE = "reproduction_state.json"
SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(model: tf.keras.Model, path: Path, state: dict[str, Any]) -> None:
    """Atomically publish a standard .keras archive plus its resume cursor."""
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.keras")
    try:
        model.save(temporary)
        with zipfile.ZipFile(temporary, "a") as archive:
            archive.writestr(CHECKPOINT_STATE, json.dumps(state, allow_nan=False))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_state(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read(CHECKPOINT_STATE))


@dataclass(frozen=True)
class PackedData:
    condition: Condition
    layout: DetectorLayout
    detectors: np.ndarray
    labels: np.ndarray
    transform: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        expected_width = (self.layout.num_detectors + 7) // 8
        if self.detectors.dtype != np.uint8 or self.detectors.ndim != 2:
            raise ValueError("Packed detector data must be a uint8 matrix")
        if self.detectors.shape[1] != expected_width or not len(self.detectors):
            raise ValueError("Invalid detector width or empty condition")
        if self.labels.shape != (len(self.detectors), 1):
            raise ValueError("Labels must have shape [shots, 1]")
        if not np.all((self.labels == 0) | (self.labels == 1)):
            raise ValueError("Labels must be binary")
        from .observables import ObservableTransform

        object.__setattr__(self, "transform", ObservableTransform.from_condition(self.condition))


class PackedSequence(tf.keras.utils.Sequence):
    """One fixed epoch of shuffled shots and homogeneous-round batches.

    A fresh instance is created for each epoch. Its order is immutable while
    Keras worker threads prefetch, so callbacks can reliably record the cursor.
    Epoch seeds also reconstruct an unfinished epoch on resume.
    """

    def __init__(
        self,
        datasets: list[PackedData],
        batch_size: int,
        seed: int,
        epoch: int,
        start_batch: int = 0,
    ) -> None:
        self.datasets = datasets
        self.batch_size = batch_size
        self.epoch = epoch
        self.start_batch = start_batch
        self.ordering_seed = [seed, epoch]
        random = np.random.default_rng(np.random.SeedSequence(self.ordering_seed))
        self.shot_orders: list[np.ndarray] = []
        batches: list[tuple[int, int, int]] = []
        for group, data in enumerate(datasets):
            # int32 indices keep a 60M-shot population's permutation under 240 MB.
            order = np.arange(len(data.labels), dtype=np.int32)
            random.shuffle(order)
            self.shot_orders.append(order)
            batches.extend(
                (group, start, min(start + batch_size, len(order)))
                for start in range(0, len(order), batch_size)
            )
        self.batches = np.asarray(batches, dtype=np.int64)
        random.shuffle(self.batches)
        self.full_length = len(self.batches)
        if not 0 <= start_batch < self.full_length:
            raise ValueError("Resume batch cursor is outside the epoch")

    def __len__(self) -> int:
        return self.full_length - self.start_batch

    def batch_samples(self, index: int) -> int:
        _, start, end = self.batches[self.start_batch + index]
        return int(end - start)

    def __getitem__(self, index: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        group, start, end = self.batches[self.start_batch + index]
        data = self.datasets[group]
        shots = self.shot_orders[group][start:end]
        packed = data.detectors[shots]
        inputs = data.layout.decode_packed_features(packed)
        labels = data.transform.path_to_global_packed(packed, data.labels[shots]).astype(np.float32)
        return inputs, {"main_output": labels, "aux_output": labels}


def _load_allowed_hardware(condition: Condition, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Index only allowed even labels; never materialize odd test-label rows."""
    indices = hardware_split_indices(split)
    width = (condition.rounds * (condition.distance**2 - 1) + 7) // 8
    detector_path = condition.path / "detection_events.b8"
    label_path = condition.path / "obs_flips_actual.01"
    if detector_path.stat().st_size != 50_000 * width:
        raise ValueError(f"Unexpected hardware detector size: {condition.name}")
    if label_path.stat().st_size != 50_000 * 2:
        raise ValueError("Expected one observable digit plus LF per official .01 row")
    detector_map = np.memmap(detector_path, dtype=np.uint8, mode="r", shape=(50_000, width))
    label_map = np.memmap(label_path, dtype=np.uint8, mode="r", shape=(50_000, 2))
    selected = np.array(label_map[indices])
    if not np.all(selected[:, 1] == ord("\n")):
        raise ValueError("Malformed allowed hardware label rows")
    if not np.all((selected[:, 0] == ord("0")) | (selected[:, 0] == ord("1"))):
        raise ValueError("Nonbinary allowed hardware labels")
    return np.array(detector_map[indices]), (selected[:, :1] - ord("0")).astype(np.uint8)


def load_populations(
    data_root: Path, cache_dir: Path | None, group: str, stage: str
) -> tuple[list[PackedData], list[PackedData], dict[str, Any]]:
    conditions = group_conditions(data_root)[group]
    by_name = {condition.name: condition for condition in conditions}
    training: list[PackedData] = []
    validation: list[PackedData] = []
    if stage == "pretrain":
        if cache_dir is None:
            raise ValueError("Pretraining requires --cache-dir containing manifest.json")
        manifest_path = cache_dir / "manifest.json"
        # Verify v2 source provenance here; the payloads are hashed exactly once
        # below immediately before loading them into RAM.
        manifest = verify_cache(data_root, cache_dir, full_hashes=False)
        if manifest["group"] != group or manifest["distance"] != conditions[0].distance:
            raise ValueError("Synthetic cache group/distance does not match the requested model")
        if set(by_name) != {entry["condition"] for entry in manifest["conditions"]}:
            raise ValueError("Synthetic cache does not cover exactly the requested conditions")
        for entry in manifest["conditions"]:
            condition = by_name[entry["condition"]]
            layout = DetectorLayout.from_condition(condition)
            if entry["num_detectors"] != layout.num_detectors:
                raise ValueError("Synthetic cache detector layout mismatch")
            loaded: dict[str, np.ndarray] = {}
            for key in ("train_detectors", "train_labels", "validation_detectors", "validation_labels"):
                filename = entry["files"][key]
                path = (cache_dir / filename).resolve()
                if path.parent != cache_dir.resolve():
                    raise ValueError("Cache files must be immediately inside their cache directory")
                if sha256(path) != entry["sha256"][filename]:
                    raise ValueError(f"Synthetic cache checksum mismatch: {filename}")
                loaded[key] = np.load(path, allow_pickle=False)
            training.append(PackedData(condition, layout, loaded["train_detectors"], loaded["train_labels"]))
            validation.append(PackedData(condition, layout, loaded["validation_detectors"], loaded["validation_labels"]))
            if len(training[-1].labels) != entry["shots"]:
                raise ValueError("Synthetic cache training shot count mismatch")
        if sum(len(data.labels) for data in training) != manifest["shots"]:
            raise ValueError("Synthetic cache total shot count mismatch")
        provenance = {
            "manifest": manifest, "manifest_sha256": sha256(manifest_path),
            "cache_directory": str(cache_dir.resolve()),
        }
    else:
        entries = []
        for condition in conditions:
            layout = DetectorLayout.from_condition(condition)
            entry: dict[str, Any] = {"condition": condition.name, "allowed_sha256": {}}
            for split, population in (("train", training), ("validation", validation)):
                detectors, labels = _load_allowed_hardware(condition, split)
                population.append(PackedData(condition, layout, detectors, labels))
                # Hash only the selected development arrays, never the test-label file.
                entry["allowed_sha256"][split] = {
                    "detectors": hashlib.sha256(detectors.tobytes()).hexdigest(),
                    "labels": hashlib.sha256(labels.tobytes()).hexdigest(),
                }
            entries.append(entry)
        provenance = {
            "conditions": entries,
            "train": "first 19,880 even shots per condition, original indices [0,39760) step2",
            "validation": "last 5,120 even shots per condition, original indices [39760,50000) step2",
            "test": "odd-index data not selected or parsed by training",
        }
    provenance["circuit_sha256"] = {
        condition.name: sha256(condition.path / "circuit_ideal.stim") for condition in conditions
    }
    provenance["observable_transforms"] = {
        data.condition.name: data.transform.metadata() for data in training
    }
    return training, validation, provenance


def evaluate_validation(
    model: tf.keras.Model,
    datasets: list[PackedData],
    can_continue: Callable[[], bool],
    batch_size: int = 2048,
) -> dict[str, Any]:
    """Compute the native two-head objective without touching fit's metrics."""
    main_loss = auxiliary_loss = 0.0
    main_errors = auxiliary_errors = shots = 0
    for data in datasets:
        for start in range(0, len(data.labels), batch_size):
            if not can_continue():
                return {"complete": False, "shots": shots}
            stop = min(start + batch_size, len(data.labels))
            packed = data.detectors[start:stop]
            inputs = data.layout.decode_packed_features(packed)
            labels = data.transform.path_to_global_packed(packed, data.labels[start:stop]).astype(np.float32)
            main, auxiliary = model(inputs, training=False)
            main_loss += float(tf.reduce_sum(tf.keras.losses.binary_crossentropy(labels, main)))
            auxiliary_loss += float(tf.reduce_sum(tf.keras.losses.binary_crossentropy(labels, auxiliary)))
            main_errors += int(np.count_nonzero((main.numpy() >= 0.5) != labels))
            auxiliary_errors += int(np.count_nonzero((auxiliary.numpy() >= 0.5) != labels))
            shots += len(labels)
    result = {
        "complete": True, "shots": shots,
        "main_loss": main_loss / shots, "aux_loss": auxiliary_loss / shots,
        "loss": (main_loss + 0.5 * auxiliary_loss) / shots,
        "main_error": main_errors / shots, "aux_error": auxiliary_errors / shots,
    }
    if not all(math.isfinite(value) for key, value in result.items() if key.endswith("loss")):
        raise FloatingPointError("Nonfinite validation loss")
    return result


class TrainingMonitor(tf.keras.callbacks.Callback):
    def __init__(
        self,
        args: argparse.Namespace,
        validation: list[PackedData],
        state: dict[str, Any],
        elapsed: Callable[[], float],
    ) -> None:
        super().__init__()
        self.args = args
        self.validation = validation
        self.state = state
        self.elapsed = elapsed
        self.sequence: PackedSequence | None = None
        self.last_validation_time = time.monotonic()
        self.last_progress_time = time.monotonic()
        self.batch_started = 0.0
        self.session_steps = 0
        self.steady_steps = self.steady_samples = 0
        self.steady_seconds = 0.0
        self.stop_reason: str | None = None
        self.signal_received: int | None = None
        # Validation/training stop before the hard allocation deadline to leave
        # time for a portable optimizer checkpoint and progress publication.
        self.checkpoint_reserve = min(30.0, args.hours * 3600 * 0.1)

    def can_continue(self) -> bool:
        return self.signal_received is None and self.elapsed() < self.args.hours * 3600 - self.checkpoint_reserve

    def snapshot(self, status: str) -> dict[str, Any]:
        return {
            **self.state,
            "status": status,
            "elapsed_seconds": self.elapsed(),
            "updated_at": utc_now(),
            "optimizer_iterations": int(self.model.optimizer.iterations.numpy()),
            "session_steps": self.session_steps,
            "steady_steps": self.steady_steps,
            "steady_samples": self.steady_samples,
            "steady_training_seconds": self.steady_seconds,
            "steady_steps_per_second": self.steady_steps / self.steady_seconds if self.steady_seconds else None,
            "steady_samples_per_second": self.steady_samples / self.steady_seconds if self.steady_seconds else None,
            "throughput_warmup_steps_excluded": 10,
            "throughput_definition": "training batch callback duration, excluding validation/checkpoint callbacks",
            "convergence_established": False,
            "stop_reason": self.stop_reason,
            "signal_received": self.signal_received,
        }

    def progress(self, status: str = "training") -> None:
        atomic_json(self.args.output / "progress.json", self.snapshot(status))
        self.last_progress_time = time.monotonic()

    def request_stop_if_needed(self) -> bool:
        if self.signal_received is not None:
            self.stop_reason = "signal"
        elif not self.can_continue():
            self.stop_reason = "walltime"
        elif self.args.max_steps is not None and self.state["global_steps"] >= self.args.max_steps:
            self.stop_reason = "max_steps"
        if self.stop_reason is not None:
            self.model.stop_training = True
            return True
        return False

    def on_train_batch_begin(self, batch: int, logs: dict[str, Any] | None = None) -> None:
        self.batch_started = time.monotonic()

    def on_train_batch_end(self, batch: int, logs: dict[str, Any] | None = None) -> None:
        assert self.sequence is not None
        duration = time.monotonic() - self.batch_started
        samples = self.sequence.batch_samples(batch)
        self.state["global_steps"] += 1
        self.state["seen_samples"] += samples
        self.state["next_batch"] = self.sequence.start_batch + batch + 1
        self.state["ordering_seed"] = self.sequence.ordering_seed
        self.session_steps += 1
        if self.session_steps > 10:
            self.steady_steps += 1
            self.steady_samples += samples
            self.steady_seconds += duration
        for name, value in (logs or {}).items():
            if not math.isfinite(float(value)):
                self.stop_reason = "nonfinite_training_metric"
                self.model.stop_training = True
                self.progress("failed")
                raise FloatingPointError(f"Nonfinite {name} at step {self.state['global_steps']}")
        self.request_stop_if_needed()
        if self.can_continue() and time.monotonic() - self.last_validation_time >= self.args.eval_seconds:
            self.validate("interval")
        if self.state["global_steps"] % 1000 == 0 or time.monotonic() - self.last_progress_time >= 60:
            self.progress()

    def validate(self, trigger: str) -> None:
        if self.state.get("last_validation_step") == self.state["global_steps"]:
            return
        started = time.monotonic()
        result = evaluate_validation(self.model, self.validation, self.can_continue)
        result.update({
            "stage": self.args.stage, "trigger": trigger, "time": utc_now(),
            "epoch": self.state["epoch"], "next_batch": self.state["next_batch"],
            "global_steps": self.state["global_steps"], "seen_samples": self.state["seen_samples"],
            "elapsed_seconds": self.elapsed(), "validation_seconds": time.monotonic() - started,
        })
        if result["complete"]:
            self.state["last_validation_step"] = self.state["global_steps"]
            self.state["latest_validation"] = result
            previous = self.state.get("best_validation_loss")
            improved = previous is None or result["loss"] < previous
            result["selected_best"] = improved
            if improved:
                self.state["best_validation_loss"] = result["loss"]
                self.state["best_step"] = self.state["global_steps"]
                save_checkpoint(self.model, self.args.output / "best.keras", self.snapshot("checkpoint"))
            save_checkpoint(self.model, self.args.output / "last.keras", self.snapshot("checkpoint"))
        with (self.args.output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
        print(json.dumps({"validation": result}, allow_nan=False), flush=True)
        self.last_validation_time = time.monotonic()
        self.progress()
        self.request_stop_if_needed()

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        assert self.sequence is not None
        if self.state["next_batch"] >= self.sequence.full_length:
            self.state["epoch"] = epoch + 1
            self.state["next_batch"] = 0
        if self.can_continue():
            self.validate("epoch_end" if self.state["next_batch"] == 0 else "partial_epoch")
        if logs is not None and self.state.get("latest_validation"):
            logs["val_loss"] = self.state["latest_validation"]["loss"]


def hardware_metadata() -> dict[str, Any]:
    try:
        nvidia_smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        nvidia_smi = None
    return {
        "hostname": platform.node(), "python": platform.python_version(),
        "tensorflow": tf.__version__, "numpy": np.__version__,
        "tensorflow_build": tf.sysconfig.get_build_info(),
        "visible_gpu_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
        "nvidia_smi": nvidia_smi,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _validate_arguments(args: argparse.Namespace) -> None:
    for name in ("hours", "eval_seconds"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
    if args.epochs < 1 or args.workers < 1:
        raise ValueError("--epochs and --workers must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Run or resume one stage; budgets and step caps include earlier sessions."""
    _validate_arguments(args)
    session_started = time.monotonic()
    devices = tf.config.list_physical_devices("GPU")
    if not devices and not args.allow_cpu:
        raise RuntimeError("No TensorFlow GPU is available; --allow-cpu is reserved for diagnostics")
    for device in devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            pass  # A model may already exist when tests call this API in-process.
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / ".training.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("Another trainer owns this output directory") from None
    try:
        return _run_locked(args, session_started)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _run_locked(args: argparse.Namespace, session_started: float) -> dict[str, Any]:
    metadata_path = args.output / "metadata.json"
    last_path = args.output / "last.keras"
    progress_path = args.output / "progress.json"
    existing_files = {path.name for path in args.output.iterdir()} - {".training.lock"}
    resume = last_path.exists()
    if existing_files and not resume:
        raise ValueError("Output is nonempty without last.keras; choose a new output directory")
    if resume and (not metadata_path.exists() or not progress_path.exists()):
        raise ValueError("Resume requires metadata.json and progress.json beside last.keras")
    old_metadata = json.loads(metadata_path.read_text()) if resume else None
    prior_progress = json.loads(progress_path.read_text()) if resume else {}
    prior_state = checkpoint_state(last_path) if resume else {}
    elapsed_before = max(float(prior_state.get("elapsed_seconds", 0)), float(prior_progress.get("elapsed_seconds", 0)))
    elapsed = lambda: elapsed_before + time.monotonic() - session_started
    if elapsed_before >= args.hours * 3600:
        raise ValueError("The total --hours budget is already consumed; extend it explicitly to resume")
    training, validation, data_provenance = load_populations(args.data_root, args.cache_dir, args.group, args.stage)
    distance = training[0].condition.distance
    if args.batch_size is None:
        args.batch_size = 64 if distance == 3 else 256
    learning_rate = 0.0005 if args.stage == "pretrain" else 0.0001
    init_record = old_metadata["identity"]["initial_model"] if resume else None
    if not resume and args.init_model is not None:
        initial_metadata_path = args.init_model.parent / "metadata.json"
        if not initial_metadata_path.is_file():
            raise ValueError("Initial model requires sibling metadata.json with group and label convention")
        initial_metadata = json.loads(initial_metadata_path.read_text())
        if initial_metadata.get("group") != args.group or initial_metadata.get("label_mode") != "qrennd_global_data_parity":
            raise ValueError("Initial model group or label convention differs from this training stage")
        if args.stage == "finetune" and initial_metadata.get("stage") != "pretrain":
            raise ValueError("Fine-tuning must initialize from a separately reported pretraining stage")
        init_record = {
            "path": str(args.init_model.resolve()), "sha256": sha256(args.init_model),
            "metadata_sha256": sha256(initial_metadata_path),
            "group": initial_metadata["group"], "stage": initial_metadata.get("stage"),
            "label_mode": initial_metadata["label_mode"],
        }
    if args.stage == "finetune" and not resume and init_record is None:
        raise ValueError("Hardware fine-tuning requires --init-model pointing to pretrained best.keras")
    source_hashes = {
        name: sha256(Path(__file__).resolve().parent / name)
        for name in ("train.py", "data.py", "neural.py", "observables.py", "cache.py")
    }
    identity = {
        "schema_version": SCHEMA_VERSION, "stage": args.stage, "group": args.group,
        "distance": distance, "seed": args.seed, "batch_size": args.batch_size,
        "native_commit": UPSTREAM_COMMIT, "native_repository": UPSTREAM_REPOSITORY,
        "model_config": model_config(distance, learning_rate),
        "data": data_provenance, "initial_model": init_record,
        "label_mode": "qrennd_global_data_parity",
        "semantic_source_sha256": {
            name: source_hashes[name] for name in ("data.py", "neural.py", "observables.py")
        },
    }
    identity_digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if resume:
        if identity != old_metadata["identity"] or identity_digest != prior_state["identity_sha256"]:
            raise ValueError("Resume settings/data differ from the existing run; use another output directory")
        if args.init_model is not None and (init_record is None or sha256(args.init_model) != init_record["sha256"]):
            raise ValueError("--init-model differs from the checkpoint's recorded initialization")
        tf.keras.utils.set_random_seed(args.seed + len(old_metadata["sessions"]) * 1_000_003)
        model = tf.keras.models.load_model(last_path)
        state = {key: prior_state[key] for key in (
            "global_steps", "seen_samples", "epoch", "next_batch", "identity_sha256",
            "best_validation_loss", "best_step", "last_validation_step", "latest_validation",
        )}
        # A crash between best.keras and last.keras replacements must not erase
        # an already-selected better validation checkpoint on the next session.
        best_path = args.output / "best.keras"
        if best_path.exists():
            best = checkpoint_state(best_path)
            if best["identity_sha256"] != identity_digest:
                raise ValueError("Best checkpoint belongs to another run")
            if state["best_validation_loss"] is None or best["best_validation_loss"] < state["best_validation_loss"]:
                state["best_validation_loss"] = best["best_validation_loss"]
                state["best_step"] = best["best_step"]
        epoch_batches = sum(math.ceil(len(data.labels) / args.batch_size) for data in training)
        if state["next_batch"] == epoch_batches:
            state["epoch"] += 1
            state["next_batch"] = 0
    else:
        if args.init_model is None:
            model = build_model(distance, args.seed, learning_rate)
        else:
            tf.keras.utils.set_random_seed(args.seed)
            model = tf.keras.models.load_model(args.init_model)
            model.optimizer.learning_rate.assign(learning_rate)
        state = {
            "identity_sha256": identity_digest, "global_steps": 0, "seen_samples": 0,
            "epoch": 0, "next_batch": 0, "best_validation_loss": None,
            "best_step": None, "last_validation_step": None, "latest_validation": None,
        }
    if model.count_params() != EXPECTED_PARAMETER_COUNTS[distance] or model.output_names != ["main_output", "aux_output"]:
        raise ValueError("Loaded model is not the paper-width native two-head decoder")
    sessions = list(old_metadata["sessions"]) if resume else []
    sessions.append({
        "started_at": utc_now(), "resume": resume,
        "resume_global_steps": state["global_steps"], "resume_seen_samples": state["seen_samples"],
        "progress_steps_after_checkpoint": max(0, prior_progress.get("global_steps", 0) - state["global_steps"]),
        "ordering_seed": [args.seed, state["epoch"]], "resume_next_batch": state["next_batch"],
        "elapsed_before_session": elapsed_before,
        "total_hours_limit": args.hours, "total_epochs_limit": args.epochs,
        "total_steps_limit": args.max_steps, "eval_seconds": args.eval_seconds,
        "hardware": hardware_metadata(),
        "source_sha256": source_hashes,
        "dropout_restart_seed": args.seed + (len(sessions) * 1_000_003 if resume else 0),
    })
    metadata = {
        "identity": identity, "identity_sha256": identity_digest,
        "group": args.group, "distance": distance,
        "stage": args.stage, "label_mode": "qrennd_global_data_parity",
        "parameter_count": model.count_params(), "sessions": sessions,
        "source_sha256": source_hashes,
        "protocol": {
            "training_samples_per_epoch": sum(len(data.labels) for data in training),
            "validation_samples": sum(len(data.labels) for data in validation),
            "selection": "main BCE + 0.5 auxiliary BCE on complete validation population",
            "selection_times": "elapsed interval and epoch/partial-epoch end, including partial epochs",
            "data_order": "fixed samples; per-epoch within-condition permutation and homogeneous-round batch permutation",
            "resume_order": "reconstruct seed=[run seed, epoch] then skip checkpointed batches; dropout RNG restarts on resume",
            "resume_source_policy": "data/neural/observables source hashes must match; train/cache source revisions recorded per session to allow operational fixes",
            "early_stopping": "omitted; explicit bounded run, convergence is not inferred from stopping",
            "finetune": "lr=1e-4; pretrained Adam state retained; hardware adaptation reported separately",
            "label_mode": "qrennd_global_data_parity",
            "observable_transform": "exact circuit-derived detector parity maps official cache labels to original global-data parity; both heads train against global parity",
            "adaptations": ["zeros for absent first-round detectors", "budget-limited training", "per-epoch shot reshuffling"],
            "cpu_diagnostic": args.allow_cpu and not bool(tf.config.list_physical_devices("GPU")),
        },
    }
    atomic_json(metadata_path, metadata)
    monitor = TrainingMonitor(args, validation, state, elapsed)
    monitor.set_model(model)
    monitor.progress("starting")
    previous_signals: dict[int, Any] = {}
    def receive_signal(number: int, frame: Any) -> None:
        monitor.signal_received = number
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        previous_signals[number] = signal.signal(number, receive_signal)
    try:
        while state["epoch"] < args.epochs and not monitor.request_stop_if_needed():
            sequence = PackedSequence(training, args.batch_size, args.seed, state["epoch"], state["next_batch"])
            monitor.sequence = sequence
            monitor.state["ordering_seed"] = sequence.ordering_seed
            model.fit(
                sequence, initial_epoch=state["epoch"], epochs=state["epoch"] + 1,
                callbacks=[monitor], shuffle=False, verbose=0, workers=args.workers,
                use_multiprocessing=False, max_queue_size=16,
            )
            if monitor.stop_reason is not None:
                break
        if monitor.stop_reason is None:
            monitor.stop_reason = "epochs"
        if state["global_steps"] and monitor.can_continue():
            monitor.validate("final")
        save_checkpoint(model, last_path, monitor.snapshot("stopped"))
        monitor.progress("stopped")
    except BaseException:
        monitor.stop_reason = monitor.stop_reason or "exception"
        # Preserve the previous good checkpoint when failure is numerical.
        if monitor.stop_reason != "nonfinite_training_metric":
            save_checkpoint(model, last_path, monitor.snapshot("interrupted"))
        monitor.progress("failed")
        raise
    finally:
        for number, previous in previous_signals.items():
            signal.signal(number, previous)
    result = monitor.snapshot("stopped")
    sessions[-1]["ended_at"] = utc_now()
    sessions[-1]["elapsed_seconds_after_session"] = result["elapsed_seconds"]
    sessions[-1]["stop_reason"] = monitor.stop_reason
    atomic_json(metadata_path, metadata)
    print(json.dumps({"training_result": result}, allow_nan=False), flush=True)
    return result


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--group", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", choices=("pretrain", "finetune"), default="pretrain")
    parser.add_argument("--init-model", type=Path)
    parser.add_argument("--hours", type=float, required=True, help="Total stage walltime budget, including resumed sessions")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-seconds", type=float, default=300)
    parser.add_argument("--max-steps", type=int, help="Total optimizer steps in this stage, including resumed sessions")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--allow-cpu", action="store_true", help="Permit CPU execution for small diagnostics only")
    return parser


def main() -> None:
    run_training(argument_parser().parse_args())


if __name__ == "__main__":
    main()
