"""Bounded native JAX/Haiku AlphaQubit training on development data only.

``last.pkl`` is a trusted, internal resume artifact and must never be loaded
from an untrusted source. Portable inference uses pickle-free ``best.npz``.
Paper ambiguities resolved here are recorded in the model/training identity.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import platform
import signal
import subprocess
import tempfile
import time
from typing import Any, Callable

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import stim

from .alphaqubit_data import AlphaQubitLayout, AlphaQubitSampler, load_hardware_packed
from .alphaqubit_model import AlphaQubitConfig, loss_terms, transformed_model
from .data import Condition, group_conditions
from .metrics import summarize_conditions


SCHEMA_VERSION = 1
LABEL_MODE = "official_path_observable_flip"
SOURCES = ("alphaqubit_train.py", "alphaqubit_model.py", "alphaqubit_data.py", "data.py", "metrics.py")
FACTORS = np.arange(5, 11, dtype=np.float64) / 10


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, writer: Callable[[Any], None]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write(path, lambda stream: stream.write((json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()))


def host_tree(tree: Any) -> Any:
    return jax.tree.map(lambda value: np.asarray(jax.device_get(value)), tree)


def save_internal(path: Path, state: dict[str, Any]) -> None:
    # Only array pytrees need conversion; Python RNG/session dictionaries must
    # retain their original scalar types for exact sampler restoration.
    payload = dict(state)
    for key in ("params", "opt_state", "ema", "anchor", "best_params", "jax_rng"):
        payload[key] = host_tree(state[key]) if state.get(key) is not None else None
    atomic_write(path, lambda stream: pickle.dump(payload, stream, protocol=5))


def load_internal(path: Path) -> dict[str, Any]:
    """Load a task-owned trusted pickle; never use this for inference inputs."""
    with path.open("rb") as stream:
        state = pickle.load(stream)
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported trusted training checkpoint schema")
    for key in ("params", "opt_state", "ema", "anchor", "best_params", "jax_rng"):
        if state.get(key) is not None:
            state[key] = jax.device_put(state[key])
    return state


def flatten_params(params: hk.Params) -> dict[str, np.ndarray]:
    return {f"{module}/{name}": np.asarray(value, dtype=np.float32)
            for module, fields in jax.device_get(params).items() for name, value in fields.items()}


def load_portable(path: Path) -> tuple[hk.Params, dict[str, Any]]:
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata["checkpoint_sha256"] != sha256(path) or metadata["label_mode"] != LABEL_MODE:
        raise ValueError("Portable checkpoint checksum/label identity mismatch")
    params: dict[str, dict[str, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            module, name = key.rsplit("/", 1)
            value = archive[key]
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("Portable parameters must be finite float32 arrays")
            params.setdefault(module, {})[name] = np.array(value)
    return jax.device_put(params), metadata


def curriculum_probabilities(seen_samples: int) -> np.ndarray:
    """Eq6/7, resolving the Fig8a counter explicitly to examples seen."""
    peak = 1 / (1 + math.exp(-min(700.0, seen_samples / 12_800_000 - 1)))
    weights = 1 + 12 * np.exp(-((FACTORS - peak) ** 2) / (2 * 0.05**2))
    return weights / weights.sum()


def learning_rate(distance: int, completed_steps: int) -> float:
    # The first update is nonzero; the 10,000th reaches the peak. Decay counters
    # count completed optimizer updates. These boundary choices are recorded.
    peak = 3.46e-4 if distance == 3 else 2.45e-4
    warmup = min((completed_steps + 1) / 10_000, 1.0)
    reductions = sum(completed_steps >= step for step in (80_000, 200_000, 400_000, 1_000_000, 2_000_000))
    return peak * warmup * 0.7**reductions


def add_anchored_decay(direction: hk.Params, params: hk.Params, anchor: hk.Params | None,
                       weight_decay: float) -> hk.Params:
    """LAMB's decay direction, exempting biases and LayerNorm offsets."""
    return {module: {name: value + (0 if name in {"b", "bias", "offset"} else weight_decay *
                                   (params[module][name] - (anchor[module][name] if anchor is not None else 0)))
                     for name, value in fields.items()} for module, fields in direction.items()}


def make_update(model: Any, microbatch: int, weight_decay: float) -> tuple[Any, Any]:
    adam = optax.scale_by_adam(b1=0.9, b2=0.95, eps=1e-6)
    trust = optax.scale_by_trust_ratio()

    @jax.jit
    def update(params: hk.Params, optimizer_state: Any, ema: hk.Params, anchor: hk.Params | None,
               features: dict[str, jax.Array], labels: jax.Array, rate: jax.Array) -> tuple[Any, ...]:
        count = labels.shape[0] // microbatch
        values = {key: value.reshape((count, microbatch, *value.shape[1:]))
                  for key, value in features.items() if key not in {"is_final", "valid_mask"}}
        static = {key: features[key] for key in ("is_final", "valid_mask")}
        targets = labels.reshape(count, microbatch, -1)
        empty_grad = jax.tree.map(jnp.zeros_like, params)
        empty_metrics = {key: jnp.asarray(0.0) for key in ("loss", "main_loss", "aux_loss", "main_error")}

        def micro_step(carry: tuple[Any, Any], batch: tuple[Any, Any]) -> tuple[Any, None]:
            gradient_sum, metric_sum = carry
            inputs, target = batch
            inputs = {**inputs, **static}
            objective = lambda weights: loss_terms(model.apply(weights, inputs), inputs, target)
            (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(params)
            return (jax.tree.map(lambda total, value: total + value / count, gradient_sum, gradients),
                    jax.tree.map(lambda total, value: total + value / count, metric_sum, metrics)), None

        (gradients, metrics), _ = jax.lax.scan(micro_step, (empty_grad, empty_metrics), (values, targets))
        direction, optimizer_state = adam.update(gradients, optimizer_state, params)
        direction = add_anchored_decay(direction, params, anchor, weight_decay)
        direction, _ = trust.update(direction, optax.EmptyState(), params)
        params = optax.apply_updates(params, jax.tree.map(lambda value: -rate * value, direction))
        ema = jax.tree.map(lambda old, new: old + 0.0001 * (new - old), ema, params)
        return params, optimizer_state, ema, {**metrics, "gradient_norm": optax.global_norm(gradients),
                                              "parameter_norm": optax.global_norm(params)}

    return adam, update


@dataclass
class Population:
    condition: Condition
    sampler: AlphaQubitSampler
    validation_detectors: np.ndarray
    validation_labels: np.ndarray
    training_detectors: np.ndarray | None = None
    training_labels: np.ndarray | None = None


def prepare_populations(data_root: Path, group: str, stage: str, seed: int,
                        eval_shots: int) -> tuple[list[Population], list[dict[str, Any]]]:
    populations, provenance = [], []
    conditions = sorted(group_conditions(data_root)[group], key=lambda condition: condition.rounds)
    if [condition.rounds for condition in conditions] != list(range(1, 26, 2)):
        raise ValueError("Training requires all thirteen round settings")
    for condition in conditions:
        sampler = AlphaQubitSampler(condition, seed + 1009 * condition.rounds, "pij")
        if stage == "pretrain":
            validation_sampler = AlphaQubitSampler(condition, seed + 1_000_000_007 + 1009 * condition.rounds, "pij")
            validation_detectors, validation_labels = validation_sampler.sample_packed(eval_shots, 1.0)
            train_detectors = train_labels = None
            validation_source = validation_sampler.state_dict()
        else:
            validation_detectors, validation_labels = load_hardware_packed(condition, "validation")
            train_detectors, train_labels = load_hardware_packed(condition, "train")
            validation_source = {"split": "last5120even", "training_split": "first19880even"}
        populations.append(Population(condition, sampler, validation_detectors, validation_labels,
                                      train_detectors, train_labels))
        provenance.append({"condition": condition.name, "sampler_identity": sampler.state_dict(),
                           "layout": sampler.layout.metadata(), "validation_source": validation_source,
                           "validation_detectors_sha256": hashlib.sha256(validation_detectors.tobytes()).hexdigest(),
                           "validation_labels_sha256": hashlib.sha256(validation_labels.tobytes()).hexdigest(),
                           "training_allowed_sha256": None if train_detectors is None else {
                               "detectors": hashlib.sha256(train_detectors.tobytes()).hexdigest(),
                               "labels": hashlib.sha256(train_labels.tobytes()).hexdigest()}})
    return populations, provenance


def fit_validation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Use the repository WLS LER fit, with the paper's explicit rejection gate."""
    try:
        summary = summarize_conditions({row["condition"]: {"errors": row["errors"], "total": row["shots"]}
                                        for row in rows}, bootstrap_samples=0)
        fit = next(iter(summary["per_group"].values()))
        selected = sorted((row for row in rows if row["rounds"] != 1), key=lambda row: row["rounds"])
        rounds = np.array([row["rounds"] for row in selected], dtype=np.float64)
        errors = np.array([row["errors"] for row in selected], dtype=np.float64)
        shots = np.array([row["shots"] for row in selected], dtype=np.float64)
        probability = (errors + 0.5) / (shots + 1)
        weights = shots * (1 - 2 * errors / shots)**2 / (4 * probability * (1 - probability))
        design = np.stack((np.ones_like(rounds), rounds), axis=1)
        stderr = float(np.sqrt(np.linalg.inv(design.T @ (weights[:, None] * design))[0, 0]))
        r_squared = fit["unweighted_log_fidelity_r_squared"]
        reasons = []
        if r_squared is None or r_squared <= 0.9:
            reasons.append("r_squared_not_above_0.9")
        if fit["intercept"] <= max(-0.02, -stderr):
            reasons.append("intercept_below_paper_cutoff")
        if not 0 <= fit["logical_error_per_round"] < 0.5:
            reasons.append("unphysical_ler")
        return {"accepted": not reasons, "rejection_reasons": reasons, **fit,
                "intercept_stderr": stderr, "selection_r_squared": r_squared,
                "fit_convention": "repository WLS; unweighted residual R²; delta-method WLS intercept stderr"}
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        return {"accepted": False, "rejection_reasons": [str(error)], "logical_error_per_round": None}


def make_validator(model: Any, microbatch: int) -> Callable[..., dict[str, Any]]:
    @jax.jit
    def predict(params: hk.Params, features: dict[str, jax.Array], labels: jax.Array) -> dict[str, jax.Array]:
        output = model.apply(params, features)
        _, metrics = loss_terms(output, features, labels)
        return {**metrics, "errors": jnp.sum((output["logits"] > 0) != labels.reshape(-1))}

    def validate(params: hk.Params, populations: list[Population], can_continue: Callable[[], bool]) -> dict[str, Any]:
        rows = []
        for population in populations:
            sums = {key: 0.0 for key in ("loss", "main_loss", "aux_loss")}
            errors = shots = 0
            for start in range(0, len(population.validation_labels), microbatch):
                if not can_continue():
                    return {"complete": False, "per_condition": rows}
                end = min(start + microbatch, len(population.validation_labels))
                features = population.sampler.layout.decode_packed_features(population.validation_detectors[start:end])
                metrics = jax.device_get(predict(params, features, population.validation_labels[start:end]))
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError("Nonfinite validation metrics")
                errors += int(metrics["errors"])
                shots += end - start
                for key in sums:
                    sums[key] += float(metrics[key]) * (end - start)
            rows.append({"condition": population.condition.name, "rounds": population.condition.rounds,
                         "shots": shots, "errors": errors, "main_error": errors / shots,
                         **{key: value / shots for key, value in sums.items()}})
        total = sum(row["shots"] for row in rows)
        return {"complete": True, "shots": total, "per_condition": rows, "fit": fit_validation(rows),
                **{key: sum(row[key] * row["shots"] for row in rows) / total
                   for key in ("loss", "main_loss", "aux_loss", "main_error")}}
    return validate


def publish_best(output: Path, state: dict[str, Any], identity: dict[str, Any]) -> None:
    if state["best_params"] is None:
        return
    path = output / "best.npz"
    flat = flatten_params(state["best_params"])
    atomic_write(path, lambda stream: np.savez(stream, **flat))
    selection = state["best_validation"]
    atomic_json(path.with_suffix(".json"), {
        "schema_version": SCHEMA_VERSION, "model_config": identity["model_config"],
        "geometry": identity["geometry"], "identity_sha256": state["identity_sha256"],
        "label_mode": LABEL_MODE, "group": identity["group"], "distance": identity["distance"],
        "stage": identity["stage"], "global_steps": selection["global_steps"],
        "seen_samples": selection["seen_samples"], "selected_LER": selection["fit"]["logical_error_per_round"],
        "validation": selection, "source_sha256": identity["source_sha256"],
        "checkpoint_sha256": sha256(path), "selection": "lowest accepted validation fitted LER using EMA"})


def hardware_metadata() -> dict[str, Any]:
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                      text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        gpu = None
    return {"hostname": platform.node(), "python": platform.python_version(), "jax": jax.__version__,
            "haiku": hk.__version__, "optax": optax.__version__, "numpy": np.__version__, "stim": stim.__version__,
            "devices": [str(device) for device in jax.devices()], "nvidia_smi": gpu,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def run_training(args: argparse.Namespace, *, config_override: AlphaQubitConfig | None = None,
                 initial_batch_size: int = 256, final_batch_size: int = 1024) -> dict[str, Any]:
    if args.hours <= 0 or args.microbatch <= 0 or args.eval_shots <= 0 or args.eval_seconds <= 0:
        raise ValueError("Hours, microbatch and evaluation settings must be positive")
    if any(value is not None and value <= 0 for value in (args.max_steps, args.max_samples)):
        raise ValueError("Step/sample limits must be positive")
    if args.stage == "finetune" and args.max_steps is None:
        args.max_steps = 30_000
    if initial_batch_size % args.microbatch or final_batch_size % args.microbatch:
        raise ValueError("Microbatch must divide both effective batch sizes")
    if not args.allow_cpu and not all(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("Training requires GPU devices; --allow-cpu is diagnostic only")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(args, config_override, initial_batch_size, final_batch_size)


def _run_locked(args: argparse.Namespace, config_override: AlphaQubitConfig | None,
                initial_batch_size: int, final_batch_size: int) -> dict[str, Any]:
    started = time.monotonic()
    checkpoint = args.output / "last.pkl"
    metadata_path = args.output / "metadata.json"
    nonempty = any(path.name != ".training.lock" for path in args.output.iterdir())
    if nonempty and not args.resume:
        raise FileExistsError("Nonempty output requires explicit --resume")
    if args.resume and not checkpoint.is_file():
        raise FileNotFoundError("--resume requires task-owned last.pkl")
    if args.stage == "finetune" and not args.resume and args.init_model is None:
        raise ValueError("Fine-tuning requires --init-model best.npz")
    prior_metadata = json.loads(metadata_path.read_text()) if args.resume else None
    initializer_sha = prior_metadata["identity"].get("initializer_checkpoint_sha256") if prior_metadata else None
    if args.init_model is not None:
        requested_sha = sha256(args.init_model)
        if args.resume and requested_sha != initializer_sha:
            raise ValueError("Resume initializer checkpoint SHA changed")
        initializer_sha = requested_sha
    populations, data_provenance = prepare_populations(args.data_root, args.group, args.stage, args.seed, args.eval_shots)
    distance = populations[0].condition.distance
    config = config_override or AlphaQubitConfig(distance, dtype=args.dtype)
    if config.distance != distance:
        raise ValueError("Model distance does not match the data")
    geometry = populations[0].sampler.layout.metadata()
    geometry_keys = ("stabilizer_positions", "stabilizer_types", "logical_axis", "logical_index")
    if any(any(pop.sampler.layout.metadata()[key] != geometry[key] for key in geometry_keys) for pop in populations):
        raise ValueError("Conditions disagree on static model geometry")
    source = {name: sha256(Path(__file__).with_name(name)) for name in SOURCES}
    protocol = {"optimizer": "Optax Adam normalization -> anchored nonbias decay -> per-leaf trust ratio -> negative LR",
                "beta1": 0.9, "beta2": 0.95, "epsilon": 1e-6,
                "weight_decay": 1e-5 if args.stage == "pretrain" else 0.08,
                "bias_exemptions": ["b", "bias", "offset"], "finetune_optimizer": "reset moments, anchor to pretrained EMA parameters",
                "ema_new_weight": 0.0001, "ema_initialization": "copy initial parameters",
                "lr_warmup_steps": 10000, "lr_first_update": "peak/10000",
                "lr_reductions_completed_steps": [80000, 200000, 400000, 1000000, 2000000],
                "lr_reduction_factor": 0.7, "batch_change_counter": args.batch_change_counter,
                "batch_change_threshold": 4000000, "initial_batch_size": initial_batch_size,
                "final_batch_size": final_batch_size, "microbatch": args.microbatch,
                "noise_curriculum_counter": "seen_samples", "noise_factors": FACTORS.tolist(),
                "noise_tc": 12800000, "noise_weight": 12, "noise_sigma": 0.05, "noise_initial_peak": 0,
                "noise_slope": 1, "condition_sampling": "uniform rounds; one homogeneous round/factor per effective batch",
                "hardware_sampling": "uniform with replacement from allowed first19880even rows per condition",
                "selection": "EMA minimum accepted repository fitted LER; invalid fits retain BCE only",
                "validation_fit": "WLS point estimate, unweighted R², delta-method intercept stderr",
                "auxiliary": "model.loss_terms; next measured bulk stabilizers only, weight0.02",
                "eval_shots_per_condition": args.eval_shots if args.stage == "pretrain" else 5120,
                "max_samples_rounding": "do not exceed cap; final effective batch rounded down to microbatch multiple",
                "jax_rng": "initialization only; model has no dropout or application RNG",
                "initialization_geometry_condition": geometry["condition"]}
    identity = {"schema_version": SCHEMA_VERSION, "group": args.group, "distance": distance, "stage": args.stage,
                "seed": args.seed, "label_mode": LABEL_MODE, "model_config": config.to_dict(), "geometry": geometry,
                "source_sha256": source, "data": data_provenance, "protocol": protocol,
                "initializer_checkpoint_sha256": initializer_sha}
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    random = np.random.default_rng(args.seed)
    model = transformed_model(config, geometry)
    adam, update = make_update(model, args.microbatch, protocol["weight_decay"])
    if args.resume:
        state = load_internal(checkpoint)
        metadata = prior_metadata
        if state["identity_sha256"] != identity_hash or metadata["identity_sha256"] != identity_hash:
            raise ValueError("Resume model/data/source/protocol identity changed")
        random.bit_generator.state = state["numpy_rng"]
        for population, sampler_state in zip(populations, state["samplers"], strict=True):
            population.sampler.load_state_dict(sampler_state)
        elapsed_before = max(state["elapsed_seconds"],
                             *(session.get("elapsed_seconds_after_session", 0.0) for session in metadata["sessions"]))
        progress_path = args.output / "progress.json"
        if progress_path.exists():
            previous_progress = json.loads(progress_path.read_text())
            if previous_progress.get("identity_sha256") != identity_hash:
                raise ValueError("Resume progress identity changed")
            elapsed_before = max(elapsed_before, previous_progress["elapsed_seconds"])
    else:
        key = jax.random.PRNGKey(args.seed)
        initial_inputs = populations[0].sampler.layout.decode_packed_features(populations[0].validation_detectors[:args.microbatch])
        params = jax.jit(model.init)(key, initial_inputs)
        anchor = None
        initialization = None
        if args.init_model is not None:
            loaded, initialization = load_portable(args.init_model)
            if initialization["group"] != args.group or initialization["model_config"] != config.to_dict():
                raise ValueError("Pretrained model group/configuration mismatch")
            if args.stage == "finetune" and initialization["stage"] != "pretrain":
                raise ValueError("Fine-tuning initializer must be a pretrained selection")
            if any(initialization["source_sha256"][name] != source[name]
                   for name in ("alphaqubit_model.py", "alphaqubit_data.py", "data.py")):
                raise ValueError("Pretrained model/data implementation source changed")
            if jax.tree.structure(loaded) != jax.tree.structure(params) or any(a.shape != b.shape for a, b in zip(jax.tree.leaves(loaded), jax.tree.leaves(params))):
                raise ValueError("Pretrained parameter structure mismatch")
            params = loaded
            anchor = params if args.stage == "finetune" else None
        jax.block_until_ready(params)
        state = {"schema_version": SCHEMA_VERSION, "identity_sha256": identity_hash,
                 "params": params, "opt_state": adam.init(params), "ema": params, "anchor": anchor,
                 "best_params": None, "best_validation": None, "latest_validation": None,
                 "jax_rng": key, "numpy_rng": copy.deepcopy(random.bit_generator.state),
                 "samplers": [population.sampler.state_dict() for population in populations],
                 "global_steps": 0, "seen_samples": 0, "elapsed_seconds": 0.0,
                 "validation_steps": [], "steady_steps": 0, "steady_samples": 0, "steady_seconds": 0.0}
        elapsed_before = 0.0
        metadata = {"schema_version": SCHEMA_VERSION, "identity": identity, "identity_sha256": identity_hash,
                    "group": args.group, "stage": args.stage, "distance": distance, "label_mode": LABEL_MODE,
                    "model_config": config.to_dict(), "parameter_count": hk.data_structures.tree_size(params),
                    "initialization": initialization, "sessions": []}
    elapsed = lambda: elapsed_before + time.monotonic() - started
    session = {"started_at": utc_now(), "resume": args.resume, "resume_global_steps": state["global_steps"],
               "resume_seen_samples": state["seen_samples"], "elapsed_before_session": elapsed_before,
               "total_hours_limit": args.hours, "max_steps": args.max_steps, "max_samples": args.max_samples,
               "hardware": hardware_metadata(), "source_sha256": source}
    metadata["sessions"].append(session)
    atomic_json(metadata_path, metadata)
    stopped: dict[str, Any] = {"signal": None, "reason": None}
    reserve = min(30.0, args.hours * 3600 * 0.1)
    can_continue = lambda: stopped["signal"] is None and elapsed() < args.hours * 3600 - reserve
    final_validation_allowed = lambda: stopped["signal"] is None and elapsed() < args.hours * 3600 - min(10.0, reserve / 3)
    def receive_signal(number: int, frame: Any) -> None:
        stopped["signal"] = number
    previous_signals = {number: signal.signal(number, receive_signal) for number in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)}
    validator = make_validator(model, args.microbatch)
    last_validation_time = time.monotonic()
    last_progress_time = time.monotonic()
    session_steps = 0

    def progress(status: str) -> dict[str, Any]:
        best = state["best_validation"]
        result = {"schema_version": SCHEMA_VERSION, "identity_sha256": identity_hash,
                  "group": args.group, "stage": args.stage, "status": status, "updated_at": utc_now(),
                  "global_steps": state["global_steps"], "seen_samples": state["seen_samples"],
                  "elapsed_seconds": elapsed(), "stop_reason": stopped["reason"], "signal_received": stopped["signal"],
                  "best_step": best["global_steps"] if best else None,
                  "best_LER": best["fit"]["logical_error_per_round"] if best else None,
                  "latest_validation": state["latest_validation"], "last_training_metrics": state.get("last_training_metrics"),
                  "steady_steps": state["steady_steps"], "steady_samples": state["steady_samples"],
                  "steady_training_seconds": state["steady_seconds"],
                  "steady_samples_per_second": state["steady_samples"] / state["steady_seconds"] if state["steady_seconds"] else None,
                  "throughput_scope": "host sampling/transfer and optimizer updates after first13 updates of each session; includes any later recompilation",
                  "convergence_established": False}
        atomic_json(args.output / "progress.json", result)
        return result

    def checkpoint_now(status: str) -> None:
        state["elapsed_seconds"] = elapsed()
        state["status"] = status
        state["stop_reason"] = stopped["reason"]
        state["signal_received"] = stopped["signal"]
        save_internal(checkpoint, state)

    def validate(trigger: str) -> None:
        nonlocal last_validation_time
        if state["global_steps"] in state["validation_steps"]:
            return
        before = time.monotonic()
        result = validator(state["ema"], populations, final_validation_allowed if trigger == "final" else can_continue)
        result.update({"global_steps": state["global_steps"], "seen_samples": state["seen_samples"],
                       "elapsed_seconds": elapsed(), "trigger": trigger, "time": utc_now(),
                       "validation_seconds": time.monotonic() - before, "selected_best": False})
        if result["complete"]:
            state["validation_steps"].append(state["global_steps"])
            state["latest_validation"] = result
            best = state["best_validation"]
            if result["fit"]["accepted"] and (best is None or result["fit"]["logical_error_per_round"] < best["fit"]["logical_error_per_round"]):
                result["selected_best"] = True
                state["best_validation"] = result
                state["best_params"] = state["ema"]
        checkpoint_now("training")
        if result["selected_best"]:
            publish_best(args.output, state, identity)
        with (args.output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(result, allow_nan=False) + "\n")
        print(json.dumps({"validation": result}, allow_nan=False), flush=True)
        last_validation_time = time.monotonic()
        progress("training")

    try:
        # Recreate a portable pair if interruption occurred between its atomic
        # NPZ and JSON publications. The trusted checkpoint is authoritative.
        if args.resume:
            publish_best(args.output, state, identity)
        progress("training")
        while can_continue():
            if args.max_steps is not None and state["global_steps"] >= args.max_steps:
                stopped["reason"] = "max_steps"
                break
            counter = state["seen_samples"] if args.batch_change_counter == "samples" else state["global_steps"]
            batch = final_batch_size if args.stage == "finetune" or counter >= 4_000_000 else initial_batch_size
            if args.max_samples is not None:
                batch = min(batch, args.max_samples - state["seen_samples"])
                batch = batch // args.microbatch * args.microbatch
            if batch <= 0:
                stopped["reason"] = "max_samples"
                break
            before = time.monotonic()
            index = int(random.integers(len(populations)))
            population = populations[index]
            factor = float(random.choice(FACTORS, p=curriculum_probabilities(state["seen_samples"]))) if args.stage == "pretrain" else 1.0
            if args.stage == "pretrain":
                features, labels = population.sampler.sample(batch, factor)
            else:
                indices = random.integers(len(population.training_labels), size=batch)
                features = population.sampler.layout.decode_packed_features(population.training_detectors[indices])
                labels = population.training_labels[indices]
            rate = learning_rate(distance, state["global_steps"])
            params, opt_state, ema, metrics = update(state["params"], state["opt_state"], state["ema"], state["anchor"],
                                                    features, labels, jnp.asarray(rate, dtype=jnp.float32))
            metrics = {key: float(value) for key, value in jax.device_get(metrics).items()}
            if not all(math.isfinite(value) for value in metrics.values()):
                raise FloatingPointError("Nonfinite training loss/gradient/parameter norm; last committed update retained")
            duration = time.monotonic() - before
            # Commit parameters and sampling cursors together only after the
            # complete effective-batch update succeeds.
            state.update(params=params, opt_state=opt_state, ema=ema,
                         numpy_rng=copy.deepcopy(random.bit_generator.state),
                         samplers=[pop.sampler.state_dict() for pop in populations])
            state["global_steps"] += 1
            state["seen_samples"] += batch
            state["last_training_metrics"] = {**metrics, "learning_rate": rate, "noise_factor": factor,
                                               "rounds": population.condition.rounds, "effective_batch": batch,
                                               "update_seconds": duration}
            session_steps += 1
            if session_steps > 13:
                state["steady_steps"] += 1
                state["steady_samples"] += batch
                state["steady_seconds"] += duration
            if can_continue() and time.monotonic() - last_validation_time >= args.eval_seconds:
                validate("interval")
            if time.monotonic() - last_progress_time >= 60 or state["global_steps"] % 100 == 0:
                progress("training")
                last_progress_time = time.monotonic()
        stopped["reason"] = stopped["reason"] or ("signal" if stopped["signal"] is not None else "walltime")
        if state["global_steps"] and final_validation_allowed():
            validate("final")
        checkpoint_now("stopped")
        publish_best(args.output, state, identity)
        result = progress("stopped")
        session.update(ended_at=utc_now(), stop_reason=stopped["reason"], elapsed_seconds_after_session=elapsed(),
                       final_global_steps=state["global_steps"], final_seen_samples=state["seen_samples"])
        atomic_json(metadata_path, metadata)
        print(json.dumps({"training_result": result}, allow_nan=False), flush=True)
        return result
    except BaseException as error:
        stopped["reason"] = "nonfinite" if isinstance(error, FloatingPointError) else "exception"
        checkpoint_now("failed")
        progress("failed")
        session.update(ended_at=utc_now(), stop_reason=stopped["reason"], error=f"{type(error).__name__}: {error}",
                       elapsed_seconds_after_session=elapsed())
        atomic_json(metadata_path, metadata)
        raise
    finally:
        for number, handler in previous_signals.items():
            signal.signal(number, handler)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("pretrain", "finetune"), default="pretrain")
    parser.add_argument("--hours", type=float, required=True, help="Cumulative stage hours, including resumed sessions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--microbatch", type=int, default=256)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--eval-shots", type=int, default=5120)
    parser.add_argument("--eval-seconds", type=float, default=600)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-change-counter", choices=("samples", "steps"), default="steps")
    parser.add_argument("--init-model", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


if __name__ == "__main__":
    run_training(argument_parser().parse_args())
