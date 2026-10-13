"""Synthetic-only artifact integrity and relocated native JAX replay checks."""
from __future__ import annotations

import hashlib
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import jax
import numpy as np
import pytest
import stim

from baselines.alphaqubit_artifact import (
    EXPECTED_GROUPS, INFERENCE_SOURCES, LABEL_MODE, SEMANTIC_SOURCES, TASK_ROOT,
    freeze_package, load_model, read_parameters, sha256_file, verify_package,
)
from baselines.alphaqubit_data import AlphaQubitLayout
from baselines.alphaqubit_model import AlphaQubitConfig, transformed_model
from baselines.alphaqubit_predict import predict_condition
from baselines.data import Condition, discover_conditions, hardware_split_indices
from baselines.score_neural import validate_inference_metadata, validate_prediction


GROUP = "d3_bX_center_3_5"


def write_json(path: Path, record: Any) -> None:
    path.write_text(json.dumps(record, sort_keys=True, allow_nan=False))


def make_condition(root: Path, rounds: int) -> Condition:
    name = f"surface_code_bX_d3_r{rounds:02d}_center_3_5"
    path = root / name
    path.mkdir(parents=True)
    ideal = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=rounds)
    ideal.to_file(path / "circuit_ideal.stim")
    # Deliberately provide only detector/circuit inputs. No raw readout, labels,
    # sweep values or training noise models exist anywhere in this fixture.
    rng = np.random.default_rng(70 + rounds)
    packed = np.packbits((rng.random((50_000, ideal.num_detectors)) < 0.1).astype(np.uint8), axis=1, bitorder="little")
    (path / "detection_events.b8").write_bytes(packed.tobytes())
    return Condition(path, name, 3, "X", rounds, GROUP)


@pytest.fixture(scope="module")
def native_fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Condition]:
    root = tmp_path_factory.mktemp("alphaqubit-native")
    condition = make_condition(root / "data", 1)
    layout = AlphaQubitLayout.from_condition(condition)
    config = AlphaQubitConfig(3, width=4, heads=1, key_size=2, value_size=2,
        layers=1, conv_width=2, widening=2, feature_layers=1, bias_width=2,
        bias_layers=1, readout_width=2, readout_layers=1)
    features = layout.decode_packed_features(np.zeros((2, 1), dtype=np.uint8))
    model = transformed_model(config, layout.metadata())
    parameters = model.init(jax.random.PRNGKey(49), features)
    source = {name: sha256_file(TASK_ROOT / "baselines" / name) for name in SEMANTIC_SOURCES}
    for stage in ("pretrain", "finetune"):
        folder = root / "models" / GROUP / stage
        folder.mkdir(parents=True)
        identity = {"group": GROUP, "stage": stage, "distance": 3, "label_mode": LABEL_MODE,
            "model_config": config.to_dict(), "geometry": layout.metadata(), "source_sha256": source,
            "data": {"synthetic_fixture": True}, "protocol": {"selection": "fixture validation selection"}}
        identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
        np.savez(folder / "best.npz", **{f"{module}/{name}": np.asarray(value, dtype=np.float32)
            for module, leaves in parameters.items() for name, value in leaves.items()})
        validation = {"global_steps": 12, "seen_samples": 48, "complete": True,
            "selected_best": True, "fit": {"accepted": True, "logical_error_per_round": 0.04},
            "per_condition": {condition.name: {"errors": 2, "total": 50}}}
        best = {"schema_version": 1, "identity_sha256": identity_sha,
            **{key: identity[key] for key in ("model_config", "geometry", "label_mode", "group", "stage", "distance", "source_sha256")},
            "global_steps": 12, "seen_samples": 48, "selected_LER": 0.04, "validation": validation,
            "selection": "lowest accepted validation fitted LER using EMA", "checkpoint_sha256": sha256_file(folder / "best.npz")}
        metadata = {"identity": identity, "identity_sha256": identity_sha,
            **{key: identity[key] for key in ("group", "stage", "distance", "label_mode")},
            "sessions": [{"ended_at": "2026-09-19T00:00:00+00:00", "stop_reason": "max_steps"}]}
        progress = {"identity_sha256": identity_sha, "status": "stopped", "stop_reason": "max_steps",
            "global_steps": 15, "best_step": 12, "best_LER": 0.04}
        for name, record in (("best.json", best), ("metadata.json", metadata), ("progress.json", progress)):
            write_json(folder / name, record)
    return root, condition


def copy_models(tmp_path: Path, native_fixture: tuple[Path, Condition]) -> Path:
    result = tmp_path / "models"
    shutil.copytree(native_fixture[0] / "models", result)
    return result


def test_freeze_both_stages_is_idempotent_and_contains_only_portable_payloads(tmp_path: Path, native_fixture: tuple[Path, Condition]) -> None:
    models = copy_models(tmp_path, native_fixture)
    output = tmp_path / "package"
    manifest = freeze_package(models, "both", output, GROUP)
    assert verify_package(output) == manifest == freeze_package(models, "both", output, GROUP)
    assert set(manifest["models"]) == {f"{GROUP}/pretrain", f"{GROUP}/finetune"}
    assert {f"baselines/{name}" for name in INFERENCE_SOURCES}.issubset(manifest["files"])
    assert {"README.md", "pyproject.toml", "uv.lock"}.issubset(manifest["files"])
    assert not any("train.py" in name or name.endswith(("pkl", "keras")) for name in manifest["files"])
    assert manifest["models"][f"{GROUP}/pretrain"]["data_provenance"] == {"synthetic_fixture": True}
    assert len(EXPECTED_GROUPS) == 10
    # Runtime-generated content is allowed, not silently mistaken for payload.
    (output / ".venv").mkdir()
    (output / ".venv" / "python").symlink_to(sys.executable)
    (output / "baselines" / "__pycache__").mkdir(exist_ok=True)
    (output / "baselines" / "__pycache__" / "module.pyc").write_bytes(b"cache")
    verify_package(output)


@pytest.mark.parametrize("corruption", ["weight", "source", "extra", "missing", "symlink", "unsafe_path", "manifest_selection"])
def test_package_rejects_artifact_tampering(tmp_path: Path, native_fixture: tuple[Path, Condition], corruption: str) -> None:
    output = tmp_path / "package"
    freeze_package(copy_models(tmp_path, native_fixture), "pretrain", output, GROUP)
    if corruption == "weight":
        (output / "models" / GROUP / "pretrain" / "model.npz").write_bytes(b"corrupt weights")
    elif corruption == "source":
        with (output / "baselines" / "alphaqubit_model.py").open("a") as handle:
            handle.write("\n# changed\n")
    elif corruption == "extra":
        (output / "undeclared.txt").write_text("extra")
    elif corruption == "missing":
        (output / "README.md").unlink()
    elif corruption == "symlink":
        (output / "README.md").unlink()
        (output / "README.md").symlink_to(native_fixture[0] / "models" / GROUP / "pretrain" / "best.json")
    elif corruption == "unsafe_path":
        manifest = json.loads((output / "manifest.json").read_text())
        manifest["files"]["../escape"] = {"bytes": 0, "sha256": "0" * 64}
        write_json(output / "manifest.json", manifest)
    else:
        manifest = json.loads((output / "manifest.json").read_text())
        manifest["models"][f"{GROUP}/pretrain"]["selection"]["selected_LER"] = 0.001
        write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError):
        verify_package(output)


@pytest.mark.parametrize("mismatch", ["identity", "checkpoint_hash", "running", "unfinished", "best_step", "best_ler", "selection", "source", "config"])
def test_freeze_rejects_unbound_or_unfinished_selections(tmp_path: Path, native_fixture: tuple[Path, Condition], mismatch: str) -> None:
    models = copy_models(tmp_path, native_fixture)
    folder = models / GROUP / "pretrain"
    records = {name: json.loads((folder / name).read_text()) for name in ("best.json", "metadata.json", "progress.json")}
    best, metadata, progress = records.values()
    if mismatch == "identity":
        best["identity_sha256"] = "0" * 64
    elif mismatch == "checkpoint_hash":
        best["checkpoint_sha256"] = "0" * 64
    elif mismatch == "running":
        progress["status"] = "training"
    elif mismatch == "unfinished":
        del metadata["sessions"][-1]["ended_at"]
    elif mismatch == "best_step":
        progress["best_step"] += 1
    elif mismatch == "best_ler":
        progress["best_LER"] = 0.03
    elif mismatch == "selection":
        best["validation"]["fit"]["accepted"] = False
    elif mismatch == "source":
        best["source_sha256"]["alphaqubit_model.py"] = "0" * 64
    else:
        best["model_config"]["width"] = 320
    for name, record in records.items():
        write_json(folder / name, record)
    with pytest.raises(ValueError):
        freeze_package(models, "pretrain", tmp_path / "rejected", GROUP)
    assert not (tmp_path / "rejected").exists()
    assert not list(tmp_path.glob(".rejected.*"))


@pytest.mark.parametrize("leaves", [{"badkey": np.ones(1, np.float32)}, {"a/w": np.ones(1, np.float64)}, {"a/w": np.array([np.nan], np.float32)}])
def test_portable_parameters_reject_malformed_or_nonfinite_leaves(leaves: dict[str, np.ndarray]) -> None:
    stream = io.BytesIO()
    np.savez(stream, **leaves)
    with pytest.raises(ValueError):
        read_parameters(stream.getvalue())


def test_active_training_lock_and_nonidentical_existing_package_refused(tmp_path: Path, native_fixture: tuple[Path, Condition]) -> None:
    models = copy_models(tmp_path, native_fixture)
    lock_path = models / GROUP / "pretrain" / ".training.lock"
    output = tmp_path / "package"
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="lock"):
            freeze_package(models, "pretrain", output, GROUP)
    manifest = freeze_package(models, "pretrain", output, GROUP)
    with pytest.raises(ValueError, match="not identical"):
        freeze_package(models, "both", output, GROUP)
    assert verify_package(output) == manifest


def test_relocated_native_cli_replays_after_original_model_deletion_without_labels(tmp_path: Path, native_fixture: tuple[Path, Condition]) -> None:
    models = copy_models(tmp_path, native_fixture)
    data = tmp_path / "detector-only"
    conditions = [make_condition(data, rounds) for rounds in range(1, 26, 2)]
    model, parameters, metadata = load_model(models / GROUP / "pretrain" / "best.npz")
    apply = jax.jit(lambda params, features: jax.nn.sigmoid(model.apply(params, features)["logits"]))
    expected_indices, expected_probabilities = predict_condition(apply, parameters, metadata["geometry"], conditions[0], "validation", 512)
    package = tmp_path / "original-package"
    freeze_package(models, "pretrain", package, GROUP)
    relocated = tmp_path / "moved" / "portable"
    relocated.parent.mkdir()
    shutil.move(str(package), relocated)
    shutil.rmtree(models)
    output = tmp_path / "predictions"
    environment = {**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": ""}
    result = subprocess.run([sys.executable, "-m", "baselines.alphaqubit_predict", "--package", ".",
        "--data-root", str(data), "--split", "validation", "--output", str(output), "--group", GROUP,
        "--batch-size", "512"], cwd=relocated, env=environment, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.splitlines()[-1])["status"] == "complete"
    record = json.loads((output / "metadata.json").read_text())
    assert record["model_label_mode"] == record["prediction_label_mode"] == LABEL_MODE
    assert record["hardware"]["devices"][0]["platform"] == "cpu"
    with np.load(output / f"{conditions[0].name}.npz") as saved:
        np.testing.assert_array_equal(saved["shot_indices"], expected_indices)
        np.testing.assert_array_equal(saved["probabilities"], expected_probabilities)
    assert not list(data.rglob("*actual*")) and not list(data.rglob("*measurement*"))
    hashes, counts = {}, {}
    condition_map = {condition.name: condition for condition in discover_conditions(data)}
    for name in condition_map:
        path = output / f"{name}.npz"
        split, ids, probabilities = validate_prediction(path)
        assert split == "validation"
        assert len(ids) == len(probabilities) == 5120
        hashes[name] = sha256_file(path)
        counts[name] = {"total": len(ids), "errors": 0}
    assert len(validate_inference_metadata(output, "validation", condition_map, hashes, counts)) == 1
    # Reject a truthful prediction mode paired with a contradictory checkpoint.
    record["checkpoint_metadata"]["label_mode"] = "qrennd_global_data_parity"
    write_json(output / "metadata.json", record)
    with pytest.raises(ValueError, match="Checkpoint identity"):
        validate_inference_metadata(output, "validation", condition_map, hashes, counts)
    verify_package(relocated)


def test_wrong_geometry_rejected_before_detector_read(native_fixture: tuple[Path, Condition]) -> None:
    root, condition = native_fixture
    model, parameters, record = load_model(root / "models" / GROUP / "pretrain" / "best.npz")
    wrong = {**record["geometry"], "logical_axis": 1 - record["geometry"]["logical_axis"]}
    with pytest.raises(ValueError, match="geometry"):
        predict_condition(model.apply, parameters, wrong, condition, "validation", 512)
