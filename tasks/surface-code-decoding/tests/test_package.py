"""Frozen checkpoint identity, tamper rejection, and relocated Keras replay."""
from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import zipfile

import numpy as np
import pytest
import stim

PROJECT = Path(__file__).resolve().parents[1] / "environment" / "baseline" / "alphaqubit" if (Path(__file__).resolve().parents[1] / "environment" / "baseline" / "alphaqubit").is_dir() else Path(__file__).resolve().parents[1]  # the baseline project: environment/baseline/alphaqubit here, /opt/task in the verifier image
sys.path.insert(0, str(PROJECT))
from baselines.package import (  # noqa: E402
    CHECKPOINT_STATE, EXPECTED_GROUPS, TASK_ROOT, freeze_package, verify_package,
)


GROUP = "d3_bX_center_3_5"


def identity_hash(identity: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def write_archive(path: Path, state: dict[str, Any], keras_bytes: bytes | None = None) -> None:
    with zipfile.ZipFile(path, "w") as output:
        if keras_bytes is None:
            for name, value in (("config.json", b"{}"), ("metadata.json", b"{}"), ("model.weights.h5", b"fixture")):
                output.writestr(name, value)
        else:
            with zipfile.ZipFile(io.BytesIO(keras_bytes)) as source:
                for name in source.namelist():
                    if name != CHECKPOINT_STATE:
                        output.writestr(name, source.read(name))
        output.writestr(CHECKPOINT_STATE, json.dumps(state))


def make_stage(models_root: Path, group: str = GROUP, keras_bytes: bytes | None = None) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    folder = models_root / group / "pretrain"
    folder.mkdir(parents=True)
    source_hashes = {name: hashlib.sha256((TASK_ROOT / "baselines" / name).read_bytes()).hexdigest()
                     for name in ("train.py", "neural.py", "cache.py", "data.py", "observables.py")}
    identity = {
        "stage": "pretrain", "group": group, "distance": int(group[1]), "seed": 0,
        "label_mode": "qrennd_global_data_parity", "native_repository": "https://github.com/Hossein-D/qrennd",
        "native_commit": "c2ed73e3ec6ee843293c8839bf059c63b2960e61",
        "data": {"manifest_sha256": "a" * 64, "cache_directory": str(models_root / "historical-cache")},
        "semantic_source_sha256": {name: source_hashes[name] for name in ("data.py", "neural.py", "observables.py")},
    }
    identity_sha = identity_hash(identity)
    metadata = {
        "identity": identity, "identity_sha256": identity_sha, "stage": "pretrain", "group": group,
        "distance": int(group[1]), "label_mode": "qrennd_global_data_parity", "source_sha256": source_hashes,
        "sessions": [{"started_at": "2026-09-19T01:00:00Z", "ended_at": "2026-09-19T02:00:00Z", "stop_reason": "max_steps"}],
        "protocol": {"selection": "main BCE + 0.5 auxiliary BCE", "adaptations": ["bounded fixture"]},
        "limitations": ["synthetic fixture; not scientific evidence"],
    }
    state = {
        "identity_sha256": identity_sha, "global_steps": 10, "best_step": 10, "last_validation_step": 10,
        "best_validation_loss": 0.2, "convergence_established": False,
        "latest_validation": {"complete": True, "selected_best": True, "global_steps": 10, "loss": 0.2},
    }
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (folder / "progress.json").write_text(json.dumps({**state, "status": "stopped", "global_steps": 20}), encoding="utf-8")
    (folder / ".training.lock").touch()
    write_archive(folder / "best.keras", state, keras_bytes)
    return folder, metadata, state


def test_exact_ten_groups_required_by_default(tmp_path: Path) -> None:
    models = tmp_path / "models"
    make_stage(models)
    with pytest.raises(ValueError, match="exactly 10"):
        freeze_package(models, "pretrain", tmp_path / "package")
    for group in EXPECTED_GROUPS:
        if group != GROUP:
            make_stage(models, group)
    result = freeze_package(models, "pretrain", tmp_path / "package")
    assert result["groups"] == list(EXPECTED_GROUPS)
    assert len(result["models"]) == 10


@pytest.mark.parametrize("invalid", ["identity", "embedded_identity", "best_step", "stage", "group", "label_mode", "unfinished", "progress", "semantic_source", "validation"])
def test_reject_inconsistent_or_active_stage(tmp_path: Path, invalid: str) -> None:
    models = tmp_path / "models"
    folder, metadata, state = make_stage(models)
    if invalid == "identity":
        metadata["identity"]["seed"] = 1
    elif invalid == "embedded_identity":
        state["identity_sha256"] = "f" * 64
    elif invalid == "best_step":
        state["best_step"] = 11
    elif invalid in {"stage", "group", "label_mode"}:
        metadata[invalid] = "wrong"
    elif invalid == "unfinished":
        del metadata["sessions"][-1]["ended_at"]
    elif invalid == "progress":
        progress = json.loads((folder / "progress.json").read_text())
        progress["status"] = "training"
        (folder / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
    elif invalid == "semantic_source":
        metadata["identity"]["semantic_source_sha256"]["data.py"] = "f" * 64
        metadata["identity_sha256"] = identity_hash(metadata["identity"])
        state["identity_sha256"] = metadata["identity_sha256"]
    elif invalid == "validation":
        state["latest_validation"]["complete"] = False
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    write_archive(folder / "best.keras", state)
    with pytest.raises(ValueError):
        freeze_package(models, "pretrain", tmp_path / "package", GROUP)
    assert not (tmp_path / "package").exists()
    assert not list(tmp_path.glob(".package.staging-*"))


def test_live_training_lock_prevents_freeze(tmp_path: Path) -> None:
    models = tmp_path / "models"
    folder, _, _ = make_stage(models)
    with (folder / ".training.lock").open("r") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        with pytest.raises(ValueError, match="training is still active"):
            freeze_package(models, "pretrain", tmp_path / "package", GROUP)


def test_idempotence_and_payload_tamper_rejection(tmp_path: Path) -> None:
    models, output = tmp_path / "models", tmp_path / "package"
    make_stage(models)
    original = freeze_package(models, "pretrain", output, GROUP)
    assert freeze_package(models, "pretrain", output, GROUP) == original
    assert verify_package(output) == original
    sidecar = json.loads((output / "models" / GROUP / "metadata.json").read_text())
    assert sidecar["checkpoint_sha256"] == original["models"][GROUP]["checkpoint_sha256"]
    assert set(path.name for path in (output / "baselines").iterdir()) == {"predict.py", "data.py", "observables.py"}
    assert (output / "UPSTREAM_LICENSE").read_bytes() == (TASK_ROOT / "baselines" / "reference" / "LICENSE").read_bytes()
    assert "UPSTREAM_LICENSE" in original["files"]
    readme = (output / "README.md").read_text()
    assert "Boris M. Varbanov" in readme and original["native_source"]["commit"] in readme
    (output / "baselines" / "predict.py").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        verify_package(output)
    with pytest.raises(ValueError, match="hash/size mismatch"):
        freeze_package(models, "pretrain", output, GROUP)


def test_refuse_different_verified_package_and_unknown_payload(tmp_path: Path) -> None:
    models, output = tmp_path / "models", tmp_path / "package"
    folder, metadata, _ = make_stage(models)
    freeze_package(models, "pretrain", output, GROUP)
    original_manifest = (output / "manifest.json").read_bytes()
    metadata["limitations"].append("a new provenance record")
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="not identical"):
        freeze_package(models, "pretrain", output, GROUP)
    assert (output / "manifest.json").read_bytes() == original_manifest
    (output / "unrecorded.py").write_text("# unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="coverage mismatch"):
        verify_package(output)


def test_relocated_standard_keras_checkpoint_replays_without_training_tree(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TF_NUM_INTRAOP_THREADS", "1")
    monkeypatch.setenv("TF_NUM_INTEROP_THREADS", "1")
    tf = pytest.importorskip("tensorflow")
    recurrent = tf.keras.Input((None, 8), name="rec_input")
    evaluation = tf.keras.Input((4,), name="eval_input")
    hidden = tf.keras.layers.Concatenate()([tf.keras.layers.GlobalAveragePooling1D()(recurrent), evaluation])
    outputs = [tf.keras.layers.Dense(1, activation="sigmoid", name=name, kernel_initializer="zeros", bias_initializer="zeros")(hidden)
               for name in ("main_output", "aux_output")]
    model = tf.keras.Model([recurrent, evaluation], outputs)
    keras_path = tmp_path / "fixture.keras"
    model.save(keras_path)
    models = tmp_path / "training-tree"
    make_stage(models, keras_bytes=keras_path.read_bytes())
    output = tmp_path / "frozen"
    freeze_package(models, "pretrain", output, GROUP)
    relocated = tmp_path / "relocated"
    shutil.move(output, relocated)
    shutil.rmtree(models)
    keras_path.unlink()
    data_root = tmp_path / "synthetic-detectors"
    for rounds in range(1, 26, 2):
        folder = data_root / f"surface_code_bX_d3_r{rounds:02d}_center_3_5"
        folder.mkdir(parents=True)
        circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=rounds)
        circuit.to_file(folder / "circuit_ideal.stim")
        np.zeros((50_000, (circuit.num_detectors + 7) // 8), dtype=np.uint8).tofile(folder / "detection_events.b8")
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-m", "baselines.predict", "--model", f"models/{GROUP}/model.keras",
         "--data-root", str(data_root), "--group", GROUP, "--split", "validation", "--output", str(tmp_path / "predictions")],
        cwd=relocated, env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1])["status"] == "complete"
    assert len(list((tmp_path / "predictions").glob("*.npz"))) == 13
    assert verify_package(relocated)["groups"] == [GROUP]
