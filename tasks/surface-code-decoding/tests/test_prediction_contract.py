"""Prediction integrity and label-free inference, using synthetic fixtures only."""
from __future__ import annotations

from pathlib import Path
import json
import sys
from typing import Any

import numpy as np
import pytest
import stim

PROJECT = Path(__file__).resolve().parents[1] / "environment" / "baseline" / "alphaqubit" if (Path(__file__).resolve().parents[1] / "environment" / "baseline" / "alphaqubit").is_dir() else Path(__file__).resolve().parents[1]  # the baseline project: environment/baseline/alphaqubit here, /opt/task in the verifier image
sys.path.insert(0, str(PROJECT))
from baselines.data import Condition, discover_conditions, hardware_split_indices  # noqa: E402
from baselines.predict import (  # noqa: E402
    atomic_prediction, predict_condition, run_prediction, sha256_file,
    validate_model_metadata,
)
from baselines.score_neural import (  # noqa: E402
    CLASSICAL_METHODS, calibration_record, discover_predictions, expected_condition_names,
    safe_summary, score_predictions, validate_inference_metadata, validate_prediction,
)


GROUP = "d3_bX_center_3_5"


def write_prediction(path: Path, indices: Any = None, probabilities: Any = None) -> None:
    if indices is None:
        indices = hardware_split_indices("validation").astype(np.uint64)
    if probabilities is None:
        probabilities = np.full(len(indices), 0.25, dtype=np.float32)
    np.savez(path, shot_indices=indices, probabilities=probabilities)


@pytest.mark.parametrize("mode", ["missing_row", "duplicate_id", "wrong_id", "reordered", "nan", "infinite", "negative", "above_one", "column_vector", "float_ids"])
def test_reject_malformed_prediction(tmp_path: Path, mode: str) -> None:
    indices = hardware_split_indices("validation").astype(np.uint64)
    probabilities = np.full(len(indices), 0.25, dtype=np.float32)
    if mode == "missing_row":
        indices, probabilities = indices[:-1], probabilities[:-1]
    elif mode == "duplicate_id":
        indices[1] = indices[0]
    elif mode == "wrong_id":
        indices[0] += np.uint64(1)
    elif mode == "reordered":
        indices = indices[::-1]
    elif mode in {"nan", "infinite", "negative", "above_one"}:
        probabilities[0] = {"nan": np.nan, "infinite": np.inf, "negative": -0.1, "above_one": 1.1}[mode]
    elif mode == "column_vector":
        probabilities = probabilities[:, None]
    elif mode == "float_ids":
        indices = indices.astype(np.float64)
    path = tmp_path / "prediction.npz"
    write_prediction(path, indices, probabilities)
    with pytest.raises(ValueError):
        validate_prediction(path)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_atomic_roundtrip_validates_original_order(tmp_path: Path, split: str) -> None:
    path = tmp_path / "prediction.npz"
    indices = hardware_split_indices(split)
    probabilities = np.linspace(0, 1, len(indices), dtype=np.float32)
    atomic_prediction(path, indices, probabilities)
    actual_split, actual_indices, actual_probabilities = validate_prediction(path)
    assert actual_split == split
    np.testing.assert_array_equal(actual_indices, indices)
    np.testing.assert_array_equal(actual_probabilities, probabilities)
    assert list(tmp_path.iterdir()) == [path]


def test_condition_coverage_rejects_missing_and_duplicate(tmp_path: Path) -> None:
    expected = expected_condition_names(GROUP)
    first = sorted(expected)[0]
    write_prediction(tmp_path / f"{first}.npz")
    with pytest.raises(ValueError, match="coverage mismatch"):
        discover_predictions(tmp_path, expected)
    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    write_prediction(duplicate / f"{first}.npz")
    with pytest.raises(ValueError, match="Duplicate prediction"):
        discover_predictions(tmp_path, {first})


def test_unfit_ler_retains_all_counts_and_marks_null() -> None:
    counts = {name: {"errors": 51, "total": 100} for name in expected_condition_names(GROUP)}
    result = safe_summary(counts, bootstrap_samples=10, seed=1)
    assert result["status"] == "unfit"
    assert "P_fail < 0.5" in result["reason"]
    assert len(result["per_condition"]) == 13
    assert sum(record["total"] for record in result["per_condition"].values()) == 1300
    assert result["per_group"][GROUP]["logical_error_per_round"] is None
    assert result["macro"]["d3"]["logical_error_per_round"] is None


def test_calibration_uses_probabilities_and_preserves_endpoint_penalty() -> None:
    result = calibration_record(np.array([0.0, 0.2, 1.0]), np.array([0, 1, 0]))
    assert result["brier_score"] == pytest.approx((0.0 + 0.8**2 + 1.0) / 3)
    assert result["negative_log_likelihood"] == pytest.approx((-np.log(1 - 1e-7) - np.log(0.2) - np.log(1e-7)) / 3)


def test_complete_synthetic_scoring_reports_unfit_and_calibration(tmp_path: Path) -> None:
    data_root, predictions = tmp_path / "data", tmp_path / "predictions"
    data_root.mkdir()
    predictions.mkdir()
    for name in expected_condition_names(GROUP):
        folder = data_root / name
        folder.mkdir()
        (folder / "obs_flips_actual.01").write_text("0\n" * 50_000, encoding="ascii")
        write_prediction(predictions / f"{name}.npz", probabilities=np.full(5120, 0.5, dtype=np.float32))
    result = score_predictions(predictions, data_root, group=GROUP, bootstrap_samples=10)
    assert result["split"] == "validation"
    assert result["total_shots"] == 13 * 5120
    assert result["summary"]["status"] == "unfit"
    assert all(record["errors"] == 5120 for record in result["summary"]["per_condition"].values())
    assert result["calibration"]["macro"]["all"]["brier_score"] == 0.25


def test_synthetic_paired_comparison_and_alignment_rejection(tmp_path: Path) -> None:
    data_root, predictions, classical = tmp_path / "data", tmp_path / "predictions", tmp_path / "classical"
    data_root.mkdir()
    predictions.mkdir()
    paired_directory = classical / "paired_errors"
    paired_directory.mkdir(parents=True)
    indices = hardware_split_indices("test")
    for name in expected_condition_names(GROUP):
        rounds = int(name.split("_r")[1].split("_")[0])
        folder = data_root / name
        folder.mkdir()
        (folder / "obs_flips_actual.01").write_text("0\n" * 50_000, encoding="ascii")
        errors = np.arange(25_000) < int(25_000 * 0.005 * rounds)
        probabilities = np.where(errors, 0.9, 0.1).astype(np.float32)
        write_prediction(predictions / f"{name}.npz", indices.astype(np.uint64), probabilities)
        np.savez(paired_directory / f"{name}.npz", shot_indices=indices,
                 actual=np.zeros(25_000, dtype=np.bool_),
                 **{f"errors__{method}": errors for method in CLASSICAL_METHODS})
    result = score_predictions(predictions, data_root, group=GROUP, classical_root=classical, bootstrap_samples=20)
    assert result["summary"]["status"] == "ok"
    assert result["summary"]["macro"]["d3"]["logical_error_per_round"] > 0
    for comparison in result["paired_comparisons"].values():
        assert comparison["status"] == "ok"
        assert comparison["comparisons"][GROUP]["logical_error_per_round_a_minus_b"] == 0
        assert comparison["comparisons"][GROUP]["ci95"] == [0.0, 0.0]
    first = paired_directory / f"{sorted(expected_condition_names(GROUP))[0]}.npz"
    with np.load(first, allow_pickle=False) as record:
        changed = {key: record[key] for key in record.files}
    changed["shot_indices"] = changed["shot_indices"][::-1]
    np.savez(first, **changed)
    with pytest.raises(ValueError, match="shot IDs or actual labels disagree"):
        score_predictions(predictions, data_root, group=GROUP, classical_root=classical, bootstrap_samples=20)


def test_predict_condition_uses_only_detectors_and_named_main_head(tmp_path: Path, monkeypatch: Any) -> None:
    """No observable-label, sweep or measurement file exists in this fixture."""
    name = "surface_code_bX_d3_r03_center_3_5"
    folder = tmp_path / name
    folder.mkdir()
    circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3)
    circuit.to_file(folder / "circuit_ideal.stim")
    width = (circuit.num_detectors + 7) // 8
    packed = np.zeros((50_000, width), dtype=np.uint8)
    packed[39_760::4, 0] = 1
    packed.tofile(folder / "detection_events.b8")
    condition = Condition(folder, name, 3, "X", 3, GROUP)

    class FakeModel:
        input_names = ["eval_input", "rec_input"]
        # Deliberately put the auxiliary head first; positional selection of
        # output zero would silently produce the wrong prediction.
        output_names = ["aux_output", "main_output"]

        def __call__(self, features: dict[str, Any], training: bool) -> list[Any]:
            assert training is False
            n = len(features["eval_input"])
            main = 0.1 + 0.7 * features["rec_input"].sum(axis=(1, 2))[:, None]
            return [np.full((n, 1), 0.99), main]

    def forbid_labels(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Inference attempted to read a shot-data label file")

    monkeypatch.setattr(stim, "read_shot_data_file", forbid_labels)
    indices, probabilities = predict_condition(FakeModel(), condition, "validation", batch_size=79)
    np.testing.assert_array_equal(indices, hardware_split_indices("validation"))
    # Account for the exact global-to-path transform rather than assuming its
    # mask is empty for the generated circuit.
    from baselines.observables import ObservableTransform
    transform = ObservableTransform.from_condition(condition)
    global_probabilities = np.tile(np.array([0.8, 0.1], dtype=np.float32), 2560)[:, None]
    expected = transform.global_to_path_probabilities_packed(packed[indices], global_probabilities)[:, 0]
    np.testing.assert_allclose(probabilities, expected, atol=1e-7)


@pytest.mark.parametrize("invalid", ["label_mode", "group", "distance", "model_sha256"])
def test_checkpoint_metadata_rejects_incompatible_model(tmp_path: Path, invalid: str) -> None:
    model_path = tmp_path / "best.keras"
    model_path.write_bytes(b"fixture checkpoint")
    record = {"label_mode": "qrennd_global_data_parity", "group": GROUP, "distance": 3,
              "model_sha256": sha256_file(model_path)}
    record[invalid] = "incompatible"
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_model_metadata(path, model_path, GROUP, 3)


@pytest.mark.parametrize("invalid", ["status", "split", "prediction_hash", "detector_hash", "circuit_hash", "model_label", "group", "checkpoint_hash", "shots", "missing_condition"])
def test_inference_metadata_rejects_incomplete_or_tampered_artifacts(tmp_path: Path, invalid: str) -> None:
    name = sorted(expected_condition_names(GROUP))[0]
    folder = tmp_path / "data" / name
    folder.mkdir(parents=True)
    (folder / "circuit_ideal.stim").write_bytes(b"circuit fixture")
    (folder / "detection_events.b8").write_bytes(b"detector fixture")
    prediction_root = tmp_path / "predictions"
    prediction_root.mkdir()
    prediction = prediction_root / f"{name}.npz"
    write_prediction(prediction)
    conditions = {name: Condition(folder, name, 3, "X", 1, GROUP)}
    counts = {name: {"total": 5120, "errors": 0}}
    hashes = {name: sha256_file(prediction)}
    metadata = {
        "status": "complete", "split": "validation", "group": GROUP,
        "prediction_label_mode": "official_path_observable_flip", "model_label_mode": "qrennd_global_data_parity",
        "model_sha256": "0" * 64,
        "checkpoint_metadata": {"label_mode": "qrennd_global_data_parity", "group": GROUP, "model_sha256": "0" * 64},
        "conditions": {name: {"shots": 5120, "prediction_sha256": hashes[name],
                              "detectors_sha256": sha256_file(folder / "detection_events.b8"),
                              "circuit_sha256": sha256_file(folder / "circuit_ideal.stim")}},
    }
    metadata_path = prediction_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert len(validate_inference_metadata(prediction_root, "validation", conditions, hashes, counts)) == 1
    if invalid == "status":
        metadata["status"] = "running"
    elif invalid == "split":
        metadata["split"] = "test"
    elif invalid == "prediction_hash":
        metadata["conditions"][name]["prediction_sha256"] = "f" * 64
    elif invalid in {"detector_hash", "circuit_hash"}:
        filename = "detection_events.b8" if invalid == "detector_hash" else "circuit_ideal.stim"
        (folder / filename).write_bytes(b"different input fixture")
    elif invalid == "model_label":
        metadata["model_label_mode"] = "official_path_observable_flip"
    elif invalid == "group":
        metadata["group"] = "d3_bZ_center_3_5"
    elif invalid == "checkpoint_hash":
        metadata["checkpoint_metadata"]["model_sha256"] = "f" * 64
    elif invalid == "shots":
        metadata["conditions"][name]["shots"] = 5119
    elif invalid == "missing_condition":
        metadata["conditions"] = {}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_inference_metadata(prediction_root, "validation", conditions, hashes, counts)


def test_portable_standard_keras_replay_without_labels(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TF_NUM_INTRAOP_THREADS", "1")
    monkeypatch.setenv("TF_NUM_INTEROP_THREADS", "1")
    tf = pytest.importorskip("tensorflow")
    data_root = tmp_path / "data"
    data_root.mkdir()
    for name in expected_condition_names(GROUP):
        rounds = int(name.split("_r")[1].split("_")[0])
        folder = data_root / name
        folder.mkdir()
        circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=rounds)
        circuit.to_file(folder / "circuit_ideal.stim")
        np.zeros((50_000, (circuit.num_detectors + 7) // 8), dtype=np.uint8).tofile(folder / "detection_events.b8")
    recurrent = tf.keras.Input(shape=(None, 8), name="rec_input")
    evaluation = tf.keras.Input(shape=(4,), name="eval_input")
    hidden = tf.keras.layers.Concatenate()([tf.keras.layers.GlobalAveragePooling1D()(recurrent), evaluation])
    main = tf.keras.layers.Dense(1, activation="sigmoid", name="main_output", kernel_initializer="zeros", bias_initializer="zeros")(hidden)
    auxiliary = tf.keras.layers.Dense(1, activation="sigmoid", name="aux_output")(hidden)
    model = tf.keras.Model([recurrent, evaluation], [auxiliary, main])
    model_path = tmp_path / "best.keras"
    model.save(model_path)
    (tmp_path / "metadata.json").write_text(json.dumps({
        "label_mode": "qrennd_global_data_parity", "group": GROUP, "distance": 3,
        "model_sha256": sha256_file(model_path),
    }), encoding="utf-8")

    def forbid_labels(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Inference attempted to read labels")

    monkeypatch.setattr(stim, "read_shot_data_file", forbid_labels)
    result = run_prediction(model_path, data_root, GROUP, "validation", tmp_path / "predictions")
    assert result["status"] == "complete"
    assert result["model_sha256"] == sha256_file(model_path)
    assert result["model_label_mode"] == "qrennd_global_data_parity"
    assert result["prediction_label_mode"] == "official_path_observable_flip"
    assert len(result["conditions"]) == 13
    for path in (tmp_path / "predictions").glob("*.npz"):
        split, _, probabilities = validate_prediction(path)
        assert split == "validation"
        np.testing.assert_array_equal(probabilities, np.full(5120, 0.5, dtype=np.float32))
    conditions = {item.name: item for item in discover_conditions(data_root)}
    hashes = {name: sha256_file(tmp_path / "predictions" / f"{name}.npz") for name in conditions}
    counts = {name: {"total": 5120, "errors": 0} for name in conditions}
    metadata = validate_inference_metadata(tmp_path / "predictions", "validation", conditions, hashes, counts)
    assert len(metadata) == 1
