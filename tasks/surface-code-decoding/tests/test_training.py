"""Small CPU exercises for fixed-data training, resume, and split isolation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import stim
import tensorflow as tf

from baselines.cache import prepare
from baselines.data import group_conditions
from baselines.train import (
    PackedSequence,
    _load_allowed_hardware,
    argument_parser,
    checkpoint_state,
    evaluate_validation,
    load_populations,
    run_training,
    sha256,
)


def small_data(tmp_path: Path) -> tuple[Path, Path, str]:
    root = tmp_path / "data"
    root.mkdir()
    for rounds in range(1, 26, 2):
        condition = root / f"surface_code_bZ_d3_r{rounds:02d}_center_3_3"
        condition.mkdir()
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=3, rounds=rounds,
            after_clifford_depolarization=0.01,
            before_measure_flip_probability=0.01,
            after_reset_flip_probability=0.01,
        )
        circuit.to_file(condition / "circuit_noisy.stim")
        circuit.without_noise().to_file(condition / "circuit_ideal.stim")
    group = "d3_bZ_center_3_3"
    cache = prepare(root, tmp_path / "cache", group, seed=13, shots=312, validation_shots=8)
    return root, cache, group


def test_sequence_covers_population_and_resumes_exact_cursor(tmp_path: Path) -> None:
    root, cache, group = small_data(tmp_path)
    training, _, _ = load_populations(root, cache, group, "pretrain")
    sequence = PackedSequence(training, batch_size=8, seed=71, epoch=2)
    resumed = PackedSequence(training, batch_size=8, seed=71, epoch=2, start_batch=2)
    assert sum(sequence.batch_samples(i) for i in range(len(sequence))) == 312
    seen: dict[int, list[int]] = {index: [] for index in range(13)}
    for group_index, start, end in sequence.batches:
        seen[group_index].extend(sequence.shot_orders[group_index][start:end].tolist())
    assert all(sorted(indices) == list(range(24)) for indices in seen.values())
    for index in range(len(resumed)):
        expected_inputs, expected_labels = sequence[index + 2]
        inputs, labels = resumed[index]
        for name in inputs:
            np.testing.assert_array_equal(inputs[name], expected_inputs[name])
        for name in labels:
            np.testing.assert_array_equal(labels[name], expected_labels[name])
        np.testing.assert_array_equal(labels["main_output"], labels["aux_output"])


def test_fixed_cache_training_checkpoint_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, cache, group = small_data(tmp_path)
    output = tmp_path / "run"
    args = argument_parser().parse_args([
        "--data-root", str(root), "--cache-dir", str(cache), "--group", group,
        "--output", str(output), "--hours", "1", "--epochs", "3",
        "--max-steps", "3", "--batch-size", "8", "--eval-seconds", "0.01",
        "--workers", "1", "--allow-cpu",
    ])
    tf.keras.backend.clear_session()
    result = run_training(args)
    assert result["stop_reason"] == "max_steps"
    assert result["global_steps"] == result["optimizer_iterations"] == 3
    assert result["seen_samples"] == 24
    assert result["next_batch"] == 3
    assert checkpoint_state(output / "last.keras")["global_steps"] == 3
    assert (output / "best.keras").is_file()
    history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
    assert any(row["trigger"] == "interval" and row["global_steps"] < 3 for row in history)
    assert all(row["complete"] and np.isfinite(row["loss"]) for row in history)

    args.max_steps = 5
    resumed = run_training(args)
    assert resumed["global_steps"] == resumed["optimizer_iterations"] == 5
    assert resumed["seen_samples"] == 40
    assert resumed["next_batch"] == 5
    assert resumed["elapsed_seconds"] > result["elapsed_seconds"]
    metadata = json.loads((output / "metadata.json").read_text())
    assert len(metadata["sessions"]) == 2
    assert metadata["sessions"][1]["resume_global_steps"] == 3
    assert metadata["identity"]["label_mode"] == "qrennd_global_data_parity"
    assert metadata["source_sha256"]["train.py"] == sha256(Path("baselines/train.py"))

    # Manual validation must not reset or update the ongoing fit metric states.
    training, validation, _ = load_populations(root, cache, group, "pretrain")
    model = tf.keras.models.load_model(output / "last.keras")
    assert any(np.any(variable.numpy()) for variable in model.optimizer.variables()[1:])
    inputs, labels = PackedSequence(training, 8, 0, 0)[0]
    model.train_on_batch(inputs, labels)
    metrics_before = [variable.numpy().copy() for metric in model.metrics for variable in metric.variables]
    validation_result = evaluate_validation(model, validation, lambda: True, batch_size=8)
    metrics_after = [variable.numpy() for metric in model.metrics for variable in metric.variables]
    assert validation_result["complete"]
    for before, after in zip(metrics_before, metrics_after):
        np.testing.assert_array_equal(before, after)

    last_hash = sha256(output / "last.keras")
    args.seed = 4
    with pytest.raises(ValueError, match="Resume settings/data differ"):
        run_training(args)
    assert sha256(output / "last.keras") == last_hash

    args.seed = 0
    with monkeypatch.context() as context:
        def changed_transform_hash(path: Path) -> str:
            return "0" * 64 if path.name == "observables.py" else sha256(path)
        context.setattr("baselines.train.sha256", changed_transform_hash)
        with pytest.raises(ValueError, match="Resume settings/data differ"):
            run_training(args)
    assert sha256(output / "last.keras") == last_hash

    # Cross an epoch boundary after restoring a partial epoch, without replaying
    # already-seen batches. Also exercise the measured-throughput warmup window.
    args.max_steps = 40
    args.eval_seconds = 60
    next_epoch = run_training(args)
    assert next_epoch["global_steps"] == next_epoch["optimizer_iterations"] == 40
    assert next_epoch["seen_samples"] == 320
    assert next_epoch["epoch"] == 1 and next_epoch["next_batch"] == 1
    assert next_epoch["steady_steps"] == 25
    assert next_epoch["steady_samples_per_second"] > 0


def test_hardware_training_does_not_parse_odd_test_labels(tmp_path: Path) -> None:
    root, _, group = small_data(tmp_path)
    condition = group_conditions(root)[group][0]
    width = condition.rounds * (condition.distance**2 - 1) // 8
    np.zeros((50_000, width), dtype=np.uint8).tofile(condition.path / "detection_events.b8")
    # Deliberately invalid odd rows fail if any full-file label parser is used.
    (condition.path / "obs_flips_actual.01").write_bytes(b"0\nx\n" * 25_000)
    train_dets, train_labels = _load_allowed_hardware(condition, "train")
    val_dets, val_labels = _load_allowed_hardware(condition, "validation")
    assert train_dets.shape == (19_880, width)
    assert val_dets.shape == (5_120, width)
    assert not train_labels.any() and not val_labels.any()


def test_corrupt_cache_and_nonempty_output_are_rejected(tmp_path: Path) -> None:
    root, cache, group = small_data(tmp_path)
    manifest = json.loads((cache / "manifest.json").read_text())
    corrupted = cache / manifest["conditions"][0]["files"]["train_labels"]
    with corrupted.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_populations(root, cache, group, "pretrain")
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "do-not-overwrite.txt").write_text("existing results")
    args = argument_parser().parse_args([
        "--data-root", str(root), "--cache-dir", str(cache), "--group", group,
        "--output", str(output), "--hours", "1", "--allow-cpu",
    ])
    with pytest.raises(ValueError, match="nonempty"):
        run_training(args)
