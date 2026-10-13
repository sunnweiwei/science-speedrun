"""Paper input semantics, exact replay, canonical geometry and even-only labels."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest
import stim

PROJECT = Path(__file__).resolve().parents[1] / "environment" / "baseline" / "alphaqubit" if (Path(__file__).resolve().parents[1] / "environment" / "baseline" / "alphaqubit").is_dir() else Path(__file__).resolve().parents[1]  # the baseline project: environment/baseline/alphaqubit here, /opt/task in the verifier image
sys.path.insert(0, str(PROJECT))
from baselines.alphaqubit_data import (  # noqa: E402
    AlphaQubitLayout, AlphaQubitSampler, _scale_dem, load_hardware, load_hardware_packed,
)
from baselines.data import Condition, discover_conditions, group_conditions, hardware_split_indices  # noqa: E402


def make_condition(root: Path, distance: int = 3, basis: str = "X", rounds: int = 3) -> Condition:
    name = f"surface_code_b{basis}_d{distance}_r{rounds:02d}_center_3_5"
    folder = root / name
    folder.mkdir()
    generator = f"surface_code:rotated_memory_{basis.lower()}"
    ideal = stim.Circuit.generated(generator, distance=distance, rounds=rounds)
    noisy = stim.Circuit.generated(generator, distance=distance, rounds=rounds,
                                   after_clifford_depolarization=0.01,
                                   before_measure_flip_probability=0.02,
                                   after_reset_flip_probability=0.01)
    ideal.to_file(folder / "circuit_ideal.stim")
    noisy.to_file(folder / "circuit_noisy.stim")
    dem = noisy.detector_error_model(decompose_errors=True)
    dem.to_file(folder / "circuit_detector_error_model.dem")
    dem.to_file(folder / "pij_from_even_for_odd.dem")
    return Condition(folder, name, distance, basis, rounds, f"d{distance}_b{basis}_center_3_5")


@pytest.fixture(scope="module")
def hardware_root() -> Path:
    value = os.environ.get("QEC_DATA_ROOT") or os.environ.get("SYCAMORE_DATA_ROOT")
    if value is None:
        pytest.skip("Set QEC_DATA_ROOT for circuit/allowed-even Sycamore integration checks")
    return Path(value)


@pytest.mark.parametrize("distance,basis,rounds", [(3, "X", 1), (3, "Z", 3), (5, "X", 3), (5, "Z", 25)])
def test_known_measurement_history_roundtrip_and_final_masks(tmp_path: Path, distance: int, basis: str, rounds: int) -> None:
    condition = make_condition(tmp_path, distance, basis, rounds)
    layout = AlphaQubitLayout.from_condition(condition)
    rng = np.random.default_rng(23)
    measurements = rng.integers(0, 2, size=(17, rounds + 1, distance**2 - 1), dtype=np.uint8)
    measurements[:, 0, ~layout.on_basis] = 0  # Published initial off-basis gauge.
    measurements[:, -1, ~layout.on_basis] = 0
    previous = np.concatenate([np.zeros_like(measurements[:, :1]), measurements[:, :-1]], axis=1)
    events = measurements ^ previous
    events[:, -1, ~layout.on_basis] = 0  # Undefined final checks are not events.
    circuit = stim.Circuit.from_file(condition.path / "circuit_ideal.stim")
    spatial = {tuple(coords): slot for slot, coords in enumerate(layout.physical_stabilizer_coords)}
    detectors = np.empty((17, circuit.num_detectors), dtype=np.uint8)
    for detector, (x, y, time) in circuit.get_detector_coordinates().items():
        detectors[:, detector] = events[:, int(time), spatial[(x, y)]]
    features = layout.decode_features(detectors)
    np.testing.assert_array_equal(features["measurements"], measurements)
    np.testing.assert_array_equal(features["events"], events)
    np.testing.assert_array_equal(features["previous_events"][:, 1:], events[:, :-1])
    assert np.count_nonzero(features["previous_events"][:, 0]) == 0
    assert features["valid_mask"].shape == (rounds + 1, distance**2 - 1)
    assert features["valid_mask"][:-1].all(), "First off-basis zeros must not use final-missing embeddings"
    np.testing.assert_array_equal(features["valid_mask"][-1], layout.on_basis)
    np.testing.assert_array_equal(features["is_final"], [False] * rounds + [True])
    for key in ("measurements", "events", "previous_events"):
        assert features[key].dtype == np.float32
    packed = np.packbits(detectors, axis=1, bitorder="little")
    for key, expected in features.items():
        np.testing.assert_array_equal(layout.decode_packed_features(packed)[key], expected)


@pytest.mark.parametrize("source", ["pij", "xeb", "dem", "circuit"])
def test_curriculum_sampler_exact_resume_and_noise_zero(tmp_path: Path, source: str) -> None:
    condition = make_condition(tmp_path)
    sampler = AlphaQubitSampler(condition, seed=71, noise_source=source)
    empty, labels = sampler.sample_packed(256, factor=0)
    assert empty.dtype == labels.dtype == np.uint8
    assert not np.any(empty) and not np.any(labels)
    sampler.sample_packed(128, factor=0.5)
    state = sampler.state_dict()
    expected = [sampler.sample_packed(shots, factor) for shots, factor in ((67, 0.7), (512, 1.0), (128, 0.5))]
    restored = AlphaQubitSampler(condition, seed=71, noise_source=source)
    restored.load_state_dict(state)
    for (shots, factor), (detectors, labels) in zip(((67, 0.7), (512, 1.0), (128, 0.5)), expected):
        actual_dets, actual_labels = restored.sample_packed(shots, factor)
        np.testing.assert_array_equal(actual_dets, detectors)
        np.testing.assert_array_equal(actual_labels, labels)
    assert sum(np.count_nonzero(values[0]) for values in expected) > 0
    assert restored.state_dict() == sampler.state_dict()
    features, y = sampler.sample(32, 0.6)
    assert features["measurements"].shape == (32, 4, 8)
    assert y.shape == (32, 1) and y.dtype == np.float32
    altered = {**state, "seed": 99}
    with pytest.raises(ValueError, match="source/configuration"):
        restored.load_state_dict(altered)
    with pytest.raises(ValueError, match="draw_index"):
        restored.load_state_dict({**state, "draw_index": -1})


def test_dem_scaling_preserves_correlated_hyperedges(tmp_path: Path) -> None:
    condition = make_condition(tmp_path, distance=5)
    dem = stim.DetectorErrorModel.from_file(condition.path / "circuit_detector_error_model.dem")
    scaled = _scale_dem(dem, 0.6)
    errors = 0
    decomposed = 0
    for original, changed in zip(dem.flattened(), scaled):
        assert original.type == changed.type
        assert original.targets_copy() == changed.targets_copy()
        if original.type == "error":
            assert changed.args_copy()[0] == pytest.approx(original.args_copy()[0] * 0.6)
            errors += 1
            decomposed += any(target.is_separator() for target in original.targets_copy())
        else:
            assert original == changed
    assert errors > 0 and decomposed > 0


def test_hardware_loader_never_parses_poisoned_odd_labels(tmp_path: Path) -> None:
    condition = make_condition(tmp_path)
    width = (condition.rounds * (condition.distance**2 - 1) + 7) // 8
    packed = np.zeros((50_000, width), dtype=np.uint8)
    packed[:, 0] = np.arange(50_000, dtype=np.uint64) % 256
    packed.tofile(condition.path / "detection_events.b8")
    labels = np.empty((50_000, 2), dtype=np.uint8)
    labels[:, 1] = 10
    labels[::2, 0] = 48 + np.arange(25_000) % 2
    labels[1::2, 0] = ord("!")  # Any full-file label parser must reject this.
    labels.tofile(condition.path / "obs_flips_actual.01")
    for split in ("train", "validation"):
        selected, actual = load_hardware_packed(condition, split)
        indices = hardware_split_indices(split)
        np.testing.assert_array_equal(selected, packed[indices])
        np.testing.assert_array_equal(actual[:, 0], labels[indices, 0] - 48)
    with pytest.raises(ValueError, match="even rows only"):
        load_hardware(condition, "test")  # type: ignore[arg-type]


def test_all_real_geometries_have_canonical_paths_and_consistent_order(hardware_root: Path) -> None:
    groups = group_conditions(hardware_root)
    assert sum(map(len, groups.values())) == 130
    for conditions in groups.values():
        reference = AlphaQubitLayout.from_condition(conditions[0])
        for condition in conditions:
            layout = AlphaQubitLayout.from_condition(condition)
            assert layout.logical_index == 0
            assert layout.logical_axis == (1 if condition.basis == "X" else 0)
            np.testing.assert_array_equal(layout.stabilizer_positions, reference.stabilizer_positions)
            assert len(set(map(tuple, layout.stabilizer_positions))) == condition.distance**2 - 1
            assert np.all((layout.stabilizer_positions >= 0) & (layout.stabilizer_positions <= condition.distance))
            assert set(map(tuple, layout.data_positions)) == {(i, j) for i in range(condition.distance) for j in range(condition.distance)}
            assert np.count_nonzero(layout.stabilizer_types == 0) == (condition.distance**2 - 1) // 2
            assert layout.metadata()["logical_index"] == 0


def read_selected_bits(path: Path, bits: int, indices: np.ndarray) -> np.ndarray:
    mapped = np.memmap(path, dtype=np.uint8, mode="r", shape=(50_000, (bits + 7) // 8))
    return np.unpackbits(np.array(mapped[indices]), axis=1, count=bits, bitorder="little").astype(np.bool_)


@pytest.mark.parametrize("distance,basis", [(3, "X"), (5, "Z")])
def test_even_measurements_sweeps_reproduce_official_events_labels_and_gauged_syndromes(hardware_root: Path, distance: int, basis: str) -> None:
    condition = next(item for item in discover_conditions(hardware_root) if item.distance == distance and item.basis == basis and item.rounds == 3)
    layout = AlphaQubitLayout.from_condition(condition)
    circuit = stim.Circuit.from_file(condition.path / "circuit_ideal.stim")
    indices = np.arange(0, 256, 2, dtype=np.int64)  # Only 128 allowed even shots.
    measurements = read_selected_bits(condition.path / "measurements.b8", circuit.num_measurements, indices)
    sweeps = read_selected_bits(condition.path / "sweep.b8", circuit.num_sweep_bits, indices)
    converter = circuit.compile_m2d_converter()
    detectors, labels = converter.convert(measurements=measurements, sweep_bits=sweeps, separate_observables=True)
    selected_dets, selected_labels = load_hardware_packed(condition, "train")
    np.testing.assert_array_equal(detectors, np.unpackbits(selected_dets[:128], axis=1, count=circuit.num_detectors, bitorder="little"))
    np.testing.assert_array_equal(labels, selected_labels[:128])
    features = layout.decode_features(detectors)

    # Independently reconstruct raw stabilizer readouts from measurement-record
    # indices, then remove the reference/sweep frame and initial off-basis gauge.
    spatial = {tuple(coords): slot for slot, coords in enumerate(layout.physical_stabilizer_coords)}
    qubits = circuit.get_final_qubit_coordinates()
    raw = np.zeros((128, condition.rounds + 1, layout.num_stabilizers), dtype=np.bool_)
    offset = frame = 0
    first_data = circuit.num_measurements - distance**2
    for instruction in circuit.flattened():
        if instruction.name == "M":
            targets = instruction.targets_copy()
            if frame < condition.rounds:
                for local, target in enumerate(targets):
                    raw[:, frame, spatial[tuple(qubits[target.value][:2])]] = measurements[:, offset + local]
            offset += len(targets)
            frame += 1
        elif instruction.name == "DETECTOR" and instruction.gate_args_copy()[2] == condition.rounds:
            x, y, _ = instruction.gate_args_copy()
            data_indices = [offset + target.value for target in instruction.targets_copy() if offset + target.value >= first_data]
            raw[:, -1, spatial[(x, y)]] = np.logical_xor.reduce(measurements[:, data_indices], axis=1)
    raw[:, :-1, ~layout.on_basis] ^= raw[:, :1, ~layout.on_basis].copy()
    reference_events, _ = converter.convert(measurements=np.zeros_like(measurements), sweep_bits=sweeps, separate_observables=True)
    reference_grid = np.zeros_like(raw)
    for detector, (x, y, time) in circuit.get_detector_coordinates().items():
        reference_grid[:, int(time), spatial[(x, y)]] = reference_events[:, detector]
    expected = raw ^ np.logical_xor.accumulate(reference_grid, axis=1)
    expected[:, -1, ~layout.on_basis] = 0
    np.testing.assert_array_equal(features["measurements"], expected)
