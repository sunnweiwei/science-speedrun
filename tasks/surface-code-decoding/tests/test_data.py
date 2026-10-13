"""Detector geometry, split isolation and experimental reference-bit checks.

Set QEC_DATA_ROOT to the extracted Zenodo release to enable the integration
checks. The small generated-circuit checks always run without downloaded data.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pytest
import stim

PROJECT = Path(__file__).resolve().parents[1] / "environment" / "baseline" / "alphaqubit" if (Path(__file__).resolve().parents[1] / "environment" / "baseline" / "alphaqubit").is_dir() else Path(__file__).resolve().parents[1]  # the baseline project: environment/baseline/alphaqubit here, /opt/task in the verifier image
sys.path.insert(0, str(PROJECT))
from baselines.data import (  # noqa: E402
    CircuitSampler,
    Condition,
    DetectorLayout,
    discover_conditions,
    group_conditions,
    hardware_split_indices,
    load_hardware,
    load_hardware_packed,
)


def make_condition(tmp_path: Path, distance: int, basis: str, rounds: int) -> Condition:
    name = f"surface_code_b{basis}_d{distance}_r{rounds:02d}_center_5_5"
    path = tmp_path / name
    path.mkdir()
    generator = f"surface_code:rotated_memory_{basis.lower()}"
    ideal = stim.Circuit.generated(generator, distance=distance, rounds=rounds)
    noisy = stim.Circuit.generated(
        generator, distance=distance, rounds=rounds,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.02,
        after_reset_flip_probability=0.01,
    )
    ideal.to_file(path / "circuit_ideal.stim")
    noisy.to_file(path / "circuit_noisy.stim")
    noisy.detector_error_model(decompose_errors=True).to_file(path / "circuit_detector_error_model.dem")
    return Condition(path, name, distance, basis, rounds, f"d{distance}_b{basis}_center_5_5")


@pytest.fixture(scope="module")
def hardware_root() -> Path:
    value = os.environ.get("QEC_DATA_ROOT") or os.environ.get("SYCAMORE_DATA_ROOT")
    if value is None:
        pytest.skip("Set QEC_DATA_ROOT to run the downloaded Sycamore integration checks")
    root = Path(value)
    if not root.is_dir():
        pytest.fail(f"Configured Sycamore directory does not exist: {root}")
    return root


def check_geometry(condition: Condition) -> None:
    layout = DetectorLayout.from_condition(condition)
    m = condition.distance**2 - 1
    circuit = stim.Circuit.from_file(condition.path / "circuit_ideal.stim")
    # Each detector fires alone once, detecting omitted, duplicated, shifted or
    # permuted detector mappings without relying on stochastic samples.
    features = layout.decode_features(np.eye(circuit.num_detectors, dtype=np.bool_))
    recurrent, evaluation = features["rec_input"], features["eval_input"]
    assert recurrent.shape == (m * condition.rounds, condition.rounds, m)
    assert evaluation.shape == (m * condition.rounds, m // 2)
    assert recurrent.dtype == evaluation.dtype == np.float32
    np.testing.assert_array_equal(recurrent.sum(axis=(1, 2)) + evaluation.sum(axis=1), 1.0)
    for detector, (x, y, time) in circuit.get_detector_coordinates().items():
        if int(time) == condition.rounds:
            assert evaluation[detector, layout.eval_coords.index((x, y))] == 1.0
        else:
            assert recurrent[detector, int(time), layout.stabilizer_coords.index((x, y))] == 1.0
    off_basis = [i for i, coord in enumerate(layout.stabilizer_coords) if coord not in layout.eval_coords]
    assert len(off_basis) == m // 2
    assert np.all(recurrent[:, 0, off_basis] == 0)


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("rounds", [1, 3, 25])
def test_generated_geometry(tmp_path: Path, distance: int, basis: str, rounds: int) -> None:
    check_geometry(make_condition(tmp_path, distance, basis, rounds))


def test_discovery_numeric_order_and_grouping(tmp_path: Path) -> None:
    make_condition(tmp_path, 3, "X", 11)
    make_condition(tmp_path, 3, "X", 3)
    make_condition(tmp_path, 5, "Z", 1)
    (tmp_path / "repetition_code_bZ_d25_r50_center_5_5").mkdir()
    (tmp_path / "README").write_text("Not a condition", encoding="utf-8")
    groups = group_conditions(tmp_path)
    assert list(groups) == ["d3_bX_center_5_5", "d5_bZ_center_5_5"]
    assert [item.rounds for item in groups["d3_bX_center_5_5"]] == [3, 11]
    assert len(discover_conditions(tmp_path)) == 3


def test_hardware_splits_are_disjoint_and_exhaustive() -> None:
    train = hardware_split_indices("train")
    validation = hardware_split_indices("validation")
    test = hardware_split_indices("test")
    assert (len(train), len(validation), len(test)) == (19_880, 5_120, 25_000)
    assert (train[0], train[-1]) == (0, 39_758)
    assert (validation[0], validation[-1]) == (39_760, 49_998)
    assert (test[0], test[-1]) == (1, 49_999)
    assert np.all(train % 2 == 0) and np.all(validation % 2 == 0)
    assert np.all(test % 2 == 1)
    np.testing.assert_array_equal(
        np.sort(np.concatenate([train, validation, test])),
        hardware_split_indices("all"),
    )
    with pytest.raises(ValueError, match="50,000 shots"):
        hardware_split_indices("train", 1_000)
    with pytest.raises(ValueError, match="Unknown hardware split"):
        hardware_split_indices("invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize("noise_source", ["circuit", "dem"])
@pytest.mark.parametrize("distance", [3, 5])
def test_sampler_labels_shapes_and_reproducibility(tmp_path: Path, noise_source: str, distance: int) -> None:
    condition = make_condition(tmp_path, distance, "Z", 3)
    sampler_a = CircuitSampler(condition, seed=23, noise_source=noise_source)
    sampler_b = CircuitSampler(condition, seed=23, noise_source=noise_source)
    for shots in (1_024, 33):
        packed, labels = sampler_a.sample_packed(shots)
        other_packed, other_labels = sampler_b.sample_packed(shots)
        np.testing.assert_array_equal(packed, other_packed)
        np.testing.assert_array_equal(labels, other_labels)
        assert packed.shape == (shots, (sampler_a.num_detectors + 7) // 8)
        assert packed.dtype == labels.dtype == np.uint8
        assert labels.shape == (shots, 1)
        assert np.all((labels == 0) | (labels == 1))
        assert np.count_nonzero(packed) > 0
        features = sampler_a.layout.decode_packed_features(packed)
        unpacked = np.unpackbits(packed, axis=1, count=sampler_a.num_detectors, bitorder="little")
        direct = sampler_a.layout.decode_features(unpacked)
        assert set(features) == {"rec_input", "eval_input"}
        for key in features:
            assert features[key].dtype == np.float32
            assert np.isfinite(features[key]).all()
            np.testing.assert_array_equal(features[key], direct[key])
    features, y = sampler_a.sample(8)
    assert features["rec_input"].shape == (8, 3, distance**2 - 1)
    assert features["eval_input"].shape == (8, (distance**2 - 1) // 2)
    assert y.shape == (8, 1) and y.dtype == np.float32


def test_hardware_all_layouts_and_round_one_order(hardware_root: Path) -> None:
    groups = group_conditions(hardware_root)
    assert len(groups) == 10
    assert sum(map(len, groups.values())) == 130
    for conditions in groups.values():
        assert [condition.rounds for condition in conditions] == list(range(1, 26, 2))
        layouts = [DetectorLayout.from_condition(condition) for condition in conditions]
        assert all(layout.stabilizer_coords == layouts[0].stabilizer_coords for layout in layouts)
        assert all(layout.eval_coords == layouts[0].eval_coords for layout in layouts)
        for condition in conditions:
            check_geometry(condition)


@pytest.mark.parametrize("distance,basis", [(3, "Z"), (5, "X")])
def test_hardware_reference_conversion_and_split_loading(
    hardware_root: Path, distance: int, basis: str
) -> None:
    condition = next(
        item for item in discover_conditions(hardware_root)
        if item.distance == distance and item.basis == basis and item.rounds == 3
    )
    circuit = stim.Circuit.from_file(condition.path / "circuit_ideal.stim")
    measurements = stim.read_shot_data_file(
        path=str(condition.path / "measurements.b8"), format="b8",
        num_measurements=circuit.num_measurements,
    )
    sweeps = stim.read_shot_data_file(
        path=str(condition.path / "sweep.b8"), format="b8",
        num_measurements=circuit.num_sweep_bits,
    )
    computed_dets, computed_obs = circuit.compile_m2d_converter().convert(
        measurements=measurements, sweep_bits=sweeps, separate_observables=True,
    )
    packed, labels = load_hardware_packed(condition, "all")
    official_dets = np.unpackbits(packed, axis=1, count=circuit.num_detectors, bitorder="little")
    assert len(labels) == 50_000
    np.testing.assert_array_equal(computed_dets, official_dets)
    np.testing.assert_array_equal(computed_obs, labels)
    # A raw-data-parity target is a different observable and would be a silent
    # protocol bug even when its tensor dimensions look correct.
    raw_parity = np.logical_xor.reduce(measurements[:, -(distance**2):], axis=1)
    assert np.count_nonzero(raw_parity != labels[:, 0]) > 1_000
    all_features = DetectorLayout.from_condition(condition).decode_features(official_dets)
    for split in ("train", "validation", "test"):
        indices = hardware_split_indices(split)
        features, y = load_hardware(condition, split)
        np.testing.assert_array_equal(y, labels[indices].astype(np.float32))
        for key in all_features:
            np.testing.assert_array_equal(features[key], all_features[key][indices])
