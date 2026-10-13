"""Independent measurement-space checks of the native/global observable map.

QEC_DATA_ROOT enables checks on every released condition using only the first
256 even training shots and 128 independent synthetic shots per condition.
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
from baselines.data import Condition, discover_conditions
from baselines.observables import ObservableTransform


def _augmented_global_observable(circuit: stim.Circuit, distance: int) -> stim.Circuit:
    augmented = circuit.copy()
    augmented.append(
        "OBSERVABLE_INCLUDE", [stim.target_rec(i) for i in range(-distance**2, 0)], 1,
    )
    return augmented


def _check_both_representations(
    transform: ObservableTransform, detectors: np.ndarray, observations: np.ndarray,
) -> None:
    path = observations[:, :1]
    expected_global = observations[:, 1:2]
    packed = np.packbits(detectors, axis=1, bitorder="little")
    global_labels = transform.path_to_global(detectors, path)
    packed_labels = transform.path_to_global_packed(packed, path)
    assert global_labels.dtype == packed_labels.dtype == np.uint8
    np.testing.assert_array_equal(global_labels, expected_global)
    np.testing.assert_array_equal(packed_labels, expected_global)
    path_probability = transform.global_to_path_probabilities(detectors, global_labels)
    packed_probability = transform.global_to_path_probabilities_packed(packed, global_labels)
    assert path_probability.dtype == packed_probability.dtype == np.float32
    np.testing.assert_array_equal(path_probability, path)
    np.testing.assert_array_equal(packed_probability, path)


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("rounds", [1, 3, 25])
def test_generated_measurement_parity_identity(
    tmp_path: Path, distance: int, basis: str, rounds: int,
) -> None:
    condition = Condition(tmp_path, "fixture", distance, basis, rounds, "fixture")
    generator = f"surface_code:rotated_memory_{basis.lower()}"
    ideal = stim.Circuit.generated(generator, distance=distance, rounds=rounds)
    ideal.to_file(tmp_path / "circuit_ideal.stim")
    noisy = stim.Circuit.generated(
        generator, distance=distance, rounds=rounds,
        after_clifford_depolarization=0.02, before_measure_flip_probability=0.02,
    )
    transform = ObservableTransform.from_condition(condition)
    augmented = _augmented_global_observable(noisy, distance)
    detectors, observations = augmented.compile_detector_sampler(seed=82).sample(
        256, separate_observables=True,
    )
    _check_both_representations(transform, detectors, observations)
    # Probability conversion is an affine reflection, not hard thresholding.
    probabilities = np.linspace(0.03, 0.97, len(detectors), dtype=np.float32)[:, None]
    reflected = transform.global_to_path_probabilities(detectors, probabilities)
    restored = transform.global_to_path_probabilities(detectors, reflected)
    np.testing.assert_allclose(restored, probabilities, atol=6e-8)
    assert np.any(np.abs(reflected - probabilities) > 0.1)
    assert transform.metadata()["num_data_qubits"] == distance**2
    with pytest.raises(ValueError, match="finite"):
        transform.global_to_path_probabilities(detectors, np.full_like(probabilities, np.nan))


def test_non_equivalent_observable_is_rejected(tmp_path: Path) -> None:
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3)
    # A single extra data measurement changes the stabilizer equivalence class.
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
    circuit.to_file(tmp_path / "circuit_ideal.stim")
    condition = Condition(tmp_path, "fixture", 3, "Z", 3, "fixture")
    with pytest.raises(ValueError, match="not equivalent"):
        ObservableTransform.from_condition(condition)


def _read_even_prefix(path: Path, width_bits: int) -> np.ndarray:
    width_bytes = (width_bits + 7) // 8
    raw = np.fromfile(path, dtype=np.uint8, count=512 * width_bytes)
    assert raw.size == 512 * width_bytes
    return np.unpackbits(
        raw.reshape(512, width_bytes)[::2], axis=1, count=width_bits, bitorder="little",
    ).astype(np.bool_)


def test_all_hardware_and_simulation_transforms() -> None:
    data_root = os.environ.get("QEC_DATA_ROOT") or os.environ.get("SYCAMORE_DATA_ROOT")
    if data_root is None:
        pytest.skip("Set QEC_DATA_ROOT to enable all-condition release integration checks")
    conditions = discover_conditions(data_root)
    assert len(conditions) == 130
    for condition in conditions:
        transform = ObservableTransform.from_condition(condition)
        circuit = stim.Circuit.from_file(condition.path / "circuit_ideal.stim")
        measurement = _read_even_prefix(condition.path / "measurements.b8", circuit.num_measurements)
        sweep = _read_even_prefix(condition.path / "sweep.b8", circuit.num_sweep_bits)
        augmented = _augmented_global_observable(circuit, condition.distance)
        detectors, observations = augmented.compile_m2d_converter().convert(
            measurements=measurement, sweep_bits=sweep, separate_observables=True,
        )
        official_detectors = _read_even_prefix(condition.path / "detection_events.b8", circuit.num_detectors)
        official_path = stim.read_shot_data_file(
            path=condition.path / "obs_flips_actual.01", format="01", num_observables=1,
        )[:512:2]
        np.testing.assert_array_equal(detectors, official_detectors)
        np.testing.assert_array_equal(observations[:, :1], official_path)
        _check_both_representations(transform, detectors, observations)
        noisy = stim.Circuit.from_file(condition.path / "circuit_noisy.stim")
        synthetic = _augmented_global_observable(noisy, condition.distance)
        dets, obs = synthetic.compile_detector_sampler(seed=701 + condition.rounds).sample(
            128, separate_observables=True,
        )
        _check_both_representations(transform, dets, obs)


def test_released_pauli_circuits_are_invariant_under_initial_sweeps() -> None:
    """Validate the cache's zero-sweep population against different initial states."""
    data_root = os.environ.get("QEC_DATA_ROOT") or os.environ.get("SYCAMORE_DATA_ROOT")
    if data_root is None:
        pytest.skip("Set QEC_DATA_ROOT to check release-specific Pauli sweep invariance")
    conditions = discover_conditions(data_root)
    for distance in (3, 5):
        for basis in ("X", "Z"):
            condition = next(item for item in conditions if (
                item.distance == distance and item.basis == basis and item.rounds == 3
            ))
            circuit = stim.Circuit.from_file(condition.path / "circuit_noisy.stim")
            expected_dem = circuit.detector_error_model(decompose_errors=True).flattened()
            expected = circuit.compile_detector_sampler(seed=41).sample(256, append_observables=True)
            patterns = (
                np.zeros(circuit.num_sweep_bits, dtype=np.bool_),
                np.ones(circuit.num_sweep_bits, dtype=np.bool_),
                np.arange(circuit.num_sweep_bits) % 2 == 1,
                np.random.default_rng(7).integers(0, 2, circuit.num_sweep_bits).astype(np.bool_),
            )
            for bits in patterns:
                fixed = stim.Circuit()
                for instruction in circuit.flattened():
                    targets = instruction.targets_copy()
                    if instruction.name == "CX" and any(target.is_sweep_bit_target for target in targets):
                        for control, target in zip(targets[::2], targets[1::2]):
                            if control.is_sweep_bit_target:
                                if bits[control.value]:
                                    fixed.append("X", [target.value])
                            else:
                                fixed.append("CX", [control, target])
                    else:
                        fixed.append(instruction)
                assert fixed.detector_error_model(decompose_errors=True).flattened() == expected_dem
                actual = fixed.compile_detector_sampler(seed=41).sample(256, append_observables=True)
                np.testing.assert_array_equal(actual, expected)
