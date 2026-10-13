"""Hard-input AlphaQubit features for the released Sycamore memory experiments.

Source: Bausch et al., Nature 635, 834–840 (2024), Methods "Soft measurement
inputs versus soft event inputs", "Input representation", "Attention bias",
Extended Data Fig. 4b, and Supplementary Algorithm 4.

Each shot has R bulk ancilla frames followed by one inferred final boundary.
Measurements are the cumulative XOR of the official detector events, in their
ideal-reference/sweep-corrected frame. The initial off-basis frame is gauged to
zero, as described for Pauli-noise data in the paper. This reconstructs the
published detector-equivalent hard input; it does not reconstruct raw analogue
measurements, leakage, or the unknown off-basis initial physical state.

Final off-basis zeros are placeholders, marked invalid for the model's learned
missing-stabilizer embedding. Labels are official logical-path flips, never
QRENND global-data parity. Hardware labels are loaded from even rows only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import NDArray
import stim

from .data import Condition, DetectorLayout, hardware_split_indices, load_release_packed


PAPER_URL = "https://www.nature.com/articles/s41586-024-08148-8"
HardwareSplit = Literal["train", "validation"]
NoiseSource = Literal["pij", "xeb", "dem", "circuit"]
Features = dict[str, NDArray[Any]]
_READOUTS = {"M", "MX", "MY", "MR", "MRX", "MRY"}
_NOISE_FILES = {
    "pij": "pij_from_even_for_odd.dem",
    "xeb": "circuit_detector_error_model.dem",
    "circuit": "circuit_noisy.stim",
}


def _square_coordinates(
    data_coords: NDArray[np.float64], stabilizer_coords: NDArray[np.float64], distance: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Undo Sycamore's 45-degree layout; also accept Stim's square test layout."""
    transforms = (
        np.array([[0.5, 0.5], [-0.5, 0.5]]),  # Mirror square columns so both official paths have index 0.
        np.array([[0.5, 0.0], [0.0, 0.5]]),   # Stim rotated_memory_x/z coordinates.
    )
    expected = {(i, j) for i in range(distance) for j in range(distance)}
    for transform in transforms:
        transformed = data_coords @ transform.T
        origin = transformed.min(axis=0)
        data = transformed - origin
        if not np.allclose(data, np.rint(data), atol=1e-8, rtol=0):
            continue
        data_positions = np.rint(data).astype(np.int64)
        if set(map(tuple, data_positions)) != expected:
            continue
        stabilizers = stabilizer_coords @ transform.T - origin + 0.5
        if not np.allclose(stabilizers, np.rint(stabilizers), atol=1e-8, rtol=0):
            continue
        positions = np.rint(stabilizers).astype(np.int64)
        if np.any(positions < 0) or np.any(positions > distance) or len(set(map(tuple, positions))) != len(positions):
            continue
        return positions, data_positions, transform
    raise ValueError("Qubit coordinates do not form the expected square surface-code lattice")


@dataclass(frozen=True)
class AlphaQubitLayout:
    """Static feature order and geometry shared by all shots of a condition.

    ``logical_axis`` is the fixed coordinate of the official logical path:
    0 denotes a row and 1 a column. ``logical_index`` identifies that row or
    column in the d×d data-qubit grid. These describe the observable itself;
    the paper's perpendicular pooling direction is a model-side operation.
    """

    condition: Condition
    detector_layout: DetectorLayout
    stabilizer_positions: NDArray[np.int64]
    data_positions: NDArray[np.int64]
    physical_stabilizer_coords: NDArray[np.float64]
    coordinate_transform: NDArray[np.float64]
    on_basis: NDArray[np.bool_]
    stabilizer_types: NDArray[np.int64]
    logical_axis: int
    logical_index: int
    logical_data_indices: tuple[int, ...]
    valid_mask: NDArray[np.bool_]
    is_final: NDArray[np.bool_]
    circuit_sha256: str

    @property
    def distance(self) -> int:
        return self.condition.distance

    @property
    def rounds(self) -> int:
        return self.condition.rounds

    @property
    def num_stabilizers(self) -> int:
        return self.distance**2 - 1

    @property
    def num_detectors(self) -> int:
        return self.detector_layout.num_detectors

    @classmethod
    def from_condition(cls, condition: Condition) -> AlphaQubitLayout:
        base = DetectorLayout.from_condition(condition)
        source = (condition.path / "circuit_ideal.stim").read_bytes()
        circuit = stim.Circuit(source.decode())
        measured: list[int] = []
        final_qubits: list[int] = []
        observable_records: set[int] = set()
        for instruction in circuit.flattened():
            if stim.gate_data(instruction.name).produces_measurements:
                if instruction.name not in _READOUTS:
                    raise ValueError("Only single-qubit ancilla/data readouts are supported")
                final_qubits = [target.value for target in instruction.targets_copy()]
                measured.extend(final_qubits)
            elif instruction.name == "OBSERVABLE_INCLUDE":
                if instruction.gate_args_copy() != [0.0]:
                    raise ValueError("Expected one official logical-path observable")
                for target in instruction.targets_copy():
                    if not target.is_measurement_record_target:
                        raise ValueError("Logical observable must be a measurement-record parity")
                    index = len(measured) + target.value
                    observable_records.symmetric_difference_update({index})
        if len(final_qubits) != condition.distance**2 or len(set(final_qubits)) != len(final_qubits):
            raise ValueError("Final readout must contain the d² data qubits")
        first_data = len(measured) - len(final_qubits)
        if len(observable_records) != condition.distance or any(index < first_data or index >= len(measured) for index in observable_records):
            raise ValueError("Official observable must be one weight-d final-data path")
        logical_data_indices = tuple(sorted(index - first_data for index in observable_records))
        qubit_coords = circuit.get_final_qubit_coordinates()
        data_coords = np.array([qubit_coords[qubit][:2] for qubit in final_qubits], dtype=np.float64)
        physical = np.array(base.stabilizer_coords, dtype=np.float64)
        positions, data_positions, transform = _square_coordinates(data_coords, physical, condition.distance)
        logical_positions = data_positions[list(logical_data_indices)]
        fixed_axes = [axis for axis in (0, 1) if len(set(logical_positions[:, axis])) == 1]
        if len(fixed_axes) != 1:
            raise ValueError("Official logical path is not one complete data-grid row or column")
        logical_axis = fixed_axes[0]
        on_basis = np.array([coord in base.eval_coords for coord in base.stabilizer_coords], dtype=np.bool_)
        memory_type = 0 if condition.basis == "X" else 1
        stabilizer_types = np.where(on_basis, memory_type, 1 - memory_type).astype(np.int64)
        valid = np.ones((condition.rounds + 1, condition.distance**2 - 1), dtype=np.bool_)
        valid[-1] = on_basis
        final = np.zeros(condition.rounds + 1, dtype=np.bool_)
        final[-1] = True
        for array in (positions, data_positions, physical, transform, on_basis, stabilizer_types, valid, final):
            array.setflags(write=False)
        return cls(condition, base, positions, data_positions, physical, transform, on_basis,
                   stabilizer_types, logical_axis, int(logical_positions[0, logical_axis]),
                   logical_data_indices, valid, final, hashlib.sha256(source).hexdigest())

    def decode_features(self, detectors: NDArray[Any]) -> Features:
        """Return float32 [batch,R+1,S] values and unbatched static masks."""
        detectors = np.asarray(detectors)
        if detectors.ndim != 2 or detectors.shape[1] != self.num_detectors:
            raise ValueError(f"Expected unpacked [batch,{self.num_detectors}] detectors")
        if not np.all((detectors == 0) | (detectors == 1)):
            raise ValueError("Hard detector values must be binary")
        native = self.detector_layout.decode_features(detectors)
        events = np.zeros((len(detectors), self.rounds + 1, self.num_stabilizers), dtype=np.uint8)
        events[:, :-1] = native["rec_input"].astype(np.uint8)
        events[:, -1, self.on_basis] = native["eval_input"].astype(np.uint8)
        measurements = np.bitwise_xor.accumulate(events, axis=1)
        measurements[:, -1, ~self.on_basis] = 0
        previous = np.zeros_like(events)
        previous[:, 1:] = events[:, :-1]
        return {
            "measurements": measurements.astype(np.float32),
            "events": events.astype(np.float32),
            "previous_events": previous.astype(np.float32),
            "valid_mask": self.valid_mask,
            "is_final": self.is_final,
        }

    def decode_packed_features(self, detectors: NDArray[np.uint8]) -> Features:
        detectors = np.asarray(detectors)
        if detectors.dtype != np.uint8 or detectors.ndim != 2 or detectors.shape[1] != (self.num_detectors + 7) // 8:
            raise ValueError("Expected uint8 Stim b8 detector bytes")
        return self.decode_features(np.unpackbits(detectors, axis=1, count=self.num_detectors, bitorder="little"))

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "paper": PAPER_URL, "condition": self.condition.name,
            "distance": self.distance, "bulk_frames": self.rounds, "total_frames": self.rounds + 1,
            "stabilizer_positions": self.stabilizer_positions.tolist(),
            "physical_stabilizer_coords": self.physical_stabilizer_coords.tolist(),
            "coordinate_transform": self.coordinate_transform.tolist(),
            "coordinate_convention": "Sycamore square row=(physical_row+physical_col)/2; column=(physical_col-physical_row)/2; translated to local grids",
            "data_positions": self.data_positions.tolist(), "on_basis": self.on_basis.tolist(),
            "stabilizer_types": self.stabilizer_types.tolist(), "stabilizer_type_encoding": "0=X,1=Z inherited CSS types",
            "logical_axis": self.logical_axis, "logical_index": self.logical_index,
            "logical_data_indices": list(self.logical_data_indices),
            "measurement_frame": "cumulative XOR of official ideal-reference/sweep-corrected events; initial off-basis frame gauged to zero",
            "final_boundary": "additional inferred on-basis frame; off-basis placeholders require learned missing embedding",
            "input_information": "hard measurements and events only; no raw data bits, soft readout or leakage",
            "label_mode": "official_path_observable_flip", "circuit_sha256": self.circuit_sha256,
        }


def load_hardware_packed(condition: Condition, split: HardwareSplit) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Select development rows before parsing labels; never parse odd labels."""
    if split not in {"train", "validation"}:
        raise ValueError("AlphaQubit development loader permits train/validation even rows only")
    if (condition.path / "arrays.npz").is_file():   # development release layout (the agent's /workspace/data)
        return load_release_packed(condition, split)
    indices = hardware_split_indices(split)
    width = (condition.rounds * (condition.distance**2 - 1) + 7) // 8
    detector_path, label_path = condition.path / "detection_events.b8", condition.path / "obs_flips_actual.01"
    if detector_path.stat().st_size != 50_000 * width or label_path.stat().st_size != 100_000:
        raise ValueError("Expected 50,000 byte-aligned detector shots and one-bit LF-delimited labels")
    packed = np.memmap(detector_path, dtype=np.uint8, mode="r", shape=(50_000, width))
    label_rows = np.memmap(label_path, dtype=np.uint8, mode="r", shape=(50_000, 2))
    selected = np.array(label_rows[indices])
    if not np.all(selected[:, 1] == 10) or not np.all((selected[:, 0] == 48) | (selected[:, 0] == 49)):
        raise ValueError("Invalid selected even-index labels")
    return np.array(packed[indices]), (selected[:, :1] - 48).astype(np.uint8)


def load_hardware(condition: Condition, split: HardwareSplit) -> tuple[Features, NDArray[np.float32]]:
    detectors, labels = load_hardware_packed(condition, split)
    return AlphaQubitLayout.from_condition(condition).decode_packed_features(detectors), labels.astype(np.float32)


def _scale_dem(dem: stim.DetectorErrorModel, factor: float) -> stim.DetectorErrorModel:
    if factor == 1:
        return dem
    scaled = stim.DetectorErrorModel()
    for instruction in dem.flattened():
        if instruction.type == "error":
            scaled.append("error", instruction.args_copy()[0] * factor, instruction.targets_copy())
        else:
            scaled.append(instruction)
    return scaled


def _scale_circuit(circuit: stim.Circuit, factor: float) -> stim.Circuit:
    if factor == 1:
        return circuit
    scaled = stim.Circuit()
    for instruction in circuit.flattened():
        arguments = instruction.gate_args_copy()
        if stim.gate_data(instruction.name).is_noisy_gate:
            arguments = [value * factor for value in arguments]
        scaled.append(instruction.name, instruction.targets_copy(), arguments)
    return scaled


class AlphaQubitSampler:
    """Streaming draws with a compact, exactly restorable per-draw RNG cursor.

    ``pij`` uses the even-fitted DEM (including validation syndromes, no odd
    labels); ``xeb``/``dem`` uses the released circuit-derived DEM. Curriculum
    factors scale each independent DEM error probability without splitting
    hyperedges. ``circuit`` instead scales the underlying gate-error channel
    probabilities and samples the released noisy circuit directly.

    A seed derived from source hash, base seed, draw index, factor and shot
    count initializes each draw. Restoring the cursor reproduces future draws
    without replaying earlier samples. Bitwise replay requires the same Stim
    version and machine instruction set, following Stim's seed guarantees.
    """

    def __init__(self, condition: Condition, seed: int, noise_source: NoiseSource = "pij") -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.condition = condition
        self.layout = AlphaQubitLayout.from_condition(condition)
        self.noise_source = "xeb" if noise_source == "dem" else noise_source
        if self.noise_source not in _NOISE_FILES:
            raise ValueError(f"Unsupported noise source: {noise_source}")
        source = (condition.path / _NOISE_FILES[self.noise_source]).read_bytes()
        self.source_sha256 = hashlib.sha256(source).hexdigest()
        self.seed = seed
        self.draw_index = 0
        self._base = stim.Circuit(source.decode()) if self.noise_source == "circuit" else stim.DetectorErrorModel(source.decode())
        if self._base.num_detectors != self.layout.num_detectors or self._base.num_observables != 1:
            raise ValueError("Simulation source dimensions disagree with ideal circuit")
        self._scaled: dict[float, Any] = {1.0: self._base}

    def state_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "condition": self.condition.name, "seed": self.seed,
                "noise_source": self.noise_source, "source_sha256": self.source_sha256,
                "circuit_sha256": self.layout.circuit_sha256, "stim_version": stim.__version__, "draw_index": self.draw_index}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        current = self.state_dict()
        if set(state) != set(current) or any(state[key] != value for key, value in current.items() if key != "draw_index"):
            raise ValueError("Sampler state source/configuration does not match")
        draw_index = state["draw_index"]
        if isinstance(draw_index, bool) or not isinstance(draw_index, int) or draw_index < 0:
            raise ValueError("Sampler draw_index must be a nonnegative integer")
        self.draw_index = draw_index

    def sample_packed(self, shots: int, factor: float = 1.0) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
            raise ValueError("shots must be a positive integer")
        factor = float(factor)
        if not math.isfinite(factor) or not 0 <= factor <= 1:
            raise ValueError("Noise factor must be finite and in [0,1]")
        if factor not in self._scaled:
            self._scaled[factor] = _scale_circuit(self._base, factor) if self.noise_source == "circuit" else _scale_dem(self._base, factor)
        seed_material = f"alphaqubit-draw-v1|{self.source_sha256}|{self.seed}|{self.draw_index}|{shots}|{factor.hex()}".encode()
        draw_seed = int.from_bytes(hashlib.blake2b(seed_material, digest_size=8).digest(), "little")
        model = self._scaled[factor]
        if self.noise_source == "circuit":
            detectors, observables = model.compile_detector_sampler(seed=draw_seed).sample(shots, bit_packed=True, separate_observables=True)
        else:
            detectors, observables, _ = model.compile_sampler(seed=draw_seed).sample(shots, bit_packed=True)
        self.draw_index += 1
        labels = np.unpackbits(observables, axis=1, count=1, bitorder="little")
        return detectors, labels

    def sample(self, shots: int, factor: float = 1.0) -> tuple[Features, NDArray[np.float32]]:
        detectors, labels = self.sample_packed(shots, factor)
        return self.layout.decode_packed_features(detectors), labels.astype(np.float32)
