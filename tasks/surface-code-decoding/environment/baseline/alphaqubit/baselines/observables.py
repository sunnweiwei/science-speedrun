"""Exact conversion between Sycamore path flips and QRENND all-data parity.

Native QRENND labels XOR all final data-qubit measurement *flips*, whereas
Sycamore's supplied label XORs a weight-d logical path. Their difference is a
stabilizer product. This module derives that product from the ideal circuit:
it solves, over GF(2), ``all_data_row XOR path_row = c @ detector_rows`` in
measurement-record space. Subtracting the ideal reference, including sweep
effects, preserves this linear relation. No raw measurements, sweep bits,
ground-truth labels, or noise parameters are used to derive the transform.

Training applies y_global = y_path XOR parity(detectors[c]). Evaluation applies
p_path = 1-p_global when this parity is one. Both native network heads can
therefore use QRENND's global-parity target while reported predictions retain
the official Sycamore observable convention.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
from numpy.typing import NDArray
import stim

from .data import Condition


_BYTE_PARITY = np.array([value.bit_count() % 2 for value in range(256)], dtype=np.uint8)
_BYTE_PARITY.setflags(write=False)
_SINGLE_QUBIT_READOUTS = {"M", "MX", "MY", "MR", "MRX", "MRY"}


def _record_mask(instruction: stim.CircuitInstruction, measurements_so_far: int) -> int:
    mask = 0
    for target in instruction.targets_copy():
        if not target.is_measurement_record_target:
            raise ValueError(f"Expected measurement-record parity, found {instruction}")
        index = measurements_so_far + target.value
        if not 0 <= index < measurements_so_far:
            raise ValueError(f"Measurement-record index is outside the circuit: {instruction}")
        mask ^= 1 << index
    return mask


def _rowspace_solution(rows: list[int], target: int) -> tuple[int, ...]:
    """Solve XOR(selected rows) == target using integer bitset elimination."""
    basis: dict[int, tuple[int, int]] = {}
    for index, row in enumerate(rows):
        combination = 1 << index
        while row:
            pivot = row.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = row, combination
                break
            prior_row, prior_combination = basis[pivot]
            row ^= prior_row
            combination ^= prior_combination
    solution = 0
    while target:
        pivot = target.bit_length() - 1
        if pivot not in basis:
            raise ValueError("Global data parity and path observable are not equivalent modulo detector parities")
        row, combination = basis[pivot]
        target ^= row
        solution ^= combination
    return tuple(index for index in range(len(rows)) if (solution >> index) & 1)


@dataclass(frozen=True)
class ObservableTransform:
    """A circuit-derived binary parity, with no dependence on shot labels."""

    num_detectors: int
    num_measurements: int
    num_data_qubits: int
    detector_indices: tuple[int, ...]
    circuit_sha256: str
    packed_byte_indices: NDArray[np.int64]
    packed_byte_masks: NDArray[np.uint8]

    @classmethod
    def from_condition(cls, condition: Condition) -> ObservableTransform:
        path = condition.path / "circuit_ideal.stim"
        source = path.read_bytes()
        circuit = stim.Circuit(source.decode("utf-8"))
        if circuit.num_observables != 1:
            raise ValueError("Observable conversion requires exactly one official path observable")
        rows: list[int] = []
        official_row = 0
        measurements = 0
        final_readout_qubits: list[int] = []
        for instruction in circuit.flattened():
            if stim.gate_data(instruction.name).produces_measurements:
                if instruction.name not in _SINGLE_QUBIT_READOUTS:
                    raise ValueError(f"Unsupported measurement operation: {instruction.name}")
                final_readout_qubits = [target.value for target in instruction.targets_copy()]
                measurements += len(final_readout_qubits)
            elif instruction.name == "DETECTOR":
                rows.append(_record_mask(instruction, measurements))
            elif instruction.name == "OBSERVABLE_INCLUDE":
                if instruction.gate_args_copy() != [0.0]:
                    raise ValueError("Expected observable index zero")
                official_row ^= _record_mask(instruction, measurements)
        num_data = condition.distance**2
        if len(final_readout_qubits) != num_data or len(set(final_readout_qubits)) != num_data:
            raise ValueError("Final measurement instruction must read each of the d² data qubits once")
        if measurements != circuit.num_measurements or len(rows) != circuit.num_detectors:
            raise ValueError("Parsed circuit dimensions disagree with Stim")
        all_data_row = ((1 << num_data) - 1) << (measurements - num_data)
        detector_indices = _rowspace_solution(rows, all_data_row ^ official_row)
        # Independently verify the returned GF(2) certificate before use.
        reconstructed = official_row
        for index in detector_indices:
            reconstructed ^= rows[index]
        if reconstructed != all_data_row:
            raise AssertionError("Incorrect observable-transform rowspace certificate")
        masks = np.zeros((circuit.num_detectors + 7) // 8, dtype=np.uint8)
        for index in detector_indices:
            masks[index // 8] ^= np.uint8(1 << (index % 8))
        byte_indices = np.flatnonzero(masks).astype(np.int64)
        byte_masks = masks[byte_indices]
        byte_indices.setflags(write=False)
        byte_masks.setflags(write=False)
        return cls(
            num_detectors=circuit.num_detectors,
            num_measurements=measurements,
            num_data_qubits=num_data,
            detector_indices=detector_indices,
            circuit_sha256=hashlib.sha256(source).hexdigest(),
            packed_byte_indices=byte_indices,
            packed_byte_masks=byte_masks,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "method": "exact GF(2) detector-rowspace certificate from ideal circuit measurement records",
            "source_target": "official logical-path measurement flips",
            "training_target": "parity of all final data-qubit measurement flips",
            "num_data_qubits": self.num_data_qubits,
            "num_detectors": self.num_detectors,
            "detector_indices": list(self.detector_indices),
            "circuit_ideal_sha256": self.circuit_sha256,
            "probability_conversion": "p_path = 1-p_global iff selected detector parity is one",
        }

    def _parity(self, detectors: NDArray[Any]) -> NDArray[np.uint8]:
        detectors = np.asarray(detectors)
        if detectors.ndim != 2 or detectors.shape[1] != self.num_detectors:
            raise ValueError(f"Expected unpacked detector shape [shots, {self.num_detectors}]")
        if not np.all((detectors == 0) | (detectors == 1)):
            raise ValueError("Unpacked detectors must be binary")
        return np.logical_xor.reduce(detectors[:, self.detector_indices], axis=1).astype(np.uint8)

    def _parity_packed(self, detectors: NDArray[np.uint8]) -> NDArray[np.uint8]:
        detectors = np.asarray(detectors)
        if detectors.dtype != np.uint8 or detectors.ndim != 2 or detectors.shape[1] != (self.num_detectors + 7) // 8:
            raise ValueError("Expected uint8 b8 detector bytes [shots, ceil(num_detectors / 8)]")
        selected = detectors[:, self.packed_byte_indices] & self.packed_byte_masks
        return np.bitwise_xor.reduce(_BYTE_PARITY[selected], axis=1)

    @staticmethod
    def _labels(labels: NDArray[Any], parity: NDArray[np.uint8]) -> NDArray[np.uint8]:
        labels = np.asarray(labels)
        if labels.shape not in {(len(parity),), (len(parity), 1)}:
            raise ValueError("Path labels must have shape [shots] or [shots, 1]")
        if not np.all((labels == 0) | (labels == 1)):
            raise ValueError("Path labels must be binary")
        return labels.reshape(-1, 1).astype(np.uint8) ^ parity[:, None]

    @staticmethod
    def _probabilities(probabilities: NDArray[Any], parity: NDArray[np.uint8]) -> NDArray[np.float32]:
        probabilities = np.asarray(probabilities)
        if probabilities.shape not in {(len(parity),), (len(parity), 1)}:
            raise ValueError("Global probabilities must have shape [shots] or [shots, 1]")
        if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("Global probabilities must be finite and lie in [0, 1]")
        probabilities = probabilities.reshape(-1, 1).astype(np.float32)
        return np.where(parity[:, None], 1 - probabilities, probabilities).astype(np.float32)

    def path_to_global(self, detectors_unpacked: NDArray[Any], path_labels: NDArray[Any]) -> NDArray[np.uint8]:
        """Convert official labels to native global-parity labels, uint8 [N, 1]."""
        return self._labels(path_labels, self._parity(detectors_unpacked))

    def path_to_global_packed(self, detectors_packed: NDArray[np.uint8], path_labels: NDArray[Any]) -> NDArray[np.uint8]:
        """Same transform directly from Stim little-endian b8 detector bytes."""
        return self._labels(path_labels, self._parity_packed(detectors_packed))

    def global_to_path_probabilities(self, detectors_unpacked: NDArray[Any], global_probs: NDArray[Any]) -> NDArray[np.float32]:
        """Convert native probabilities to official-path probabilities, float32 [N, 1]."""
        return self._probabilities(global_probs, self._parity(detectors_unpacked))

    def global_to_path_probabilities_packed(self, detectors_packed: NDArray[np.uint8], global_probs: NDArray[Any]) -> NDArray[np.float32]:
        """Probability conversion directly from Stim little-endian b8 bytes."""
        return self._probabilities(global_probs, self._parity_packed(detectors_packed))
