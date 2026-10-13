"""Sycamore surface-code data in the two-input QRENND decoder format.

The recurrent input contains detector events, with absent first-round checks
set to zero. The evaluation input contains the final, on-basis detectors.
Targets are the release's *observable flips*, including the circuit reference
and sweep-bit correction; raw final-data measurement parity is not a target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

import numpy as np
from numpy.typing import NDArray
import stim


HardwareSplit = Literal["train", "validation", "test", "all"]
NoiseSource = Literal["circuit", "dem"]
Features = dict[str, NDArray[np.float32]]
_CONDITION_PATTERN = re.compile(
    r"surface_code_b(?P<basis>[XZ])_d(?P<distance>\d+)_r(?P<rounds>\d+)"
    r"_center_(?P<row>\d+)_(?P<col>\d+)"
)


@dataclass(frozen=True)
class Condition:
    """A single distance, basis, patch and round-count in the release."""

    path: Path
    name: str
    distance: int
    basis: str
    rounds: int
    group_id: str


def is_release_root(data_root: str | Path) -> bool:
    """A development release root (`/workspace/data` in the agent container) rather than the raw Zenodo directory."""
    return (Path(data_root) / "train" / "manifest.json").is_file()


def discover_conditions(data_root: str | Path) -> list[Condition]:
    """Discover surface-code experiments, excluding repetition-code data.

    Accepts the raw Zenodo directory or a development release root. For the latter each
    condition's path is its `train/conditions/<name>` folder, which holds the ideal circuit
    and the even-fitted DEM; train/validation arrays are read by `load_release_packed`.
    """
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Sycamore data directory does not exist: {root}")
    scan = root / "train" / "conditions" if is_release_root(root) else root
    conditions: list[Condition] = []
    for path in scan.iterdir():
        match = _CONDITION_PATTERN.fullmatch(path.name)
        if match is None or not path.is_dir():
            continue
        fields = match.groupdict()
        distance, rounds = int(fields["distance"]), int(fields["rounds"])
        if distance < 3 or distance % 2 != 1 or rounds < 1:
            raise ValueError(f"Unsupported surface-code condition: {path.name}")
        conditions.append(
            Condition(
                path=path,
                name=path.name,
                distance=distance,
                basis=fields["basis"],
                rounds=rounds,
                group_id=(f"d{distance}_b{fields['basis']}_center_"
                          f"{fields['row']}_{fields['col']}"),
            )
        )
    if not conditions:
        raise ValueError(f"No surface-code conditions found in {root}")
    return sorted(conditions, key=lambda item: (item.group_id, item.rounds))


def group_conditions(data_root: str | Path) -> dict[str, list[Condition]]:
    """Group conditions sharing one decoder's geometry and memory basis."""
    groups: dict[str, list[Condition]] = {}
    for condition in discover_conditions(data_root):
        groups.setdefault(condition.group_id, []).append(condition)
    return groups


@dataclass(frozen=True)
class DetectorLayout:
    """Map official detector order to stable spatial and temporal axes.

    Coordinates are sorted lexicographically within a patch. The recurrent
    axis always contains all d²−1 stabilizers, including at R=1, where half
    have no initial detector. Those coordinates come from the first ancilla
    measurement instruction, not from the incomplete first detector slice.
    """

    distance: int
    rounds: int
    num_detectors: int
    stabilizer_coords: tuple[tuple[float, float], ...]
    eval_coords: tuple[tuple[float, float], ...]
    rec_detector_indices: NDArray[np.int64]
    eval_detector_indices: NDArray[np.int64]

    @classmethod
    def from_condition(cls, condition: Condition) -> DetectorLayout:
        circuit = stim.Circuit.from_file(condition.path / "circuit_ideal.stim")
        if circuit.num_observables != 1:
            raise ValueError("This adapter requires exactly one logical observable")
        num_stabilizers = condition.distance**2 - 1
        qubit_coords = circuit.get_final_qubit_coordinates()
        stabilizer_coords: tuple[tuple[float, float], ...] | None = None
        for instruction in circuit.flattened():
            if instruction.name in {"M", "MZ", "MR", "MRZ", "MX", "MRX"}:
                targets = instruction.targets_copy()
                if len(targets) != num_stabilizers:
                    raise ValueError("First measurement must read every ancilla")
                stabilizer_coords = tuple(sorted(
                    tuple(qubit_coords[target.value][:2]) for target in targets
                ))
                break
        if stabilizer_coords is None or len(set(stabilizer_coords)) != num_stabilizers:
            raise ValueError("Cannot identify all stabilizer coordinates")

        coordinate_to_slot = {coord: i for i, coord in enumerate(stabilizer_coords)}
        recurrent = np.full((condition.rounds, num_stabilizers), -1, dtype=np.int64)
        final: dict[tuple[float, float], int] = {}
        detector_coords = circuit.get_detector_coordinates()
        for detector, coords in detector_coords.items():
            if len(coords) != 3 or coords[2] != int(coords[2]):
                raise ValueError(f"Invalid detector coordinate {coords}")
            spatial = (coords[0], coords[1])
            if spatial not in coordinate_to_slot:
                raise ValueError(f"Detector is not on an ancilla: {coords}")
            time = int(coords[2])
            if 0 <= time < condition.rounds:
                slot = coordinate_to_slot[spatial]
                if recurrent[time, slot] != -1:
                    raise ValueError(f"Duplicate detector coordinate: {coords}")
                recurrent[time, slot] = detector
            elif time == condition.rounds:
                if spatial in final:
                    raise ValueError(f"Duplicate final detector coordinate: {coords}")
                final[spatial] = detector
            else:
                raise ValueError(f"Detector time is outside the condition: {coords}")

        eval_coords = tuple(sorted(final))
        initial_coords = tuple(stabilizer_coords[i] for i in np.flatnonzero(recurrent[0] >= 0))
        if len(final) != num_stabilizers // 2 or initial_coords != eval_coords:
            raise ValueError("Initial and final detectors must contain the same half of checks")
        if np.any(recurrent[1:] < 0):
            raise ValueError("Every intermediate round must contain every stabilizer")
        if circuit.num_detectors != condition.rounds * num_stabilizers:
            raise ValueError("Unexpected number of surface-code detectors")
        evaluation = np.array([final[coord] for coord in eval_coords], dtype=np.int64)
        recurrent.setflags(write=False)
        evaluation.setflags(write=False)
        return cls(
            distance=condition.distance,
            rounds=condition.rounds,
            num_detectors=circuit.num_detectors,
            stabilizer_coords=stabilizer_coords,
            eval_coords=eval_coords,
            rec_detector_indices=recurrent,
            eval_detector_indices=evaluation,
        )

    def decode_features(self, dets: NDArray[np.generic]) -> Features:
        """Convert unpacked [shots, detectors] bits to float32 model inputs."""
        dets = np.asarray(dets)
        if dets.ndim != 2 or dets.shape[1] != self.num_detectors:
            raise ValueError(f"Expected [shots, {self.num_detectors}] unpacked detectors")
        # Indexing the absent first-round checks at zero is safe only because
        # these locations are immediately overwritten with the zero padding.
        recurrent = dets[:, np.maximum(self.rec_detector_indices, 0)].astype(np.float32)
        recurrent[:, self.rec_detector_indices < 0] = 0.0
        evaluation = dets[:, self.eval_detector_indices].astype(np.float32)
        return {"rec_input": recurrent, "eval_input": evaluation}

    def decode_packed_features(self, dets: NDArray[np.uint8]) -> Features:
        """Decode Stim b8 detector bytes (little-endian bits within each byte)."""
        dets = np.asarray(dets)
        if dets.dtype != np.uint8 or dets.ndim != 2 or dets.shape[1] != (self.num_detectors + 7) // 8:
            raise ValueError("Expected uint8 [shots, ceil(num_detectors / 8)] detector bytes")
        return self.decode_features(
            np.unpackbits(dets, axis=1, count=self.num_detectors, bitorder="little")
        )


def hardware_split_indices(split: HardwareSplit, num_shots: int = 50_000) -> NDArray[np.int64]:
    """Return original shot indices, retaining the published even/odd split.

    Training uses the first 19,880 even-index shots, validation the remaining
    5,120 even-index shots, and testing all 25,000 odd-index shots.
    """
    if split == "all":
        return np.arange(num_shots, dtype=np.int64)
    if num_shots != 50_000:
        raise ValueError(f"The predefined hardware splits require 50,000 shots, got {num_shots}")
    if split == "train":
        return np.arange(0, 39_760, 2, dtype=np.int64)
    if split == "validation":
        return np.arange(39_760, 50_000, 2, dtype=np.int64)
    if split == "test":
        return np.arange(1, 50_000, 2, dtype=np.int64)
    raise ValueError(f"Unknown hardware split: {split!r}")


def load_release_packed(condition: Condition, split: HardwareSplit) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Train/validation rows of a development release condition, in the same order as the raw split indices."""
    if split not in {"train", "validation"}:
        raise ValueError("A development release holds train/validation rows only")
    arrays = condition.path.parents[2] / split / "conditions" / condition.name / "arrays.npz"
    with np.load(arrays, allow_pickle=False) as payload:
        detectors = np.asarray(payload["detectors"], dtype=np.uint8)
        labels = np.asarray(payload["labels"], dtype=np.uint8).reshape(-1, 1)
    if len(detectors) != len(hardware_split_indices(split)) or labels.shape != (len(detectors), 1):
        raise ValueError(f"Unexpected {split} release arrays: {condition.name}")
    return detectors, labels


def load_hardware_packed(
    condition: Condition, split: HardwareSplit
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Load released detector bytes and observable labels in matching shot order."""
    if (condition.path / "arrays.npz").is_file():
        return load_release_packed(condition, split)
    num_detectors = condition.rounds * (condition.distance**2 - 1)
    detector_bytes = (num_detectors + 7) // 8
    raw = np.fromfile(condition.path / "detection_events.b8", dtype=np.uint8)
    if raw.size % detector_bytes:
        raise ValueError(f"Truncated detector data: {condition.name}")
    packed = raw.reshape(-1, detector_bytes)
    labels = stim.read_shot_data_file(
        path=str(condition.path / "obs_flips_actual.01"),
        format="01", num_observables=1,
    ).astype(np.uint8)
    if labels.shape != (len(packed), 1):
        raise ValueError(f"Detector and observable shot counts disagree: {condition.name}")
    indices = hardware_split_indices(split, len(packed))
    return packed[indices], labels[indices]


def load_hardware(condition: Condition, split: HardwareSplit) -> tuple[Features, NDArray[np.float32]]:
    """Load recurrent/evaluation inputs and the official [shots, 1] targets."""
    packed, labels = load_hardware_packed(condition, split)
    layout = DetectorLayout.from_condition(condition)
    return layout.decode_packed_features(packed), labels.astype(np.float32)


class CircuitSampler:
    """Seeded simulation from the released noisy circuit or its circuit DEM.

    DEM sampling preserves each hyperedge's joint detector and observable
    flips. It never samples a reweighted DEM fitted to hardware test shots.
    Reproducibility requires the same Stim version and sequence of shot counts,
    because Stim does not guarantee random streams across batching choices.
    """

    def __init__(self, condition: Condition, seed: int, noise_source: NoiseSource = "circuit") -> None:
        self.condition = condition
        self.layout = DetectorLayout.from_condition(condition)
        self.num_detectors = self.layout.num_detectors
        self.noise_source = noise_source
        if noise_source == "circuit":
            circuit = stim.Circuit.from_file(condition.path / "circuit_noisy.stim")
            if circuit.num_detectors != self.num_detectors or circuit.num_observables != 1:
                raise ValueError("Noisy and ideal circuit dimensions disagree")
            self._sampler = circuit.compile_detector_sampler(seed=seed)
        elif noise_source == "dem":
            dem = stim.DetectorErrorModel.from_file(condition.path / "circuit_detector_error_model.dem")
            if dem.num_detectors != self.num_detectors or dem.num_observables != 1:
                raise ValueError("Circuit DEM and ideal circuit dimensions disagree")
            self._sampler = dem.compile_sampler(seed=seed)
        else:
            raise ValueError(f"Unknown noise source: {noise_source!r}")

    def sample_packed(self, shots: int) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """Return detector b8 bytes and unpacked uint8 [shots, 1] labels."""
        if shots < 0:
            raise ValueError("shots must be nonnegative")
        if self.noise_source == "circuit":
            detectors, observables = self._sampler.sample(
                shots, separate_observables=True, bit_packed=True,
            )
        else:
            detectors, observables, _ = self._sampler.sample(shots, bit_packed=True)
        labels = np.unpackbits(observables, axis=1, count=1, bitorder="little")
        return detectors, labels

    def sample(self, shots: int) -> tuple[Features, NDArray[np.float32]]:
        """Return simulated model inputs and official observable-flip targets."""
        detectors, labels = self.sample_packed(shots)
        return self.layout.decode_packed_features(detectors), labels.astype(np.float32)
