"""Extract one public validation shot without exposing any test data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import stim


def extract(condition: Path) -> dict[str, Any]:
    shot = 39760
    if condition.name != "surface_code_bZ_d3_r09_center_3_5":
        raise ValueError("This figure fixes the d3/Z/r9/center_3_5 condition")
    circuit = stim.Circuit.from_file(condition / "circuit_ideal.stim")
    coords = circuit.get_final_qubit_coordinates()
    ancillas: list[int] = []
    for instruction in circuit.flattened():
        if instruction.name in {"M", "MZ", "MR", "MRZ", "MX", "MRX"}:
            ancillas = [target.value for target in instruction.targets_copy()]
            break
    width = (circuit.num_detectors + 7) // 8
    with (condition / "detection_events.b8").open("rb") as handle:
        handle.seek(shot * width)
        packed = np.frombuffer(handle.read(width), dtype=np.uint8)
    bits = np.unpackbits(packed, bitorder="little")[:circuit.num_detectors]
    with (condition / "obs_flips_actual.01").open() as handle:
        for index, line in enumerate(handle):
            if index == shot:
                target = int(line.strip())
                break
        else:
            raise ValueError("Validation label is missing")
    detector_coords = circuit.get_detector_coordinates()
    filenames = ["circuit_ideal.stim", "detection_events.b8", "obs_flips_actual.01"]
    return {
        "schema": "surface-code-science-example/v1",
        "split": "validation",
        "source_url": "https://zenodo.org/records/6804040",
        "source_license": "CC-BY-4.0",
        "source_attribution": "Google Quantum AI, Sycamore surface-code memory experiments",
        "condition": condition.name,
        "source_shot_index_zero_based": shot,
        "selection": "First validation shot for the fixed d3/Z/r9/center_3_5 condition; no prediction-based selection",
        "source_sha256": {name: hashlib.sha256((condition / name).read_bytes()).hexdigest() for name in filenames},
        "distance": 3,
        "basis": "Z",
        "rounds": 9,
        "data_qubits": [xy[:2] for q, xy in coords.items() if q not in ancillas],
        "check_qubits": [coords[q][:2] for q in ancillas],
        "detectors": [{"id": i, "coords": detector_coords[i], "event": int(bit)} for i, bit in enumerate(bits)],
        "target_logical_flip": target,
        "model_probability": None,
        "output_note": "Decoder probability and threshold are symbolic, not an inference result. The label is a public validation target.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("example.json"))
    args = parser.parse_args()
    args.output.write_text(json.dumps(extract(args.condition), indent=2) + "\n")


if __name__ == "__main__":
    main()
