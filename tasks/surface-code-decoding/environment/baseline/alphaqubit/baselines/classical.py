"""Reproduce PyMatching and score released Sycamore predictions on odd shots.

Usage: python classical.py --data-root /path/to/sycamore --output /path/to/run

Calibration is selected using labels from the last 5,120 even-indexed shots
per condition. The released even-calibrated DEM uses all even syndromes,
including validation syndromes; this is transductive unsupervised calibration,
not a fully held-out calibration set. It uses no odd-shot syndromes or labels.
Released BP and tensor-network predictions use the corresponding cross-fit
DEM. Released MWPM and correlated-matching predictions use the circuit DEM.
The historical correlated decoder and PyMatching 2.3's correlated decoder
are distinct implementations and are labeled separately.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import socket
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pymatching
import stim

if __package__:
    from .metrics import parse_condition, summarize_conditions
else:
    from metrics import parse_condition, summarize_conditions


DEMS = {
    "circuit": "circuit_detector_error_model.dem",
    "even_calibrated": "pij_from_even_for_odd.dem",
}
REFERENCE_FILES = {
    "released_mwpm": "obs_flips_predicted_by_pymatching.01",
    "released_correlated_matching": "obs_flips_predicted_by_correlated_matching.01",
    "released_belief_matching": "obs_flips_predicted_by_belief_matching.01",
    "released_tensor_network": "obs_flips_predicted_by_tensor_network_contraction.01",
}
DECODERS = ("pymatching_mwpm", "pymatching_correlated")
EXPECTED_SHOTS = 50_000
VALIDATION_SHOTS = 5_120


def expected_conditions() -> set[str]:
    return {
        f"surface_code_b{basis}_d{distance}_r{rounds:02d}_center_{row}_{col}"
        for distance, centers in ((3, ((3, 5), (5, 3), (5, 7), (7, 5))), (5, ((5, 5),)))
        for row, col in centers
        for basis in ("X", "Z")
        for rounds in range(1, 26, 2)
    }


def discover_conditions(data_root: Path) -> list[Path]:
    """Fail incomplete/wrong releases before producing a partial score."""
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root is not a directory: {data_root}")
    paths = {path.name: path for path in data_root.glob("surface_code_*") if path.is_dir()}
    expected = expected_conditions()
    if set(paths) != expected:
        raise ValueError(
            f"Expected all 130 Sycamore conditions. Missing={sorted(expected - set(paths))}; "
            f"unexpected={sorted(set(paths) - expected)}"
        )
    required = (
        "circuit_ideal.stim", "detection_events.b8", "obs_flips_actual.01",
        *DEMS.values(), *REFERENCE_FILES.values(),
    )
    for name, folder in paths.items():
        parse_condition(name)
        for filename in required:
            if not (folder / filename).is_file():
                raise FileNotFoundError(folder / filename)
    return [paths[name] for name in sorted(paths)]


def split_indices(split: str) -> NDArray[np.int64]:
    """Public split contract shared with learned-decoder evaluators."""
    even = np.arange(0, EXPECTED_SHOTS, 2, dtype=np.int64)
    if split == "train":
        return even[:-VALIDATION_SHOTS]
    if split == "validation":
        return even[-VALIDATION_SHOTS:]
    if split == "test":
        return np.arange(1, EXPECTED_SHOTS, 2, dtype=np.int64)
    raise ValueError(f"Unknown split: {split!r}")


def _read_observables(path: Path) -> NDArray[np.bool_]:
    values = stim.read_shot_data_file(path=path, format="01", num_observables=1)
    if values.shape != (EXPECTED_SHOTS, 1):
        raise ValueError(f"{path}: expected {(EXPECTED_SHOTS, 1)}, found {values.shape}")
    return values[:, 0]


def load_condition(
    folder: Path, indices: NDArray[np.int64]
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    circuit = stim.Circuit.from_file(folder / "circuit_ideal.stim")
    if circuit.num_observables != 1:
        raise ValueError(f"{folder}: expected one observable")
    detection_events = stim.read_shot_data_file(
        path=folder / "detection_events.b8", format="b8", num_detectors=circuit.num_detectors
    )
    if detection_events.shape != (EXPECTED_SHOTS, circuit.num_detectors):
        raise ValueError(f"{folder}: wrong detection-event shape {detection_events.shape}")
    actual = _read_observables(folder / "obs_flips_actual.01")
    return detection_events[indices], actual[indices]


def decode_candidates(
    folder: Path, detection_events: NDArray[np.bool_], actual: NDArray[np.bool_]
) -> tuple[dict[str, NDArray[np.bool_]], dict[str, Any]]:
    errors: dict[str, NDArray[np.bool_]] = {}
    timing: dict[str, Any] = {}
    for calibration, filename in DEMS.items():
        dem = stim.DetectorErrorModel.from_file(folder / filename)
        if dem.num_detectors != detection_events.shape[1] or dem.num_observables != 1:
            raise ValueError(f"{folder / filename}: DEM dimensions do not match data")
        for decoder in DECODERS:
            correlated = decoder == "pymatching_correlated"
            method = f"{decoder}_{calibration}"
            start = time.perf_counter()
            matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=correlated)
            build_seconds = time.perf_counter() - start
            start = time.perf_counter()
            predicted = matching.decode_batch(detection_events, enable_correlations=correlated)
            decode_seconds = time.perf_counter() - start
            if predicted.shape != (len(actual), 1):
                raise ValueError(f"{method}: unexpected prediction shape {predicted.shape}")
            errors[method] = predicted[:, 0] != actual
            timing[method] = {
                "build_seconds": build_seconds,
                "decode_batch_seconds": decode_seconds,
                "shots": len(actual),
                "offline_batch_microseconds_per_shot": 1e6 * decode_seconds / len(actual),
            }
    return errors, timing


def _count(errors: NDArray[np.bool_]) -> dict[str, int]:
    return {"errors": int(np.count_nonzero(errors)), "total": len(errors)}


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _validation_objective(summary: Mapping[str, Any]) -> float:
    # Equal weight for the two distances; each distance first averages its
    # separate basis/patch fits. No odd labels enter this choice.
    return float(np.mean([
        summary["macro"][f"d{distance}"]["logical_error_per_round"] for distance in (3, 5)
    ]))


def run(data_root: Path, output: Path, *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    if pymatching.__version__ != "2.3.1":
        raise RuntimeError(f"Reproduction pins PyMatching 2.3.1, found {pymatching.__version__}")
    if bootstrap_samples <= 0:
        raise ValueError("Final evaluation requires a positive bootstrap sample count")
    folders = discover_conditions(data_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory to preserve existing results: {output}")
    output.mkdir(parents=True, exist_ok=True)
    paired_directory = output / "paired_errors"
    paired_directory.mkdir()
    started = time.perf_counter()
    metadata: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root.resolve()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "pymatching": pymatching.__version__,
        "stim": stim.__version__,
        "numpy": np.__version__,
        "conditions": len(folders),
        "split": {
            "index_base": 0,
            "train": "first 19,880 even-indexed shots per condition",
            "validation": "last 5,120 even-indexed shots per condition",
            "test": "all 25,000 odd-indexed shots per condition",
            "calibration_caveat": (
                "Released even-calibrated DEM uses ALL even syndromes, including validation. "
                "It uses neither validation labels nor odd test syndromes/labels. "
                "Validation is label-held-out, not fully held out from unsupervised DEM estimation."
            ),
        },
        "historical_predictions": {
            "released_mwpm": "released PyMatching predictions configured with circuit DEM",
            "released_correlated_matching": "Google internal historical decoder; circuit DEM; distinct from modern PyMatching correlations",
            "released_belief_matching": "Google internal BP/MWPM decoder; odd predictions use even-calibrated DEM",
            "released_tensor_network": "Google internal tensor contraction decoder; odd predictions use even-calibrated DEM",
        },
        "latency_scope": "single offline batch including decode call overhead; not streaming latency or real-time deployment evidence",
    }
    _write_json(output / "metadata.json", metadata)
    validation_counts: dict[str, dict[str, dict[str, int]]] = {}
    validation_timing: dict[str, Any] = {}
    for index, folder in enumerate(folders):
        dets, actual = load_condition(folder, split_indices("validation"))
        errors, timing = decode_candidates(folder, dets, actual)
        validation_timing[folder.name] = timing
        for method, failures in errors.items():
            validation_counts.setdefault(method, {})[folder.name] = _count(failures)
        print(json.dumps({"stage": "validation", "complete": index + 1, "total": len(folders), "condition": folder.name}), flush=True)
    validation_scores = {
        method: summarize_conditions(counts, bootstrap_samples=0, seed=seed)
        for method, counts in validation_counts.items()
    }
    selected: dict[str, str] = {}
    for decoder in DECODERS:
        candidates = [f"{decoder}_{calibration}" for calibration in DEMS]
        # Stable tie-breaking favors the circuit DEM, declared first above.
        selected[f"{decoder}_selected"] = min(candidates, key=lambda method: _validation_objective(validation_scores[method]))
    validation = {
        "selection_rule": "separately for MWPM and correlated, choose one global calibration minimizing the equal-distance mean of macro LER on validation; ties favor circuit DEM",
        "objective_values": {method: _validation_objective(score) for method, score in validation_scores.items()},
        "selected": selected,
        "locked_before_odd_evaluation_utc": datetime.now(timezone.utc).isoformat(),
        "scores": validation_scores,
        "timing": validation_timing,
    }
    _write_json(output / "validation.json", validation)
    print(json.dumps({"stage": "selection_locked", "selected": selected}), flush=True)

    test_counts: dict[str, dict[str, dict[str, int]]] = {}
    test_timing: dict[str, Any] = {}
    test_indices = split_indices("test")
    for index, folder in enumerate(folders):
        dets, actual = load_condition(folder, test_indices)
        errors, timing = decode_candidates(folder, dets, actual)
        test_timing[folder.name] = timing
        for method, filename in REFERENCE_FILES.items():
            prediction = _read_observables(folder / filename)[test_indices]
            errors[method] = prediction != actual
        for alias, method in selected.items():
            errors[alias] = errors[method]
        for method, failures in errors.items():
            test_counts.setdefault(method, {})[folder.name] = _count(failures)
        np.savez_compressed(
            paired_directory / f"{folder.name}.npz",
            shot_indices=test_indices,
            actual=actual,
            **{f"errors__{method}": failures for method, failures in errors.items()},
        )
        print(json.dumps({"stage": "test", "complete": index + 1, "total": len(folders), "condition": folder.name}), flush=True)
    _write_json(output / "condition_counts.json", test_counts)
    _write_json(output / "timing.json", test_timing)
    scores = {
        method: summarize_conditions(counts, bootstrap_samples=bootstrap_samples, seed=seed)
        for method, counts in test_counts.items()
    }
    result = {
        "metadata": metadata,
        "selected": selected,
        "primary_comparators": list(selected),
        "diagnostic_comparators": [method for method in scores if method not in selected],
        "scores": scores,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "paired_error_format": "paired_errors/<condition>.npz: shot_indices, actual, errors__<method>; all methods share the saved indices",
    }
    _write_json(output / "scores.json", result)
    print(json.dumps({"stage": "complete", "output": str(output.resolve()), "wall_seconds": result["wall_seconds"]}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    run(args.data_root, args.output, bootstrap_samples=args.bootstrap_samples, seed=args.seed)


if __name__ == "__main__":
    main()
