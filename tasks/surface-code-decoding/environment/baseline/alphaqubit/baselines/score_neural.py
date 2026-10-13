"""Trusted scoring of complete, identified neural probability submissions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import stim

if __package__:
    from .data import discover_conditions, hardware_split_indices
    from .metrics import compare_paired_conditions, parse_condition, summarize_conditions, wilson_interval
    from .predict import atomic_json, sha256_file
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from baselines.data import discover_conditions, hardware_split_indices
    from baselines.metrics import compare_paired_conditions, parse_condition, summarize_conditions, wilson_interval
    from baselines.predict import atomic_json, sha256_file


CLASSICAL_METHODS = (
    "pymatching_mwpm_selected", "pymatching_correlated_selected",
    "released_tensor_network", "released_belief_matching",
)
NLL_EPSILON = 1e-7


def expected_condition_names(group: str | None = None) -> set[str]:
    expected = {
        f"surface_code_b{basis}_d{distance}_r{rounds:02d}_center_{row}_{col}"
        for distance, centers in ((3, ((3, 5), (5, 3), (5, 7), (7, 5))), (5, ((5, 5),)))
        for row, col in centers for basis in ("X", "Z") for rounds in range(1, 26, 2)
    }
    if group is not None:
        expected = {name for name in expected if parse_condition(name).group == group}
        if not expected:
            raise ValueError(f"Unknown Sycamore group: {group}")
    return expected


def validate_prediction(path: Path) -> tuple[str, NDArray[np.uint64], NDArray[np.float32]]:
    """Infer the split only if every original shot ID exactly matches it."""
    with np.load(path, allow_pickle=False) as record:
        if set(record.files) != {"shot_indices", "probabilities"}:
            raise ValueError(f"{path}: require exactly shot_indices and probabilities")
        indices, probabilities = record["shot_indices"], record["probabilities"]
    if indices.dtype != np.uint64 or indices.ndim != 1:
        raise ValueError(f"{path}: shot_indices must be a uint64 vector")
    if probabilities.dtype != np.float32 or probabilities.ndim != 1 or probabilities.shape != indices.shape:
        raise ValueError(f"{path}: probabilities must be a same-length float32 vector")
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError(f"{path}: probabilities must be finite and in [0, 1]")
    for split in ("validation", "test"):
        if np.array_equal(indices, hardware_split_indices(split)):
            return split, indices, probabilities
    raise ValueError(f"{path}: missing, duplicated, reordered or invalid original shot indices")


def discover_predictions(root: Path, expected: set[str]) -> dict[str, Path]:
    records: dict[str, Path] = {}
    for path in sorted(root.rglob("*.npz")):
        if path.stem in records:
            raise ValueError(f"Duplicate prediction condition: {path.stem}")
        records[path.stem] = path
    if set(records) != expected:
        raise ValueError(
            f"Prediction coverage mismatch; missing={sorted(expected - set(records))}; "
            f"unexpected={sorted(set(records) - expected)}"
        )
    return records


def calibration_record(probabilities: NDArray[Any], actual: NDArray[Any]) -> dict[str, float]:
    p, y = probabilities.astype(np.float64), actual.astype(np.float64)
    clipped = np.clip(p, NLL_EPSILON, 1 - NLL_EPSILON)
    return {
        "negative_log_likelihood": float(np.mean(-y * np.log(clipped) - (1 - y) * np.log1p(-clipped))),
        "brier_score": float(np.mean((p - y)**2)),
    }


def validate_inference_metadata(
    predictions: Path, split: str, conditions: dict[str, Any],
    provenance: dict[str, str], counts: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    """Bind completed inference provenance to exact predictions and input files."""
    metadata: list[dict[str, Any]] = []
    metadata_conditions: set[str] = set()
    for path in sorted(predictions.rglob("metadata.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "complete" or record.get("split") != split:
            raise ValueError(f"Prediction metadata is incomplete or has wrong split: {path}")
        if record.get("prediction_label_mode") != "official_path_observable_flip":
            raise ValueError(f"Prediction metadata has incompatible observable convention: {path}")
        if record.get("model_label_mode") not in {"qrennd_global_data_parity", "official_path_observable_flip"}:
            raise ValueError(f"Prediction metadata has incompatible model label convention: {path}")
        checkpoint = record.get("checkpoint_metadata", {})
        digest = record.get("model_sha256", "")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Prediction metadata lacks a valid model SHA256: {path}")
        if checkpoint.get("model_sha256") != digest or checkpoint.get("label_mode") != record["model_label_mode"] or checkpoint.get("group") != record.get("group"):
            raise ValueError(f"Checkpoint identity and inference metadata disagree: {path}")
        for name, condition_metadata in record.get("conditions", {}).items():
            if name in metadata_conditions or name not in provenance:
                raise ValueError(f"Duplicate or unexpected condition in prediction metadata: {name}")
            if parse_condition(name).group != record.get("group"):
                raise ValueError(f"Prediction metadata group disagrees with condition: {name}")
            if condition_metadata.get("prediction_sha256") != provenance[name]:
                raise ValueError(f"Prediction does not match recorded SHA256: {name}")
            if condition_metadata.get("shots") != counts[name]["total"]:
                raise ValueError(f"Prediction metadata shot count disagrees: {name}")
            for key, filename in (("detectors_sha256", "detection_events.b8"), ("circuit_sha256", "circuit_ideal.stim")):
                if condition_metadata.get(key) != sha256_file(conditions[name].path / filename):
                    raise ValueError(f"Inference input does not match recorded {key}: {name}")
            metadata_conditions.add(name)
        metadata.append(record)
    if metadata and metadata_conditions != set(provenance):
        raise ValueError("Inference metadata does not cover every prediction condition")
    return metadata


def safe_summary(counts: dict[str, dict[str, int]], bootstrap_samples: int, seed: int) -> dict[str, Any]:
    """Preserve undefined fits and every count; never drop offending rounds."""
    try:
        return {"status": "ok", **summarize_conditions(counts, bootstrap_samples=bootstrap_samples, seed=seed)}
    except ValueError as error:
        result: dict[str, Any] = {"status": "unfit", "reason": str(error), "per_condition": {}, "per_group": {}, "macro": {}}
    groups: dict[str, dict[str, dict[str, int]]] = {}
    for name, record in counts.items():
        condition = parse_condition(name)
        groups.setdefault(condition.group, {})[name] = record
        result["per_condition"][name] = {
            **record, "rounds": condition.rounds, "distance": condition.distance,
            "basis": condition.basis, "center": [condition.row, condition.col],
            "included_in_fit": condition.rounds != 1,
            "shot_error_rate": record["errors"] / record["total"],
            "shot_error_rate_ci95": wilson_interval(record["errors"], record["total"]),
        }
    for group, group_counts in groups.items():
        try:
            group_score = summarize_conditions(group_counts, bootstrap_samples=bootstrap_samples, seed=seed)
            result["per_group"][group] = {"status": "ok", **group_score["per_group"][group]}
        except ValueError as error:
            result["per_group"][group] = {
                "status": "unfit", "reason": str(error),
                "logical_error_per_round": None, "logical_error_per_round_ci95": None,
            }
    for distance in sorted({parse_condition(name).distance for name in counts}):
        distance_counts = {name: record for name, record in counts.items() if parse_condition(name).distance == distance}
        try:
            distance_score = summarize_conditions(distance_counts, bootstrap_samples=bootstrap_samples, seed=seed)
            result["macro"][f"d{distance}"] = {"status": "ok", **distance_score["macro"][f"d{distance}"]}
        except ValueError as error:
            result["macro"][f"d{distance}"] = {
                "status": "unfit", "reason": str(error),
                "logical_error_per_round": None, "ci95": None,
            }
    return result


def score_predictions(
    predictions: Path, data_root: Path, group: str | None = None,
    classical_root: Path | None = None, bootstrap_samples: int = 2000, seed: int = 0,
) -> dict[str, Any]:
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be nonnegative")
    if classical_root is not None and bootstrap_samples == 0:
        raise ValueError("Paired comparisons require positive bootstrap_samples")
    expected = expected_condition_names(group)
    conditions = {item.name: item for item in discover_conditions(data_root) if item.name in expected}
    if set(conditions) != expected:
        raise ValueError(f"Missing expected data conditions: {sorted(expected - set(conditions))}")
    files = discover_predictions(predictions, expected)
    counts: dict[str, dict[str, int]] = {}
    calibration: dict[str, dict[str, float]] = {}
    errors: dict[str, NDArray[Any]] = {}
    classical_errors: dict[str, dict[str, NDArray[Any]]] = {method: {} for method in CLASSICAL_METHODS}
    provenance: dict[str, str] = {}
    selected_split: str | None = None
    for name, path in sorted(files.items()):
        split, indices, probabilities = validate_prediction(path)
        if selected_split is not None and split != selected_split:
            raise ValueError("All predictions must use the same validation/test split")
        selected_split = split
        official = stim.read_shot_data_file(
            path=str(conditions[name].path / "obs_flips_actual.01"), format="01", num_observables=1,
        )
        if official.shape != (50_000, 1):
            raise ValueError(f"{name}: expected 50,000 official observable labels")
        actual = official[indices, 0]
        errors[name] = (probabilities >= 0.5) != actual
        counts[name] = {"errors": int(errors[name].sum()), "total": len(actual)}
        calibration[name] = calibration_record(probabilities, actual)
        provenance[name] = sha256_file(path)
        if classical_root is not None:
            if split != "test":
                raise ValueError("The classical paired-error release covers test shots only")
            paired_directory = classical_root if classical_root.name == "paired_errors" else classical_root / "paired_errors"
            with np.load(paired_directory / f"{name}.npz", allow_pickle=False) as paired:
                if not np.array_equal(paired["shot_indices"], indices) or not np.array_equal(paired["actual"], actual):
                    raise ValueError(f"{name}: classical shot IDs or actual labels disagree")
                for method in CLASSICAL_METHODS:
                    values = paired[f"errors__{method}"]
                    if values.shape != actual.shape or not np.all(np.isin(values, [0, 1])):
                        raise ValueError(f"{name}: invalid classical errors for {method}")
                    classical_errors[method][name] = values.astype(np.bool_)

    summary = safe_summary(counts, bootstrap_samples, seed)
    for name, record in calibration.items():
        summary["per_condition"][name].update(record)
    calibration_macro: dict[str, Any] = {}
    for key, names in [("all", list(calibration)), *[
        (f"d{distance}", [name for name in calibration if parse_condition(name).distance == distance])
        for distance in (3, 5)
    ]]:
        if names:
            calibration_macro[key] = {
                metric: float(np.mean([calibration[name][metric] for name in names]))
                for metric in ("negative_log_likelihood", "brier_score")
            }
    comparisons: dict[str, Any] = {}
    if classical_root is not None:
        for method in CLASSICAL_METHODS:
            try:
                comparisons[method] = {
                    "status": "ok", "A": "neural", "B": method,
                    **compare_paired_conditions(errors, classical_errors[method], bootstrap_samples=bootstrap_samples, seed=seed),
                }
            except ValueError as error:
                comparisons[method] = {"status": "unfit", "reason": str(error), "A": "neural", "B": method}
    assert selected_split is not None  # Exact nonempty condition coverage was already checked.
    metadata = validate_inference_metadata(predictions, selected_split, conditions, provenance, counts)
    return {
        "schema_version": 1, "scored_utc": datetime.now(timezone.utc).isoformat(),
        "split": selected_split, "group": group, "conditions": len(counts),
        "total_shots": sum(record["total"] for record in counts.values()),
        "decision_rule": "predict observable flip iff converted official-path probability >= 0.5",
        "summary": summary,
        "calibration": {"macro": calibration_macro, "aggregation": "equal weight per condition", "nll_clip_epsilon": NLL_EPSILON},
        "paired_comparisons": comparisons,
        "prediction_sha256": provenance, "inference_metadata": metadata,
        "inference_provenance_status": "verified_against_inputs_and_predictions" if metadata else "not_supplied",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--classical-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = score_predictions(args.predictions, args.data_root, args.group, args.classical_root, args.bootstrap_samples, args.seed)
    atomic_json(args.output, result)
    print(json.dumps({"status": result["summary"]["status"], "output": str(args.output), "split": result["split"]}))


if __name__ == "__main__":
    main()
