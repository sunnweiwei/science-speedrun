"""Trusted scoring with frozen reference anchors and scientific diagnostics."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import zipfile
import zlib
from pathlib import Path
from typing import Any
import numpy as np
from qecbench.data import contained, load_condition, load_manifest, read_npz, sha256, write_json


class InvalidPrediction(ValueError):
    """A model output violated the public submission contract."""


def normalize(performance: float, config: dict[str, Any]) -> dict[str, float]:
    lower, upper = float(config["lower_bound"]), float(config["upper_bound"])
    if not np.isfinite([performance, lower, upper]).all() or not 0 <= lower < upper <= 1:
        raise ValueError("Invalid score anchors or performance")
    value = (performance - lower) / (upper - lower)
    # No upper clip (score_config v1.1.0): the tensor-network anchor scales the score to 1 but is not a ceiling.
    return {"performance": performance, "score_unclipped": value, "score": float(max(value, 0.0)),
            "reference_lower_bound": lower, "reference_upper_bound": upper}


def probabilities(output: Path, row: dict[str, Any], expected_ids: np.ndarray) -> np.ndarray:
    try:
        archive = read_npz(contained(output, f"{row['name']}.npz"), row["shots"] * 12)
        if set(archive) != {"sample_ids", "probabilities"}:
            raise ValueError("Expected sample_ids and probabilities only")
        ids, values = archive["sample_ids"], archive["probabilities"]
        if ids.dtype != np.uint64 or ids.shape != expected_ids.shape or len(np.unique(ids)) != len(ids):
            raise ValueError("Invalid or duplicate output sample IDs")
        if values.dtype != np.float32 or values.shape != ids.shape or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Expected finite float32[N] probabilities in [0,1]")
        order, expected_order = np.argsort(ids), np.argsort(expected_ids)
        if not np.array_equal(ids[order], expected_ids[expected_order]):
            raise ValueError("Output IDs do not exactly match input IDs")
        aligned = np.empty_like(values)
        aligned[expected_order] = values[order]
        return aligned
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile, zlib.error) as error:
        raise InvalidPrediction(f"{row['name']}: {error}") from error


def score(inputs: Path, output: Path, labels_root: Path, config_path: Path,
          require_full_suite: bool = True, validation: bool = False) -> dict[str, Any]:
    manifest = load_manifest(inputs, allow_labels=validation)
    config = json.loads(config_path.read_text())
    if config.get("schema") != "surface-code-decoding-score/v1":
        raise ValueError("Unsupported scoring configuration")
    input_is_labeled = manifest["split"] == "validation"
    if validation:
        validation_manifest = load_manifest(labels_root, allow_labels=True)
        if validation_manifest["split"] != "validation" or manifest["split"] not in ("validation", "validation-inputs"):
            raise ValueError("Validation evaluator requires a validation label root and validation inputs")
        targets = {row["name"]: row for row in validation_manifest["conditions"]}
    else:
        target_manifest = json.loads(contained(labels_root, "manifest.json").read_text())
        if target_manifest.get("schema") != "surface-code-decoding-labels/v1" or target_manifest.get("input_manifest_sha256") != sha256(inputs / "manifest.json"):
            raise ValueError("Trusted labels do not match inference input manifest")
        targets = {row["name"]: row for row in target_manifest["conditions"]}
        if set(targets) != {row["name"] for row in manifest["conditions"]}:
            raise ValueError("Trusted labels have incorrect condition coverage")
    rows = manifest["conditions"]
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row["rounds"])
    if require_full_suite:
        from baselines.alphaqubit_artifact import EXPECTED_GROUPS
        if set(groups) != set(EXPECTED_GROUPS) or any(sorted(rounds) != list(range(1, 26, 2)) for rounds in groups.values()):
            raise ValueError("Official score requires all 130 fixed conditions")
    # Validate every trusted input and label before looking at any model output.
    # A malformed first prediction must not hide corruption in a later label file.
    prepared: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in rows:
        data = load_condition(inputs, row, labels=input_is_labeled)
        ids = data["sample_ids"]
        if validation:
            reference = load_condition(labels_root, targets[row["name"]], labels=True)
            order = np.argsort(reference["sample_ids"])
            sorted_ids = reference["sample_ids"][order]
            indices = np.searchsorted(sorted_ids, ids)
            if np.any(indices >= len(sorted_ids)) or not np.array_equal(sorted_ids[indices], ids):
                raise ValueError("Staged validation IDs do not belong to the validation split")
            target = reference["labels"][order[indices]]
        else:
            target_row = targets[row["name"]]
            path = contained(labels_root, target_row["arrays"])
            if sha256(path) != target_row["arrays_sha256"]:
                raise ValueError("Trusted label checksum mismatch")
            truth = read_npz(path, row["shots"] * 9)
            if set(truth) != {"sample_ids", "labels"} or not np.array_equal(truth["sample_ids"], ids):
                raise ValueError("Trusted label IDs do not match input IDs")
            target = truth["labels"]
        if target.dtype != np.uint8 or target.shape != ids.shape or np.any(target > 1):
            raise ValueError("Trusted labels are not binary uint8[N]")
        prepared[row["name"]] = (ids, target)
    counts: dict[str, Any] = {}
    failures: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    brier, nll, shot_count = 0.0, 0.0, 0
    bins_n, bins_p, bins_y = np.zeros(10), np.zeros(10), np.zeros(10)
    for row in rows:
        ids, target = prepared[row["name"]]
        values = probabilities(output, row, ids)
        errors = int(np.count_nonzero((values >= config["threshold"]) != target))
        counts[row["name"]] = {"errors": errors, "total": len(ids)}
        failures[row["distance"]][row["group"]].append(errors / len(ids))
        values64 = values.astype(np.float64)
        brier += float(np.square(values64 - target).sum())
        clipped = np.clip(values64, 1e-7, 1 - 1e-7)
        nll -= float((target * np.log(clipped) + (1 - target) * np.log1p(-clipped)).sum())
        bins = np.minimum((values64 * 10).astype(int), 9)
        bins_n += np.bincount(bins, minlength=10)
        bins_p += np.bincount(bins, weights=values64, minlength=10)
        bins_y += np.bincount(bins, weights=target, minlength=10)
        shot_count += len(ids)
    by_distance = {f"d{distance}": float(np.mean([np.mean(values) for values in by_group.values()]))
                   for distance, by_group in failures.items()}
    balanced = float(np.mean(list(by_distance.values())))
    scientific: dict[str, Any]
    try:
        from baselines.metrics import summarize_conditions
        scientific = summarize_conditions(counts, bootstrap_samples=0)
        scientific["status"] = "available"
    except ValueError as error:
        # Poor valid models can have P_fail>=0.5. Their primary score is valid,
        # even though exponential log-fidelity fitting is then undefined.
        scientific = {"status": "unavailable", "reason": str(error)}
    return {"schema": "surface-code-decoding-report/v1", "status": "valid",
            "model_quality_measured": True, "shots": shot_count, "conditions": len(rows),
            "metric_version": config["version"], **normalize(1 - balanced, config),
            "balanced_logical_failure": balanced, "failure_by_distance": by_distance,
            "per_condition": counts, "scientific_metrics": scientific,
            "calibration": {"aggregation": "pooled shots (diagnostic only)", "brier_score": brier / shot_count,
                "negative_log_likelihood": nll / shot_count,
                "expected_calibration_error_10_bins": float(np.abs(bins_p - bins_y).sum() / shot_count)},
            "score_config_sha256": sha256(config_path), "input_manifest_sha256": sha256(inputs / "manifest.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--allow-subset-diagnostic", action="store_true")
    args = parser.parse_args()
    result = score(args.input, args.predictions, args.labels, args.config,
                   not args.allow_subset_diagnostic, args.validation)
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("status", "score", "performance", "score_unclipped")}))


if __name__ == "__main__":
    main()
