"""CPU evidence gate for expanding the bounded AlphaQubit allocation.

This is an allocation decision, not a paper-accuracy or convergence test. It
replays only independent synthetic validation data and standard PyMatching;
model counts come from the source-bound, finalized EMA selection record.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import pymatching
import stim

from .alphaqubit_artifact import (SEMANTIC_SOURCES, _checkpoint_record, atomic_bytes,
                                 atomic_json, freeze_package, sha256_file)
from .alphaqubit_data import AlphaQubitSampler
from .data import group_conditions
from .metrics import summarize_conditions


GROUPS = ("d3_bX_center_3_5", "d5_bX_center_5_5")
THRESHOLDS = {
    "minimum_optimizer_steps": 10_000,
    "minimum_seen_examples": 4_000_000,
    "minimum_complete_validation_snapshots": 3,
    "minimum_main_BCE_relative_improvement": 0.10,
    "maximum_already_strong_LER_to_matching_ratio": 1.25,
    "maximum_LER_to_matching_ratio": 2.0,
    "minimum_shot_error_relative_improvement_over_no_correction": 0.10,
    "minimum_fit_R_squared_exclusive": 0.9,
    "minimum_fit_intercept": "strictly above max(-0.02, -intercept_stderr)",
    "validation_shots_per_condition": 5120,
}


def _json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(path.read_text(), parse_constant=reject)


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    if len(rows) != 13 or sorted(row["rounds"] for row in rows) != list(range(1, 26, 2)):
        raise ValueError("Validation must contain all thirteen distinct round settings")
    result = {}
    for row in rows:
        name_round = re.search(r"_r(\d+)_", row["condition"])
        if name_round is None or int(name_round[1]) != row["rounds"]:
            raise ValueError("Validation condition name and round count differ")
        errors, shots = row["errors"], row["shots"]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (errors, shots)):
            raise ValueError("Validation counts must be integers")
        if not 0 <= errors <= shots or shots != THRESHOLDS["validation_shots_per_condition"]:
            raise ValueError("Validation counts/shot population differ from the gate protocol")
        result[row["condition"]] = {"errors": errors, "total": shots}
    if len(result) != 13:
        raise ValueError("Repeated validation condition")
    return result


def checked_fit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute the repository point fit and the declared paper fit gate."""
    counts = _counts(rows)
    summary = summarize_conditions(counts, bootstrap_samples=0)
    if len(summary["per_group"]) != 1:
        raise ValueError("Validation mixes model groups")
    fit = next(iter(summary["per_group"].values()))
    selected = sorted((row for row in rows if row["rounds"] != 1), key=lambda row: row["rounds"])
    rounds = np.array([row["rounds"] for row in selected], dtype=np.float64)
    errors = np.array([row["errors"] for row in selected], dtype=np.float64)
    shots = np.array([row["shots"] for row in selected], dtype=np.float64)
    probability = (errors + 0.5) / (shots + 1)
    weights = shots * (1 - 2 * errors / shots)**2 / (4 * probability * (1 - probability))
    design = np.stack((np.ones_like(rounds), rounds), axis=1)
    stderr = float(np.sqrt(np.linalg.inv(design.T @ (weights[:, None] * design))[0, 0]))
    r_squared, ler = fit["unweighted_log_fidelity_r_squared"], fit["logical_error_per_round"]
    accepted = (r_squared is not None and r_squared > 0.9
                and fit["intercept"] > max(-0.02, -stderr) and 0 <= ler < 0.5)
    return {**fit, "accepted": accepted, "intercept_stderr": stderr}


def policy_decision(measurements: dict[str, Any]) -> dict[str, Any]:
    """Pure policy, separated from artifact/data verification for audit tests."""
    reasons = []
    for key, threshold, reason in (
        ("global_steps", "minimum_optimizer_steps", "fewer_than_10000_optimizer_steps"),
        ("seen_samples", "minimum_seen_examples", "fewer_than_4M_training_examples"),
        ("complete_validations", "minimum_complete_validation_snapshots", "fewer_than_three_complete_validations"),
    ):
        if measurements.get(key, -1) < THRESHOLDS[threshold]:
            reasons.append(reason)
    if measurements.get("accepted_best_fit") is not True:
        reasons.append("no_accepted_best_validation_fit")
    first, candidate = measurements.get("first_main_BCE"), measurements.get("candidate_main_BCE")
    improvement = ((first - candidate) / first if isinstance(first, (int, float)) and first > 0
                   and isinstance(candidate, (int, float)) and math.isfinite(first) and math.isfinite(candidate) else None)
    ler, matching = measurements.get("selected_LER"), measurements.get("matching_LER")
    valid_lers = all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value < 0.5
                     for value in (ler, matching))
    ratio = (ler / matching if valid_lers and matching > 0 else 0.0 if valid_lers and ler == matching == 0 else None)
    already_strong = ratio is not None and ratio <= THRESHOLDS["maximum_already_strong_LER_to_matching_ratio"]
    if ratio is None or ratio > THRESHOLDS["maximum_LER_to_matching_ratio"]:
        reasons.append("selected_LER_exceeds_twice_matching_or_fit_unavailable")
    if not already_strong and (improvement is None or improvement < THRESHOLDS["minimum_main_BCE_relative_improvement"]):
        reasons.append("insufficient_BCE_improvement_and_not_already_strong")
    errors, uncorrected = measurements.get("selected_errors"), measurements.get("no_correction_errors")
    shot_improvement = ((uncorrected - errors) / uncorrected
                        if isinstance(uncorrected, int) and uncorrected > 0 and isinstance(errors, int) and errors >= 0 else None)
    if shot_improvement is None or shot_improvement < THRESHOLDS["minimum_shot_error_relative_improvement_over_no_correction"]:
        reasons.append("less_than_10_percent_shot_error_improvement_over_no_correction")
    return {"passed": not reasons, "status": "eligible" if not reasons else "needs_diagnosis",
            "rejection_reasons": reasons, "thresholds": dict(THRESHOLDS),
            "measurements": {**measurements, "bce_relative_improvement": improvement,
                             "model_to_matching_LER_ratio": ratio, "already_strong": already_strong,
                             "shot_error_relative_improvement": shot_improvement}}


def _validation_evidence(history: list[dict[str, Any]], best: dict[str, Any],
                         progress: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    complete = [row for row in history if row.get("complete") is True]
    steps = [row["global_steps"] for row in complete]
    if steps != sorted(set(steps)) or any(step > progress["global_steps"] for step in steps):
        raise ValueError("Complete validation cursors are duplicated or inconsistent")
    if not complete:
        raise ValueError("No complete validation snapshots")
    for row in complete:
        counts = _counts(row["per_condition"])
        total = sum(record["total"] for record in counts.values())
        for item in row["per_condition"]:
            if (not math.isclose(item["main_error"], item["errors"] / item["shots"], rel_tol=1e-9, abs_tol=1e-10)
                    or any(item[key] < 0 for key in ("main_loss", "loss", "aux_loss"))):
                raise ValueError("Validation loss/error-rate disagrees with raw counts")
        for key in ("main_loss", "loss", "aux_loss", "main_error"):
            weighted = sum(item[key] * item["shots"] for item in row["per_condition"]) / total
            if not math.isclose(weighted, row[key], rel_tol=1e-9, abs_tol=1e-10):
                raise ValueError(f"Inconsistent validation aggregate {key}")
        if row["shots"] != total or row["main_loss"] < 0:
            raise ValueError("Inconsistent validation shots or negative BCE")
        if row.get("fit", {}).get("accepted") is True:
            fit = checked_fit(row["per_condition"])
            if not fit["accepted"] or not math.isclose(fit["logical_error_per_round"], row["fit"]["logical_error_per_round"], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("Recorded accepted fit does not reproduce from validation counts")
    selected = best["validation"]
    if selected not in complete or progress.get("latest_validation") != complete[-1]:
        raise ValueError("Selected/latest validation record is absent from complete history")
    accepted = [row for row in complete if row.get("fit", {}).get("accepted") is True]
    # Only the selected checkpoint and last fit-consistent validation are
    # candidates; this is not a search for the lowest BCE over the trajectory.
    candidates = [selected, accepted[-1]]
    candidate = min(candidates, key=lambda row: row["main_loss"])
    return complete, candidate


def evaluate_group(asset_root: Path, group: str) -> dict[str, Any]:
    folder = asset_root / "alphaqubit-v1" / "models" / group / "pretrain"
    lock_path = folder / ".training.lock"
    with lock_path.open("rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        sources = {f"baselines/{name}": Path(__file__).with_name(name).read_bytes() for name in SEMANTIC_SOURCES}
        _, checkpoint = _checkpoint_record(folder, group, "pretrain", sources)
        best, metadata, progress = (_json(folder / name) for name in ("best.json", "metadata.json", "progress.json"))
        history_path = folder / "history.jsonl"
        history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
        identity = metadata["identity"]
        if not _finite([best, metadata, progress, history]):
            raise ValueError("Nonfinite training/validation state")
        if any(session.get("stop_reason") in {"exception", "nonfinite", "failed"} or session.get("error")
               for session in metadata["sessions"]):
            raise ValueError("Training history contains a failed/nonfinite session")
        final_session = metadata["sessions"][-1]
        if (final_session.get("final_global_steps"), final_session.get("final_seen_samples")) != (progress["global_steps"], progress["seen_samples"]):
            raise ValueError("Final session and progress cursors differ")
        for name, digest in identity["source_sha256"].items():
            if Path(name).name != name or sha256_file(Path(__file__).with_name(name)) != digest:
                raise ValueError(f"Current training source differs from checkpoint: {name}")
        complete, candidate = _validation_evidence(history, best, progress)
        condition_list = sorted(group_conditions(asset_root / "data" / "sycamore")[group], key=lambda item: item.rounds)
        records = {record["condition"]: record for record in identity["data"]}
        selected_counts = _counts(best["validation"]["per_condition"])
        if len(records) != 13 or set(records) != {condition.name for condition in condition_list} or set(records) != set(selected_counts):
            raise ValueError("Model/data validation conditions differ")
        if identity["protocol"]["eval_shots_per_condition"] != 5120:
            raise ValueError("Representative validation protocol is not 5120 shots per condition")
        matching_rows, uncorrected, validation_provenance = [], {}, []
        for condition in condition_list:
            record = records[condition.name]
            sampler = AlphaQubitSampler(condition, identity["seed"] + 1_000_000_007 + 1009 * condition.rounds, "pij")
            detectors, labels = sampler.sample_packed(5120, 1.0)
            detector_sha = hashlib.sha256(detectors.tobytes()).hexdigest()
            label_sha = hashlib.sha256(labels.tobytes()).hexdigest()
            if (detector_sha != record["validation_detectors_sha256"] or label_sha != record["validation_labels_sha256"]
                    or sampler.state_dict() != record["validation_source"] or sampler.layout.metadata() != record["layout"]):
                raise ValueError(f"Independent validation/source replay mismatch: {condition.name}")
            dem_path = condition.path / "pij_from_even_for_odd.dem"
            decoder = pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel.from_file(dem_path))
            prediction = decoder.decode_batch(detectors, bit_packed_shots=True)
            if prediction.shape != labels.shape:
                raise ValueError("Matching observable dimensions differ from validation labels")
            errors = int(np.count_nonzero(prediction != labels))
            matching_rows.append({"condition": condition.name, "rounds": condition.rounds, "errors": errors, "shots": len(labels)})
            uncorrected[condition.name] = {"errors": int(labels.sum()), "total": len(labels)}
            validation_provenance.append({"condition": condition.name, "shots": len(labels),
                "detectors_sha256": detector_sha, "labels_sha256": label_sha,
                "pij_sha256": sha256_file(dem_path), "sampler": sampler.state_dict()})
        matching_counts = _counts(matching_rows)
        matching_fit = next(iter(summarize_conditions(matching_counts, bootstrap_samples=0)["per_group"].values()))
        measured = {"global_steps": progress["global_steps"], "seen_samples": progress["seen_samples"],
                    "complete_validations": len(complete), "first_validation_step": complete[0]["global_steps"],
                    "candidate_validation_step": candidate["global_steps"],
                    "first_main_BCE": complete[0]["main_loss"], "candidate_main_BCE": candidate["main_loss"],
                    "accepted_best_fit": True, "selected_LER": best["selected_LER"],
                    "matching_LER": matching_fit["logical_error_per_round"],
                    "selected_errors": sum(row["errors"] for row in selected_counts.values()),
                    "no_correction_errors": sum(row["errors"] for row in uncorrected.values()),
                    "total_shots": sum(row["total"] for row in selected_counts.values())}
        # The resumed training stage overwrites best/metadata/history. Preserve
        # the exact representative selection before releasing its dependency.
        attempt = f"step-{progress['global_steps']}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
        snapshot = asset_root / "alphaqubit-v1" / "representative-snapshots" / group / attempt
        package = snapshot / "package"
        freeze_package(folder.parents[1], "pretrain", package, group)
        atomic_bytes(snapshot / "history.jsonl", history_path.read_bytes())
        return {**policy_decision(measured), "matching_fit": matching_fit,
                "counts": {"selected": selected_counts, "matching": matching_counts, "no_correction": uncorrected},
                "provenance": {"identity_sha256": checkpoint["identity_sha256"],
                    "artifact_sha256": {name: sha256_file(folder / name) for name in
                                        ("best.npz", "best.json", "metadata.json", "progress.json", "history.jsonl")},
                    "model_counts_source": "finalized source-bound EMA selection record; no CPU model replay",
                    "retained_representative_package": str(package),
                    "retained_manifest_sha256": sha256_file(package / "manifest.json"),
                    "retained_history": str(snapshot / "history.jsonl"),
                    "source_sha256": identity["source_sha256"], "validation": validation_provenance,
                    "matching": {"package": pymatching.__version__, "correlations": False, "noise_source": "even-fitted pij DEM"}}}


def run_gate(asset_root: Path, output: Path | None = None) -> dict[str, Any]:
    started = time.monotonic()
    results = {}
    for group in GROUPS:
        try:
            results[group] = evaluate_group(asset_root, group)
        except Exception as error:
            results[group] = {"passed": False,
                "status": "insufficient_evidence" if isinstance(error, (FileNotFoundError, BlockingIOError)) else "needs_diagnosis",
                "rejection_reasons": [f"{type(error).__name__}: {error}"], "thresholds": dict(THRESHOLDS)}
    passed = all(result["passed"] for result in results.values())
    result = {"schema_version": 1, "purpose": "allocation gate only; not a scientific result", "passed": passed,
              "decision": "expand" if passed else "no_expansion",
              "interpretation": "Evidence supports bounded continued allocation" if passed else
                  "Do not expand automatically; inspect missing evidence or diagnose/extend representatives. This is not a negative scientific result.",
              "evaluated_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
              "thresholds": dict(THRESHOLDS), "groups": results,
              "source_sha256": {"alphaqubit_gate.py": sha256_file(Path(__file__))},
              "validation_scope": "Independent full-noise synthetic pij validation only; no experimental odd-fold labels"}
    atomic_json(output or asset_root / "alphaqubit-v1" / "expansion-gate.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_gate(args.asset_root, args.output)
    print(json.dumps(result, allow_nan=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
