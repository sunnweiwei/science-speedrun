#!/usr/bin/env python3
"""Trusted prediction-only grader; never imports a submission."""
import argparse
import json
import math
import time
from pathlib import Path

from common import PRIVATE_SCHEMA, SNAPSHOT_SCHEMA, jsonl, sha256, write_json
from metrics import (
    PPI_SYMMETRIC_TASKS,
    TASKS,
    aggregate_pillars,
    flatten_metrics,
    score_task,
)


def harbor_reward(values):
    """Harbor parses reward.json as a flat object of finite numbers; strings, booleans, None and
    nested details belong in report.json / status.json."""
    return {
        k: float(v)
        for k, v in values.items()
        if type(v) in (int, float) and math.isfinite(v)
    }


def _load_predictions_for_truth(prediction_path, truth):
    found = {}
    for row in jsonl(prediction_path):
        if (
            set(row) != {"example_id", "prediction"}
            or row["example_id"] not in truth
            or row["example_id"] in found
            or not isinstance(row["prediction"], dict)
        ):
            raise ValueError("malformed/unknown/duplicate prediction")
        found[row["example_id"]] = row["prediction"]
    return found


def grade_task(
    task,
    prediction_path,
    label_path,
    ppi_affinity_std=None,
    swapped_prediction_path=None,
    require_symmetric_ppi=False,
):
    try:
        truth = {row["example_id"]: row["label"] for row in jsonl(label_path)}
        found = _load_predictions_for_truth(prediction_path, truth)
        found_swapped = None
        if task in PPI_SYMMETRIC_TASKS:
            if swapped_prediction_path is not None and (
                require_symmetric_ppi or swapped_prediction_path.is_file()
            ):
                found_swapped = _load_predictions_for_truth(swapped_prediction_path, truth)
            elif require_symmetric_ppi:
                raise ValueError(f"missing swapped orientation predictions for {task}")
        return score_task(
            task, truth, found, ppi_affinity_std, found_swapped=found_swapped
        ) | {"invalid_reason": None}
    except (OSError, ValueError, TypeError, KeyError, OverflowError, json.JSONDecodeError) as error:
        return {"reward": 0.0, "skill": 0.0, "invalid_reason": str(error)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--labels", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--output", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--wait-seconds", type=int, default=43200)
    parser.add_argument(
        "--allow-single-orientation-ppi",
        action="store_true",
        help="Allow grading legacy snapshots that lack swapped PPI orientation files.",
    )
    args = parser.parse_args()
    require_sym = not args.allow_single_orientation_ppi

    ready = args.sealed.parent / "READY"
    invalid = args.sealed.parent / "INVALID"
    deadline = time.monotonic() + args.wait_seconds
    while not ready.is_file():
        if invalid.is_file():
            return
        if time.monotonic() > deadline:
            raise TimeoutError("snapshot not published")
        time.sleep(1)
    snapshot = json.loads((args.sealed / "snapshot.json").read_text())
    private = json.loads((args.labels / "manifest.json").read_text())
    names = {f"{task}.jsonl" for task in TASKS}
    swapped_names = {f"{task}_swapped.jsonl" for task in PPI_SYMMETRIC_TASKS}
    if (
        ready.read_text().strip() != sha256(args.sealed / "snapshot.json")
        or snapshot.get("schema") != SNAPSHOT_SCHEMA
        or private.get("schema") != PRIVATE_SCHEMA
        or private.get("public_input_manifest_sha256") != snapshot.get("input_manifest_sha256")
    ):
        raise ValueError("snapshot/private mismatch")
    allowed_keys = {"schema", "input_manifest_sha256", "files", "swapped_ppi_files"}
    required_keys = {"schema", "input_manifest_sha256", "files"}
    if require_sym:
        required_keys.add("swapped_ppi_files")
    if (
        not required_keys <= set(snapshot) <= allowed_keys
        or not set(snapshot["files"]) <= names
    ):
        raise ValueError("invalid snapshot files")
    for name, digest in snapshot["files"].items():
        if sha256(args.sealed / name) != digest:
            raise ValueError("sealed prediction hash mismatch")
    swapped_files = snapshot.get("swapped_ppi_files")
    if swapped_files is not None:
        if not isinstance(swapped_files, dict) or not set(swapped_files) <= swapped_names:
            raise ValueError("invalid swapped PPI snapshot files")
        for name, digest in swapped_files.items():
            if sha256(args.sealed / name) != digest:
                raise ValueError("sealed swapped prediction hash mismatch")
    for name, digest in private["files"].items():
        if name not in names or sha256(args.labels / name) != digest:
            raise ValueError("private label hash mismatch")

    std = private.get("fit_standard_deviation", {}).get("ppi_affinity")
    details = {}
    for task in TASKS:
        swapped_path = (
            args.sealed / f"{task}_swapped.jsonl"
            if task in PPI_SYMMETRIC_TASKS
            else None
        )
        details[task] = grade_task(
            task,
            args.sealed / f"{task}.jsonl",
            args.labels / f"{task}.jsonl",
            std,
            swapped_prediction_path=swapped_path,
            require_symmetric_ppi=(require_sym and task in PPI_SYMMETRIC_TASKS),
        )

    agg = aggregate_pillars(details)
    score = agg["score"]
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "report.json",
        {
            "schema": "peer-protein-report/v4",
            "score": score,
            "reward": score,
            "signed_score": agg["signed_score"],
            "pillars": agg["pillars"],
            "task_skills": agg["task_skills"],
            "symmetric_ppi_enforced": require_sym,
            "tasks": details,
        },
    )
    flat = {
        "score": score,
        "reward": score,
        "signed_score": agg["signed_score"],
    }
    for pillar, val in agg["pillars"].items():
        slug = pillar.lower().replace(" ", "_").replace("-", "_")
        flat[f"pillar_{slug}"] = val
    for task, result in details.items():
        flatten_metrics(task, result, flat)
    write_json(args.output / "reward.json", harbor_reward(flat))


if __name__ == "__main__":
    main()
