#!/usr/bin/env python3
"""Trusted prediction-only grader; never imports a submission."""
import argparse
import json
import time
from pathlib import Path

from common import PRIVATE_SCHEMA, SNAPSHOT_SCHEMA, jsonl, sha256, write_json
from metrics import TASKS, flatten_metrics, score_task


def harbor_reward(values):
    """Harbor parses reward.json as a flat object of finite numbers; strings, None and nested details
    belong in report.json / status.json."""
    import math as _math
    return {k: v for k, v in values.items() if isinstance(v, (int, float)) and _math.isfinite(v)}


def grade_task(task, prediction_path, label_path, ppi_affinity_std):
    try:
        truth = {row["example_id"]: row["label"] for row in jsonl(label_path)}
        found = {}
        for row in jsonl(prediction_path):
            if (set(row) != {"example_id", "prediction"} or row["example_id"] not in truth
                    or row["example_id"] in found or not isinstance(row["prediction"], dict)):
                raise ValueError("malformed/unknown/duplicate prediction")
            found[row["example_id"]] = row["prediction"]
        return score_task(task, truth, found, ppi_affinity_std) | {"invalid_reason": None}
    except (OSError, ValueError, TypeError, KeyError, OverflowError, json.JSONDecodeError) as error:
        return {"reward": 0.0, "invalid_reason": str(error)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--labels", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--output", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--wait-seconds", type=int, default=43200)
    args = parser.parse_args()
    ready = args.sealed.parent / "READY"
    deadline = time.monotonic() + args.wait_seconds
    while not ready.is_file():
        if time.monotonic() > deadline:
            raise TimeoutError("snapshot not published")
        time.sleep(1)
    snapshot = json.loads((args.sealed / "snapshot.json").read_text())
    private = json.loads((args.labels / "manifest.json").read_text())
    names = {f"{task}.jsonl" for task in TASKS}
    if (ready.read_text().strip() != sha256(args.sealed / "snapshot.json")
            or snapshot.get("schema") != SNAPSHOT_SCHEMA
            or private.get("schema") != PRIVATE_SCHEMA
            or private.get("public_input_manifest_sha256") != snapshot.get("input_manifest_sha256")):
        raise ValueError("snapshot/private mismatch")
    if set(snapshot) != {"schema", "input_manifest_sha256", "files"} or not set(snapshot["files"]) <= names:
        raise ValueError("invalid snapshot files")
    for name, digest in snapshot["files"].items():
        if sha256(args.sealed / name) != digest:
            raise ValueError("sealed prediction hash mismatch")
    for name, digest in private["files"].items():
        if name not in names or sha256(args.labels / name) != digest:
            raise ValueError("private label hash mismatch")
    std = private["fit_standard_deviation"]["ppi_affinity"]
    details = {task: grade_task(task, args.sealed / f"{task}.jsonl", args.labels / f"{task}.jsonl", std)
               for task in TASKS}
    reward = sum(value["reward"] for value in details.values()) / len(TASKS)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "report.json",
               {"schema": "peer-protein-report/v1", "tasks": details, "reward": reward})
    flat = {"reward": reward}
    for task, result in details.items():
        flatten_metrics(task, result, flat)
    write_json(args.output / "reward.json", harbor_reward(flat))


if __name__ == "__main__":
    main()
