#!/usr/bin/env python3
"""Trusted labels-only service; never executes submitted model code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import duckdb
import numpy as np

from common import sha256, write_json

sys.path.insert(0, "/evaluation")
from panel_scoring import PRIMARY  # noqa: E402
from risk_scoring import read_normalization, score  # noqa: E402


def validate_private_split(input_dir: Path, trusted: Path,
                           normalization: Path) -> None:
    manifest = json.loads((input_dir / "manifest.json").read_text())
    if manifest.get("split") != "test":
        raise ValueError("Trusted grader accepts only the test input manifest")
    diseases = manifest.get("scored_diseases")
    if not isinstance(diseases, list) or not diseases:
        raise ValueError("Public test manifest has no scored diseases")
    public = (input_dir / "samples.parquet").as_posix()
    labels = (trusted / "label_windows.parquet").as_posix()
    boundaries = (trusted / "sample_boundaries.parquet").as_posix()
    db = duckdb.connect()
    coverage = db.execute(f"""SELECT
      (SELECT count(*) FROM read_parquet('{public}')),
      (SELECT count(DISTINCT sample_id) FROM read_parquet('{public}')),
      (SELECT count(*) FROM read_parquet('{labels}')),
      (SELECT count(DISTINCT (sample_id,disease_code)) FROM read_parquet('{labels}')),
      (SELECT count(*) FROM read_parquet('{boundaries}')),
      (SELECT count(*) FROM read_parquet('{boundaries}') WHERE split<>'test'),
      (SELECT count(*) FROM read_parquet('{public}') p ANTI JOIN
         read_parquet('{labels}') l USING(sample_id)),
      (SELECT count(*) FROM read_parquet('{public}') p ANTI JOIN
         read_parquet('{boundaries}') b USING(sample_id))""").fetchone()
    db.close()
    expected_pairs = coverage[0] * len(diseases)
    if (not coverage[0] or coverage[0] != coverage[1]
            or coverage[2] != expected_pairs or coverage[3] != expected_pairs
            or coverage[4] != coverage[0] or any(coverage[5:])):
        raise ValueError(f"Test public/trusted split coverage is invalid: {coverage}")
    read_normalization(normalization, diseases)


def grade(args: argparse.Namespace, ready: dict) -> None:
    run_id = ready["run_id"]
    target = args.output / run_id
    target.mkdir(parents=True)
    try:
        snapshot_path = args.sealed / "snapshot.json"
        if sha256(snapshot_path) != ready["snapshot_sha256"]:
            raise ValueError("Sealed snapshot hash mismatch")
        snapshot = json.loads(snapshot_path.read_text())
        if (snapshot.get("schema") != "natural-disease-progression-prediction-snapshot/v1"
                or snapshot.get("run_id") != run_id):
            raise ValueError("Sealed snapshot identity mismatch")
        if snapshot.get("input_manifest_sha256") != sha256(args.input / "manifest.json"):
            raise ValueError("Sealed prediction input mismatch")
        prediction = args.sealed / "predictions.parquet"
        if snapshot.get("predictions_sha256") != sha256(prediction):
            raise ValueError("Sealed prediction hash mismatch")
        if {path.name for path in args.sealed.iterdir()} != {
                "snapshot.json", "predictions.parquet"}:
            raise ValueError("Unexpected file in sealed prediction snapshot")
        if not args.normalization.is_file():
            raise FileNotFoundError("Frozen train-only normalization is missing")
        validate_private_split(args.input, args.trusted, args.normalization)
        try:
            report = score(args.input, args.trusted, prediction, args.bootstrap,
                           args.normalization)
            reward = report["panel"][PRIMARY]
            if reward is None or not np.isfinite(reward) or not 0.0 <= reward <= 1.0:
                raise ValueError(f"Primary {PRIMARY} is unavailable")
            report.update({"schema": "natural-disease-progression-report/v3",
                           "status": "scored", "model_quality_measured": True,
                           "run_id": run_id, "reward": reward})
            write_json(target / "report.json", report)
            write_json(target / "reward.json", {"reward": reward})
        except ValueError as error:
            write_json(target / "report.json", {
                "schema": "natural-disease-progression-report/v3",
                "status": "invalid_prediction", "model_quality_measured": False,
                "run_id": run_id, "error": str(error),
            })
            write_json(target / "reward.json", {"reward": 0.0})
    except Exception as error:
        write_json(target / "report.json", {
            "schema": "natural-disease-progression-report/v3",
            "status": "infrastructure_error", "model_quality_measured": False,
            "run_id": run_id, "error": f"{type(error).__name__}: {error}",
        })
    write_json(target / "DONE", {"run_id": run_id})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/input"))
    parser.add_argument("--trusted", type=Path, default=Path("/trusted/test"))
    parser.add_argument("--normalization", type=Path,
                        default=Path("/trusted/risk_normalization.parquet"))
    parser.add_argument("--sealed", type=Path, default=Path("/sealed/run"))
    parser.add_argument("--output", type=Path, default=Path("/sealed/results"))
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--wait-seconds", type=int, default=42600)
    args = parser.parse_args()
    deadline = time.monotonic() + args.wait_seconds
    ready_path = args.sealed.parent / "READY"
    while time.monotonic() < deadline:
        if ready_path.is_file():
            ready = json.loads(ready_path.read_text())
            run_id = ready.get("run_id", "")
            if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
                raise ValueError("Invalid trusted run identity")
            grade(args, ready)
            return
        time.sleep(0.25)
    raise TimeoutError("Inference supervisor did not seal a submission")


if __name__ == "__main__":
    main()
