#!/usr/bin/env python3
"""Trusted labels-only service; never executes a submitted model or source."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any
PROJECT = Path(__file__).resolve().parents[1] / "environment" / "baseline" / "alphaqubit" if (Path(__file__).resolve().parents[1] / "environment" / "baseline" / "alphaqubit").is_dir() else Path(__file__).resolve().parents[1]  # the baseline project: environment/baseline/alphaqubit here, /opt/task in the verifier image
sys.path.insert(0, str(PROJECT))
from qecbench.data import contained, sha256, write_json
from qecbench.scoring import InvalidPrediction, score


def grade(sealed: Path, inputs: Path, labels: Path, config: Path, output: Path,
          ready: dict[str, Any]) -> None:
    run_id = ready["run_id"]
    target = output / run_id
    target.mkdir(parents=True)
    try:
        snapshot_path = contained(sealed, "snapshot.json")
        if sha256(snapshot_path) != ready["snapshot_sha256"]:
            raise ValueError("Sealed snapshot hash mismatch")
        snapshot = json.loads(snapshot_path.read_text())
        if snapshot.get("schema") != "surface-code-decoding-snapshot/v1" or snapshot.get("run_id") != run_id:
            raise ValueError("Sealed snapshot identity mismatch")
        if snapshot.get("input_manifest_sha256") != sha256(inputs / "manifest.json"):
            raise ValueError("Input/snapshot mismatch")
        actual = {path.name for path in sealed.iterdir() if path.name != "snapshot.json"}
        if actual != set(snapshot["files"]):
            raise ValueError("Sealed snapshot file coverage mismatch")
        for name, digest in snapshot["files"].items():
            if sha256(contained(sealed, name)) != digest:
                raise ValueError("Sealed prediction checksum mismatch")
        try:
            result = score(inputs, sealed, labels, config)
        except InvalidPrediction as error:
            result = {"schema": "surface-code-decoding-report/v1", "status": "invalid_prediction",
                "model_quality_measured": False, "score": 0.0, "score_unclipped": None,
                "performance": None, "error": str(error)}
        result["run_id"] = run_id
        write_json(target / "report.json", result)
        write_json(target / "reward.json", {"reward": result["performance"]})  # raw P; the normalized score stays in report.json
    except Exception as error:
        write_json(target / "report.json", {"schema": "surface-code-decoding-report/v1", "run_id": run_id,
            "status": "infrastructure_error", "model_quality_measured": False,
            "error": f"{type(error).__name__}: {error}"})
    write_json(target / "DONE", {"run_id": run_id})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", type=Path, default=Path("/sealed/run"))
    parser.add_argument("--input", type=Path, default=Path("/benchmark"))
    parser.add_argument("--labels", type=Path, default=Path("/private"))
    parser.add_argument("--config", type=Path, default=Path("/opt/task/score_config.json"))
    parser.add_argument("--output", type=Path, default=Path("/sealed/results"))
    parser.add_argument("--wait-seconds", type=int, default=3900)
    args = parser.parse_args()
    deadline = time.monotonic() + args.wait_seconds
    while time.monotonic() < deadline:
        ready = args.sealed.parent / "READY"
        if ready.is_file():
            payload = json.loads(ready.read_text())
            run_id = payload.get("run_id", "")
            if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
                raise ValueError("Invalid trusted run identity")
            grade(args.sealed, args.input, args.labels, args.config, args.output, payload)
            return
        time.sleep(0.25)
    raise TimeoutError("Inference supervisor did not seal a submission")


if __name__ == "__main__":
    main()
