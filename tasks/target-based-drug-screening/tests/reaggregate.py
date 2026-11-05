#!/usr/bin/env python3
"""Migrate a trusted full-suite report without inference or overwriting its source."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

evaluation = Path(__file__).resolve().parents[1] / "environment/evaluation"
if evaluation.is_dir():
    sys.path.insert(0, str(evaluation))
from aggregation import summarize_targets


def reaggregate(report, source_sha256):
    if report.get("valid") is not True:
        raise ValueError("only valid, trusted full-suite reports can be migrated")
    summary = summarize_targets(report["per_target"], require_full_suite=True)
    result = {**report, **summary, "reward": summary["dataset_balanced_bedroc"]}
    result["migration"] = {
        "kind": "reaggregation of stored per-target metrics; no new inference",
        "source_report_sha256": source_sha256,
        "source_scoring_protocol": report.get("scoring_protocol", "legacy-ef1/(1+ef1)/v1"),
        "source_primary_metric": report.get("primary_metric"),
        "source_reward": report["reward"],
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    result = reaggregate(json.loads(raw), hashlib.sha256(raw).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents replacing an original report through an alias.
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
