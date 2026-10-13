#!/usr/bin/env python3
"""Merge MIST-CF JSONL shards and aggregate fixed-denominator diagnostics."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()

    rows = {}
    for path in args.predictions:
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                identifier = str(row["identifier"])
                if identifier in rows:
                    raise ValueError(f"duplicate prediction: {identifier}")
                rows[identifier] = row
    if len(rows) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} predictions, found {len(rows)}")

    metric_rows = [json.loads(path.read_text()) for path in args.metrics]
    processed = sum(int(row["processed"]) for row in metric_rows)
    if processed != args.expected_count:
        raise ValueError(f"metric denominator is {processed}, expected {args.expected_count}")
    def weighted(key: str) -> float | None:
        available = [row for row in metric_rows if row.get(key) is not None]
        if not available:
            return None
        weights = [
            int(row.get("labelled_diagnostic_rows", row["processed"]))
            for row in available
        ]
        denominator = sum(weights)
        if denominator == 0:
            return None
        return sum(float(row[key]) * weight for row, weight in zip(available, weights)) / denominator

    combined = {
        "processed": processed,
        "labelled_diagnostic_rows": sum(
            int(row.get("labelled_diagnostic_rows", row["processed"]))
            if row.get("top1_formula_accuracy") is not None else 0
            for row in metric_rows
        ),
        "candidate_coverage": weighted("candidate_coverage"),
        "top1_formula_accuracy": weighted("top1_formula_accuracy"),
        "top5_formula_accuracy": weighted("top5_formula_accuracy"),
        "empty_candidate_sets": sum(int(row["empty_candidate_sets"]) for row in metric_rows),
        "ppm": sorted({int(row["ppm"]) for row in metric_rows}),
        "max_candidates": sorted({int(row["max_candidates"]) for row in metric_rows}),
        "shards": len(metric_rows),
        "sum_shard_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in metric_rows),
        "max_shard_elapsed_seconds": max(float(row["elapsed_seconds"]) for row in metric_rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for identifier in sorted(rows):
            stream.write(json.dumps(rows[identifier], sort_keys=True, separators=(",", ":")) + "\n")
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.write_text(json.dumps(combined, indent=2) + "\n")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
