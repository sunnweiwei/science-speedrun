#!/usr/bin/env python3
"""Development self-evaluation on labeled splits (same metrics as the grader).

Example:
    python3 score.py --predictions OUT/predictions --labels /workspace/data/tune \
        --stats /workspace/data/stats.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import (  # noqa: E402
    PPI_SYMMETRIC_TASKS,
    TASKS,
    aggregate_pillars,
    flatten_metrics,
    score_task,
)


def read_jsonl(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def load_prediction_map(path):
    found = {}
    for row in read_jsonl(path):
        if set(row) != {"example_id", "prediction"} or row["example_id"] in found:
            raise ValueError("malformed or duplicate prediction row")
        found[row["example_id"]] = row["prediction"]
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--stats", type=Path, default=Path("/workspace/data/stats.json"))
    parser.add_argument(
        "--swapped-predictions",
        type=Path,
        default=None,
        help="Optional directory of swapped (sequence_b, sequence_a) predictions for yeast_ppi/human_ppi.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    std = (
        json.loads(args.stats.read_text())["ppi_affinity"]["std"]
        if args.stats and args.stats.is_file()
        else None
    )
    details = {}
    for task in TASKS:
        try:
            truth = {r["example_id"]: r["label"] for r in read_jsonl(args.labels / f"{task}.jsonl")}
            found = load_prediction_map(args.predictions / f"{task}.jsonl")
            found_swapped = None
            if task in PPI_SYMMETRIC_TASKS:
                swapped_candidate = (
                    args.swapped_predictions / f"{task}.jsonl"
                    if args.swapped_predictions is not None
                    else args.predictions / f"{task}_swapped.jsonl"
                )
                if swapped_candidate.is_file():
                    found_swapped = load_prediction_map(swapped_candidate)
            details[task] = score_task(
                task, truth, found, std, found_swapped=found_swapped
            ) | {"invalid_reason": None}
        except (OSError, ValueError, TypeError, KeyError, OverflowError, json.JSONDecodeError) as error:
            details[task] = {"reward": 0.0, "skill": 0.0, "invalid_reason": str(error)}
    agg = aggregate_pillars(details)
    report = {
        "tasks": details,
        "pillars": agg["pillars"],
        "signed_score": agg["signed_score"],
        "score": agg["score"],
        "reward": agg["reward"],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        flat = {"score": report["score"], "reward": report["reward"], "signed_score": report["signed_score"]}
        for task, result in details.items():
            flatten_metrics(task, result, flat)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
