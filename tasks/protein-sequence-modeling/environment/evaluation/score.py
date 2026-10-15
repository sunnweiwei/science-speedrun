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
from metrics import TASKS, flatten_metrics, score_task  # noqa: E402


def read_jsonl(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--stats", type=Path, default=Path("/workspace/data/stats.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    std = json.loads(args.stats.read_text())["ppi_affinity"]["std"]
    details = {}
    for task in TASKS:
        try:
            truth = {r["example_id"]: r["label"] for r in read_jsonl(args.labels / f"{task}.jsonl")}
            found = {}
            for row in read_jsonl(args.predictions / f"{task}.jsonl"):
                if set(row) != {"example_id", "prediction"} or row["example_id"] in found:
                    raise ValueError("malformed or duplicate prediction row")
                found[row["example_id"]] = row["prediction"]
            details[task] = score_task(task, truth, found, std) | {"invalid_reason": None}
        except (OSError, ValueError, TypeError, KeyError, OverflowError, json.JSONDecodeError) as error:
            details[task] = {"reward": 0.0, "invalid_reason": str(error)}
    report = {"tasks": details,
              "reward": sum(v["reward"] for v in details.values()) / len(TASKS)}
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        flat = {"reward": report["reward"]}
        for task, result in details.items():
            flatten_metrics(task, result, flat)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
