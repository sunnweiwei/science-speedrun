#!/usr/bin/env python3
"""Minimal fixed-statistic contract reference; it does not train a model.

Uses stats.json (fit-split label statistics, produced by build_stats.py or the
materializer) when present next to this script; otherwise falls back to
uniform scores and zero-valued regressions.
"""
import json
import sys
from pathlib import Path

CLASSES = {"solubility": 2, "subcellular_localization": 10, "binary_localization": 2,
           "fold": 1195, "yeast_ppi": 2, "human_ppi": 2}
REGRESSION = ("fluorescence", "stability", "beta_lactamase", "ppi_affinity")
TASKS = (*REGRESSION[:3], *CLASSES, "secondary_structure", "ppi_affinity")


def load_stats():
    path = Path(__file__).resolve().parent / "stats.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: predict.py INPUT_ROOT OUTPUT_ROOT")
    input_root, output_root = Path(sys.argv[1]), Path(sys.argv[2])
    destination = output_root / "predictions"
    destination.mkdir(parents=True, exist_ok=False)
    stats = load_stats()
    for task in TASKS:
        with (input_root / f"{task}.jsonl").open() as source, \
             (destination / f"{task}.jsonl").open("w") as sink:
            for raw in source:
                row = json.loads(raw)
                if task in CLASSES:
                    scores = stats.get(task, {}).get("class_counts") or [1] * CLASSES[task]
                    prediction = {"scores": scores}
                elif task == "secondary_structure":
                    counts = stats.get(task, {}).get("class_counts") or [1, 1, 1]
                    prediction = {"ss3_scores": [counts] * len(row["features"]["sequence"])}
                else:
                    prediction = {"value": stats.get(task, {}).get("mean", 0.0)}
                sink.write(json.dumps({"example_id": row["example_id"], "prediction": prediction},
                                      separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
