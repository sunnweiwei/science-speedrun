#!/usr/bin/env python3
"""Validate and merge DPLM-2 FoldBench inference shards for official scoring."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
from typing import Any


TRACK = "monomer_protein"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--targets-dir", type=Path, required=True)
    parser.add_argument("--scoring-targets-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--algorithm", default="dplm2_50m")
    parser.add_argument("--expected-shards", type=int, default=7)
    parser.add_argument("--expected-max-iter", type=int, default=500)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = arguments()
    audits = []
    predictions = []
    for shard in range(args.expected_shards):
        audit_path = args.run_dir / f"inference_audit.shard-{shard}.json"
        prediction_path = args.run_dir / f"prediction_reference.shard-{shard}.csv"
        if not audit_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"incomplete FoldBench shard {shard}")
        audit = json.loads(audit_path.read_text())
        expected = {
            "schema_version": "foldbench-inference-shard-audit/v1",
            "model_id": "dplm2_50m",
            "track": TRACK,
            "num_shards": args.expected_shards,
            "shard_index": shard,
            "max_iter": args.expected_max_iter,
        }
        for key, value in expected.items():
            if audit.get(key) != value:
                raise ValueError(
                    f"shard {shard}: {key}={audit.get(key)!r}, expected {value!r}"
                )
        rows = read_csv(prediction_path)
        if len(rows) != audit["predictions_written"]:
            raise ValueError(f"shard {shard}: prediction count disagrees with audit")
        audits.append(audit)
        predictions.extend(rows)

    invariant_keys = [
        "coordinate_free_inputs",
        "target_count",
        "supported_target_count",
        "rejected_targets",
        "seed",
    ]
    for key in invariant_keys:
        if any(audit[key] != audits[0][key] for audit in audits[1:]):
            raise ValueError(f"shard audits disagree on {key}")
    if sum(audit["selected_targets"] for audit in audits) != audits[0][
        "supported_target_count"
    ]:
        raise ValueError("shards do not partition all supported targets")

    prediction_ids = [row["pdb_id"].lower() for row in predictions]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("duplicate PDB IDs across inference shards")
    if len(predictions) != audits[0]["supported_target_count"]:
        raise ValueError("prediction union does not cover all supported targets")
    for row in predictions:
        path = Path(row["prediction_path"])
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing prediction artifact: {path}")

    source_targets = read_csv(args.targets_dir / f"{TRACK}.csv")
    rejected_ids = {
        item["pdb_id"].lower() for item in audits[0]["rejected_targets"]
    }
    scoring_targets = [
        row for row in source_targets if row["pdb_id"].lower() not in rejected_ids
    ]
    target_ids = {row["pdb_id"].lower() for row in scoring_targets}
    if target_ids != set(prediction_ids):
        raise ValueError("prediction IDs do not exactly match supported target subset")

    args.scoring_targets_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.scoring_targets_dir / f"{TRACK}.csv",
        scoring_targets,
        list(source_targets[0]),
    )
    algorithm_dir = args.evaluation_dir / args.algorithm
    write_csv(
        algorithm_dir / "prediction_reference.csv",
        sorted(predictions, key=lambda row: row["pdb_id"].lower()),
        ["pdb_id", "seed", "sample", "ranking_score", "prediction_path"],
    )
    coverage = {
        "schema_version": "foldbench-capability-coverage/v1",
        "model_id": "dplm2_50m",
        "algorithm": args.algorithm,
        "track": TRACK,
        "official_target_count": audits[0]["target_count"],
        "supported_target_count": audits[0]["supported_target_count"],
        "prediction_count": len(predictions),
        "coverage": len(predictions) / audits[0]["target_count"],
        "rejected_targets": audits[0]["rejected_targets"],
        "inference": {
            "max_iter": audits[0]["max_iter"],
            "seed": audits[0]["seed"],
            "shards": args.expected_shards,
        },
        "score_scope": "supported_subset",
        "unsupported_tracks": [
            "monomer_rna",
            "monomer_dna",
            "interface_protein_protein",
            "interface_antibody_antigen",
            "interface_protein_peptide",
            "interface_protein_rna",
            "interface_protein_dna",
            "interface_protein_ligand",
        ],
    }
    coverage_path = algorithm_dir / "coverage.json"
    coverage_path.write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")
    shutil.copy2(coverage_path, args.run_dir / "coverage.json")
    print(json.dumps(coverage, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
