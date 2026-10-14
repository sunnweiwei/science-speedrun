#!/usr/bin/env python3
"""Adapt current upstream Protenix outputs to FoldBench's prediction index.

The CIF normalization is the same normalization used by FoldBench's official
``algorithms/Protenix/postprocess.py``.  File discovery accepts both the older
FoldBench integration names and current upstream Protenix names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import biotite.structure.io.pdbx as pdbx
import pandas as pd


def normalize_cif(source: Path, destination: Path) -> None:
    cif_file = pdbx.CIFFile.read(source)
    block = cif_file.block
    atom_site = block.get("atom_site")
    row_count = len(atom_site["group_PDB"])
    atom_site["occupancy"] = pdbx.CIFColumn(
        pdbx.CIFData(["1" for _ in range(row_count)])
    )
    atom_site["B_iso_or_equiv"] = pdbx.CIFColumn(
        pdbx.CIFData(["0" for _ in range(row_count)])
    )

    entity_types: dict[str, tuple[str, str]] = {}
    entity_ids = atom_site["label_entity_id"].as_array().tolist()
    groups = atom_site["group_PDB"].as_array().tolist()
    for entity_id, group in zip(entity_ids, groups, strict=True):
        if entity_id not in entity_types:
            entity_types[entity_id] = (
                "polymer" if group == "ATOM" else "non-polymer",
                group,
            )
        elif entity_types[entity_id][1] != group:
            entity_types[entity_id] = ("polymer", group)
    block["entity"] = pdbx.CIFCategory(
        {
            "id": list(entity_types),
            "type": [entity_types[key][0] for key in entity_types],
        }
    )
    cif_file.write(destination)


def first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", default=["101"])
    parser.add_argument("--samples", nargs="+", type=int, default=[0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs: list[dict[str, Any]] = json.loads(args.input_json.read_text())
    args.evaluation_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for item in inputs:
        name = item["name"]
        for seed in args.seeds:
            prediction_root = args.prediction_dir / name / f"seed_{seed}" / "predictions"
            for sample in args.samples:
                source = first_existing(
                    [
                        prediction_root / f"{name}_sample_{sample}.cif",
                        prediction_root / f"{name}_seed_{seed}_sample_{sample}.cif",
                    ]
                )
                confidence = first_existing(
                    [
                        prediction_root
                        / f"{name}_summary_confidence_sample_{sample}.json",
                        prediction_root
                        / f"{name}_seed_{seed}_summary_confidence_sample_{sample}.json",
                    ]
                )
                if source is None or confidence is None:
                    continue
                destination = source.with_name(f"{source.stem}_foldbench.cif")
                normalize_cif(source, destination)
                confidence_data = json.loads(confidence.read_text())
                rows.append(
                    {
                        "pdb_id": name,
                        "seed": seed,
                        "sample": sample,
                        "ranking_score": confidence_data.get("ranking_score", 0),
                        "prediction_path": str(destination),
                    }
                )

    if not rows:
        raise RuntimeError("no Protenix predictions were found")
    output = args.evaluation_dir / "prediction_reference.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"wrote {len(rows)} predictions to {output}")


if __name__ == "__main__":
    main()
