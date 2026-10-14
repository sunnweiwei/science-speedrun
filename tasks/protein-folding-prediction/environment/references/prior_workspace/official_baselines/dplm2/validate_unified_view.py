#!/usr/bin/env python3
"""Load the finalized common-data view through DPLM-2's native dataset code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from official_baselines.dplm2.data_adapter import install


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-view", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def shapes(value):
    if isinstance(value, dict):
        return {key: shapes(item) for key, item in value.items()}
    if isinstance(value, list):
        return {"type": "list", "length": len(value)}
    return {"type": type(value).__name__, "shape": list(value.shape)}


def main() -> None:
    args = arguments()
    audit = json.loads((args.dataset_view / "adapter_audit.json").read_text())
    install()
    from byprot.datamodules.dataset.tokenized_protein import (
        DPLM2Collater,
        TokenizedProteinDataset,
    )

    datasets = {
        split: TokenizedProteinDataset(
            str(args.dataset_view.parent),
            split,
            args.dataset_view.name,
            max_len=128,
            vocab_file=str(args.config),
        )
        for split in ("train", "valid")
    }
    counts = {split: len(dataset) for split, dataset in datasets.items()}
    expected = {
        split: audit["splits"][split]["records"] for split in ("train", "valid")
    }
    if counts != expected:
        raise AssertionError(f"native dataset counts {counts} != audit {expected}")
    samples = [datasets["train"][index] for index in range(4)]
    batch = DPLM2Collater(datasets["train"].tokenizer)(samples)
    report = {
        "schema_version": "foldingtrain-dplm2-native-loader-smoke-v1",
        "quality_comparable": False,
        "dataset_view": str(args.dataset_view),
        "adapter_schema": audit["schema_version"],
        "records": counts,
        "sample_ids": [sample["pdb_name"] for sample in samples],
        "official_dataset_class": (
            f"{type(datasets['train']).__module__}."
            f"{type(datasets['train']).__name__}"
        ),
        "official_collater_class": (
            "byprot.datamodules.dataset.tokenized_protein.DPLM2Collater"
        ),
        "batch": shapes(batch),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
