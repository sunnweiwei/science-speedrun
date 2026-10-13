#!/usr/bin/env python3
"""Fail early if a development mount lacks complete, separated inputs."""
from __future__ import annotations
import argparse
from pathlib import Path
from qecbench.data import load_condition, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    expected = {"train": 2584400, "validation": 665600}
    for split, shots in expected.items():
        manifest = load_manifest(args.data_root / split, allow_labels=True)
        if manifest["split"] != split or len(manifest["conditions"]) != 130 or sum(row["shots"] for row in manifest["conditions"]) != shots:
            raise ValueError(f"Unexpected {split} data coverage")
        for row in manifest["conditions"]:
            load_condition(args.data_root / split, row, labels=True)
    reference = args.data_root / "reference"
    for required in ("package-template/infer.sh", "results/validation_report.json", "results/test_summary.json"):
        if not (reference / required).is_file():
            raise FileNotFoundError(f"Reference material is missing: reference/{required}")
    if (args.data_root / "starter").exists() or any(reference.rglob("model.npz")):
        raise ValueError("Development mount must not include trained model weights")
    if any((args.data_root / name).exists() for name in ("test", "verifier", "grader")):
        raise ValueError("Development mount must not include evaluation role directories")
    print("Verified development inputs: 130 conditions, separated train/validation, reference results without weights")


if __name__ == "__main__":
    main()
