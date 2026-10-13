"""Create an input-only view of validation, optionally shuffled/subsampled."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import shutil
import numpy as np
from qecbench.data import load_condition, load_manifest, sha256, write_json


def stage(source: Path, output: Path, limit: int | None = None, seed: int = 0,
          group: str | None = None) -> None:
    manifest = copy.deepcopy(load_manifest(source, allow_labels=True))
    if manifest["split"] != "validation" or output.exists():
        raise ValueError("Require validation source and new output directory")
    rng = np.random.default_rng(seed)
    output.mkdir(parents=True)
    manifest["split"] = "validation-inputs"
    records = [row for row in manifest["conditions"] if group is None or row["group"] == group]
    if not records:
        raise ValueError("No matching conditions")
    for record in records:
        arrays = load_condition(source, record, labels=True)
        order = rng.permutation(len(arrays["sample_ids"]))
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive")
            order = order[:limit]
        destination = output / record["arrays"]
        destination.parent.mkdir(parents=True)
        np.savez(destination, sample_ids=arrays["sample_ids"][order], detectors=arrays["detectors"][order])
        shutil.copyfile(source / record["circuit"], output / record["circuit"])
        record.update(shots=len(order), arrays_sha256=sha256(destination))
        record.pop("noise", None)
        record.pop("noise_sha256", None)
    manifest["conditions"] = list(reversed(records))
    manifest["shots"] = sum(row["shots"] for row in records)
    write_json(output / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group")
    args = parser.parse_args()
    stage(args.source, args.output, args.limit, args.seed, args.group)


if __name__ == "__main__":
    main()
