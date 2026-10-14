#!/usr/bin/env python3
"""Create a small official-format Protenix index without changing the loader."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdb-ids", nargs="+", required=True)
    args = parser.parse_args()
    wanted = {pdb_id.lower() for pdb_id in args.pdb_ids}

    with gzip.open(args.input, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row["pdb_id"].lower() in wanted]
        fieldnames = reader.fieldnames
    if fieldnames is None or not rows:
        raise ValueError("no matching Protenix index rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "pdb_ids": sorted(wanted),
                "rows": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
