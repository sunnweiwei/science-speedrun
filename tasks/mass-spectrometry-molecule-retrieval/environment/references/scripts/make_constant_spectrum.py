#!/usr/bin/env python3
"""Create the spectrum-destruction control required by the reference audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mzs", nargs="+", type=float, default=[25.0, 50.0])
    parser.add_argument("--intensities", nargs="+", type=float, default=[0.5, 1.0])
    args = parser.parse_args()
    if len(args.mzs) != len(args.intensities) or not args.mzs:
        raise ValueError("mzs and intensities must be nonempty and equal-length")
    if args.mzs != sorted(args.mzs) or min(args.mzs) <= 0:
        raise ValueError("mzs must be positive and sorted")
    if min(args.intensities) < 0 or max(args.intensities) <= 0:
        raise ValueError("intensities must be nonnegative with a positive maximum")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    count = 0
    with args.queries.open(encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8"
    ) as destination:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "query_id" not in row:
                raise ValueError(f"invalid query at line {line_number}")
            row["mzs"] = args.mzs
            row["intensities"] = args.intensities
            destination.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    temporary.replace(args.output)
    print(json.dumps({"queries": count, "mzs": args.mzs, "intensities": args.intensities}))


if __name__ == "__main__":
    main()
