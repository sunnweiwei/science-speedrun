#!/usr/bin/env python3
"""CLI for the trusted scorer."""

import argparse
from pathlib import Path
import json

from panel_scoring import PRIMARY
from risk_scoring import score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--trusted", type=Path, required=True)
    parser.add_argument("--normalization", type=Path,
                        help="Frozen train-only risk normalization parquet")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="Optional patient bootstrap replicates; disabled by default")
    args = parser.parse_args()
    report = score(args.input, args.trusted, args.predictions, args.bootstrap,
                   args.normalization)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    primary = report["panel"][PRIMARY]
    reward = primary if primary is not None else 0.0
    (args.output / "reward.json").write_text(
        json.dumps({"reward": reward}, indent=2) + "\n")


if __name__ == "__main__":
    main()
