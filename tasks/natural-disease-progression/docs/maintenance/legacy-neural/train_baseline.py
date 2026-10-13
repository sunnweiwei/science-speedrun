#!/usr/bin/env python3
"""Train the shared risk baseline on the train split only."""

import argparse
import json
from pathlib import Path

try:
    from model import train, train_rate_baseline
except ImportError:
    from baseline import train, train_rate_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", choices=["full", "diagnosis", "demographic", "prevalence"],
                        default="full")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Legacy v1 baseline only; use longitudinal.py for v2 early stopping")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.baseline in {"demographic", "prevalence"}:
        report = train_rate_baseline(args.materialized / "train",
            args.materialized / "manifest.json", args.output, args.baseline)
    else:
        report = train(args.materialized / "train", args.materialized / "manifest.json", args.output,
                       modalities="all" if args.baseline == "full" else "diagnosis",
                       epochs=args.epochs, batch_size=args.batch_size,
                       max_samples=args.max_samples, device_name=args.device)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
