#!/usr/bin/env python3
"""Development-facing entry point for the shared CrystalGen evaluator."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, "/opt/crystalgen")
from scoring.evaluate import evaluate, write_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--generated", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--output", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--skip-relaxation", action="store_true",
                        help="Development diagnostic only: score the supplied geometry without pre-relaxing")
    args = parser.parse_args()
    rewards, report = evaluate(
        args.input,
        args.generated,
        Path("/opt/lemat"),
        Path("/opt/lemat/.venv/bin/python"),
        artifact_root=args.output,
        skip_relaxation=args.skip_relaxation,
    )
    write_results(args.output, rewards, report)


if __name__ == "__main__":
    main()
