#!/usr/bin/env python3
"""Export the exact final-test relaxation without running the expensive scorer."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, "/opt/crystalgen")
from scoring.contract import load_output, load_requests
from scoring.relaxation import relax_records, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--generated", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--output", required=True, type=lambda value: Path(value).resolve())
    args = parser.parse_args()
    requests = load_requests(args.input)
    records, errors = load_output(args.generated, requests)
    _, summary = relax_records(records, sum(r["n_samples"] for r in requests.values()),
                               args.output, Path("/opt/lemat/.venv/bin/python"))
    write_json(args.output / "output_errors.json", errors)
    print(f"Relaxed {summary['relaxed']}/{summary['requested']} requested structures. "
          f"CIFs: {args.output / 'generated'}; diagnostics: {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
