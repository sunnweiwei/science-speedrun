#!/usr/bin/env python3
"""Score perturbation outputs on the development variant evaluations with the verifier's code.

    python /workspace/baseline/alphagenome-reference/perturb.py /workspace/data OUT --checkpoint last.pt \
        --queries /workspace/data/variant_evals/valid/public/perturbations.parquet
    python /workspace/evaluation/score_variants.py --perturbations OUT/perturbations.zarr \
        --labels /workspace/data/variant_evals/valid --output REPORT_DIR

The development split (`variant_evals/valid`) covers the same 13 evaluations as the hidden test
split, at about 40 % of its size, on disjoint variants. Each metric is reported next to the
published anchors on the same rows.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from variant_scoring import score_perturbations, validate_perturbations  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--perturbations", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="directory with public/ and labels/")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import pandas as pd
    validate_perturbations(args.perturbations, pd.read_parquet(args.labels / "public/perturbations.parquet"))
    report = score_perturbations(args.perturbations, args.labels)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {"model": item["model"], **item["anchors"]} for name, item in report.items()},
                     indent=2))


if __name__ == "__main__":
    main()
