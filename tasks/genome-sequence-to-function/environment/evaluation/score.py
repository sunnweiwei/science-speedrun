#!/usr/bin/env python3
"""Score a prediction directory with the verifier's own code.

    python /workspace/evaluation/score.py --predictions OUT/predictions.zarr \
        --labels /workspace/data --split valid --output REPORT_DIR

Validation labels are mounted read-only under /workspace/data; the test split is graded only by
the verifier.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import score, validate_predictions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split", default="valid", choices=("train", "valid", "test"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import pandas as pd
    ids = pd.read_parquet(args.labels / f"labels/{args.split}.parquet").example_id.astype(str).tolist()
    validate_predictions(args.predictions, ids)
    report = score(args.predictions, args.labels, args.split)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    reward = report["overall_mean_family_pearson"]
    (args.output / "reward.json").write_text(json.dumps({"reward": reward}, indent=2) + "\n")
    print(json.dumps({"reward": reward, **{k: v["mean_pearson"] for k, v in report["families"].items()}},
                     indent=2))


if __name__ == "__main__":
    main()
