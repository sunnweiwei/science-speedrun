#!/usr/bin/env python3
"""Score track predictions (and optionally perturbation outputs) with the verifier's own code.

    # Combined SSI V4 two-axis validation score (0.5 * tracks_axis + 0.5 * variants_axis):
    python /workspace/evaluation/score.py --predictions OUT/predictions.zarr \
        --perturbations PERTURB_OUT/perturbations.zarr \
        --labels /workspace/data --split valid --output REPORT_DIR

    # Fast tracks-only validation check (reports tracks_axis across 5 modalities):
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
from scoring import score, ssi_v4_genome_score, validate_predictions  # noqa: E402
from variant_scoring import score_perturbations, validate_perturbations  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split", default="valid", choices=("train", "valid", "test"))
    parser.add_argument(
        "--perturbations",
        type=Path,
        default=None,
        help="Optional perturbations.zarr to compute the full 0.5*tracks + 0.5*variants SSI V4 score.",
    )
    parser.add_argument(
        "--variant-labels",
        type=Path,
        default=None,
        help="Optional variant_evals directory (defaults to LABELS/variant_evals/valid or LABELS/variant_evals).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import pandas as pd

    ids = pd.read_parquet(args.labels / f"labels/{args.split}.parquet").example_id.astype(str).tolist()
    validate_predictions(args.predictions, ids)
    report = score(args.predictions, args.labels, args.split)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.perturbations is not None:
        v_labels = args.variant_labels
        if v_labels is None:
            candidate = args.labels / "variant_evals" / args.split
            v_labels = candidate if candidate.is_dir() else (args.labels / "variant_evals")
        validate_perturbations(args.perturbations, pd.read_parquet(v_labels / "public/perturbations.parquet"))
        report["variant_effects"] = score_perturbations(args.perturbations, v_labels)
        v4 = ssi_v4_genome_score(report["families"], report["variant_effects"])
        report["ssi_v4"] = v4
        reward_payload = {
            "score": v4["score"],
            "reward": v4["score"],
            "signed_score": v4["signed_score"],
            "tracks_axis": v4["tracks_axis"],
            "variants_axis": v4["variants_axis"],
            "overall_mean_family_pearson": report["overall_mean_family_pearson"],
        }
    else:
        reward_payload = {
            "tracks_axis": report["tracks_axis"],
            "overall_mean_family_pearson": report["overall_mean_family_pearson"],
            "reward": report["tracks_axis"],
        }

    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "reward.json").write_text(json.dumps(reward_payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                **reward_payload,
                "track_modalities": report["track_modalities"],
                **{k: v["mean_pearson"] for k, v in report["families"].items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
