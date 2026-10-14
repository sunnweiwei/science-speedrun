#!/usr/bin/env python3
"""Serve a trained Protenix checkpoint through the public folding contract.

This adapter deliberately performs only schema conversion and artifact
normalization.  Model construction, featurization, diffusion, confidence and
checkpoint loading continue to execute in the pinned upstream Protenix code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from folding_adapters import adapt_request, normalize_predictions
from folding_contract import validate_request
from official_baselines.protenix.config import MODEL_NAME
from official_baselines.protenix.postprocess_foldbench import normalize_cif


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    return parser.parse_args()


def _prediction_paths(
    root: Path, seed: int, sample: int
) -> tuple[Path, Path]:
    prediction_root = root / "native" / "prediction" / f"seed_{seed}" / "predictions"
    structure = prediction_root / f"prediction_sample_{sample}.cif"
    confidence = prediction_root / f"prediction_summary_confidence_sample_{sample}.json"
    return structure, confidence


def main() -> None:
    args = arguments()
    if args.diffusion_steps < 1:
        raise ValueError("--diffusion-steps must be positive")
    checkpoint = args.checkpoint_dir / f"{MODEL_NAME}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    request = json.loads(args.request.read_text())
    validate_request(request)
    native = adapt_request("protenix", request)
    # Never allow a public request identifier to become a filesystem path in
    # the upstream dumper.  The identifier is restored in the result contract.
    native["payload"][0]["name"] = "prediction"
    input_path = args.output_dir / "input.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path.write_text(json.dumps(native["payload"], indent=2) + "\n")

    upstream_argv = [
        "protenix-contract-inference",
        "--model_name",
        MODEL_NAME,
        "--input_json_path",
        str(input_path),
        "--dump_dir",
        str(args.output_dir / "native"),
        "--load_checkpoint_dir",
        str(args.checkpoint_dir),
        "--use_seeds_in_json",
        "true",
        "--use_msa",
        "false",
        "--use_template",
        "false",
        "--use_rna_msa",
        "false",
        "--model.N_cycle",
        "4",
        "--sample_diffusion.N_step",
        str(args.diffusion_steps),
        "--sample_diffusion.N_sample",
        str(request["run"]["num_samples"]),
        "--triangle_attention",
        "cuequivariance",
        "--triangle_multiplicative",
        "cuequivariance",
        "--dtype",
        "bf16",
    ]
    original_argv = sys.argv
    try:
        sys.argv = upstream_argv
        from official_baselines.protenix.inference import main as run_upstream

        run_upstream()
    finally:
        sys.argv = original_argv

    error_dir = args.output_dir / "native" / "ERR"
    errors = sorted(error_dir.glob("*")) if error_dir.exists() else []
    if errors:
        details = "\n".join(path.read_text(errors="replace") for path in errors)
        raise RuntimeError(f"upstream Protenix inference failed:\n{details}")

    predictions = []
    normalized_dir = args.output_dir / "predictions"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    for seed in request["run"]["seeds"]:
        for sample in range(request["run"]["num_samples"]):
            structure, confidence = _prediction_paths(args.output_dir, seed, sample)
            if not structure.is_file() or not confidence.is_file():
                raise RuntimeError(
                    "upstream Protenix did not emit the declared prediction: "
                    f"{structure}, {confidence}"
                )
            destination = normalized_dir / f"seed_{seed}_sample_{sample}.cif"
            normalize_cif(structure, destination)
            confidence_data = json.loads(confidence.read_text())
            predictions.append(
                {
                    "seed": seed,
                    "sample": sample,
                    "structure_path": str(destination.resolve()),
                    "ranking_score": confidence_data.get("ranking_score"),
                    "confidence": confidence_data,
                }
            )

    result = normalize_predictions(
        "protenix",
        MODEL_NAME,
        request,
        predictions,
        revision="c3bfc365b3e1341a11935eddfe7bfdc308092147",
        provenance={
            "checkpoint": str(checkpoint.resolve()),
            "diffusion_steps": args.diffusion_steps,
            "cycles": 4,
            "templates_used": False,
        },
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
