#!/usr/bin/env python3
"""Serve the trained scaled OpenDDE graph through the folding contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from folding_adapters import adapt_request, normalize_predictions
from folding_contract import validate_request
from official_baselines.opendde.config import (
    MODEL_NAME_50M,
    UPSTREAM_REVISION,
    configure_ccd,
    scaled_config,
)
from official_baselines.protenix.postprocess_foldbench import normalize_cif


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ccd-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    return parser.parse_args()


def _prediction_paths(root: Path, seed: int, sample: int) -> tuple[Path, Path]:
    prediction_root = root / "native" / "prediction" / f"seed_{seed}" / "predictions"
    return (
        prediction_root / f"prediction_sample_{sample}.cif",
        prediction_root / f"prediction_summary_confidence_sample_{sample}.json",
    )


def main() -> None:
    args = arguments()
    if args.diffusion_steps < 1:
        raise ValueError("--diffusion-steps must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    configure_ccd(args.ccd_dir)

    request = json.loads(args.request.read_text())
    validate_request(request)
    native = adapt_request("opendde", request)
    native["payload"][0]["name"] = "prediction"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.output_dir / "input.json"
    input_path.write_text(json.dumps(native["payload"], indent=2) + "\n")

    configs = scaled_config()
    configs.model_name = MODEL_NAME_50M
    configs.input_json_path = str(input_path)
    configs.dump_dir = str(args.output_dir / "native")
    configs.load_checkpoint_path = str(args.checkpoint)
    configs.load_strict = True
    configs.seeds = list(request["run"]["seeds"])
    configs.model.N_cycle = 1
    configs.sample_diffusion.N_step = args.diffusion_steps
    configs.sample_diffusion.N_sample = request["run"]["num_samples"]
    configs.dtype = "bf16"
    configs.device = "cuda"
    configs.use_msa = False
    configs.data.msa.enable_prot_msa = False
    configs.use_template = False
    configs.data.template.enable_prot_template = False
    configs.use_rna_msa = False
    configs.need_atom_confidence = False
    configs.data.ccd_components_file = str(args.ccd_dir / "components.cif")
    configs.data.ccd_components_rdkit_mol_file = str(
        args.ccd_dir / "components.cif.rdkit_mol.pkl"
    )

    from runner.inference import InferenceRunner, infer_predict

    runner = InferenceRunner(configs)
    try:
        infer_predict(runner, runner.configs)
    finally:
        runner.close()

    error_dir = args.output_dir / "native" / "ERR"
    errors = sorted(error_dir.glob("*")) if error_dir.exists() else []
    if errors:
        details = "\n".join(path.read_text(errors="replace") for path in errors)
        raise RuntimeError(f"upstream OpenDDE inference failed:\n{details}")

    predictions = []
    normalized_dir = args.output_dir / "predictions"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    for seed in request["run"]["seeds"]:
        for sample in range(request["run"]["num_samples"]):
            structure, confidence = _prediction_paths(args.output_dir, seed, sample)
            if not structure.is_file() or not confidence.is_file():
                raise RuntimeError(
                    "released OpenDDE runner did not emit the declared prediction: "
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
                    "ranking_score": confidence_data.get(
                        "final_score", confidence_data.get("ranking_score")
                    ),
                    "confidence": confidence_data,
                }
            )

    result = normalize_predictions(
        "opendde",
        MODEL_NAME_50M,
        request,
        predictions,
        revision=UPSTREAM_REVISION,
        provenance={
            "checkpoint": str(args.checkpoint.resolve()),
            "diffusion_steps": args.diffusion_steps,
            "cycles": 1,
            "templates_used": False,
        },
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
