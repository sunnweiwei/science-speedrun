#!/usr/bin/env python3
"""Serve the trained scaled ESMFold2 graph through the folding contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from folding_adapters import adapt_request, normalize_predictions
from folding_contract import validate_request
from official_baselines.esmfold2.config import (
    ESMC_REPOSITORY,
    ESMC_REVISION,
    MODEL_NAME_50M,
    TRANSFORMERS_REVISION,
)
from official_baselines.esmfold2.trainable_forward import build_model, persistent_state_dict


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--ccd-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--loops", type=int, default=3)
    parser.add_argument("--sampling-steps", type=int, default=20)
    return parser.parse_args()


def _native_input(request: dict):
    """Translate every public biomolecule entity through Biohub's own API."""

    from esm.models.esmfold2 import (
        DNAInput,
        LigandInput,
        Modification,
        ProteinInput,
        RNAInput,
        StructurePredictionInput,
    )

    classes = {
        "protein": ProteinInput,
        "rna": RNAInput,
        "dna": DNAInput,
        "ligand": LigandInput,
    }
    payload = adapt_request("esmfold2", request)["payload"]
    sequences = []
    for value in payload["sequences"]:
        kind = value["type"]
        kwargs = {"id": value["id"]}
        if kind == "ligand":
            kwargs.update({key: value.get(key) for key in ("smiles", "ccd")})
        else:
            kwargs["sequence"] = value["sequence"]
            kwargs["modifications"] = [
                Modification(**modification)
                for modification in value.get("modifications", [])
            ]
        sequences.append(classes[kind](**kwargs))
    return StructurePredictionInput(sequences=sequences)


def main() -> None:
    args = arguments()
    if args.loops < 0 or args.sampling_steps < 1:
        raise ValueError("loops must be non-negative and sampling steps positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    ccd = args.ccd_dir / "ccd.pkl"
    if not ccd.is_file():
        raise FileNotFoundError(ccd)

    import numpy as np
    import torch
    from esm.models.esmfold2 import ESMFold2InputBuilder
    from huggingface_hub import snapshot_download

    request = json.loads(args.request.read_text())
    validate_request(request)
    native_input = _native_input(request)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise ValueError("deployment checkpoint has no model state")

    model = build_model()
    model.load_state_dict(persistent_state_dict(payload["model"]), strict=True)
    model = model.to("cuda")
    esmc = snapshot_download(
        ESMC_REPOSITORY,
        revision=ESMC_REVISION,
        repo_type="model",
        cache_dir=args.hf_home,
        local_files_only=True,
    )
    model.load_esmc(esmc, precision="bf16")
    model.set_kernel_backend("fused")
    model.eval()
    builder = ESMFold2InputBuilder(ccd_cache=args.ccd_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    predictions = []
    for seed in request["run"]["seeds"]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        output = builder.fold(
            model,
            native_input,
            seed=seed,
            num_loops=args.loops,
            num_sampling_steps=args.sampling_steps,
            num_diffusion_samples=request["run"]["num_samples"],
            complex_id="prediction",
        )
        outputs = output if isinstance(output, list) else [output]
        if len(outputs) != request["run"]["num_samples"]:
            raise RuntimeError(
                f"released ESMFold2 returned {len(outputs)} samples, expected "
                f"{request['run']['num_samples']}"
            )
        for sample, native in enumerate(outputs):
            prediction = prediction_dir / f"seed_{seed}_sample_{sample}.cif"
            prediction.write_text(native.complex.to_mmcif())
            ptm = float(native.ptm or 0.0)
            iptm = None if native.iptm is None else float(native.iptm)
            ranking = ptm if iptm is None else 0.2 * ptm + 0.8 * iptm
            predictions.append(
                {
                    "seed": seed,
                    "sample": sample,
                    "structure_path": str(prediction.resolve()),
                    "ranking_score": ranking,
                    "confidence": {"ptm": ptm, "iptm": iptm},
                }
            )
        torch.cuda.empty_cache()

    result = normalize_predictions(
        "esmfold2",
        MODEL_NAME_50M,
        request,
        predictions,
        revision=TRANSFORMERS_REVISION,
        provenance={
            "checkpoint": str(args.checkpoint.resolve()),
            "esmc_revision": ESMC_REVISION,
            "loops": args.loops,
            "sampling_steps": args.sampling_steps,
            "task_projection": "full public target through Biohub StructurePredictionInput",
        },
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
