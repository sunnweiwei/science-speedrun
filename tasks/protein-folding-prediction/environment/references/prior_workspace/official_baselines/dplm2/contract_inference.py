#!/usr/bin/env python3
"""Run a trained DPLM-2 bundle behind the folding request/result contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

from folding_adapters import adapt_request, normalize_predictions
from folding_contract import validate_request


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--struct-tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--max-iter", type=int, default=500)
    return parser.parse_args()


def load_generation_module():
    path = Path(os.environ.get("DPLM_ROOT", "/opt/dplm")) / "generate_dplm2.py"
    spec = importlib.util.spec_from_file_location("pinned_dplm2_generation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned DPLM-2 generation module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = arguments()
    if args.max_iter < 1:
        raise ValueError("--max-iter must be positive")
    for path in (args.model_dir, args.struct_tokenizer):
        if not path.is_dir():
            raise FileNotFoundError(path)

    import torch
    from biotite.structure.io import load_structure, save_structure
    from byprot.models.dplm2.dplm2 import (
        MultimodalDiffusionProteinLanguageModel,
    )

    request = json.loads(args.request.read_text())
    validate_request(request)
    native = adapt_request("dplm2", request)
    sequence = native["payload"]["sequence"]
    seed = native["runtime"]["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = MultimodalDiffusionProteinLanguageModel.from_pretrained(
        str(args.model_dir),
        cfg_override={
            "tokenizer": {"vocab_file": str(args.model_dir)},
            "struct_tokenizer": {"exp_path": str(args.struct_tokenizer)},
        },
    ).eval().cuda()
    # The released local-path helper expects a training-run ``.hydra`` layout,
    # while the official Hub artifact stores config.yaml and checkpoint at its
    # repository root. Reuse the already-audited loader used to tokenize the
    # frozen training view, which instantiates the same upstream VQModel.
    from official_baselines.dplm2.tokenize_unified_view import load_tokenizer

    model._struct_tokenizer = load_tokenizer(
        args.struct_tokenizer, next(model.parameters()).device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fasta = args.output_dir / "request.fasta"
    # A fixed safe native header prevents request IDs from becoming paths.
    fasta.write_text(f">prediction\n{sequence}\n")
    generation_args = SimpleNamespace(
        task="folding",
        batch_size=1,
        max_iter=args.max_iter,
        temperature=1.0,
        unmasking_strategy="stochastic1.0",
        sampling_strategy="annealing@2.0:0.1",
    )
    upstream = load_generation_module()
    batches, names = upstream.initialize_conditional_generation(
        str(fasta), model.tokenizer, next(model.parameters()).device,
        generation_args, model=model,
    )
    if len(batches) != 1 or names != [["prediction"]]:
        raise RuntimeError("unexpected DPLM-2 conditional-generation batching")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.generate(
            input_tokens=batches[0]["input_tokens"],
            max_iter=args.max_iter,
            temperature=1.0,
            unmasking_strategy="stochastic1.0",
            sampling_strategy="annealing@2.0:0.1",
            partial_masks=batches[0]["partial_mask"],
        )
    native_output = args.output_dir / "native"
    upstream.save_results(
        tokenizer=model.tokenizer,
        struct_tokenizer=model.struct_tokenizer,
        save_dir=str(native_output),
        task="folding",
        outputs=outputs,
        headers=["prediction"],
        save_pdb=True,
    )
    pdb_files = list(native_output.rglob("*.pdb"))
    if len(pdb_files) != 1:
        raise RuntimeError(f"expected one decoded PDB, found {pdb_files}")
    prediction = args.output_dir / "prediction.cif"
    save_structure(prediction, load_structure(pdb_files[0]))
    result = normalize_predictions(
        "dplm2",
        args.model_dir.name,
        request,
        [{
            "seed": seed,
            "sample": 0,
            "structure_path": str(prediction.resolve()),
            # The released DPLM-2 folding path emits no confidence head.  This
            # bundle produces exactly one sample, so a finite constant defines
            # the required FoldBench ordering without changing sample choice.
            "ranking_score": 0.0,
            "confidence": {},
        }],
        provenance={
            "checkpoint": str(args.model_dir.resolve()),
            "struct_tokenizer": str(args.struct_tokenizer.resolve()),
            "max_iter": args.max_iter,
        },
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
