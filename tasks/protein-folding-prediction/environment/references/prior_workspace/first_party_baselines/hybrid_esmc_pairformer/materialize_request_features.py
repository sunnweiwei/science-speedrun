#!/usr/bin/env python3
"""Materialize the hybrid's exact frozen ESMC mixed-256 request features.

Training features and inference features meet at the same boundary: official
ESMFold2-Fast ``base_z_linear`` followed by its learned ``base_z_combine``.
The frozen ESMC-6B and ESMFold2-Fast parameters are not part of the 50M
folding-parameter budget.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from common import ESMC_REVISION, FOLDING_REVISION, load_model
from first_party_baselines.hybrid_esmc_pairformer.config import FEATURE_ID


EMBEDDING_DIM = 256


def _protein_sequences(request: dict[str, Any]) -> list[str]:
    sequences = {
        entity["sequence"]
        for entity in request["target"]["entities"]
        if entity["type"] == "protein"
    }
    return sorted(sequences, key=lambda sequence: (len(sequence), sequence))


def _inputs(sequence: str, device: str) -> dict[str, Any]:
    from transformers.models.esmfold2.protein_utils import prepare_protein_features

    return {
        key: value.to(device)
        for key, value in prepare_protein_features(sequence).items()
    }


def materialize(
    request: dict[str, Any], output_root: Path, *, device: str = "cuda"
) -> Path:
    """Write request-local safetensors plus a validated cache manifest."""
    import torch
    from safetensors.torch import save_file

    sequences = _protein_sequences(request)
    if any(len(sequence) > 2048 for sequence in sequences):
        raise ValueError("hybrid ESMC inference supports protein entities up to 2048 residues")
    if sequences and not torch.cuda.is_available():
        raise RuntimeError("frozen ESMC-6B feature generation requires CUDA")

    output_root.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    model = None
    try:
        if sequences:
            model = load_model(device, precision="bf16").eval()
        with torch.inference_mode():
            for sequence in sequences:
                values = _inputs(sequence, device)
                hidden = model._compute_lm_hidden_states(
                    values["input_ids"],
                    values["asym_id"],
                    values["residue_index"],
                    values["mol_type"],
                    values["token_attention_mask"],
                    lm_mask_pct=0.0,
                )
                projected = model.language_model.base_z_linear(hidden)
                weights = model.language_model.base_z_combine.softmax(0)
                mixed = torch.einsum("n,blnd->bld", weights, projected)[0]
                mixed = mixed[: len(sequence)].to(torch.bfloat16).cpu().contiguous()
                if tuple(mixed.shape) != (len(sequence), EMBEDDING_DIM):
                    raise RuntimeError(
                        f"unexpected mixed feature shape {tuple(mixed.shape)}"
                    )
                digest = hashlib.sha256(sequence.encode()).hexdigest()
                path = output_root / f"{digest}.safetensors"
                save_file(
                    {"single": mixed},
                    str(path),
                    metadata={
                        "schema_version": FEATURE_ID,
                        "sequence_sha256": digest,
                        "esmc_revision": ESMC_REVISION,
                        "folding_revision": FOLDING_REVISION,
                        "boundary": "LanguageModelShim.base_z_linear+base_z_combine;before_base_z_mlp",
                        "kernel_backend": "pytorch-sdpa-no-xformers-no-transformer-engine",
                    },
                )
                records.append(
                    {
                        "sequence": sequence,
                        "sequence_sha256": digest,
                        "feature_version": FEATURE_ID,
                        "local_status": "ready",
                        "shape": [len(sequence), EMBEDDING_DIM],
                        "local_path": str(path.resolve()),
                    }
                )
                del values, hidden, projected, mixed
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = output_root / "manifest.jsonl"
    with manifest.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    return manifest
