"""Predict both sequences of every perturbation (contract: perturbations/v1).

    python perturb.py INPUT_ROOT OUTPUT_ROOT --checkpoint last.pt

INPUT_ROOT holds `public/perturbations.parquet` and `reference/GRCh38.p13.genome.fa` (in the
workspace, `/workspace/data` works too: the queries are then read from `variant_evals/public/`). Each row is a
1,048,576-bp window starting at `input_start` and an edit spanning [edit_start, edit_end) in genomic
0-based coordinates:

- `snv`: sequence A carries `allele_a` at edit_start, sequence B carries `allele_b` (both are written
  explicitly; either may differ from the reference genome);
- `mask`: sequence A is the reference, sequence B has the interval replaced by N.

For each perturbation the output group `perturbations/p{index:06d}` holds `a/<family>` and
`b/<family>` for every family in `families`: dense families sliced to bins
[dense_bin_start, dense_bin_end) of the central target window, and `splice_sites` sliced to bp
[splice_start, splice_end). Values are the same quantities as predictions.zarr (asinh space for
dense tracks, splice-site scores for splice_sites).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from numcodecs import Blosc
import numpy as np
import pandas as pd
import pysam
import torch
import zarr

from reference.alphagenome_50m.dataset import AlphaGenomeDataset
from reference.alphagenome_50m.model import CompactModel, ModelConfig

PERTURBATION_SCHEMA = "genome-sequence-to-function-perturbations/v1"
COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


def sequences(fasta, row) -> tuple[str, str]:
    length = fasta.get_reference_length(row.chrom)
    start, end = int(row.input_start), int(row.input_start) + 1_048_576
    sequence = fasta.fetch(row.chrom, max(0, start), min(length, end)).upper()
    sequence = "N" * max(0, -start) + sequence + "N" * max(0, end - length)
    lo, hi = int(row.edit_start) - start, int(row.edit_end) - start
    if row.edit == "snv":
        return sequence[:lo] + row.allele_a + sequence[hi:], sequence[:lo] + row.allele_b + sequence[hi:]
    if row.edit == "mask":
        return sequence, sequence[:lo] + "N" * (hi - lo) + sequence[hi:]
    raise ValueError(f"unknown edit {row.edit!r}")


def main(input_root: Path, output_root: Path, checkpoint: Path, queries: Path | None = None) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = ModelConfig(**payload["config"])
    model = CompactModel(config)
    model.load_state_dict(payload["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    if queries is None:
        queries = input_root / "public/perturbations.parquet"
        if not queries.exists():
            queries = input_root / "variant_evals/public/perturbations.parquet"
    rows = pd.read_parquet(queries)
    fasta = pysam.FastaFile(str(input_root / "reference/GRCh38.p13.genome.fa"))
    output_root.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(output_root / "perturbations.zarr"), mode="w")
    target_start = (config.input_length - config.target_length) // 2
    with torch.inference_mode():
        for index, row in enumerate(rows.itertuples(index=False)):
            pair = sequences(fasta, row)
            dna = torch.from_numpy(np.stack([AlphaGenomeDataset._one_hot(s) for s in pair])).to(device)
            trunk = model._trunk(dna)
            group = root.create_group(f"perturbations/p{index:06d}")
            features = None
            for family in row.families:
                if family == "splice_sites":
                    logits = model._splice_site_logits(trunk)[:, target_start:target_start + config.target_length]
                    values = logits[:, row.splice_start:row.splice_end]
                else:
                    features = model._dense_features(trunk) if features is None else features
                    values = model.dense_heads[family](features)[:, row.dense_bin_start:row.dense_bin_end]
                # float32: one variant's effect is often below float16 resolution (about 0.004 at
                # asinh values of 4-8), so A and B would round to the same numbers.
                values = values.float().cpu().numpy()
                for side, array in zip("ab", values):
                    group.create_dataset(f"{side}/{family}", data=array, compressor=COMPRESSOR)
            print(json.dumps({"event": "perturb", "done": index + 1, "total": len(rows)}), flush=True)
    root.attrs.update({"schema": PERTURBATION_SCHEMA,
                       "perturbation_ids": rows.perturbation_id.astype(str).tolist()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent / "last.pt")
    parser.add_argument("--queries", type=Path, help="another query table, e.g. the development split")
    args = parser.parse_args()
    main(args.input_root, args.output_root, args.checkpoint, args.queries)
