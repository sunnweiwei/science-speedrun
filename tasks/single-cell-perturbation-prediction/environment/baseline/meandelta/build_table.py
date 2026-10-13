#!/usr/bin/env python3
"""Training step for B1/B1g: the cross-context mean-delta table.

For every target measured in >= MIN_CELLS cells in a training context, take that context's
pseudobulk delta (mean log1p CP10k of perturbed cells minus control) per gene, and average the
deltas over the contexts that measured the target. Genes absent from a context's panel do not
contribute to that context's average. Writes delta_table.npz next to this script.

    python build_table.py --data /workspace/data --contexts k562 jurkat hct116 hek293t
"""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np

MIN_CELLS = 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/workspace/data"))
    ap.add_argument("--contexts", nargs="+", required=True, help="training contexts to pool (LOCO: leave one out)")
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "delta_table.npz")
    a = ap.parse_args()
    genes: dict[str, int] = {}
    per_ctx = []
    for ctx in a.contexts:
        pb = ad.read_h5ad(a.data / "train" / ctx / "pseudobulk.h5ad")
        pb = pb[(pb.obs["n_cells"] >= MIN_CELLS) & ~pb.obs["is_control"]].copy()
        for g in pb.var_names:
            genes.setdefault(str(g), len(genes))
        per_ctx.append((ctx, pb))
    targets: dict[str, int] = {}
    for _, pb in per_ctx:
        for t in pb.obs_names:
            targets.setdefault(str(t), len(targets))
    G, T = len(genes), len(targets)
    total = np.zeros((T, G), dtype=np.float64)
    count = np.zeros((T, G), dtype=np.int32)
    n_ctx = np.zeros(T, dtype=np.int32)
    for ctx, pb in per_ctx:
        gcol = np.array([genes[str(g)] for g in pb.var_names])
        trow = np.array([targets[str(t)] for t in pb.obs_names])
        delta = np.asarray(pb.layers["delta_log1p_cp10k"], dtype=np.float64)
        total[np.ix_(trow, gcol)] += delta
        count[np.ix_(trow, gcol)] += 1
        n_ctx[trow] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), 0.0).astype(np.float32)
    np.savez(a.output, genes=np.array(list(genes), dtype=object), targets=np.array(list(targets), dtype=object),
                        delta=mean, n_contexts=n_ctx, contexts=np.array(a.contexts, dtype=object))
    (a.output.with_suffix(".json")).write_text(json.dumps({"contexts": a.contexts, "n_targets": T, "n_genes": G,
                                                            "min_cells": MIN_CELLS}, indent=2) + "\n")
    print(f"delta table: {T} targets x {G} genes from {a.contexts} -> {a.output}")


if __name__ == "__main__":
    main()
