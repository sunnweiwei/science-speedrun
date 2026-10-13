#!/usr/bin/env python3
"""B2 data preparation: one h5ad per training context in the layout arc-state expects.

  .X               log1p(CP10k) on the common output panel (genes present in every listed context)
  .obsm["X_state"] SE-600M embedding (from the shipped se_embeddings.npy; kept for experiments,
                   not used by the shipped gene-space ST)
  .obs             gene (target), cell_type (context), gem_group (batch)
Also writes zeroshot.toml, esm2_pert_features.pt (ESM-2 features for every target symbol in the
release + controls as zeros) and output_genes.json.

    python prepare_data.py --data /workspace/data --contexts k562 jurkat hct116 hek293t --out ./st_data
"""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse


def normlog(X):
    X = sparse.csr_matrix(X, dtype=np.float32)
    totals = np.asarray(X.sum(axis=1)).ravel(); totals[totals == 0] = 1
    Y = sparse.diags((1e4 / totals).astype(np.float32)) @ X
    Y.data = np.log1p(Y.data)
    return Y.tocsr()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/workspace/data"))
    ap.add_argument("--contexts", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "st_data")
    ap.add_argument("--max-cells-per-context", type=int, default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    panels = []
    for ctx in a.contexts:
        v = ad.read_h5ad(a.data / "train" / ctx / "cells.h5ad", backed="r").var_names
        panels.append(list(map(str, v)))
    master = pd.read_csv(a.data / "gene_panel.csv", index_col=0)
    common = [g for g in master.index if all(g in set(p) for p in panels)]
    common_set = set(common)
    (a.out / "output_genes.json").write_text(json.dumps(common) + "\n")
    esm = torch.load(a.data / "priors" / "esm2_gene_embeddings.pt", map_location="cpu")
    feats = {}
    for ctx in a.contexts:
        cells = ad.read_h5ad(a.data / "train" / ctx / "cells.h5ad")
        emb = np.load(a.data / "train" / ctx / "se_embeddings.npy").astype(np.float32)
        assert emb.shape[0] == cells.n_obs
        if a.max_cells_per_context and cells.n_obs > a.max_cells_per_context:
            rng = np.random.default_rng(0)
            keep = np.sort(rng.choice(cells.n_obs, a.max_cells_per_context, replace=False))
            cells, emb = cells[keep].copy(), emb[keep]
        cols = [i for i, g in enumerate(map(str, cells.var_names)) if g in common_set]
        X = normlog(cells.X)[:, cols]
        out = ad.AnnData(X=X, obs=pd.DataFrame({
            "gene": cells.obs["target_gene"].astype(str).to_numpy(),
            "cell_type": ctx,
            "gem_group": cells.obs["batch"].astype(str).to_numpy()}, index=cells.obs_names.astype(str)),
            var=pd.DataFrame(index=[str(cells.var_names[i]) for i in cols]))
        out.var["gene_name"] = out.var_names
        out.obsm["X_state"] = emb
        out.uns["log1p"] = {"base": None}
        out.write_h5ad(a.out / f"{ctx}.h5ad")
        for t in out.obs["gene"].unique():
            if t != "non-targeting" and t in esm:
                feats[t] = esm[t].float()
        print(f"{ctx}: {out.n_obs} cells x {out.n_vars} genes, {out.obs['gene'].nunique() - 1} targets")
    dim = next(iter(esm.values())).shape[0]
    feats["non-targeting"] = torch.zeros(dim)
    # every target in the release (all contexts) plus every gene with an ESM-2 embedding, so that
    # withheld / unseen targets get a real feature at inference time
    for g, v in esm.items():
        feats.setdefault(g, v.float())
    torch.save(feats, a.out / "esm2_pert_features.pt")
    toml = ["[datasets]"] + [f'{c} = "{(a.out / (c + ".h5ad")).resolve()}"' for c in a.contexts] + ["", "[training]"] + \
           [f'{c} = "train"' for c in a.contexts]
    (a.out / "zeroshot.toml").write_text("\n".join(toml) + "\n")
    print(f"common output panel: {len(common)} genes; pert features: {len(feats)}")


if __name__ == "__main__":
    main()
