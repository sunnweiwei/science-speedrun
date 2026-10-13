#!/usr/bin/env python3
"""B1 / B1g inference: shift resampled basal cells by the cross-context mean delta.

Mode is read from mode.json next to this script: {"gated": false} = B1, {"gated": true} = B1g.
B1g gates the delta: it is applied only if the target gene is detected in the basal state (or is
absent from the panel, in which case nothing is known and the delta is applied), and only on
genes detected in >= 1% of basal cells. Unknown targets -> no effect.
"""
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent


def normlog(X: sparse.csr_matrix):
    totals = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    totals[totals == 0] = 1.0
    Y = sparse.diags(1e4 / totals) @ X
    Y.data = np.log1p(Y.data)
    return Y.tocsr(), totals


def main(input_root: str, output_root: str) -> None:
    inp, out = Path(input_root), Path(output_root)
    mode = json.loads((HERE / "mode.json").read_text()) if (HERE / "mode.json").exists() else {"gated": False}
    table = np.load(HERE / "delta_table.npz", allow_pickle=True)
    t_index = {str(t): i for i, t in enumerate(table["targets"])}
    g_index = {str(g): i for i, g in enumerate(table["genes"])}
    delta_table = table["delta"]  # materialise once; NpzFile re-decompresses on every access
    basal = ad.read_h5ad(inp / "basal.h5ad")
    requests = [json.loads(l) for l in (inp / "requests.jsonl").read_text().splitlines() if l.strip()]
    X = sparse.csr_matrix(basal.X, dtype=np.float32)
    genes = [str(g) for g in basal.var_names]
    col = np.array([g_index.get(g, -1) for g in genes])
    known = col >= 0
    logX, totals = normlog(X)
    detect = np.asarray((X > 0).mean(axis=0)).ravel()
    gene_gate = (detect >= 0.01).astype(np.float32) if mode["gated"] else np.ones(len(genes), dtype=np.float32)
    blocks, ids, targets = [], [], []
    for r in requests:
        rng = np.random.default_rng(int(r.get("seed", 0)))
        n = int(r["n_cells"])
        idx = rng.choice(basal.n_obs, size=n, replace=basal.n_obs < n)
        target = r["perturbation"]["target"]
        cells = logX[idx].toarray()
        if target in t_index:
            delta = np.zeros(len(genes), dtype=np.float32)
            delta[known] = delta_table[t_index[target], col[known]]
            if mode["gated"]:
                if target in basal.var_names and detect[basal.var_names.get_loc(target)] < 0.01:
                    delta[:] = 0.0  # target not expressed here: expect no effect
                delta *= gene_gate
            cells = np.clip(cells + delta[None, :], 0.0, None)
        counts = np.expm1(cells) * (totals[idx][:, None] / 1e4)
        blocks.append(sparse.csr_matrix(counts.astype(np.float32)))
        ids += [r["request_id"]] * n; targets += [target] * n
    pred = ad.AnnData(X=sparse.vstack(blocks).tocsr(),
                      obs=pd.DataFrame({"request_id": ids, "target_gene": targets}, index=[f"p{i}" for i in range(len(ids))]),
                      var=pd.DataFrame(index=basal.var_names.astype(str)))
    out.mkdir(parents=True, exist_ok=True)
    pred.write_h5ad(out / "predictions.h5ad")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
