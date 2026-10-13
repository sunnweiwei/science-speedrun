#!/usr/bin/env python3
"""Glue around `state tx infer` for the delivery contract (see infer.sh)."""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent


def normlog(X):
    X = sparse.csr_matrix(X, dtype=np.float32)
    totals = np.asarray(X.sum(axis=1)).ravel().astype(np.float32); totals[totals == 0] = 1
    Y = sparse.diags(1e4 / totals) @ X
    Y.data = np.log1p(Y.data)
    return Y.tocsr(), totals


def read_requests(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def prepare(a):
    basal = ad.read_h5ad(a.input / "basal.h5ad")
    basal.var["gene_name"] = basal.var_names.astype(str)
    basal.write_h5ad(a.work / "basal_in.h5ad")


def pad(a):
    """Resample basal cells per request (seeded) and label them with the target; SE embedding rows follow."""
    basal = ad.read_h5ad(a.input / "basal.h5ad")
    emb = np.load(a.work / "basal_se.npy").astype(np.float32) if (a.work / "basal_se.npy").exists() else None
    requests = read_requests(a.input / "requests.jsonl")
    genes = json.loads((HERE / "st_data" / "output_genes.json").read_text())
    in_genes = list(map(str, basal.var_names))
    logX, totals = normlog(basal.X)
    rows, labels, rid, tot = [], [], [], []
    for r in requests:
        rng = np.random.default_rng(int(r.get("seed", 0)))
        idx = rng.choice(basal.n_obs, size=int(r["n_cells"]), replace=basal.n_obs < int(r["n_cells"]))
        rows.append(idx); labels += [r["perturbation"]["target"]] * len(idx); rid += [r["request_id"]] * len(idx)
    idx_all = np.concatenate(rows)
    # controls first (state tx infer needs a control pool), then the virtual perturbed copies
    n_ctrl = min(basal.n_obs, 2000)
    ctrl_idx = np.arange(n_ctrl)
    all_idx = np.concatenate([ctrl_idx, idx_all])
    obs = pd.DataFrame({"gene": ["non-targeting"] * n_ctrl + labels, "request_id": [""] * n_ctrl + rid,
                        "gem_group": basal.obs["batch"].astype(str).to_numpy()[all_idx],
                        "source_row": all_idx}, index=[f"v{i}" for i in range(len(all_idx))])
    # .X on the model's output panel (missing genes -> 0); the input panel is restored in `finish`
    col = {g: i for i, g in enumerate(in_genes)}
    sel = [col.get(g, -1) for g in genes]
    X = sparse.lil_matrix((len(all_idx), len(genes)), dtype=np.float32)
    present = [i for i, c in enumerate(sel) if c >= 0]
    X[:, present] = logX[all_idx][:, [sel[i] for i in present]]
    padded = ad.AnnData(X=X.tocsr(), obs=obs, var=pd.DataFrame(index=genes))
    if emb is not None:
        padded.obsm["X_state"] = emb[all_idx]
    padded.write_h5ad(a.work / "padded.h5ad")
    np.save(a.work / "totals.npy", totals[all_idx])


def finish(a):
    basal = ad.read_h5ad(a.input / "basal.h5ad")
    pred = ad.read_h5ad(a.work / "pred.h5ad")
    padded = ad.read_h5ad(a.work / "padded.h5ad")
    totals = np.load(a.work / "totals.npy")
    genes = list(map(str, padded.var_names))
    in_genes = list(map(str, basal.var_names))
    keep = padded.obs["request_id"].astype(str).to_numpy() != ""
    P = pred.X  # gene-space ST: `state tx infer` overwrites .X with the predicted log1p CP10k expression
    if P is None or P.shape[1] != len(genes):
        raise RuntimeError("state tx infer did not return gene-space predictions on the model panel")
    P = np.asarray(P.todense() if sparse.issparse(P) else P, dtype=np.float32)[keep]
    P = np.clip(P, 0.0, None)
    logB, _ = normlog(basal.X)
    src = padded.obs["source_row"].to_numpy()[keep]
    out_log = logB[src].toarray()  # start from the basal cell itself (no effect on genes outside the model panel)
    col = {g: i for i, g in enumerate(in_genes)}
    for j, g in enumerate(genes):
        if g in col:
            out_log[:, col[g]] = P[:, j]
    counts = np.expm1(out_log) * (totals[keep][:, None] / 1e4)
    result = ad.AnnData(X=sparse.csr_matrix(counts.astype(np.float32)),
                        obs=pd.DataFrame({"request_id": padded.obs["request_id"].astype(str).to_numpy()[keep],
                                          "target_gene": padded.obs["gene"].astype(str).to_numpy()[keep]},
                                         index=[f"p{i}" for i in range(int(keep.sum()))]),
                        var=pd.DataFrame(index=basal.var_names.astype(str)))
    a.output.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(a.output / "predictions.h5ad")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "pad", "finish"])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    {"prepare": prepare, "pad": pad, "finish": finish}[a.stage](a)


if __name__ == "__main__":
    main()
