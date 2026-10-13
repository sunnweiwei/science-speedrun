"""In-memory representation of one cellular context and the shared gene vocabulary."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

CONTROL = "non-targeting"


@dataclasses.dataclass
class Context:
    """Raw-count cells of one context on (a subset of) the master gene panel.

    X: CSR raw counts (cells x genes), genes in master-panel order.
    obs: target_gene (master symbol; CONTROL for controls), target_ensembl, guide_id, batch,
         total_counts, pct_counts_mt, plus source-specific QC columns prefixed `qc_`.
    var: index = master symbol; ensembl_id; master_index (position in the master panel).
    """

    name: str
    X: sparse.csr_matrix
    obs: pd.DataFrame
    var: pd.DataFrame
    chemistry: str
    source: str

    def to_anndata(self, rows: np.ndarray | None = None) -> ad.AnnData:
        X = self.X if rows is None else self.X[rows]
        obs = self.obs if rows is None else self.obs.iloc[rows]
        adata = ad.AnnData(X=X.astype(np.float32), obs=obs.copy(), var=self.var.copy())
        adata.obs_names = adata.obs_names.astype(str)
        adata.var_names = adata.var_names.astype(str)
        adata.uns["context"] = self.name
        adata.uns["chemistry"] = self.chemistry
        return adata


def log1p_cp10k(X: sparse.csr_matrix) -> sparse.csr_matrix:
    """cell-eval's `normlog` convention: counts per 10k, then log1p."""
    X = sparse.csr_matrix(X, dtype=np.float32)
    totals = np.asarray(X.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    scale = sparse.diags((1e4 / totals).astype(np.float32))
    Y = scale @ X
    Y.data = np.log1p(Y.data)
    return Y.tocsr()


def group_sums(X: sparse.csr_matrix, groups: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-group column sums. Returns (group_names, sums[G x genes], counts[G])."""
    codes, names = pd.factorize(groups.astype(str), sort=True)
    indicator = sparse.csr_matrix((np.ones(len(codes), dtype=np.float32), (codes, np.arange(len(codes)))),
                                  shape=(len(names), len(codes)))
    sums = np.asarray((indicator @ X).todense() if sparse.issparse(X) else indicator @ X, dtype=np.float64)
    counts = np.bincount(codes, minlength=len(names))
    return np.asarray(names, dtype=object), sums, counts


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")
