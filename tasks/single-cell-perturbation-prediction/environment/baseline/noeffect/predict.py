#!/usr/bin/env python3
"""B0: for each request, resample `n_cells` basal cells (seeded by the request) unchanged."""
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def main(input_root: str, output_root: str) -> None:
    inp, out = Path(input_root), Path(output_root)
    basal = ad.read_h5ad(inp / "basal.h5ad")
    requests = [json.loads(l) for l in (inp / "requests.jsonl").read_text().splitlines() if l.strip()]
    X = sparse.csr_matrix(basal.X, dtype=np.float32)
    blocks, ids, targets = [], [], []
    for r in requests:
        rng = np.random.default_rng(int(r.get("seed", 0)))
        idx = rng.choice(basal.n_obs, size=int(r["n_cells"]), replace=basal.n_obs < int(r["n_cells"]))
        blocks.append(X[idx]); ids += [r["request_id"]] * len(idx); targets += [r["perturbation"]["target"]] * len(idx)
    pred = ad.AnnData(X=sparse.vstack(blocks).tocsr(),
                      obs=pd.DataFrame({"request_id": ids, "target_gene": targets}, index=[f"p{i}" for i in range(len(ids))]),
                      var=pd.DataFrame(index=basal.var_names.astype(str)))
    out.mkdir(parents=True, exist_ok=True)
    pred.write_h5ad(out / "predictions.h5ad")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
