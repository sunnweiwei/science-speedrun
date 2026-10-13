"""Gene-level priors shipped with the release (dev/priors/).

- esm2_gene_embeddings.pt : the ESM-2 per-gene embeddings distributed with SE-600M (symbol -> 5120-d)
- go_annotations.tsv.gz   : GO term memberships (symbol, GO id, aspect, evidence) from goa_human
- depmap_train_lines.csv  : DepMap 24Q4 Chronos gene effect for the TRAINING lines only; the
                            test contexts' identities are hidden, so their rows are not shipped.
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import pandas as pd

from .context import write_json

TRAIN_LINES = {"k562": "K562", "jurkat": "JURKAT", "hct116": "HCT116", "hek293t": None}  # HEK293T is not in DepMap


def stage_priors(cache: Path, root: Path) -> None:
    out = root / "dev" / "priors"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cache / "se600m" / "protein_embeddings.pt", out / "esm2_gene_embeddings.pt")

    rows = []
    with gzip.open(cache / "go" / "goa_human.gaf.gz", "rt") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 15 or f[0] != "UniProtKB":
                continue
            if "NOT" in f[3].split("|"):
                continue
            rows.append((f[2], f[4], f[8], f[6]))
    go = pd.DataFrame(rows, columns=["symbol", "go_id", "aspect", "evidence"]).drop_duplicates()
    go.to_csv(out / "go_annotations.tsv.gz", sep="\t", index=False, compression="gzip")

    model = pd.read_csv(cache / "depmap" / "Model.csv")
    effect = pd.read_csv(cache / "depmap" / "CRISPRGeneEffect.csv", index_col=0)
    effect.columns = [c.split(" (")[0] for c in effect.columns]
    picked = {}
    for ctx, name in TRAIN_LINES.items():
        if name is None:
            continue
        hit = model[model["StrippedCellLineName"] == name]
        if len(hit) != 1 or hit["ModelID"].iloc[0] not in effect.index:
            continue
        picked[ctx] = hit["ModelID"].iloc[0]
    dep = effect.loc[list(picked.values())].T
    dep.columns = list(picked.keys())
    dep.index.name = "symbol"
    dep.to_csv(out / "depmap_train_lines.csv")
    write_json(out / "manifest.json", {
        "esm2_gene_embeddings.pt": {"n_genes": None, "dim": 5120, "source": "arcinstitute/SE-600M protein_embeddings.pt"},
        "go_annotations.tsv.gz": {"n_rows": int(len(go)), "n_genes": int(go["symbol"].nunique()), "source": "goa_human 2026-05-21"},
        "depmap_train_lines.csv": {"lines": picked, "n_genes": int(dep.shape[0]), "source": "DepMap 24Q4 CRISPRGeneEffect (Chronos)",
                                    "note": "training contexts only; HEK293T absent from DepMap; test contexts withheld by design"},
    })
    (out / "README.md").write_text(
        "# Gene priors\n\n"
        "- `esm2_gene_embeddings.pt`: dict gene symbol -> 5120-d ESM-2 embedding (from the SE-600M release).\n"
        "- `go_annotations.tsv.gz`: symbol, go_id, aspect (P/F/C), evidence; NOT-qualified rows removed.\n"
        "- `depmap_train_lines.csv`: Chronos gene-effect scores (more negative = more essential) for the\n"
        "  training contexts that exist in DepMap. Test contexts are anonymous, so no DepMap rows are provided\n"
        "  for them; a model may only use dependency information it can infer from basal expression.\n")
    print(f"[priors] GO rows {len(go)}, DepMap lines {picked}")
