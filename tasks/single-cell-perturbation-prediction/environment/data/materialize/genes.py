"""Master gene panel: one row per Ensembl gene, unique symbol, fixed order.

Order: the VCC 2025 H1 panel first, in its native order (so the H1 test context's `var`
equals the official VCC gene list), then every other gene sorted by symbol. Symbols come
from the VCC panel, then the 10x 2024-A annotation shipped with X-Atlas/Orion, then the
source's own annotation; duplicated symbols are disambiguated with the Ensembl ID.
Every join between sources goes through Ensembl IDs, never through symbols.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_VERSION = re.compile(r"\.\d+$")


def strip_version(ensembl: pd.Series | np.ndarray | list) -> pd.Series:
    return pd.Series(ensembl).astype(str).str.replace(_VERSION, "", regex=True)


def build_master_panel(vcc_var: pd.DataFrame, orion_genes: pd.DataFrame,
                       replogle_vars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """vcc_var: index=symbol, column gene_id. orion_genes: ensembl_id, gene_name.
    replogle_vars: name -> DataFrame(gene_id, gene_name)."""
    rows: dict[str, dict] = {}
    order: list[str] = []
    vcc_ens = strip_version(vcc_var["gene_id"].to_numpy())
    for symbol, ens in zip(vcc_var.index.astype(str), vcc_ens):
        rows[ens] = {"ensembl_id": ens, "symbol": symbol, "symbol_source": "vcc2025",
                     "in_vcc_panel": True}
        order.append(ens)
    orion_map = dict(zip(strip_version(orion_genes["ensembl_id"].to_numpy()),
                         orion_genes["gene_name"].astype(str)))
    extra: dict[str, dict] = {}
    for name, var in replogle_vars.items():
        ens_all = strip_version(var["gene_id"].to_numpy())
        for ens, sym in zip(ens_all, var["gene_name"].astype(str)):
            if ens in rows:
                rows[ens].setdefault("in_replogle_panels", set()).add(name)
                continue
            entry = extra.setdefault(ens, {"ensembl_id": ens, "in_vcc_panel": False,
                                           "in_replogle_panels": set()})
            entry["in_replogle_panels"].add(name)
            if ens in orion_map and orion_map[ens] != ens:
                entry["symbol"], entry["symbol_source"] = orion_map[ens], "orion_2024A"
            else:
                entry.setdefault("symbol", sym)
                entry.setdefault("symbol_source", f"replogle:{name}")
    for ens in sorted(extra, key=lambda e: (extra[e]["symbol"], e)):
        rows[ens] = extra[ens]
        order.append(ens)
    for ens in order:
        rows[ens]["in_orion"] = ens in orion_map
        rows[ens]["in_replogle_panels"] = ",".join(sorted(rows[ens].get("in_replogle_panels", set())))
    panel = pd.DataFrame([rows[e] for e in order])
    # unique symbols: keep the VCC / first occurrence, suffix the rest with the Ensembl ID
    seen: set[str] = set()
    symbols = []
    for sym, ens in zip(panel["symbol"], panel["ensembl_id"]):
        if sym in seen or sym == "" or sym == "nan":
            sym = f"{sym}_{ens}" if sym not in ("", "nan") else ens
        seen.add(sym)
        symbols.append(sym)
    panel["symbol"] = symbols
    panel["master_index"] = np.arange(len(panel))
    panel.index = pd.Index(panel["symbol"], name="symbol")
    assert panel.index.is_unique and panel["ensembl_id"].is_unique
    return panel


def build_gene_universe(master: pd.DataFrame, orion_genes: pd.DataFrame,
                        replogle_vars: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Ensembl -> canonical symbol for every gene any source knows about, including genes
    outside the expression panel (CRISPRi targets are often not expressed). Precedence:
    master panel symbol, then 10x 2024-A (Orion), then the source annotation."""
    universe = dict(zip(master["ensembl_id"], master.index.astype(str)))
    for ens, sym in zip(strip_version(orion_genes["ensembl_id"].to_numpy()), orion_genes["gene_name"].astype(str)):
        universe.setdefault(ens, sym)
    for var in replogle_vars.values():
        for ens, sym in zip(strip_version(var["gene_id"].to_numpy()), var["gene_name"].astype(str)):
            universe.setdefault(ens, sym)
    return universe


def panel_view(master: pd.DataFrame, ensembl_ids: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Map a source's genes (Ensembl) to master positions. Returns (source_positions,
    master_positions) for genes present in the master panel, ordered by master position."""
    ens = strip_version(ensembl_ids).to_numpy()
    lookup = dict(zip(master["ensembl_id"], master["master_index"]))
    src_pos = np.array([i for i, e in enumerate(ens) if e in lookup], dtype=np.int64)
    master_pos = np.array([lookup[ens[i]] for i in src_pos], dtype=np.int64)
    order = np.argsort(master_pos, kind="stable")
    return src_pos[order], master_pos[order]
