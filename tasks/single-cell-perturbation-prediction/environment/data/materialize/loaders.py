"""Source loaders. Each returns the kept cells as a Context plus pseudobulk sums over ALL
cells of the source (so the per-target tables are never subsampled)."""

from __future__ import annotations

import dataclasses
import glob
import multiprocessing as mp
import os
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

from .context import CONTROL, Context, group_sums, log1p_cp10k
from .genes import panel_view, strip_version


@dataclasses.dataclass
class Pseudobulk:
    targets: np.ndarray          # master symbols (CONTROL included)
    target_ensembl: np.ndarray
    n_cells: np.ndarray
    sum_log1p: np.ndarray        # T x G_ctx, float64
    sum_counts: np.ndarray       # T x G_ctx, float64

    def merge(self, other: "Pseudobulk") -> "Pseudobulk":
        idx = {t: i for i, t in enumerate(self.targets)}
        targets = list(self.targets)
        ens = list(self.target_ensembl)
        n = list(self.n_cells)
        s1, s2 = [self.sum_log1p], [self.sum_counts]
        extra_rows_l, extra_rows_c = [], []
        for j, t in enumerate(other.targets):
            if t in idx:
                i = idx[t]
                n[i] += other.n_cells[j]
                self.sum_log1p[i] += other.sum_log1p[j]
                self.sum_counts[i] += other.sum_counts[j]
            else:
                idx[t] = len(targets)
                targets.append(t); ens.append(other.target_ensembl[j]); n.append(other.n_cells[j])
                extra_rows_l.append(other.sum_log1p[j]); extra_rows_c.append(other.sum_counts[j])
        if extra_rows_l:
            s1.append(np.vstack(extra_rows_l)); s2.append(np.vstack(extra_rows_c))
        return Pseudobulk(np.asarray(targets, dtype=object), np.asarray(ens, dtype=object),
                          np.asarray(n, dtype=np.int64), np.vstack(s1), np.vstack(s2))


@dataclasses.dataclass
class Loaded:
    context: Context
    pseudobulk: Pseudobulk
    stats: dict


def _choose_rows(target: pd.Series, keep_per_target: int | None, keep_controls: int | None,
                 seed: int, weight: pd.Series | None = None) -> np.ndarray:
    """Seeded per-target cap. Returns sorted row positions."""
    rng = np.random.default_rng(seed)
    keep = []
    for t, idx in target.groupby(target.to_numpy(), sort=True).indices.items():
        cap = keep_controls if t == CONTROL else keep_per_target
        if cap is None or len(idx) <= cap:
            keep.append(idx)
        else:
            keep.append(rng.choice(idx, size=cap, replace=False))
    return np.sort(np.concatenate(keep))


def _resolve_targets(symbols: pd.Series, ensembl: pd.Series, universe: dict[str, str]) -> tuple[pd.Series, pd.Series]:
    """Map source target symbols to canonical symbols through Ensembl IDs when possible."""
    ens = strip_version(ensembl.fillna("nan").astype(str))
    ens = ens.where(~ens.isin(["nan", "None", "", CONTROL]), other="")
    out = []
    for sym, e in zip(symbols.astype(str), ens):
        if sym == CONTROL:
            out.append(CONTROL)
        elif e and e in universe:
            out.append(universe[e])
        else:
            out.append(sym)
    return pd.Series(out, index=symbols.index), pd.Series(ens.to_numpy(), index=symbols.index)


def _read_cat(group: h5py.Group, key: str) -> np.ndarray:
    """Old-style anndata categorical (obs/<key> codes + obs/__categories/<key>)."""
    codes = group[key][:]
    cats = group["__categories"][key][:]
    cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cats], dtype=object)
    out = np.empty(len(codes), dtype=object)
    valid = codes >= 0
    out[valid] = cats[codes[valid]]
    out[~valid] = "nan"
    return out


# --------------------------------------------------------------------------- Replogle-format
def load_replogle_format(path: Path, name: str, master: pd.DataFrame, universe: dict[str, str],
                         chemistry: str, source: str, keep_per_target: int | None,
                         keep_controls: int | None, seed: int, chunk_rows: int = 40000,
                         max_mito: float | None = None) -> Loaded:
    """Replogle 2022 / Nadig 2025 processed h5ad: dense float32 raw counts, old anndata layout."""
    with h5py.File(path, "r") as f:
        obs = f["obs"]
        target_src = pd.Series(_read_cat(obs, "gene"))
        target_ens = pd.Series(_read_cat(obs, "gene_id"))
        guide = pd.Series(_read_cat(obs, "sgID_AB"))
        batch = pd.Series(obs["gem_group"][:]).astype(str)
        umi = pd.Series(obs["UMI_count"][:].astype(np.float32))
        mito = pd.Series(obs["mitopercent"][:].astype(np.float32))
        barcode = pd.Series([b.decode() if isinstance(b, bytes) else str(b) for b in obs["cell_barcode"][:]])
        var_ens = pd.Series(f["var"]["gene_id"][:]).map(lambda b: b.decode() if isinstance(b, bytes) else str(b))
        n_cells, n_genes = f["X"].shape
    target, target_ens = _resolve_targets(target_src, target_ens, universe)
    src_pos, master_pos = panel_view(master, var_ens)
    var = master.iloc[master_pos][["ensembl_id", "master_index"]].copy()

    qc_ok = np.ones(n_cells, dtype=bool)
    if max_mito is not None:
        qc_ok &= mito.to_numpy() <= max_mito
    stats = {"source_cells": int(n_cells), "source_genes": int(n_genes),
             "genes_in_master": int(len(src_pos)), "qc_dropped": int((~qc_ok).sum())}
    eligible = pd.Series(np.where(qc_ok, target.to_numpy(), None))
    keep_rows = _choose_rows(eligible[qc_ok], keep_per_target, keep_controls, seed)
    keep_rows = np.flatnonzero(qc_ok)[keep_rows]
    keep_mask = np.zeros(n_cells, dtype=bool); keep_mask[keep_rows] = True

    codes, names = pd.factorize(target.to_numpy(), sort=True)
    G = len(src_pos)
    sum_log1p = np.zeros((len(names), G), dtype=np.float64)
    sum_counts = np.zeros((len(names), G), dtype=np.float64)
    n_by_target = np.zeros(len(names), dtype=np.int64)
    kept_blocks = []
    with h5py.File(path, "r") as f:
        X = f["X"]
        for start in range(0, n_cells, chunk_rows):
            stop = min(start + chunk_rows, n_cells)
            block = np.asarray(X[start:stop])[:, src_pos]
            ok = qc_ok[start:stop]
            block_csr = sparse.csr_matrix(block[ok])
            c = codes[start:stop][ok]
            ind = sparse.csr_matrix((np.ones(len(c), dtype=np.float32), (c, np.arange(len(c)))),
                                    shape=(len(names), len(c)))
            sum_counts += np.asarray((ind @ block_csr).todense())
            sum_log1p += np.asarray((ind @ log1p_cp10k(block_csr)).todense())
            n_by_target += np.bincount(c, minlength=len(names))
            km = keep_mask[start:stop]
            if km.any():
                kept_blocks.append(sparse.csr_matrix(block[km]))
            del block
    X_keep = sparse.vstack(kept_blocks).tocsr() if kept_blocks else sparse.csr_matrix((0, G), dtype=np.float32)
    ens_by_target = dict(zip(target.to_numpy(), target_ens.to_numpy()))
    pb = Pseudobulk(targets=np.asarray(names, dtype=object),
                    target_ensembl=np.asarray([ens_by_target.get(t, "") for t in names], dtype=object),
                    n_cells=n_by_target, sum_log1p=sum_log1p, sum_counts=sum_counts)
    obs_df = pd.DataFrame({
        "target_gene": target.to_numpy()[keep_rows], "target_ensembl": target_ens.to_numpy()[keep_rows],
        "guide_id": guide.to_numpy()[keep_rows], "batch": batch.to_numpy()[keep_rows],
        "total_counts": umi.to_numpy()[keep_rows], "pct_counts_mt": 100.0 * mito.to_numpy()[keep_rows],
        "source_target_symbol": target_src.to_numpy()[keep_rows],
    }, index=pd.Index([f"{name}:{b}" for b in barcode.to_numpy()[keep_rows]], name="cell"))
    obs_df["is_control"] = obs_df["target_gene"] == CONTROL
    stats.update({"kept_cells": int(len(keep_rows)), "targets": int(len(names) - 1),
                  "control_cells_source": int(n_by_target[list(names).index(CONTROL)])})
    return Loaded(Context(name, X_keep, obs_df, var, chemistry, source), pb, stats)


# --------------------------------------------------------------------------- VCC 2025 (H1)
def load_vcc2025(paths: list[Path], name: str, master: pd.DataFrame, universe: dict[str, str],
                 keep_per_target: int | None, keep_controls: int | None, seed: int) -> Loaded:
    parts = []
    for p in paths:
        a = ad.read_h5ad(p)
        a.obs["source_object"] = p.name
        parts.append(a)
    var0 = parts[0].var.copy()
    for a in parts[1:]:
        assert list(a.var_names) == list(var0.index), "VCC objects must share the gene panel"
    adata = ad.concat(parts, join="inner", index_unique=None, merge="first")
    adata.var = var0
    # non-targeting cells are shared between the official objects; keep one copy
    adata = adata[~adata.obs_names.duplicated(keep="first")].copy()
    src_pos, master_pos = panel_view(master, adata.var["gene_id"])
    assert len(src_pos) == adata.n_vars, "the H1 panel defines the head of the master panel"
    X = sparse.csr_matrix(adata.X)[:, src_pos].tocsr()
    var = master.iloc[master_pos][["ensembl_id", "master_index"]].copy()
    target_src = adata.obs["target_gene"].astype(str).reset_index(drop=True)
    ens_lookup = dict(zip(adata.var_names.astype(str), strip_version(adata.var["gene_id"])))
    target_ens = target_src.map(lambda s: ens_lookup.get(s, "nan"))
    target, target_ens = _resolve_targets(target_src, target_ens, universe)
    names, sum_counts, n_by_target = group_sums(X, target)
    _, sum_log1p, _ = group_sums(log1p_cp10k(X), target)
    ens_by_target = dict(zip(target.to_numpy(), target_ens.to_numpy()))
    pb = Pseudobulk(np.asarray(names, dtype=object),
                    np.asarray([ens_by_target.get(t, "") for t in names], dtype=object),
                    n_by_target, sum_log1p, sum_counts)
    keep_rows = _choose_rows(target, keep_per_target, keep_controls, seed)
    obs_df = pd.DataFrame({
        "target_gene": target.to_numpy()[keep_rows], "target_ensembl": target_ens.to_numpy()[keep_rows],
        "guide_id": adata.obs["guide_id"].astype(str).to_numpy()[keep_rows],
        "batch": adata.obs["batch"].astype(str).to_numpy()[keep_rows],
        "total_counts": np.asarray(X[keep_rows].sum(axis=1)).ravel().astype(np.float32),
        "pct_counts_mt": np.full(len(keep_rows), np.nan, dtype=np.float32),
        "source_target_symbol": target_src.to_numpy()[keep_rows],
        "source_object": adata.obs["source_object"].astype(str).to_numpy()[keep_rows],
    }, index=pd.Index([f"{name}:{b}" for b in adata.obs_names.to_numpy()[keep_rows]], name="cell"))
    obs_df["is_control"] = obs_df["target_gene"] == CONTROL
    stats = {"source_cells": int(adata.n_obs), "source_genes": int(adata.n_vars),
             "genes_in_master": int(len(src_pos)), "kept_cells": int(len(keep_rows)),
             "targets": int(len(names) - 1), "control_cells_source": int(n_by_target[list(names).index(CONTROL)]),
             "objects": [p.name for p in paths]}
    return Loaded(Context(name, X[keep_rows].tocsr(), obs_df, var, "10x_flex", "vcc2025"), pb, stats)


# --------------------------------------------------------------------------- X-Atlas/Orion
_ORION_META_COLS = ["gene_target", "sample", "total_counts", "pct_counts_mt", "pass_guide_filter", "num_features"]


def _orion_shards(root: Path, line: str) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(root / "data" / f"{line}_Batch*.parquet")))


def _orion_meta(shard: Path) -> pd.DataFrame:
    t = pq.read_table(shard, columns=_ORION_META_COLS).to_pandas()
    t["shard"] = shard.name
    t["row"] = np.arange(len(t), dtype=np.int64)
    return t


_ORION_STATE: dict = {}


def _orion_init(token_to_col: np.ndarray, n_cols: int, keep: dict, target_codes: dict, n_targets: int):
    _ORION_STATE.update(token_to_col=token_to_col, n_cols=n_cols, keep=keep, target_codes=target_codes,
                        n_targets=n_targets)


def _orion_shard_pass(shard: Path):
    """One shard: pseudobulk partial sums (rows = targets present in the shard, float32) and the
    kept cells. Returned compactly so the parent can accumulate shard by shard."""
    st = _ORION_STATE
    t = pq.read_table(shard, columns=["gene_token_id", "gene_expression", "cell_barcode", "sample",
                                      "guide_target", "gene_target", "total_counts", "pct_counts_mt"])
    tok = t.column("gene_token_id").combine_chunks()
    val = t.column("gene_expression").combine_chunks()
    offsets = np.asarray(tok.offsets, dtype=np.int64)
    tokens = np.asarray(tok.values, dtype=np.int64)
    values = np.asarray(val.values, dtype=np.float32)
    cols = st["token_to_col"][tokens]
    good = cols >= 0
    row_ids = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    X = sparse.csr_matrix((values[good], (row_ids[good], cols[good])), shape=(len(offsets) - 1, st["n_cols"]))
    del tokens, values, cols, good, row_ids
    targets = t.column("gene_target").to_pandas().astype(str).to_numpy()
    keep_rows = np.asarray(st["keep"].get(shard.name, []), dtype=np.int64)
    eligible = st["keep"]["eligible"].get(shard.name)
    elig_mask = np.zeros(X.shape[0], dtype=bool)
    if eligible is not None:
        elig_mask[eligible] = True
    codes = np.array([st["target_codes"].get(x, -1) for x in targets])
    use = elig_mask & (codes >= 0)
    c = codes[use]
    present, local = np.unique(c, return_inverse=True)
    ind = sparse.csr_matrix((np.ones(len(c), dtype=np.float32), (local, np.flatnonzero(use))),
                            shape=(len(present), X.shape[0]))
    sum_counts = np.asarray((ind @ X).todense(), dtype=np.float32)
    sum_log1p = np.asarray((ind @ log1p_cp10k(X)).todense(), dtype=np.float32)
    n = np.bincount(local, minlength=len(present))
    kept = X[keep_rows] if len(keep_rows) else None
    obs = None
    if len(keep_rows):
        obs = pd.DataFrame({
            "cell_barcode": t.column("cell_barcode").to_pandas().to_numpy()[keep_rows],
            "sample": t.column("sample").to_pandas().to_numpy()[keep_rows],
            "guide_target": t.column("guide_target").to_pandas().to_numpy()[keep_rows],
            "gene_target": targets[keep_rows],
            "total_counts": t.column("total_counts").to_pandas().to_numpy()[keep_rows].astype(np.float32),
            "pct_counts_mt": t.column("pct_counts_mt").to_pandas().to_numpy()[keep_rows].astype(np.float32),
        })
    return shard.name, present, sum_counts, sum_log1p, n, kept, obs


def load_orion(root: Path, line: str, name: str, master: pd.DataFrame, universe: dict[str, str],
               keep_per_target: int | None, keep_controls: int | None, seed: int, workers: int = 32, max_mito: float = 20.0,
               min_counts: float = 2000.0, max_shards: int | None = None) -> Loaded:
    shards = _orion_shards(root, line)
    if max_shards:
        shards = shards[:max_shards]
    gm = pd.read_parquet(root / "metadata" / "gene_metadata.parquet")
    gm["ensembl_id"] = strip_version(gm["ensembl_id"])
    src_pos, master_pos = panel_view(master, gm["ensembl_id"])
    token_ids = gm["gene_token_id"].to_numpy()[src_pos]
    token_to_col = np.full(int(gm["gene_token_id"].max()) + 1, -1, dtype=np.int64)
    token_to_col[token_ids] = np.arange(len(token_ids))
    var = master.iloc[master_pos][["ensembl_id", "master_index"]].copy()
    sym_to_ens = gm.drop_duplicates("gene_name").set_index("gene_name")["ensembl_id"].to_dict()

    with mp.Pool(workers) as pool:
        meta = pd.concat(pool.map(_orion_meta, shards), ignore_index=True)
    meta["source_symbol"] = meta["gene_target"].astype(str)
    target_src = meta["source_symbol"].where(
        ~meta["source_symbol"].str.lower().isin(["non-targeting", "nontargeting", "ntc"]), other=CONTROL)
    target_ens = target_src.map(lambda s: CONTROL if s == CONTROL else sym_to_ens.get(s, "nan"))
    target, target_ens = _resolve_targets(target_src, target_ens, universe)
    meta["target_gene"], meta["target_ensembl"] = target.to_numpy(), target_ens.to_numpy()
    qc_ok = (meta["pass_guide_filter"] == 1) & (meta["pct_counts_mt"] <= max_mito) & (meta["total_counts"] >= min_counts)
    stats = {"source_cells": int(len(meta)), "source_genes": int(len(gm)), "genes_in_master": int(len(src_pos)),
             "qc_dropped": int((~qc_ok).sum()), "shards": len(shards)}
    elig = meta[qc_ok]
    keep_idx = _choose_rows(elig["target_gene"].reset_index(drop=True), keep_per_target, keep_controls, seed)
    keep_global = elig.index.to_numpy()[keep_idx]
    keep = {"eligible": {s: g["row"].to_numpy() for s, g in elig.groupby("shard")}}
    for s, g in meta.loc[keep_global].groupby("shard"):
        keep[s] = g["row"].to_numpy()
    codes, names = pd.factorize(elig["target_gene"].to_numpy(), sort=True)
    target_codes = {n: i for i, n in enumerate(names)}
    # workers see the raw parquet symbol; map it straight to the resolved target's code
    src_codes = dict(zip(elig["source_symbol"].to_numpy(), codes))
    sum_counts = np.zeros((len(names), len(token_ids)), dtype=np.float64)
    sum_log1p = np.zeros_like(sum_counts)
    n_by_target = np.zeros(len(names), dtype=np.int64)
    blocks, obs_parts, shard_order = [], [], []
    done = 0
    with mp.Pool(workers, initializer=_orion_init,
                 initargs=(token_to_col, len(token_ids), keep, src_codes, len(names))) as pool:
        for shard_name, present, sc, sl, n, kept, obs in pool.imap_unordered(_orion_shard_pass, shards, chunksize=1):
            sum_counts[present] += sc; sum_log1p[present] += sl; n_by_target[present] += n
            if kept is not None:
                blocks.append(kept); obs_parts.append(obs); shard_order.append(shard_name)
            done += 1
            if done % 25 == 0:
                print(f"[orion:{line}] {done}/{len(shards)} shards", flush=True)
    X_keep = sparse.vstack(blocks).tocsr()
    obs = pd.concat(obs_parts, ignore_index=True)
    t_src = obs["gene_target"].astype(str)
    t_src = t_src.where(~t_src.str.lower().isin(["non-targeting", "nontargeting", "ntc"]), other=CONTROL)
    t_ens = t_src.map(lambda s: CONTROL if s == CONTROL else sym_to_ens.get(s, "nan"))
    t_res, t_ens = _resolve_targets(t_src, t_ens, universe)
    obs_df = pd.DataFrame({
        "target_gene": t_res.to_numpy(), "target_ensembl": t_ens.to_numpy(),
        "guide_id": obs["guide_target"].astype(str).to_numpy(), "batch": obs["sample"].astype(str).to_numpy(),
        "total_counts": obs["total_counts"].to_numpy(), "pct_counts_mt": obs["pct_counts_mt"].to_numpy(),
        "source_target_symbol": obs["gene_target"].astype(str).to_numpy(),
    }, index=pd.Index([f"{name}:{b}" for b in obs["cell_barcode"].astype(str)], name="cell"))
    obs_df["is_control"] = obs_df["target_gene"] == CONTROL
    ens_by_target = dict(zip(elig["target_gene"], elig["target_ensembl"]))
    pb = Pseudobulk(np.asarray(names, dtype=object),
                    np.asarray([ens_by_target.get(t, "") for t in names], dtype=object),
                    n_by_target, sum_log1p, sum_counts)
    stats.update({"kept_cells": int(X_keep.shape[0]), "targets": int(len(names) - 1),
                  "control_cells_source": int(n_by_target[target_codes[CONTROL]])})
    return Loaded(Context(name, X_keep, obs_df, var, "10x_gemx_5p", "xatlas_orion"), pb, stats)
