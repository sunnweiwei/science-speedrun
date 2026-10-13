"""Release builder: harmonised intermediates -> DATA_ROOT layout (see ../README.md)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from . import config as cfg
from .context import CONTROL, Context, log1p_cp10k, write_json
from .loaders import Pseudobulk

WORK_PANEL = "master_panel.csv"


# --------------------------------------------------------------------------- intermediates
def write_intermediate(work: Path, name: str, context: Context, pb: Pseudobulk, stats: dict) -> None:
    out = work / name
    out.mkdir(parents=True, exist_ok=True)
    adata = context.to_anndata()
    adata.write_h5ad(out / "cells.h5ad")
    np.savez(out / "pseudobulk.npz", targets=pb.targets.astype(str), target_ensembl=pb.target_ensembl.astype(str),
             n_cells=pb.n_cells, sum_log1p=pb.sum_log1p.astype(np.float32), sum_counts=pb.sum_counts.astype(np.float32))
    context.var.to_csv(out / "var.csv")
    write_json(out / "stats.json", {**stats, "chemistry": context.chemistry, "source": context.source})


def read_pseudobulk(work: Path, name: str) -> Pseudobulk:
    z = np.load(work / name / "pseudobulk.npz", allow_pickle=False)
    return Pseudobulk(z["targets"].astype(object), z["target_ensembl"].astype(object), z["n_cells"],
                      z["sum_log1p"].astype(np.float64), z["sum_counts"].astype(np.float64))


def read_cells(work: Path, name: str) -> ad.AnnData:
    return ad.read_h5ad(work / name / "cells.h5ad")


# --------------------------------------------------------------------------- pseudobulk tables
class Table:
    """Per-context pseudobulk means with control statistics and effect sizes."""

    def __init__(self, context: str, pb: Pseudobulk, var: pd.DataFrame, control_cells: sparse.csr_matrix):
        self.context = context
        self.var = var
        self.targets = np.asarray(pb.targets, dtype=object)
        self.target_ensembl = np.asarray(pb.target_ensembl, dtype=object)
        self.n_cells = pb.n_cells.astype(np.int64)
        n = np.maximum(self.n_cells, 1)[:, None]
        self.mean_log1p = (pb.sum_log1p / n).astype(np.float32)
        self.mean_counts = (pb.sum_counts / n).astype(np.float32)
        ci = self.index(CONTROL)
        self.control_mean_log1p = self.mean_log1p[ci].copy()
        self.control_mean_counts = self.mean_counts[ci].copy()
        self.n_control = int(self.n_cells[ci])
        ctrl = log1p_cp10k(control_cells)
        mean = np.asarray(ctrl.mean(axis=0)).ravel()
        sq = np.asarray(ctrl.multiply(ctrl).mean(axis=0)).ravel()
        self.control_sd_log1p = np.sqrt(np.maximum(sq - mean ** 2, 0)).astype(np.float32)
        self.delta = self.mean_log1p - self.control_mean_log1p[None, :]
        # z-score of the pseudobulk delta against control variability, accounting for depth
        se = self.control_sd_log1p[None, :] * np.sqrt(1.0 / n + 1.0 / max(self.n_control, 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            self.z = np.where(se > 0, self.delta / se, 0.0).astype(np.float32)
        self.n_sig = (np.abs(self.z) > cfg.STRATA["z_threshold"]).sum(axis=1).astype(np.int64)
        self.effect_l2 = np.linalg.norm(self.delta, axis=1).astype(np.float32)
        self.knockdown = self._knockdown()

    def index(self, target: str) -> int:
        hit = np.flatnonzero(self.targets == target)
        if len(hit) != 1:
            raise KeyError(target)
        return int(hit[0])

    def has(self, target: str, min_cells: int = 1) -> bool:
        hit = np.flatnonzero(self.targets == target)
        return len(hit) == 1 and self.n_cells[hit[0]] >= min_cells

    def _knockdown(self) -> np.ndarray:
        pos = {e: i for i, e in enumerate(self.var["ensembl_id"])}
        out = np.full(len(self.targets), np.nan, dtype=np.float32)
        for i, (t, e) in enumerate(zip(self.targets, self.target_ensembl)):
            j = pos.get(e)
            if t == CONTROL or j is None or self.control_mean_counts[j] <= 0:
                continue
            out[i] = 1.0 - self.mean_counts[i, j] / self.control_mean_counts[j]
        return out

    def to_anndata(self, drop: set[str] | None = None) -> ad.AnnData:
        keep = np.array([t not in (drop or set()) for t in self.targets])
        obs = pd.DataFrame({"target_ensembl": self.target_ensembl[keep], "n_cells": self.n_cells[keep],
                            "knockdown_frac": self.knockdown[keep], "n_sig_genes_z": self.n_sig[keep],
                            "effect_l2_delta": self.effect_l2[keep], "is_control": self.targets[keep] == CONTROL},
                           index=pd.Index(self.targets[keep].astype(str), name="target_gene"))
        var = self.var.copy()
        var["control_mean_log1p_cp10k"] = self.control_mean_log1p
        var["control_sd_log1p_cp10k"] = self.control_sd_log1p
        var["control_mean_counts"] = self.control_mean_counts
        a = ad.AnnData(X=self.mean_log1p[keep].copy(), obs=obs, var=var)
        a.layers["mean_counts"] = self.mean_counts[keep].copy()
        a.layers["delta_log1p_cp10k"] = self.delta[keep].astype(np.float32)
        a.layers["z_vs_control"] = self.z[keep].copy()
        a.uns.update({"context": self.context, "n_control_cells": self.n_control,
                      "z_threshold": cfg.STRATA["z_threshold"],
                      "note": "X = mean over ALL source cells of log1p(CP10k); n_sig_genes_z counts |z|>threshold"})
        return a

    def signature(self, target: str, top: int = 500) -> dict:
        i = self.index(target)
        z = self.z[i]
        sig = np.flatnonzero(np.abs(z) > cfg.STRATA["z_threshold"])
        sig = sig[np.argsort(-np.abs(z[sig]))][:top]
        genes = self.var.index.to_numpy().astype(str)
        return {"up": genes[sig[z[sig] > 0]].tolist(), "down": genes[sig[z[sig] < 0]].tolist(),
                "n_cells": int(self.n_cells[i]), "n_sig": int(self.n_sig[i])}


def strata_thresholds(train_tables: dict[str, Table]) -> dict[str, tuple[float, float]]:
    s = cfg.STRATA
    out = {}
    for c, t in train_tables.items():
        measured = (t.n_cells >= s["min_cells_measured"]) & (t.targets != CONTROL)
        vals = t.n_sig[measured]
        out[c] = (float(np.quantile(vals, s["weak_quantile"])), float(np.quantile(vals, s["strong_quantile"])))
    return out


def stratum(target: str, train_tables: dict[str, Table], thresholds: dict[str, tuple[float, float]]) -> tuple[str, dict]:
    s = cfg.STRATA
    per = {c: int(t.n_sig[t.index(target)]) for c, t in train_tables.items() if t.has(target, s["min_cells_measured"])}
    if len(per) < 2:
        return "unmeasured", per
    strong = [v >= thresholds[c][1] for c, v in per.items()]
    weak = [v <= thresholds[c][0] for c, v in per.items()]
    # a conserved call tolerates one dissenting context once three or more are measured
    need = len(per) - (1 if len(per) >= 3 else 0)
    if sum(strong) >= need:
        return "conserved-strong", per
    if sum(weak) >= need:
        return "weak", per
    return "context-variable", per


def stratified_sample(rng: np.random.Generator, candidates: np.ndarray, score: np.ndarray, n: int) -> np.ndarray:
    """Sample n items with equal shares from the tertiles of `score`."""
    if len(candidates) <= n:
        return candidates
    order = np.argsort(score, kind="stable")
    bins = np.array_split(order, 3)
    out, want = [], [n // 3, n // 3, n - 2 * (n // 3)]
    for b, w in zip(bins, want):
        out.append(rng.choice(candidates[b], size=min(w, len(b)), replace=False))
    picked = np.concatenate(out)
    if len(picked) < n:
        rest = np.setdiff1d(candidates, picked)
        picked = np.concatenate([picked, rng.choice(rest, size=n - len(picked), replace=False)])
    return np.sort(picked)


# --------------------------------------------------------------------------- request views
def requests_rows(context_id: str, targets: list[str], n_cells: int, seed: int) -> list[dict]:
    return [{"request_id": f"{context_id}:{i:04d}", "context": context_id,
             "perturbation": {"type": "crispri", "target": t}, "n_cells": n_cells, "seed": seed + i}
            for i, t in enumerate(targets)]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def subset(adata: ad.AnnData, mask: np.ndarray) -> ad.AnnData:
    out = adata[mask].copy()
    out.obs_names = out.obs_names.astype(str)
    return out


def public_view(adata: ad.AnnData, context_id: str, chemistry: str, targets: list[str], n_cells: int,
                max_basal: int, seed: int, out: Path, anonymous: bool) -> tuple[ad.AnnData, list[dict]]:
    rng = np.random.default_rng(seed)
    ctrl = np.flatnonzero(adata.obs["is_control"].to_numpy())
    if len(ctrl) > max_basal:
        ctrl = np.sort(rng.choice(ctrl, size=max_basal, replace=False))
    basal = subset(adata, ctrl)
    keep_cols = ["target_gene", "batch", "total_counts", "is_control"]
    basal.obs = basal.obs[keep_cols].copy()
    if anonymous:
        # batch labels could name the cell line / chemistry; recode
        codes = {b: f"b{i:02d}" for i, b in enumerate(sorted(basal.obs["batch"].unique()))}
        basal.obs["batch"] = basal.obs["batch"].map(codes).astype(str)
        basal.obs_names = [f"{context_id}:c{i:06d}" for i in range(basal.n_obs)]
        basal.uns = {"context": context_id, "chemistry": chemistry}
    basal.var = basal.var[["ensembl_id"]].copy()
    out.mkdir(parents=True, exist_ok=True)
    basal.write_h5ad(out / "basal.h5ad")
    rows = requests_rows(context_id, targets, n_cells, seed)
    write_jsonl(out / "requests.jsonl", rows)
    write_json(out / "context.json", {"context_id": context_id, "chemistry": chemistry, "n_genes": basal.n_vars,
                                      "n_basal_cells": basal.n_obs, "n_requests": len(rows),
                                      "perturbation_type": "crispri"})
    return basal, rows


def responses_view(adata: ad.AnnData, context_id: str, targets: list[str], cap: int | None, seed: int,
                   anonymous: bool) -> ad.AnnData:
    rng = np.random.default_rng(seed + 1)
    tg = adata.obs["target_gene"].astype(str).to_numpy()
    rows = []
    for t in targets:
        idx = np.flatnonzero(tg == t)
        if cap is not None and len(idx) > cap:
            idx = np.sort(rng.choice(idx, size=cap, replace=False))
        rows.append(idx)
    resp = subset(adata, np.concatenate(rows))
    resp.obs = resp.obs[["target_gene", "target_ensembl", "batch", "total_counts", "is_control"]].copy()
    if anonymous:
        codes = {b: f"b{i:02d}" for i, b in enumerate(sorted(resp.obs["batch"].unique()))}
        resp.obs["batch"] = resp.obs["batch"].map(codes).astype(str)
        resp.obs_names = [f"{context_id}:r{i:06d}" for i in range(resp.n_obs)]
        resp.uns = {"context": context_id}
    resp.var = resp.var[["ensembl_id"]].copy()
    return resp


# --------------------------------------------------------------------------- main build
def build(work: Path, root: Path, source_manifest: dict) -> None:
    rng = np.random.default_rng(cfg.SEED)
    master = pd.read_csv(work / WORK_PANEL, index_col=0)
    by_context: dict[str, list[dict]] = {}
    for s in cfg.SOURCES:
        by_context.setdefault(s["context"], []).append(s)
    roles = {c: ss[0]["role"] for c, ss in by_context.items()}
    stats = {s["name"]: json.loads((work / s["name"] / "stats.json").read_text()) for s in cfg.SOURCES}

    # ---- phase A: pseudobulk tables for every context (all cells; W not yet removed)
    tables: dict[str, Table] = {}
    cells_cache: dict[str, ad.AnnData] = {}
    for context, ss in by_context.items():
        parts = [read_cells(work, s["name"]) for s in ss]
        for s, a in zip(ss, parts):
            a.obs["source_screen"] = s["name"]
            a.obs["batch"] = s["name"] + ":" + a.obs["batch"].astype(str)
        # two screens of one context (K562 GW + essential) are joined on the genes both
        # screens measured; a zero-filled outer join would fake absent measurements
        adata = parts[0] if len(parts) == 1 else ad.concat(parts, join="inner", merge="first", index_unique=None)
        adata.var = master.loc[adata.var_names, ["ensembl_id", "master_index"]].copy()
        adata.var.index.name = "symbol"
        adata = _reorder_master(adata, master)
        pb = _align_pb(read_pseudobulk(work, ss[0]["name"]), work, ss[0]["name"], adata.var)
        for s in ss[1:]:
            pb = pb.merge(_align_pb(read_pseudobulk(work, s["name"]), work, s["name"], adata.var))
        ctrl = sparse.csr_matrix(adata.X[adata.obs["is_control"].to_numpy()])
        tables[context] = Table(context, pb, adata.var, ctrl)
        cells_cache[context] = adata
        print(f"[build] {context}: {adata.n_obs} kept cells, {adata.n_vars} genes, {len(pb.targets) - 1} targets")

    train_ctx = [c for c, r in roles.items() if r == "train"]
    test_ctx = [c for c, r in roles.items() if r == "test"]
    train_tables = {c: tables[c] for c in train_ctx}
    thresholds = strata_thresholds(train_tables)
    print(f"[build] strata thresholds (weak<=, strong>=) per training context: {thresholds}")

    # ---- withheld targets W: from the H1 list, measurable in the near/mid test contexts
    far = [c for c in test_ctx if by_context[c][0]["tier"] == "far"][0]
    h1 = tables[far]
    h1_targets = np.array([t for t in h1.targets if t != CONTROL], dtype=object)
    other_tests = [c for c in test_ctx if c != far]
    eligible = np.array([t for t in h1_targets
                         if all(tables[c].has(t, cfg.TEST["min_cells_withheld"]) for c in other_tests)
                         and sum(tables[c].has(t, cfg.STRATA["min_cells_measured"]) for c in train_ctx) >= 2],
                        dtype=object)
    h1_nsig = np.array([h1.n_sig[h1.index(t)] for t in eligible])
    W = stratified_sample(rng, eligible, h1_nsig, cfg.TEST["n_withheld"])
    W_set = set(W.tolist())
    print(f"[build] withheld set W: {len(W)} of {len(eligible)} eligible H1 targets")

    # ---- test contexts
    private_root = root / "private"
    dev_root = root / "dev"
    manifest_tests = {}
    for c in test_ctx:
        s = by_context[c][0]
        table, adata = tables[c], cells_cache[c]
        chem = s["chemistry"]
        if s["tier"] == "far":
            targets = sorted(h1_targets.tolist())
        else:
            seen_pool = np.array([t for t in table.targets if t != CONTROL and t not in W_set
                                  and table.has(t, cfg.TEST["min_cells_test_target"])], dtype=object)
            own_nsig = np.array([table.n_sig[table.index(t)] for t in seen_pool])
            seen = stratified_sample(rng, seen_pool, own_nsig, cfg.TEST["n_requests"] - len(W))
            targets = sorted(set(seen.tolist()) | W_set)
        n_cells = cfg.TEST["n_cells_request"][chem]
        basal, rows = public_view(adata, c, chem, targets, n_cells, cfg.TEST["max_basal_cells"], cfg.SEED,
                                  dev_root / "test" / c / "public", anonymous=True)
        resp = responses_view(adata, c, targets, cfg.TEST["max_response_cells"], cfg.SEED, anonymous=True)
        priv = private_root / c
        priv.mkdir(parents=True, exist_ok=True)
        resp.write_h5ad(priv / "responses.h5ad")
        basal.write_h5ad(priv / "basal.h5ad")
        lb = set(rng.choice(np.array([r["request_id"] for r in rows]), size=cfg.TEST["n_leaderboard"], replace=False).tolist())
        req_info = []
        for r in rows:
            t = r["perturbation"]["target"]
            label, per = stratum(t, train_tables, thresholds)
            req_info.append({"request_id": r["request_id"], "target": t, "subset": "leaderboard" if r["request_id"] in lb else "final",
                             "withheld": t in W_set, "seen_in_training": (t not in W_set) and len(per) > 0,
                             "stratum": label, "train_n_sig": per, "n_response_cells": int((resp.obs["target_gene"] == t).sum()),
                             "own_n_sig": int(table.n_sig[table.index(t)]), "own_knockdown": float(table.knockdown[table.index(t)])})
        sigs = {t: {tc: train_tables[tc].signature(t) for tc in train_ctx if train_tables[tc].has(t, cfg.STRATA["min_cells_measured"])}
                for t in targets}
        write_json(priv / "train_signatures.json", sigs)
        write_json(priv / "manifest.json", {
            "release": cfg.RELEASE, "context_id": c, "identity": s["identity"], "tier": s["tier"], "chemistry": chem,
            "source": s["source"], "responses_sha256": sha256(priv / "responses.h5ad"), "basal_sha256": sha256(priv / "basal.h5ad"),
            "public_basal_sha256": sha256(dev_root / "test" / c / "public" / "basal.h5ad"),
            "requests_sha256": sha256(dev_root / "test" / c / "public" / "requests.jsonl"),
            "n_requests": len(rows), "n_leaderboard": len(lb), "n_withheld": int(sum(1 for t in targets if t in W_set)),
            "requests": req_info, "strata_config": cfg.STRATA, "strata_thresholds": thresholds})
        manifest_tests[c] = {"tier": s["tier"], "chemistry": chem, "n_requests": len(rows), "n_basal_cells": basal.n_obs,
                             "n_genes": basal.n_vars, "n_response_cells": resp.n_obs,
                             "strata_counts": pd.Series([r["stratum"] for r in req_info]).value_counts().to_dict(),
                             "n_withheld": int(sum(1 for t in targets if t in W_set))}
        print(f"[build] test {c} ({s['tier']}): {len(rows)} requests, basal {basal.n_obs}, responses {resp.n_obs}")
        del cells_cache[c]

    write_json(private_root / "withheld_targets.json", {"W": sorted(W_set), "eligible_pool": sorted(eligible.tolist())})

    # ---- training contexts: remove W, write cells / pseudobulk / dev split
    manifest_train = {}
    for c in train_ctx:
        s = by_context[c][0]
        table, adata = tables[c], cells_cache[c]
        keep = ~adata.obs["target_gene"].isin(W_set).to_numpy()
        removed = int((~keep).sum())
        adata = subset(adata, keep)
        out = dev_root / "train" / c
        out.mkdir(parents=True, exist_ok=True)
        adata.obs = adata.obs[["target_gene", "target_ensembl", "guide_id", "batch", "source_screen", "total_counts",
                               "pct_counts_mt", "is_control"]].copy()
        adata.uns = {"context": c, "chemistry": s["chemistry"], "source": s["source"]}
        adata.write_h5ad(out / "cells.h5ad")
        table.to_anndata(drop=W_set).write_h5ad(out / "pseudobulk.h5ad")
        # dev split: a test-shaped view of this context's kept cells
        counts = adata.obs.loc[~adata.obs["is_control"], "target_gene"].value_counts()
        pool = np.array(sorted(counts[counts >= cfg.DEV["min_cells"]].index.tolist()), dtype=object)
        nsig = np.array([table.n_sig[table.index(t)] for t in pool])
        dev_targets = sorted(stratified_sample(rng, pool, nsig, cfg.DEV["n_requests"]).tolist())
        dev_dir = out / "dev_split"
        basal, rows = public_view(adata, c, s["chemistry"], dev_targets, cfg.DEV["n_cells_request"],
                                  cfg.DEV["max_basal_cells"], cfg.SEED, dev_dir, anonymous=False)
        resp = responses_view(adata, c, dev_targets, None, cfg.SEED, anonymous=False)
        resp.write_h5ad(dev_dir / "responses.h5ad")
        write_json(dev_dir / "manifest.json", {"context": c, "n_requests": len(rows), "n_basal_cells": basal.n_obs,
                                               "n_response_cells": resp.n_obs, "note": "responses are the same cells as cells.h5ad"})
        manifest_train[c] = {"chemistry": s["chemistry"], "source": s["source"], "n_cells": adata.n_obs, "n_genes": adata.n_vars,
                             "n_targets": int(adata.obs.loc[~adata.obs["is_control"], "target_gene"].nunique()),
                             "n_control_cells": int(adata.obs["is_control"].sum()), "cells_removed_for_W": removed,
                             "pseudobulk_targets": int(len(table.targets) - 1 - sum(1 for t in table.targets if t in W_set)),
                             "sources": [x["name"] for x in by_context[c]],
                             "caps": {x["name"]: {"keep_per_target": x["keep_per_target"], "keep_controls": x["keep_controls"]} for x in by_context[c]}}
        print(f"[build] train {c}: {adata.n_obs} cells ({removed} removed for W), dev split {len(rows)} requests")
        del cells_cache[c]

    # ---- shared metadata
    master_out = master[["ensembl_id", "symbol_source", "in_vcc_panel", "in_orion", "in_replogle_panels", "master_index"]].copy()
    master_out["in_replogle_panels"] = master_out["in_replogle_panels"].map(_anonymise_panels)
    master_out["symbol_source"] = master_out["symbol_source"].map(_anonymise_symbol_source)
    master_out.to_csv(dev_root / "gene_panel.csv")
    panels = {c: set(pd.read_csv(work / by_context[c][0]["name"] / "var.csv", index_col=0).index) for c in by_context}
    inter = sorted(set.intersection(*panels.values()))
    write_json(dev_root / "gene_panel_intersection.json", {"n_genes": len(inter), "genes": inter,
                                                           "per_context_n_genes": {c: len(p) for c, p in panels.items()}})
    overlap = {}
    for c in test_ctx:
        pm = json.loads((private_root / c / "manifest.json").read_text())
        overlap[c] = {"n_seen_in_training": sum(r["seen_in_training"] for r in pm["requests"]),
                      "n_never_seen": sum(not r["seen_in_training"] for r in pm["requests"])}
    write_json(dev_root / "splits.json", {
        "release": cfg.RELEASE, "seed": cfg.SEED, "train_contexts": train_ctx, "test_contexts": manifest_tests,
        "withheld_targets": {"n": len(W_set), "note": "removed from every training context; identities are private"},
        "target_overlap": overlap, "test_config": cfg.TEST, "dev_config": cfg.DEV, "strata_config": cfg.STRATA})
    write_json(root / "manifest.json", {
        "release": cfg.RELEASE, "seed": cfg.SEED, "sources": source_manifest, "source_stats": stats,
        "train_contexts": manifest_train, "test_contexts": manifest_tests,
        "master_panel": {"n_genes": len(master), "n_vcc_panel": int(master["in_vcc_panel"].sum())},
        "orion_qc": cfg.ORION_QC, "compression_note": "raw cells are a seeded per-target sample (caps above); "
                                                      "pseudobulk tables cover every source cell; test contexts are complete"})
    write_json(dev_root / "manifest.json", cfg.agent_manifest(json.loads((root / "manifest.json").read_text())))


def _anonymise_panels(labels) -> str:
    """Panel membership as shipped in gene_panel.csv: test sources appear under their context id."""
    if not isinstance(labels, str):
        return ""
    return ",".join(sorted(cfg.TEST_ALIAS.get(x, x) for x in labels.split(",") if x))


def _anonymise_symbol_source(src) -> str:
    """`replogle:<source name>` -> `replogle:<context id>` for test sources."""
    loader, _, name = str(src).partition(":")
    return f"{loader}:{cfg.TEST_ALIAS.get(name, name)}" if name else str(src)


def _reorder_master(adata: ad.AnnData, master: pd.DataFrame) -> ad.AnnData:
    order = np.argsort(adata.var["master_index"].to_numpy(), kind="stable")
    if not np.array_equal(order, np.arange(adata.n_vars)):
        adata = adata[:, order].copy()
    return adata


def _align_pb(pb: Pseudobulk, work: Path, name: str, var: pd.DataFrame) -> Pseudobulk:
    """Project one screen's pseudobulk sums onto the context panel `var` (drops genes outside it)."""
    v = pd.read_csv(work / name / "var.csv", index_col=0)
    pos = var.index.get_indexer(v.index)
    sel = pos >= 0
    L = np.zeros((len(pb.targets), len(var))); C = np.zeros_like(L)
    L[:, pos[sel]] = pb.sum_log1p[:, sel]; C[:, pos[sel]] = pb.sum_counts[:, sel]
    return Pseudobulk(pb.targets, pb.target_ensembl, pb.n_cells.copy(), L, C)
