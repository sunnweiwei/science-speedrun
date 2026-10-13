#!/usr/bin/env python3
"""Single scoring code path for development (LOCO dev splits) and the isolated verifier.

Wraps the pinned official `cell-eval` (0.8.2). One call scores one context:

    report = score_context(predictions, basal, responses, requests, outdir, subsets=..., baseline=...)

Both prediction and ground truth are put on log1p(CP10k) here, then cell-eval computes the
Generalist panel; every metric is normalised against the no-effect baseline (basal cells
resampled per request) with cell-eval's own `score_agg_metrics` rule:
    higher-is-better metrics: (m - m_base) / (1 - m_base);  error metrics: 1 - m / m_base;
clipped at 0, NaN -> 0. The context score is the mean of the normalised panel.

Usage (dev):  python cellscore.py --predictions P.h5ad --basal B.h5ad --responses R.h5ad \
                  --requests requests.jsonl --outdir OUT [--subsets subsets.json]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from scipy import sparse

CONTROL = "non-targeting"
PERT_COL = "target_gene"

# The Generalist panel: DES, PDS, MAE plus the four extra Generalist metrics of VCC 2025.
PANEL = {
    "des": "overlap_at_N",
    "pds": "discrimination_score_l1",
    "mae": "mae",
    "pearson_delta": "pearson_delta",
    "spearman_lfc": "de_spearman_lfc_sig",
    "auprc_de": "pr_auc",
    "spearman_effect": "de_spearman_sig",
}
PER_PERT_METRICS = ["overlap_at_N", "discrimination_score_l1", "mae", "pearson_delta", "de_spearman_lfc_sig", "pr_auc"]
GLOBAL_METRICS = ["de_spearman_sig"]
ERROR_METRICS = {"mae"}
# cell-eval "full" minus the metrics that are not in the panel and dominate runtime
# (e-distance / clustering need all pairwise cell distances against 10k controls per target)
SKIP_METRICS = ["pearson_edistance", "clustering_agreement", "discrimination_score_l2", "discrimination_score_cosine",
                "mse", "mse_delta", "mae_delta", "overlap_at_50", "overlap_at_100", "overlap_at_200", "overlap_at_500",
                "precision_at_50", "precision_at_100", "precision_at_200", "precision_at_500"]


def cell_eval_version() -> str:
    return importlib.metadata.version("cell-eval")


def read_requests(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [r["request_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate request_id")
    return rows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def normlog(X) -> sparse.csr_matrix:
    X = sparse.csr_matrix(X, dtype=np.float32)
    totals = np.asarray(X.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    Y = sparse.diags((1e4 / totals).astype(np.float32)) @ X
    Y.data = np.log1p(Y.data)
    return Y.tocsr()


def _check_matrix(adata: ad.AnnData, name: str) -> None:
    data = adata.X.data if sparse.issparse(adata.X) else np.asarray(adata.X)
    if not np.isfinite(data).all() or (data < 0).any():
        raise ValueError(f"{name}: negative or non-finite values")


def assemble_prediction(pred: ad.AnnData, basal: ad.AnnData, requests: list[dict], keep: set[str] | None = None) -> ad.AnnData:
    """Prediction cells + the basal cells as `non-targeting`, on log1p(CP10k)."""
    if list(map(str, pred.var_names)) != list(map(str, basal.var_names)):
        raise ValueError("prediction genes/order differ from basal input")
    target_of = {r["request_id"]: r["perturbation"]["target"] for r in requests}
    rid = pred.obs["request_id"].astype(str)
    if set(rid) != set(target_of):
        raise ValueError("prediction request_ids differ from requests.jsonl")
    counts = rid.value_counts().to_dict()
    for r in requests:
        if counts.get(r["request_id"], 0) != r["n_cells"]:
            raise ValueError(f"request {r['request_id']}: expected {r['n_cells']} cells, got {counts.get(r['request_id'], 0)}")
    _check_matrix(pred, "predictions"); _check_matrix(basal, "basal")
    targets = rid.map(target_of).astype(str)
    mask = np.ones(pred.n_obs, dtype=bool) if keep is None else rid.isin(keep).to_numpy()
    p = ad.AnnData(X=normlog(pred.X[mask]), obs=pd.DataFrame({PERT_COL: targets.to_numpy()[mask]},
                                                              index=[f"p{i}" for i in range(int(mask.sum()))]),
                   var=pd.DataFrame(index=basal.var_names.astype(str)))
    b = ad.AnnData(X=normlog(basal.X), obs=pd.DataFrame({PERT_COL: CONTROL}, index=[f"c{i}" for i in range(basal.n_obs)]),
                   var=pd.DataFrame(index=basal.var_names.astype(str)))
    out = ad.concat([p, b], join="inner")
    out.obs[PERT_COL] = out.obs[PERT_COL].astype(str)
    out.var_names = out.var_names.astype(str)
    return out


def assemble_real(responses: ad.AnnData, basal: ad.AnnData, requests: list[dict], keep: set[str] | None = None) -> ad.AnnData:
    if list(map(str, responses.var_names)) != list(map(str, basal.var_names)):
        raise ValueError("responses genes/order differ from basal")
    targets = {r["perturbation"]["target"] for r in requests if keep is None or r["request_id"] in keep}
    mask = responses.obs[PERT_COL].astype(str).isin(targets).to_numpy()
    r = ad.AnnData(X=normlog(responses.X[mask]),
                   obs=pd.DataFrame({PERT_COL: responses.obs[PERT_COL].astype(str).to_numpy()[mask]},
                                    index=[f"r{i}" for i in range(int(mask.sum()))]),
                   var=pd.DataFrame(index=basal.var_names.astype(str)))
    b = ad.AnnData(X=normlog(basal.X), obs=pd.DataFrame({PERT_COL: CONTROL}, index=[f"c{i}" for i in range(basal.n_obs)]),
                   var=pd.DataFrame(index=basal.var_names.astype(str)))
    out = ad.concat([r, b], join="inner")
    out.obs[PERT_COL] = out.obs[PERT_COL].astype(str)
    missing = targets - set(out.obs[PERT_COL])
    if missing:
        raise ValueError(f"responses lack targets: {sorted(missing)[:5]}...")
    return out


def noeffect_prediction(basal: ad.AnnData, requests: list[dict]) -> ad.AnnData:
    """B0: resample basal cells unchanged, `n_cells` per request, seeded by the request."""
    blocks, ids = [], []
    for r in requests:
        rng = np.random.default_rng(int(r.get("seed", 0)))
        idx = rng.choice(basal.n_obs, size=int(r["n_cells"]), replace=basal.n_obs < int(r["n_cells"]))
        blocks.append(sparse.csr_matrix(basal.X[idx]))
        ids.extend([r["request_id"]] * len(idx))
    X = sparse.vstack(blocks).tocsr()
    obs = pd.DataFrame({"request_id": ids, PERT_COL: [r["perturbation"]["target"] for r in requests for _ in range(int(r["n_cells"]))]},
                       index=[f"n{i}" for i in range(len(ids))])
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=basal.var_names.astype(str)))


def run_cell_eval(pred: ad.AnnData, real: ad.AnnData, outdir: Path, threads: int, prefix: str,
                  de_real: Path | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """de_real: optional cached real-side DE table (cell-eval's `<prefix>_real_de.csv`); it depends
    only on the ground truth and control cells, so the grader may reuse it across submissions."""
    from cell_eval import MetricsEvaluator
    outdir.mkdir(parents=True, exist_ok=True)
    evaluator = MetricsEvaluator(adata_pred=pred, adata_real=real, control_pert=CONTROL, pert_col=PERT_COL,
                                 num_threads=threads, outdir=str(outdir), prefix=prefix, allow_discrete=False,
                                 de_real=str(de_real) if de_real and Path(de_real).exists() else None)
    results, agg = evaluator.compute(profile="full", skip_metrics=SKIP_METRICS, write_csv=True, basename="results.csv")
    return results, agg


def _mean_over(results: pl.DataFrame, perts: set[str] | None) -> dict[str, float]:
    df = results.to_pandas()
    if perts is not None:
        df = df[df["perturbation"].isin(perts)]
    out = {}
    for m in PER_PERT_METRICS:
        out[m] = float(np.nanmean(df[m].to_numpy(dtype=float))) if m in df else float("nan")
    return out


def _global(agg: pl.DataFrame) -> dict[str, float]:
    df = agg.to_pandas().set_index("statistic")
    return {m: float(df.loc["mean", m]) if m in df.columns else float("nan") for m in GLOBAL_METRICS}


def normalise(user: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    out = {}
    for m, u in user.items():
        b = base.get(m, float("nan"))
        if m in ERROR_METRICS:
            v = 1.0 - (u / b) if b else float("nan")
        else:
            v = (u - b) / (1.0 - b) if (1.0 - b) != 0 else float("nan")
        out[m] = float(np.clip(0.0 if not np.isfinite(v) else v, 0.0, None))
    return out


def summarise(user_results: pl.DataFrame, user_agg: pl.DataFrame, base_results: pl.DataFrame, base_agg: pl.DataFrame,
              perts: set[str] | None) -> dict:
    """Full-run summary (perts=None): all seven panel metrics. Group summary (perts given): only the
    per-perturbation metrics, averaged over the group — the global metric (Spearman of effect size
    across targets) is a property of the whole run and would leak into every group."""
    if perts is None:
        raw_u = {**_mean_over(user_results, None), **_global(user_agg)}
        raw_b = {**_mean_over(base_results, None), **_global(base_agg)}
        panel = PANEL
    else:
        raw_u = _mean_over(user_results, perts)
        raw_b = _mean_over(base_results, perts)
        panel = {k: v for k, v in PANEL.items() if v in PER_PERT_METRICS}
    norm = normalise(raw_u, raw_b)
    panel_norm = {k: norm[v] for k, v in panel.items()}
    return {"raw": {k: raw_u[v] for k, v in panel.items()}, "baseline_raw": {k: raw_b[v] for k, v in panel.items()},
            "normalised": panel_norm, "score": float(np.mean(list(panel_norm.values()))),
            "metrics_in_score": list(panel),
            "n_perturbations": int(user_results.select("perturbation").n_unique()) if perts is None else len(perts)}


def score_context(predictions: Path, basal: Path, responses: Path, requests_path: Path, outdir: Path,
                  subsets: dict[str, list[str]] | None = None, groups: dict[str, dict[str, str]] | None = None,
                  threads: int | None = None, baseline_dir: Path | None = None) -> dict:
    """subsets: name -> request_ids scored as a self-contained cell-eval run (e.g. leaderboard/final).
    groups: name -> {request_id: label}; per-label means of per-perturbation metrics from the full run."""
    threads = threads or max(1, len(os.sched_getaffinity(0)) - 1)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    requests = read_requests(requests_path)
    pred = ad.read_h5ad(predictions); bas = ad.read_h5ad(basal); resp = ad.read_h5ad(responses)
    target_of = {r["request_id"]: r["perturbation"]["target"] for r in requests}
    base_pred = noeffect_prediction(bas, requests)

    def one(keep: set[str] | None, tag: str):
        cache = baseline_dir / tag if baseline_dir else None
        de_real = cache / "real_de.csv" if cache else None
        u_res, u_agg = run_cell_eval(assemble_prediction(pred, bas, requests, keep), assemble_real(resp, bas, requests, keep),
                                     outdir / tag / "user", threads, tag, de_real)
        if cache and (cache / "results.csv").exists() and (cache / "agg_results.csv").exists():
            b_res, b_agg = pl.read_csv(cache / "results.csv"), pl.read_csv(cache / "agg_results.csv")
        else:
            b_res, b_agg = run_cell_eval(assemble_prediction(base_pred, bas, requests, keep), assemble_real(resp, bas, requests, keep),
                                         outdir / tag / "noeffect", threads, tag, de_real)
        return u_res, u_agg, b_res, b_agg

    report = {"cell_eval_version": cell_eval_version(), "panel": PANEL, "subsets": {}, "groups": {}}
    u_res, u_agg, b_res, b_agg = one(None, "all")
    report["all"] = summarise(u_res, u_agg, b_res, b_agg, None)
    for name, ids in (subsets or {}).items():
        keep = set(ids)
        s_res, s_agg, sb_res, sb_agg = one(keep, name)
        report["subsets"][name] = summarise(s_res, s_agg, sb_res, sb_agg, None)
    for gname, labels in (groups or {}).items():
        report["groups"][gname] = {}
        for label in sorted(set(labels.values())):
            perts = {target_of[rid] for rid, lab in labels.items() if lab == label}
            report["groups"][gname][label] = summarise(u_res, u_agg, b_res, b_agg, perts)
    per_pert = u_res.to_pandas()
    per_pert.to_csv(outdir / "per_perturbation.csv", index=False)
    (outdir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--basal", required=True, type=Path)
    ap.add_argument("--responses", required=True, type=Path)
    ap.add_argument("--requests", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--subsets", type=Path, help="json: name -> [request_id]")
    ap.add_argument("--groups", type=Path, help="json: name -> {request_id: label}")
    ap.add_argument("--baseline-dir", type=Path)
    ap.add_argument("--threads", type=int)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    subsets = json.loads(a.subsets.read_text()) if a.subsets else None
    groups = json.loads(a.groups.read_text()) if a.groups else None
    report = score_context(a.predictions, a.basal, a.responses, a.requests, a.outdir, subsets, groups, a.threads, a.baseline_dir)
    print(json.dumps({"score": report["all"]["score"], "normalised": report["all"]["normalised"]}, indent=2))


if __name__ == "__main__":
    main()
