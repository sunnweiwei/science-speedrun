#!/usr/bin/env python3
"""Trusted grader for timeseries-forecast/v1 (see ../contract.md).

    python3 score.py --truth TRUTH_DIR --output OUTPUT_DIR [--report report.json] [--csv results.csv]

Per config (dataset, frequency, term) the two GIFT-Eval headline metrics are
computed as gluonts 0.15.1 does under `evaluate_model(..., axis=None,
mask_invalid_label=True)`, to float32 precision (gluonts accumulates in the
data's float32, this grader in float64, which is the whole 1.9e-7 residual in
docs/validation/golden_gluonts_full.json):

  MASE   = mean over every (instance, variate, step) with a finite label and a
           positive seasonal error of |label - q0.5| / seasonal_error(instance, variate)
           (numpy.ma masks the division by a zero seasonal error, so gluonts drops those elements)
  WQL    = (1/9) * sum_q  [ sum 2|(label - pred_q) (1{pred_q >= label} - q)| / sum |label| ]
           (sums over the same finite-label elements; the leaderboard calls this CRPS)

Each metric is divided by the task's Seasonal Naive value for the same config
(seasonal_naive_reference.json, produced by environment/baseline/seasonal_naive
through this very grader), clipped to [REL_MIN, REL_MAX], and aggregated with a
geometric mean over configs, which is the leaderboard's aggregation. The reward is

  reward = clip(1 - sqrt(gmean(rel MASE) * gmean(rel WQL)), 0, 1)

i.e. the geometric skill score over Seasonal Naive (0 = no better than seasonal
naive, 1 = perfect). A config whose prediction file is missing, unreadable, mis-shaped or non-finite
scores REL_MAX on both metrics; it never aborts the run. A config that is only
partly forecast (the verifier ran out of time before some of its cohorts) is
scored on the instances it covers and blended toward REL_MAX in proportion to
what is missing, in log space so the blend is consistent with the geometric
aggregation:  log rel = f * log(rel_covered) + (1 - f) * log(REL_MAX).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ts_io  # noqa: E402

REL_MIN, REL_MAX = 1e-3, 10.0
BASELINE_PATH = Path(__file__).with_name("seasonal_naive_reference.json")
QUANTILES = np.array(ts_io.QUANTILE_LEVELS)


def sort_quantiles(q: np.ndarray):
    """Monotone rearrangement along the quantile axis; returns (sorted, fraction of crossing elements)."""
    s = np.sort(q, axis=-2)
    crossing = np.any(s != q, axis=-2)
    return s, float(crossing.mean()) if crossing.size else 0.0


def config_metrics(labels: np.ndarray, seasonal_error: np.ndarray, mean: np.ndarray, quantiles: np.ndarray) -> dict:
    """labels (N, D, H), seasonal_error (N, D), mean (N, D, H), quantiles (N, D, 9, H) -> MASE and WQL as gluonts computes them."""
    lab = labels.astype(np.float64)
    valid = np.isfinite(lab)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {"mase": float("nan"), "wql": float("nan"), "n_valid_labels": 0, "n_masked_labels": int((~valid).sum())}
    q = quantiles.astype(np.float64)
    median = q[:, :, 4, :]
    se = seasonal_error.astype(np.float64)[:, :, None]
    # gluonts divides a masked array by the seasonal error; numpy.ma masks the elements where the divisor is 0
    # (and a fully-masked history yields 0), so those elements drop out of both the sum and the count.
    mase_valid = valid & np.broadcast_to(np.isfinite(se) & (se > 0), lab.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        ase = np.abs(lab - median) / se
    ase_valid = ase[mase_valid]
    mase = float(ase_valid.sum() / ase_valid.size) if ase_valid.size else float("nan")
    abs_label_sum = float(np.abs(lab[valid]).sum())
    wsql = []
    for j, level in enumerate(QUANTILES):
        pred = q[:, :, j, :]
        ql = 2.0 * np.abs((lab - pred) * ((pred >= lab).astype(np.float64) - level))
        wsql.append(float(ql[valid].sum()) / abs_label_sum if abs_label_sum > 0 else float("nan"))
    return {"mase": mase, "wql": float(np.mean(wsql)), "wsql_by_level": wsql,
            "n_valid_labels": n_valid, "n_masked_labels": int((~valid).sum()),
            "mae_median": float(np.abs(lab - median)[valid].mean()),
            "n_mase_elements": int(mase_valid.sum()),
            "n_seasonal_error_zero": int((seasonal_error == 0).sum()), "n_seasonal_error_nan": int(np.isnan(seasonal_error).sum())}


def blend(rel: float, fraction: float) -> float:
    """Geometric blend toward REL_MAX for the instances that were never forecast (see the module docstring)."""
    if fraction >= 1.0:
        return rel
    if fraction <= 0.0 or not np.isfinite(rel) or rel <= 0:
        return REL_MAX
    return float(np.clip(np.exp(fraction * np.log(rel) + (1.0 - fraction) * np.log(REL_MAX)), REL_MIN, REL_MAX))


def json_safe(obj):
    """Replace non-finite floats with null so the reports are valid JSON for strict parsers (jq, JSON.parse)."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def gmean(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.exp(np.mean(np.log(x)))) if x.size else float("nan")


def relative(value: float, base: float) -> float:
    if not np.isfinite(value) or not np.isfinite(base) or base <= 0:
        return REL_MAX
    return float(np.clip(value / base, REL_MIN, REL_MAX))


def score(truth_dir: Path, output_dir: Path, baseline: dict | None = None, allow_missing_baseline: bool = False, configs=None,
          coverage: dict | None = None) -> dict:
    """Grade OUTPUT_DIR against TRUTH_DIR.

    `configs` restricts scoring to a subset of config ids (validation during training). `coverage` is
    {config_id: boolean mask over instances} and is supplied by the verifier when a run was cut short; it is never
    read from the package's own output, which the package controls.
    """
    truth_dir, output_dir = Path(truth_dir), Path(output_dir)
    index = json.loads((truth_dir / "index.json").read_text())
    ids = [c["config_id"] for c in index["configs"]]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate config_id in the truth index")
    if configs:
        wanted = set(configs)
        index["configs"] = [c for c in index["configs"] if c["config_id"] in wanted]
    if baseline is None:
        if BASELINE_PATH.exists():
            baseline = json.loads(BASELINE_PATH.read_text())["configs"]
        elif allow_missing_baseline:
            baseline = {}
        else:
            raise RuntimeError(f"Seasonal Naive reference {BASELINE_PATH} missing")
    scored, failures = [], []
    for cfg in index["configs"]:
        cid = cfg["config_id"]
        labels, se, meta = ts_io.read_truth_file(truth_dir / "cases" / f"{cid}.h5")
        entry = {**cfg, "leaderboard_key": ts_io.leaderboard_key(cfg["dataset"], cfg["term"]), "status": "ok"}
        base = baseline.get(cid)
        if base is None:
            if not allow_missing_baseline:
                raise RuntimeError(f"no Seasonal Naive reference for config {cid}")
            base = {"mase": 1.0, "wql": 1.0}
        entry["baseline"] = {"mase": base["mase"], "wql": base["wql"]}
        pred_path = output_dir / "cases" / f"{cid}.h5"
        reason, covered = None, None
        try:
            if not pred_path.exists():
                reason = "missing prediction file"
            else:
                # read the shapes first: a mis-declared dataset must not be materialized
                mean, quantiles = ts_io.read_prediction_file(pred_path, expect=labels.shape)
                covered = np.ones(labels.shape[0], bool) if not coverage or cid not in coverage else np.asarray(coverage[cid], bool)
                if covered.shape != (labels.shape[0],):
                    raise RuntimeError(f"coverage mask for {cid} has shape {covered.shape}, expected {(labels.shape[0],)}")
                if not (np.isfinite(mean[covered]).all() and np.isfinite(quantiles[covered]).all()):
                    reason = "non-finite values in prediction"
        except Exception as exc:  # noqa: BLE001 - an unreadable or mis-shaped file is a model failure, not a grader failure
            reason = f"{exc}" if isinstance(exc, ValueError) and str(exc).startswith("bad shapes") else \
                     f"unreadable prediction file: {type(exc).__name__}: {exc}"
        if reason is None:
            quantiles, entry["quantile_crossing_fraction"] = sort_quantiles(quantiles)
            fraction = float(covered.mean())
            entry["coverage"] = {"instances_scored": int(covered.sum()), "instances": int(labels.shape[0]), "fraction": fraction}
            entry["metrics"] = config_metrics(labels[covered], se[covered], mean[covered], quantiles[covered])
            entry["relative"] = {k: blend(relative(entry["metrics"][k], base[k]), fraction) for k in ("mase", "wql")}
            if not np.isfinite(entry["metrics"]["mase"]) or not np.isfinite(entry["metrics"]["wql"]):
                entry["status"] = "failed"; entry["reason"] = "non-finite metric"
                entry["relative"] = {"mase": REL_MAX, "wql": REL_MAX}
                failures.append({"config_id": cid, "reason": entry["reason"]})
            elif fraction < 1.0:
                entry["status"] = "partial"; entry["reason"] = f"{entry['coverage']['instances_scored']}/{labels.shape[0]} instances forecast"
                failures.append({"config_id": cid, "reason": entry["reason"]})
        else:
            entry["status"] = "failed"; entry["reason"] = reason
            entry["relative"] = {"mase": REL_MAX, "wql": REL_MAX}
            failures.append({"config_id": cid, "reason": reason})
        scored.append(entry)
    configs = scored
    rel_m = [c["relative"]["mase"] for c in configs]; rel_w = [c["relative"]["wql"] for c in configs]
    gm_m, gm_w = gmean(rel_m), gmean(rel_w)
    skill = 1.0 - float(np.sqrt(gm_m * gm_w))
    reward = float(np.clip(skill, 0.0, 1.0)) if configs else 0.0
    groups = {}
    for key, fn in (("term", lambda c: c["term"]), ("domain", lambda c: c["domain"]), ("frequency", lambda c: ts_io.parse_freq(c["freq"])[1]),
                    ("variates", lambda c: "univariate" if c["n_variates"] == 1 else "multivariate")):
        groups[key] = {}
        for c in configs:
            groups[key].setdefault(fn(c), []).append(c)
        groups[key] = {g: {"n_configs": len(v), "rel_mase_gmean": gmean([c["relative"]["mase"] for c in v]),
                           "rel_wql_gmean": gmean([c["relative"]["wql"] for c in v])} for g, v in sorted(groups[key].items())}
    return {"contract": ts_io.CONTRACT, "split": index.get("split"), "reward": reward, "skill_score": skill,
            "rel_mase_gmean": gm_m, "rel_wql_gmean": gm_w, "n_configs": len(configs), "n_failed": sum(1 for c in configs if c["status"] == "failed"),
            "n_instances": sum(c["n_instances"] for c in configs),
            "n_instances_scored": sum(c.get("coverage", {}).get("instances_scored", c["n_instances"]) for c in configs),
            "n_partial": sum(1 for c in configs if c["status"] == "partial"), "failures": failures, "groups": groups,
            "quantile_crossing_fraction_mean": float(np.mean([c.get("quantile_crossing_fraction", 0.0) for c in configs])) if configs else 0.0,
            "configs": configs}


def write_results_csv(report: dict, path: Path, model: str):
    """Per-config values in the leaderboard's all_results.csv column names (the two scored metrics only)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", "eval_metrics/MASE[0.5]", "eval_metrics/mean_weighted_sum_quantile_loss", "domain", "num_variates"])
        for c in report["configs"]:
            m = c.get("metrics", {})
            w.writerow([c["leaderboard_key"], model, m.get("mase", ""), m.get("wql", ""), c["domain"], c["n_variates"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--input", default="", help="ignored (kept for older command lines)")
    ap.add_argument("--report", default=""); ap.add_argument("--csv", default=""); ap.add_argument("--model-name", default="submission")
    ap.add_argument("--baseline", default="", help="alternative Seasonal Naive reference JSON")
    ap.add_argument("--allow-missing-baseline", action="store_true")
    args = ap.parse_args()
    baseline = json.loads(Path(args.baseline).read_text())["configs"] if args.baseline else None
    report = score(args.truth, args.output, baseline, args.allow_missing_baseline)
    if args.report:
        Path(args.report).write_text(json.dumps(json_safe(report), indent=1, allow_nan=False) + "\n")
    if args.csv:
        write_results_csv(report, Path(args.csv), args.model_name)
    print(json.dumps({k: report[k] for k in ("reward", "skill_score", "rel_mase_gmean", "rel_wql_gmean", "n_configs", "n_failed")}))


if __name__ == "__main__":
    main()
