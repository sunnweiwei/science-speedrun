#!/usr/bin/env python3
"""Recompute the leaderboard's aggregate for every published system from the pinned Space snapshot.

    python3 make_leaderboard_snapshot.py --leaderboard /data/leaderboard --out leaderboard_snapshot.json

For each results/<model>/all_results.csv: geometric mean over the 97 configs of MASE[0.5] and of
mean_weighted_sum_quantile_loss relative to results/seasonal_naive (the leaderboard's own normalization), plus the task
reward formula applied to those two numbers. This is the source of the README's Table 2.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

M, W = "eval_metrics/MASE[0.5]", "eval_metrics/mean_weighted_sum_quantile_loss"
REL_MIN, REL_MAX = 1e-3, 10.0   # environment/evaluation/score.py


def read(p):
    rows = {}
    for r in csv.DictReader(open(p)):
        rows.setdefault(r["dataset"], r)  # first row per config
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaderboard", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.leaderboard) / "results"
    base = read(root / "seasonal_naive" / "all_results.csv")
    assert len(base) == 97
    systems, dirs_total = [], 0
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        dirs_total += 1
        csvp = d / "all_results.csv"
        if not csvp.exists():
            continue
        cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
        rows = read(csvp)
        common = [k for k in base if k in rows]
        # the task grader clips every per-config ratio before aggregating; do the same here or the table is not its reward
        clip = lambda v: float(np.clip(v, REL_MIN, REL_MAX)) if np.isfinite(v) and v > 0 else REL_MAX  # noqa: E731
        rel_m = [clip(float(rows[k][M]) / float(base[k][M])) for k in common]
        rel_w = [clip(float(rows[k][W]) / float(base[k][W])) for k in common]
        ok = len(common) == 97
        gm = float(np.exp(np.mean(np.log(rel_m)))) if ok else None
        gw = float(np.exp(np.mean(np.log(rel_w)))) if ok else None
        systems.append({"dir": d.name, "model": cfg.get("model", d.name), "model_type": cfg.get("model_type"), "testdata_leakage": cfg.get("testdata_leakage"),
                        "org": cfg.get("org"), "n_configs": len(common), "complete": ok, "rel_mase_gmean": gm, "rel_wql_gmean": gw,
                        "reward": float(np.clip(1 - np.sqrt(gm * gw), 0, 1)) if ok else None})
    systems.sort(key=lambda s: (not s["complete"], s["rel_mase_gmean"] if s["rel_mase_gmean"] is not None else 9))
    complete = [s for s in systems if s["complete"]]
    out = {"what": "leaderboard aggregate recomputed from the pinned Space snapshot (results/<model>/all_results.csv, normalized by results/seasonal_naive, geometric mean over the 97 configs) and the task reward formula",
           "protocol": "board protocol (models run by their authors on the full history); same configs, metrics and normalizer as the task",
           "space": "Salesforce/GIFT-Eval @ a5019972f79e650b1ede252885df5f515fa115b8",
           "n_result_dirs": dirs_total, "n_with_csv": len(systems), "n_complete_97": len(complete),
           "n_agentic_complete": sum(1 for s in complete if s["model_type"] == "agentic"),
           "best_non_agentic": next(s for s in complete if s["model_type"] != "agentic"),
           "best_agentic": next((s for s in complete if s["model_type"] == "agentic"), None), "systems": systems}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("n_result_dirs", "n_with_csv", "n_complete_97", "n_agentic_complete")}), out["best_non_agentic"]["model"], out["best_agentic"]["model"])


if __name__ == "__main__":
    main()
