#!/usr/bin/env python3
"""Compare a package's per-config metrics (grader report) with a leaderboard results CSV.

    python3 compare_with_leaderboard.py --report REPORT.json --leaderboard-dir /data/leaderboard/results/seasonal_naive \
        --out docs/validation/seasonal_naive_replay.json --name "Seasonal Naive (environment/baseline/seasonal_naive)"

Writes per-config ours vs published values, relative deviations, the aggregate relative MASE / WQL vs the
leaderboard's Seasonal Naive row (geometric means) and the task reward, so the README tables can be regenerated.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M, W = "eval_metrics/MASE[0.5]", "eval_metrics/mean_weighted_sum_quantile_loss"


def read_csv(p):
    return {r["dataset"]: r for r in csv.DictReader(open(p))}


def gmean(x):
    return float(np.exp(np.mean(np.log(x))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True); ap.add_argument("--leaderboard-dir", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--baseline-csv", default="")
    ap.add_argument("--extra", default="", help="JSON string with extra fields (runtime, settings)")
    args = ap.parse_args()
    rep = json.loads(Path(args.report).read_text())
    lb = read_csv(Path(args.leaderboard_dir) / "all_results.csv")
    cfg = json.loads((Path(args.leaderboard_dir) / "config.json").read_text()) if (Path(args.leaderboard_dir) / "config.json").exists() else {}
    base_csv = args.baseline_csv or str(Path(args.leaderboard_dir).parent / "seasonal_naive" / "all_results.csv")
    base = read_csv(base_csv)
    rows, dm, dw = [], [], []
    for c in rep["configs"]:
        k = c["leaderboard_key"]; m = c.get("metrics", {})
        pub_m, pub_w = float(lb[k][M]), float(lb[k][W])
        r = {"config_id": c["config_id"], "leaderboard_key": k, "task_mase": m.get("mase"), "task_wql": m.get("wql"),
             "published_mase": pub_m, "published_wql": pub_w, "status": c["status"]}
        if m:
            r["rel_dev_mase"] = abs(m["mase"] - pub_m) / pub_m; r["rel_dev_wql"] = abs(m["wql"] - pub_w) / pub_w
            dm.append(r["rel_dev_mase"]); dw.append(r["rel_dev_wql"])
        rows.append(r)
    pub_rel_m = gmean([float(lb[k][M]) / float(base[k][M]) for k in lb if k in base])
    pub_rel_w = gmean([float(lb[k][W]) / float(base[k][W]) for k in lb if k in base])
    out = {"package": args.name, "leaderboard_model": cfg.get("model"), "leaderboard_config": cfg, "n_configs": len(rows),
           "task": {"reward": rep["reward"], "skill_score": rep["skill_score"], "rel_mase_gmean": rep["rel_mase_gmean"], "rel_wql_gmean": rep["rel_wql_gmean"],
                    "n_failed": rep["n_failed"], "quantile_crossing_fraction_mean": rep.get("quantile_crossing_fraction_mean")},
           "published": {"rel_mase_gmean": pub_rel_m, "rel_wql_gmean": pub_rel_w, "reward_from_published": float(np.clip(1 - np.sqrt(pub_rel_m * pub_rel_w), 0, 1))},
           "deviation": {"mase_max": max(dm) if dm else None, "mase_median": float(np.median(dm)) if dm else None,
                         "wql_max": max(dw) if dw else None, "wql_median": float(np.median(dw)) if dw else None,
                         "n_within_1e-6": int(sum(1 for a, b in zip(dm, dw) if max(a, b) < 1e-6)),
                         "n_over_5pct": int(sum(1 for a, b in zip(dm, dw) if max(a, b) > 0.05)),
                         "worst_configs": sorted(rows, key=lambda r: -max(r.get("rel_dev_mase", 0), r.get("rel_dev_wql", 0)))[:8]},
           "groups": rep["groups"], "configs": rows}
    if args.extra:
        out["extra"] = json.loads(args.extra)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("task", "published", "deviation")}, default=str)[:1500])


if __name__ == "__main__":
    main()
