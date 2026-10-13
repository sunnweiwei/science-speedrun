#!/usr/bin/env python3
"""Build a tiny synthetic example of INPUT_DIR / TRUTH_DIR and a valid OUTPUT_DIR without any real data.

    python3 make_example.py --out /tmp/timeseries-example [--seed 0]

Three small configs (univariate hourly with missing values, univariate daily, 7-variate 15-minute) with random
seasonal series; the truth holds the labels and the seasonal errors exactly as make_cases.py would write them, the
output is a Seasonal Naive-like forecast. Used by the unit tests and by the contract documentation.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import ts_io  # noqa: E402

CONFIGS = [
    {"dataset": "example_hourly/H", "freq": "H", "term": "short", "n_variates": 1, "n_series": 5, "length": 800, "missing": 0.05, "domain": "Example"},
    {"dataset": "example_daily/D", "freq": "D", "term": "short", "n_variates": 1, "n_series": 3, "length": 400, "missing": 0.0, "domain": "Example"},
    {"dataset": "example_multi/15T", "freq": "15T", "term": "medium", "n_variates": 7, "n_series": 1, "length": 6000, "missing": 0.01, "domain": "Example"},
]


def make(out: Path, seed: int, windows_cap: int = 20):
    rng = np.random.default_rng(seed)
    inp, truth, outp = out / "input", out / "truth", out / "output"
    staged = []
    for c in CONFIGS:
        n, unit = ts_io.parse_freq(c["freq"])
        L = ts_io.prediction_length(c["dataset"], c["freq"], c["term"]); m = ts_io.seasonality(c["freq"])
        W = min(ts_io.n_windows(c["dataset"], c["length"], L), windows_cap)
        cid = ts_io.config_id(c["dataset"], unit, c["term"])
        contexts, labels, se, ids, wins, starts, hist, cohorts = [], [], [], [], [], [], [], []
        means, quants = [], []
        for s in range(c["n_series"]):
            t = np.arange(c["length"])
            base = rng.uniform(10, 100)
            series = np.stack([base + rng.uniform(1, 10) * np.sin(2 * np.pi * t / max(m, 7) + rng.uniform(0, 6)) + rng.normal(0, 1, len(t)).cumsum() * 0.1
                               for _ in range(c["n_variates"])]).astype(np.float32)
            if c["missing"]:
                series[rng.random(series.shape) < c["missing"]] = np.nan
            for k, cut in enumerate(ts_io.test_cutoffs(c["length"], L, W)):
                h = series[:, :cut]
                contexts.append(h[:, -ts_io.CONTEXT_CAP:]); labels.append(series[:, cut:cut + L]); se.append(ts_io.seasonal_error(h, m))
                ids.append(f"item_{s}"); wins.append(k); hist.append(cut); cohorts.append(k)  # rank of the cutoff time per item, ascending, as make_cases.py assigns it
                starts.append(ts_io.timestamp_after(np.datetime64("2020-01-01T00:00:00"), c["freq"], cut))
                last = np.where(np.isfinite(h[:, -m:]), h[:, -m:], np.nanmean(h, axis=1, keepdims=True))
                mean = np.tile(last, (1, L // m + 1))[:, :L]
                means.append(mean); quants.append(np.stack([mean + (q - 0.5) * 4 for q in ts_io.QUANTILE_LEVELS], axis=1))
        meta = {"dataset": c["dataset"], "freq": c["freq"], "freq_unit": unit, "term": c["term"], "config_id": cid,
                "leaderboard_key": ts_io.leaderboard_key(c["dataset"], c["term"]), "prediction_length": L, "seasonality": m,
                "windows": W, "n_variates": c["n_variates"], "domain": c["domain"], "split": "example", "context_cap": ts_io.CONTEXT_CAP}
        ts_io.write_case_file(inp / "cases" / f"{cid}.h5", meta, contexts, ids, wins, starts, hist, cohorts)
        ts_io.write_truth_file(truth / "cases" / f"{cid}.h5", np.stack(labels), np.stack(se), meta)
        ts_io.write_prediction_file(outp / "cases" / f"{cid}.h5", np.stack(means), np.stack(quants), {"model": "example"})
        staged.append({"config_id": cid, "leaderboard_key": meta["leaderboard_key"], "dataset": c["dataset"], "term": c["term"], "freq": c["freq"],
                       "prediction_length": L, "seasonality": m, "n_variates": c["n_variates"], "n_instances": len(contexts), "domain": c["domain"]})
    index = {"contract": ts_io.CONTRACT, "split": "example", "context_cap": ts_io.CONTEXT_CAP, "quantile_levels": list(ts_io.QUANTILE_LEVELS),
             "n_cohorts": max(int(ts_io.n_windows(c["dataset"], c["length"], ts_io.prediction_length(c["dataset"], c["freq"], c["term"]))) for c in CONFIGS), "configs": staged}
    (inp / "index.json").write_text(json.dumps(index, indent=1) + "\n"); (truth / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    (outp / "model_manifest.json").write_text(json.dumps({"name": "example", "contract": ts_io.CONTRACT, "parameters": 0, "pretrained_assets": [], "sources_used": []}))
    return index


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    index = make(Path(args.out), args.seed)
    print(json.dumps({"configs": [c["config_id"] for c in index["configs"]], "n_instances": sum(c["n_instances"] for c in index["configs"])}))


if __name__ == "__main__":
    main()
