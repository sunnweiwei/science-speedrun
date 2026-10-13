#!/usr/bin/env python3
"""Seasonal Naive reference package for timeseries-forecast/v1.

Reproduces the leaderboard's `Seasonal_Naive` row (gift-eval notebooks/naive.ipynb):
statsforecast `SeasonalNaive(season_length=m)` with prediction intervals at
levels {20, 40, 60, 80} mapped to the nine deciles, NaN history imputed with
the last observed value (gluonts `LastValueImputation`), one model per
univariate series (multivariate configs are flattened per variate). Its
per-config MASE / WQL through the task grader define the normalizer
`evaluation/seasonal_naive_reference.json`.

    bash infer.sh INPUT_DIR OUTPUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

HERE = Path(__file__).resolve().parent
for cand in (HERE.parent.parent / "evaluation", Path("/workspace/evaluation"), HERE):
    if (cand / "ts_io.py").exists():
        sys.path.insert(0, str(cand)); break
import ts_io  # noqa: E402

LEVELS = [20, 40, 60, 80]


def last_value_impute(x: np.ndarray) -> np.ndarray:
    """gluonts LastValueImputation: forward-fill, with leading NaNs replaced by the first finite value (or 0)."""
    x = x.astype(np.float32).copy()
    if not np.isnan(x).any():
        return x
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    idx = np.where(finite, np.arange(len(x)), -1)
    np.maximum.accumulate(idx, out=idx)
    first = np.argmax(finite)
    filled = np.where(idx >= 0, x[np.maximum(idx, 0)], x[first])
    return filled.astype(np.float32)


def quantile_columns(level_cols, prediction_length):
    """Map statsforecast lo/hi interval columns to the nine deciles in order (gluonts StatsForecastPredictor)."""
    out = np.zeros((len(ts_io.QUANTILE_LEVELS), prediction_length), np.float64)
    for j, q in enumerate(ts_io.QUANTILE_LEVELS):
        interval = round(200 * (max(q, 1 - q) - 0.5))
        side = "hi" if q > 0.5 else "lo"
        out[j] = level_cols["mean"] if interval == 0 else level_cols[f"{side}-{interval}"]
    return out


def run_config(case: ts_io.CaseFile, out_path: Path):
    L, m, D = case.prediction_length, int(case.meta["seasonality"]), case.n_variates
    freq = ts_io._pandas_freq(case.meta["freq"])
    frames, keys = [], []
    for i, ctx in enumerate(case.contexts()):
        for d in range(D):
            y = last_value_impute(ctx[d])
            uid = f"{i}_{d}"
            frames.append(pd.DataFrame({"unique_id": uid, "ds": pd.date_range(end=pd.Timestamp(case.forecast_start[i]), periods=len(y) + 1, freq=freq)[:-1].to_numpy(), "y": y}))
            keys.append((i, d))
    sf = StatsForecast(models=[SeasonalNaive(season_length=m)], freq=freq, n_jobs=1)
    res = sf.forecast(df=pd.concat(frames, ignore_index=True), h=L, level=LEVELS)
    if "unique_id" not in res.columns:
        res = res.reset_index()
    cols = [c for c in res.columns if c.startswith("SeasonalNaive")]
    mean = np.zeros((case.n, D, L), np.float32); quant = np.zeros((case.n, D, len(ts_io.QUANTILE_LEVELS), L), np.float32)
    grouped = {uid: g for uid, g in res.groupby("unique_id", sort=False)}
    for (i, d) in keys:
        g = grouped[f"{i}_{d}"]
        assert len(g) == L
        level_cols = {"mean": g["SeasonalNaive"].to_numpy(np.float64)}
        for c in cols:
            if "-" in c:
                level_cols[c.split("-", 1)[1]] = g[c].to_numpy(np.float64)
        mean[i, d] = level_cols["mean"]
        quant[i, d] = quantile_columns(level_cols, L)
    ts_io.write_prediction_file(out_path, mean, quant, {"model": "seasonal_naive_reference"})


def main():
    inp, outp = Path(sys.argv[1]), Path(sys.argv[2])
    index = json.loads((inp / "index.json").read_text())
    for cfg in index["configs"]:
        case = ts_io.CaseFile(inp / "cases" / f"{cfg['config_id']}.h5")
        run_config(case, outp / "cases" / f"{cfg['config_id']}.h5")
        case.close()
        print(f"  {cfg['config_id']}: {cfg['n_instances']} instances", file=sys.stderr)
    (outp / "model_manifest.json").write_text(json.dumps({"name": "seasonal_naive_reference", "contract": ts_io.CONTRACT,
                                                          "parameters": 0, "pretrained_assets": [], "sources_used": []}, indent=1))


if __name__ == "__main__":
    main()
