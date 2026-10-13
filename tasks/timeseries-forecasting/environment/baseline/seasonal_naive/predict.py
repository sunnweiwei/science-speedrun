#!/usr/bin/env python3
"""Seasonal Naive reference package for timeseries-forecast/v1 (numpy re-implementation of the leaderboard's model).

The leaderboard's `Seasonal_Naive` row (gift-eval notebooks/naive.ipynb) is
statsforecast `SeasonalNaive(season_length=m)` with prediction intervals at
levels {20, 40, 60, 80} mapped onto the nine deciles, on the history with
NaN filled by gluonts `LastValueImputation`. Its formulas (statsforecast
2.1.1, utils._seasonal_naive / _calculate_sigma / _calculate_intervals):

    mean[k]   = y[n - m + (k mod m)]                       (last season tiled)
    sigma     = sqrt( sum_{t>=m} (y[t] - y[t-m])^2 / (n - m) )
    sigma_h[k]= sigma * sqrt(floor(k / m) + 1)
    q_level   = mean +- z(level) * sigma_h,  z(level) = Phi^-1(0.5 + level/200)
    decile q  -> level round(200 (max(q, 1-q) - 0.5)); q = 0.5 -> mean

`predict_statsforecast.py` is the original statsforecast-backed version; both
give identical files on the smoke split (docs/validation/). This version runs the
full test split in about a minute. Because packages only see the last
16,384 observations (contract.md), sigma is computed on that window; the
leaderboard used the full history.

    bash infer.sh INPUT_DIR OUTPUT_DIR
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for cand in (HERE.parent.parent / "evaluation", Path("/workspace/evaluation"), HERE):
    if (cand / "ts_io.py").exists():
        sys.path.insert(0, str(cand)); break
import ts_io  # noqa: E402


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation refined with one Newton step)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p)); x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p > 1 - 0.02425:
        q = math.sqrt(-2 * math.log(1 - p)); x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    else:
        q = p - 0.5; r = q * q
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    e = 0.5 * math.erfc(-x / math.sqrt(2)) - p
    return x - e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)


Z = {q: (0.0 if q == 0.5 else norm_ppf(0.5 + round(200 * (max(q, 1 - q) - 0.5)) / 200) * (1 if q > 0.5 else -1)) for q in ts_io.QUANTILE_LEVELS}


def last_value_impute(x: np.ndarray) -> np.ndarray:
    """gluonts LastValueImputation on a (R, T) array: forward-fill; leading NaNs take the first finite value; all-NaN rows -> 0."""
    x = x.astype(np.float32).copy()
    finite = np.isfinite(x)
    idx = np.where(finite, np.arange(x.shape[1])[None, :], -1)
    idx = np.maximum.accumulate(idx, axis=1)
    first = np.argmax(finite, axis=1)
    out = np.where(idx >= 0, np.take_along_axis(x, np.maximum(idx, 0), axis=1), x[np.arange(len(x)), first][:, None])
    out[~finite.any(axis=1)] = 0.0
    return out


def seasonal_naive(y: np.ndarray, m: int, h: int):
    """y (R, T) imputed float32 -> mean (R, h), sigma_h (R, h) as float64 (statsforecast computes in the input dtype; float32 here)."""
    y = y.astype(np.float32)
    n = y.shape[1]
    season = np.full((len(y), m), np.nan, np.float32)
    avail = min(m, n)
    if n < m:
        season[:, m - avail:] = y[:, -avail:]
    else:
        season[:, :avail] = y[:, -avail:]
    mean = np.tile(season, (1, math.ceil(h / m)))[:, :h]
    if n > m:
        res = y[:, m:] - y[:, :-m]
        sigma = np.sqrt(np.nansum(res.astype(np.float32) ** 2, axis=1) / (n - m))
    else:
        sigma = np.zeros(len(y), np.float32)
    k = np.floor(np.arange(h) / m)
    sigma_h = sigma[:, None] * np.sqrt(k + 1)[None, :]
    return mean, sigma_h


def run_config(case: ts_io.CaseFile, out_path: Path, chunk_elems: int = 50_000_000):
    """Process instances in chunks (rows x max length <= chunk_elems floats) so memory stays flat on the large configs."""
    L, m, D = case.prediction_length, int(case.meta["seasonality"]), case.n_variates
    mean = np.zeros((case.n * D, L), np.float32); quant = np.zeros((case.n * D, len(ts_io.QUANTILE_LEVELS), L), np.float32)
    order = np.argsort(case.context_length, kind="stable")  # similar lengths together -> less padding
    T_max = int(case.context_length.max())
    rows_per_chunk = max(1, chunk_elems // (T_max * D))
    values = case._f["values"]
    for s in range(0, case.n, rows_per_chunk):
        inst = order[s:s + rows_per_chunk]
        T = int(case.context_length[inst].max())
        rows = np.full((len(inst) * D, T), np.nan, np.float32)
        for r, i in enumerate(inst):
            ctx = values[case.offsets[i]:case.offsets[i + 1]].reshape(D, -1)
            rows[r * D:(r + 1) * D, T - ctx.shape[1]:] = ctx
        y = last_value_impute(rows)
        # rows were left-padded with NaN to a common length and the imputation turned that padding into the first
        # value, which statsforecast never saw: compute per distinct length so sigma uses exactly the given history.
        lengths = np.repeat(case.context_length[inst], D)
        flat = np.repeat(inst, D) * D + np.tile(np.arange(D), len(inst))
        for length in np.unique(lengths):
            sel = np.nonzero(lengths == length)[0]
            mu, sh = seasonal_naive(y[sel, T - length:], m, L)
            mu = np.where(np.isfinite(mu), mu, y[sel, -1:].repeat(L, axis=1))  # history shorter than a season: last value
            mean[flat[sel]] = mu
            quant[flat[sel]] = np.stack([mu + Z[q] * sh for q in ts_io.QUANTILE_LEVELS], axis=1).astype(np.float32)
    ts_io.write_prediction_file(out_path, mean.reshape(case.n, D, L), quant.reshape(case.n, D, len(ts_io.QUANTILE_LEVELS), L), {"model": "seasonal_naive_reference"})


def main():
    inp, outp = Path(sys.argv[1]), Path(sys.argv[2])
    index = json.loads((inp / "index.json").read_text())
    for cfg in index["configs"]:
        case = ts_io.CaseFile(inp / "cases" / f"{cfg['config_id']}.h5")
        run_config(case, outp / "cases" / f"{cfg['config_id']}.h5")
        case.close()
        print(f"  {cfg['config_id']}: {cfg['n_instances']} instances", file=sys.stderr)
    (outp / "model_manifest.json").write_text((HERE / "model_manifest.json").read_text())


if __name__ == "__main__":
    main()
