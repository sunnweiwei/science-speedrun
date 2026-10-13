#!/usr/bin/env python3
"""Golden check of the task grader against gluonts 0.15.1 + gift-eval on the same forecasts.

Runs in the pinned reference environment (gluonts 0.15.1, datasets 2.17.1,
pandas 2.2.3, gift-eval @ db743860): for every staged config it rebuilds the
official `Dataset(name, term, to_univariate=D>1).test_data`, wraps the
prediction file of a package as gluonts QuantileForecast objects, calls
`evaluate_model(..., axis=None, mask_invalid_label=True, seasonality=...)`
exactly as the GIFT-Eval notebooks do, and compares MASE[0.5] and
mean_weighted_sum_quantile_loss with the values in the task grader's report.

    GIFT_EVAL=/data/gifteval python3 golden_gluonts_check.py --input INPUT --output OUTPUT_DIR --report REPORT.json --out golden.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "environment" / "evaluation"))
import ts_io  # noqa: E402


class FileForecasts:
    """QuantileForecast per gluonts test instance, read from a timeseries-forecast/v1 prediction file."""

    def __init__(self, case: ts_io.CaseFile, pred_path: Path, univariate_entries: bool):
        from gluonts.model.forecast import QuantileForecast
        import pandas as pd

        mean, quant = ts_io.read_prediction_file(pred_path)
        self.forecasts = []
        keys = [str(q) for q in ts_io.QUANTILE_LEVELS] + ["mean"]
        n, d = mean.shape[0], mean.shape[1]
        n_series = n // max(1, int(case.meta["windows"]))
        order = []
        if univariate_entries:  # gluonts MultivariateToUnivariate: item-major, dim, then windows
            for s in range(n_series):
                for v in range(d):
                    for k in range(int(case.meta["windows"])):
                        order.append((s * int(case.meta["windows"]) + k, v))
        else:
            order = [(i, None) for i in range(n)]
        for i, v in order:
            if v is None and d == 1:
                arr = np.concatenate([quant[i, 0], mean[i, 0][None]], axis=0)  # univariate entry: (10, H)
            elif v is None:
                arr = np.concatenate([quant[i].transpose(1, 0, 2), mean[i][None]], axis=0)  # (10, D, H)
                arr = arr.transpose(0, 2, 1)  # gluonts multivariate: (num_keys, H, D)
            else:
                arr = np.concatenate([quant[i, v], mean[i, v][None]], axis=0)  # (10, H)
            self.forecasts.append(QuantileForecast(forecast_arrays=arr, forecast_keys=keys,
                                                   start_date=pd.Period(case.forecast_start[i], freq=ts_io._pandas_freq(case.meta["freq"])),
                                                   item_id=case.item_id[i]))

    def __iter__(self):
        return iter(self.forecasts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--output", required=True); ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--configs", default="")
    args = ap.parse_args()
    from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
    from gluonts.model import evaluate_forecasts
    from gift_eval.data import Dataset

    report = json.loads(Path(args.report).read_text())
    ours = {c["config_id"]: c for c in report["configs"]}
    index = json.loads((Path(args.input) / "index.json").read_text())
    rows, worst = [], 0.0
    for cfg in index["configs"]:
        if args.configs and cfg["config_id"] not in args.configs.split(","):
            continue
        case = ts_io.CaseFile(Path(args.input) / "cases" / f"{cfg['config_id']}.h5")
        ds0 = Dataset(name=cfg["dataset"], term=cfg["term"], to_univariate=False)
        univariate = ds0.target_dim > 1
        ds = Dataset(name=cfg["dataset"], term=cfg["term"], to_univariate=univariate) if univariate else ds0
        assert ds.prediction_length == cfg["prediction_length"] and ds.windows == int(case.meta["windows"]), (cfg["config_id"], ds.prediction_length, ds.windows)
        fc = FileForecasts(case, Path(args.output) / "cases" / f"{cfg['config_id']}.h5", univariate)
        res = evaluate_forecasts(fc, test_data=ds.test_data, metrics=[MASE(), MeanWeightedSumQuantileLoss(quantile_levels=list(ts_io.QUANTILE_LEVELS))],
                                 axis=None, batch_size=512, mask_invalid_label=True, allow_nan_forecast=False, seasonality=cfg["seasonality"])
        g_mase, g_wql = float(res["MASE[0.5]"].iloc[0]), float(res["mean_weighted_sum_quantile_loss"].iloc[0])
        o = ours[cfg["config_id"]]["metrics"]
        dm, dw = abs(g_mase - o["mase"]) / g_mase, abs(g_wql - o["wql"]) / g_wql
        worst = max(worst, dm, dw)
        rows.append({"config_id": cfg["config_id"], "gluonts_mase": g_mase, "task_mase": o["mase"], "gluonts_wql": g_wql, "task_wql": o["wql"],
                     "rel_diff_mase": dm, "rel_diff_wql": dw, "n_forecasts": len(fc.forecasts), "univariate_entries": univariate})
        print(f"  {cfg['config_id']:44s} MASE {g_mase:.6f} vs {o['mase']:.6f} ({dm:.1e})  WQL {g_wql:.6f} vs {o['wql']:.6f} ({dw:.1e})", file=sys.stderr)
        case.close()
    out = {"gluonts_version": __import__("gluonts").__version__, "n_configs": len(rows), "max_rel_diff": worst, "rows": rows}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"n_configs": len(rows), "max_rel_diff": worst}))


if __name__ == "__main__":
    main()
