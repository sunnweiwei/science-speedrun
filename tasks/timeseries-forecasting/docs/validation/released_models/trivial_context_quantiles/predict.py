#!/usr/bin/env python3
"""Trivial control for timeseries-forecast/v1: forecast the context's own distribution, ignoring time.

The point forecast is the mean of the instance's finite context values and the nine quantiles are the
context's empirical deciles, repeated for every horizon step. It uses no temporal structure at all, so its
score is the floor a package must clear for the metric to be measuring forecasting rather than plumbing
(`CONTRIBUTING.md` asks for a random or constant control alongside the trained starter).

    bash infer.sh INPUT_DIR OUTPUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for cand in (HERE.parent.parent / "evaluation", Path("/workspace/evaluation"), HERE):
    if (cand / "ts_io.py").exists():
        sys.path.insert(0, str(cand)); break
import ts_io  # noqa: E402


def run_config(case: ts_io.CaseFile, out_path: Path):
    n, d, h = case.n, case.n_variates, case.prediction_length
    mean = np.zeros((n, d, h), np.float32)
    quant = np.zeros((n, d, len(ts_io.QUANTILE_LEVELS), h), np.float32)
    for i, ctx in enumerate(case.contexts()):
        for v in range(d):
            x = ctx[v][np.isfinite(ctx[v])]
            if x.size == 0:
                continue
            mean[i, v] = float(x.mean())
            quant[i, v] = np.quantile(x.astype(np.float64), ts_io.QUANTILE_LEVELS)[:, None]
    ts_io.write_prediction_file(out_path, mean, quant, {"model": "trivial_context_quantiles"})


def main():
    inp, outp = Path(sys.argv[1]), Path(sys.argv[2])
    index = json.loads((inp / "index.json").read_text())
    for cfg in index["configs"]:
        case = ts_io.CaseFile(inp / "cases" / f"{cfg['config_id']}.h5")
        run_config(case, outp / "cases" / f"{cfg['config_id']}.h5")
        case.close()
    (outp / "model_manifest.json").write_text((HERE / "model_manifest.json").read_text())


if __name__ == "__main__":
    main()
