#!/usr/bin/env python3
"""Chronos-Bolt (base) zero-shot reference package for timeseries-forecast/v1.

Replays the leaderboard row `chronos_bolt_base` of GIFT-Eval, produced by
gift-eval `notebooks/chronos.ipynb`:

  * `BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base", device_map="cuda:0")`
    -> `ChronosBoltPipeline` (`ForecastType.QUANTILES`, native quantiles
    [0.1 .. 0.9], model context 2048, native horizon 64);
  * `pipeline.predict_quantiles(context, prediction_length=H,
    quantile_levels=[0.1 .. 0.9])`, which is exactly the notebook's
    `pipeline.predict(...)` plus the identity quantile lookup (the nine
    requested levels are the nine the model was trained on), and which
    returns the 0.5 quantile as the point forecast -- the same thing gluonts
    does for the notebook's `QuantileForecast` (it has no `mean`, so
    `Forecast.mean` falls back to the median);
  * `limit_prediction_length` is left at its default `False`, so horizons
    longer than 64 are produced by the pipeline's own autoregressive
    long-horizon heuristic (quantile paths fed back as context);
  * multivariate configs are flattened to independent univariate series
    (the notebook's `to_univariate = target_dim > 1`);
  * NaNs in the history are passed straight through -- Chronos-Bolt treats
    them as missing observations.

The checkpoint is downloaded at run time with `huggingface_hub` at a pinned
revision; no weights live in this repository.

    bash infer.sh INPUT_DIR OUTPUT_DIR
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for cand in (HERE.parent.parent / "evaluation", Path("/workspace/evaluation"), HERE):
    if (cand / "ts_io.py").exists():
        sys.path.insert(0, str(cand))
        break
import ts_io  # noqa: E402

MODEL_REPO = "amazon/chronos-bolt-base"
MODEL_REVISION = "5d9f166d69f47aef3401367a7b842e78fe97b121"
MODEL_LICENSE = "Apache-2.0"
CHRONOS_VERSION = "2.3.2"  # pip install chronos-forecasting==2.3.2
DEVICE_MAP = os.environ.get("CHRONOS_DEVICE", "cuda:0")
# Numerically inert for Chronos-Bolt: series never interact inside a batch (the
# left NaN padding is masked), so this only trades memory for speed. The
# notebook uses 1024; the default here is smaller because the GPU is shared.
BATCH_SIZE = int(os.environ.get("CHRONOS_BATCH_SIZE", "256"))
QUANTILE_LEVELS = list(ts_io.QUANTILE_LEVELS)


def pretrained_root() -> Path:
    """Where maintainers stage external-target weights: $TIMESERIES_PRETRAINED_DIR, else /data/pretrained."""
    return Path(os.environ.get("TIMESERIES_PRETRAINED_DIR", "/data/pretrained"))


def model_dir() -> str:
    """Download `MODEL_REPO` at `MODEL_REVISION` (cached) and return the local path."""
    local = os.environ.get("CHRONOS_BOLT_DIR") or str(pretrained_root() / "chronos-bolt-base")
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=MODEL_REPO, revision=MODEL_REVISION, local_dir=local)


def load_pipeline():
    from chronos import BaseChronosPipeline, ChronosBoltPipeline

    pipe = BaseChronosPipeline.from_pretrained(model_dir(), device_map=DEVICE_MAP)
    assert isinstance(pipe, ChronosBoltPipeline), f"expected ChronosBoltPipeline, got {type(pipe)}"
    assert list(pipe.quantiles) == QUANTILE_LEVELS, f"unexpected native quantiles {pipe.quantiles}"
    return pipe


def predict_batched(pipe, series: list[torch.Tensor], horizon: int, batch_size: int):
    """`predict_quantiles` over `series` in chunks; returns (M, 9, H) and (M, H) float32."""
    quants, means = [], []
    i, bs = 0, batch_size
    while i < len(series):
        chunk = series[i:i + bs]
        try:
            q, m = pipe.predict_quantiles(chunk, prediction_length=horizon, quantile_levels=QUANTILE_LEVELS)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"    OOM -> batch_size {bs}", file=sys.stderr, flush=True)
            continue
        quants.append(q.numpy().transpose(0, 2, 1).astype(np.float32))  # (B, H, 9) -> (B, 9, H)
        means.append(m.numpy().astype(np.float32))
        i += len(chunk)
    return np.concatenate(quants), np.concatenate(means)


def run_config(pipe, case: ts_io.CaseFile, out_path: Path) -> dict:
    n, d, h = case.n, case.n_variates, case.prediction_length
    ctx_len = pipe.model_context_length
    # Flatten to univariate series, oldest-first, truncated to the model context.
    # The pipeline left-pads a batch to its longest member and then keeps the last
    # `model_context_length` steps, so pre-truncating each series is a no-op on the
    # result and only saves host memory.
    series: list[torch.Tensor] = []
    for i in range(n):
        ctx = case.context(i)
        for v in range(d):
            series.append(torch.from_numpy(np.ascontiguousarray(ctx[v, -ctx_len:])))
    # The long-horizon heuristic multiplies the effective batch by len(quantiles).
    bs = BATCH_SIZE if h <= pipe.model_prediction_length else max(1, BATCH_SIZE // len(QUANTILE_LEVELS))
    q, m = predict_batched(pipe, series, h, bs)
    quantiles = q.reshape(n, d, len(QUANTILE_LEVELS), h)
    mean = m.reshape(n, d, h)
    if not (np.isfinite(quantiles).all() and np.isfinite(mean).all()):
        raise RuntimeError(f"{case.meta['config_id']}: non-finite forecast")
    ts_io.write_prediction_file(out_path, mean, quantiles, {"model": "chronos_bolt_base"})
    return {"n_series_forecast": len(series), "batch_size": bs}


def manifest() -> dict:
    return {
        "name": "chronos_bolt_base_zero_shot",
        "contract": ts_io.CONTRACT,
        "parameters": 205292928,
        "pretrained_assets": [{"repo": MODEL_REPO, "revision": MODEL_REVISION, "license": MODEL_LICENSE}],
        "sources_used": [],
        "settings": {
            "library": f"chronos-forecasting=={CHRONOS_VERSION}",
            "pipeline": "chronos.ChronosBoltPipeline via BaseChronosPipeline.from_pretrained",
            "device_map": DEVICE_MAP,
            "torch_dtype": "float32 (checkpoint default)",
            "call": "predict_quantiles(context, prediction_length=H, quantile_levels=[0.1..0.9])",
            "quantile_levels": QUANTILE_LEVELS,
            "point_forecast": "0.5 quantile (what gluonts uses as the mean of a QuantileForecast)",
            "model_context_length": 2048,
            "model_prediction_length": 64,
            "limit_prediction_length": False,
            "long_horizon": "pipeline autoregressive quantile-path heuristic for H > 64",
            "multivariate": "to_univariate: each variate forecast independently",
            "nan_handling": "passed through to the model as missing observations",
            "batch_size": BATCH_SIZE,
            "num_samples": None,
        },
    }


def main():
    inp, outp = Path(sys.argv[1]), Path(sys.argv[2])
    index = json.loads((inp / "index.json").read_text())
    pipe = load_pipeline()
    outp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for k, cfg in enumerate(index["configs"], 1):
        t = time.time()
        case = ts_io.CaseFile(inp / "cases" / f"{cfg['config_id']}.h5")
        info = run_config(pipe, case, outp / "cases" / f"{cfg['config_id']}.h5")
        case.close()
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        print(f"  [{k}/{len(index['configs'])}] {cfg['config_id']}: {cfg['n_instances']} instances "
              f"-> {info['n_series_forecast']} univariate series, bs={info['batch_size']}, "
              f"{time.time() - t:.1f}s, peak {peak:.2f} GiB", file=sys.stderr, flush=True)
    man = manifest()
    man["runtime_seconds"] = round(time.time() - t0, 1)
    if torch.cuda.is_available():
        man["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    (outp / "model_manifest.json").write_text(json.dumps(man, indent=1) + "\n")


if __name__ == "__main__":
    main()
