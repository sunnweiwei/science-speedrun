#!/usr/bin/env python3
"""TimesFM-2.5 zero-shot reference package for timeseries-forecast/v1.

Replays the GIFT-Eval leaderboard row `TimesFM-2.5` (gift-eval
`notebooks/timesfm2p5.ipynb`): the released checkpoint
`google/timesfm-2.5-200m-pytorch` run zero-shot through
`timesfm.timesfm_2p5.timesfm_2p5_torch.TimesFM_2p5_200M_torch` with the
notebook's `ForecastConfig`.  Multivariate configs are flattened to
independent univariate series, exactly as the notebook does with
`to_univariate=True`.

    bash infer.sh INPUT_DIR OUTPUT_DIR

Environment overrides (all optional):
    TIMESFM25_CHECKPOINT   local checkpoint dir (default: TIMESERIES_PRETRAINED_DIR/timesfm-2.5-200m-pytorch)
    TIMESERIES_PRETRAINED_DIR      where the checkpoint is downloaded (default: /data/pretrained)
    TIMESFM25_TOKEN_BUDGET batch-size budget in (series * context) elements, default 196608
    TIMESFM25_MAX_BATCH    hard cap on the per-core batch size, default 128 (the notebook value)
    TIMESFM25_TORCH_COMPILE  1 to enable torch.compile (default 0, see README)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for cand in (HERE.parent.parent / "evaluation", Path("/workspace/evaluation"), HERE):
    if (cand / "ts_io.py").exists():
        sys.path.insert(0, str(cand))
        break
import ts_io  # noqa: E402

REPO_ID = "google/timesfm-2.5-200m-pytorch"
REVISION = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
TIMESFM_GIT = "https://github.com/google-research/timesfm"
TIMESFM_COMMIT = "e31dadd84cb26bd5153fde6687502b8312e918fb"  # timesfm 3.0.2

# --- notebook settings (gift-eval/notebooks/timesfm2p5.ipynb) -------------------
PREDICTOR_CHUNK = 1024      # the notebook's `predict(..., batch_size=1024)`
MAX_CONTEXT_CAP = 15360     # `max_context=min(15360, patch-rounded max length of the chunk)`
MAX_HORIZON = 1024          # the notebook's value
# `TimesFM_2p5.decode` is autoregressive over output patches of 128 steps and the
# continuous-quantile head reads a fixed 1024-step quantile spread, so the first H
# forecast steps do not depend on max_horizon: rounding it up from H instead of
# fixing it at 1024 is bit-identical and skips up to 7 AR patches per series.
# Set TIMESFM25_MAX_HORIZON=1024 to run the notebook's literal setting.
MAX_HORIZON_MODE = os.environ.get("TIMESFM25_MAX_HORIZON", "auto")
NOTEBOOK_PER_CORE_BATCH = 128
FORECAST_FLAGS = dict(
    infer_is_positive=True,
    use_continuous_quantile_head=True,
    fix_quantile_crossing=True,
    force_flip_invariance=True,
    return_backcast=False,
    normalize_inputs=True,
)

TOKEN_BUDGET = int(os.environ.get("TIMESFM25_TOKEN_BUDGET", 196608))
MAX_BATCH = int(os.environ.get("TIMESFM25_MAX_BATCH", NOTEBOOK_PER_CORE_BATCH))
TORCH_COMPILE = os.environ.get("TIMESFM25_TORCH_COMPILE", "0") == "1"


def pretrained_root() -> Path:
    """Where maintainers stage external-target weights: $TIMESERIES_PRETRAINED_DIR, else /data/pretrained."""
    return Path(os.environ.get("TIMESERIES_PRETRAINED_DIR", "/data/pretrained"))


def checkpoint_dir() -> Path:
    env = os.environ.get("TIMESFM25_CHECKPOINT")
    if env:
        return Path(env)
    root = pretrained_root()
    dst = root / "timesfm-2.5-200m-pytorch"
    if not (dst / "model.safetensors").exists():
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=REPO_ID, revision=REVISION, local_dir=str(dst))
    return dst


def load_model():
    import torch
    from timesfm.timesfm_2p5 import timesfm_2p5_torch

    tfm = timesfm_2p5_torch.TimesFM_2p5_200M_torch(torch_compile=TORCH_COMPILE)
    tfm.load_checkpoint(str(checkpoint_dir()))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[timesfm-2.5] loaded on {dev} (torch_compile={TORCH_COMPILE})", flush=True)
    return tfm


def per_core_batch(max_context: int) -> int:
    """Chunking only: `TimesFM_2p5.forecast` pads/masks every series to `max_context`
    independently of the batch size, so this never changes the numbers."""
    return int(np.clip(TOKEN_BUDGET // max(max_context, 1), 1, MAX_BATCH))


def forecast_chunk(tfm, series: list[np.ndarray], horizon: int) -> np.ndarray:
    """-> (n, horizon, 10): [mean, q0.1 .. q0.9], as `TimesFM_2p5.forecast` returns."""
    import torch
    from timesfm import configs

    patch = tfm.model.p
    if MAX_HORIZON_MODE == "auto":
        max_horizon = int(np.ceil(horizon / tfm.model.o) * tfm.model.o)
    else:
        max_horizon = int(MAX_HORIZON_MODE)
    max_len = max(len(s) for s in series)
    max_context = min(MAX_CONTEXT_CAP, max(patch, math.ceil(max_len / patch) * patch))
    batch = per_core_batch(max_context)
    while True:
        try:
            tfm.compile(
                forecast_config=configs.ForecastConfig(
                    max_context=max_context, max_horizon=max_horizon,
                    per_core_batch_size=batch, **FORECAST_FLAGS,
                )
            )
            _, full = tfm.forecast(horizon=horizon, inputs=list(series))
            return np.asarray(full)
        except torch.cuda.OutOfMemoryError:
            if batch <= 1:
                raise
            batch = max(1, batch // 2)
            torch.cuda.empty_cache()
            print(f"  [oom] retrying with per_core_batch_size={batch}", flush=True)


def run_config(tfm, case: ts_io.CaseFile, out_path: Path):
    H, D, N = case.prediction_length, case.n_variates, case.n
    mean = np.zeros((N, D, H), np.float32)
    quant = np.zeros((N, D, 9, H), np.float32)
    flat = [(i, d) for i in range(N) for d in range(D)]  # notebook's to_univariate flattening
    pos = 0
    while pos < len(flat):
        keys = flat[pos:pos + PREDICTOR_CHUNK]
        series, last = [], []
        cache_i, cache_ctx = -1, None
        for i, d in keys:
            if i != cache_i:
                cache_i, cache_ctx = i, case.context(i)
            x = np.asarray(cache_ctx[d], dtype=np.float32)
            fin = x[np.isfinite(x)]
            if fin.size == 0:
                # An all-NaN context crashes `TimesFM_2p5.forecast` (its
                # `linear_interpolation` guard `if non_nans_values:` raises on an
                # empty numpy array). TimesFM-3's forecaster zeroes such a series,
                # so do the same here. 649 of the 371,330 univariate series of the
                # test split are affected (bitbrains_* variates and one
                # electricity/W series); their labels are NaN and masked anyway.
                x = np.zeros_like(x)
            series.append(x)
            last.append(float(fin[-1]) if fin.size else 0.0)
        full = forecast_chunk(tfm, series, H)
        for k, (i, d) in enumerate(keys):
            q = full[k, :H, 1:].T                      # (9, H) -> q0.1 .. q0.9
            m = full[k, :H, 0]                         # the model's mean channel
            if not np.isfinite(q).all():
                q = np.where(np.isfinite(q), q, last[k])
            if not np.isfinite(m).all():
                m = np.where(np.isfinite(m), m, q[4])
            quant[i, d] = q
            mean[i, d] = m
        pos += len(keys)
    ts_io.write_prediction_file(out_path, mean, quant)
    return N * D


def main():
    input_dir, output_dir = Path(sys.argv[1]), Path(sys.argv[2])
    index = json.loads((input_dir / "index.json").read_text())
    (output_dir / "cases").mkdir(parents=True, exist_ok=True)
    tfm = load_model()
    t_all = time.time()
    n_series_total = 0
    for k, cfg in enumerate(index["configs"], 1):
        cid = cfg["config_id"]
        out_path = output_dir / "cases" / f"{cid}.h5"
        if out_path.exists() and os.environ.get("TIMESERIES_RESUME", "1") == "1":
            print(f"[{k}/{len(index['configs'])}] {cid}: already done", flush=True)
            continue
        t0 = time.time()
        case = ts_io.CaseFile(input_dir / "cases" / f"{cid}.h5")
        tmp = out_path.with_suffix(".h5.part")
        n = run_config(tfm, case, tmp)
        case.close()
        tmp.replace(out_path)
        n_series_total += n
        print(f"[{k}/{len(index['configs'])}] {cid}: {n} univariate series in {time.time()-t0:.1f}s", flush=True)
    manifest = json.loads((HERE / "model_manifest.json").read_text())
    manifest["runtime"] = {"wall_clock_s": round(time.time() - t_all, 1), "univariate_series": n_series_total}
    (output_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"total {time.time()-t_all:.1f}s for {n_series_total} univariate series", flush=True)


if __name__ == "__main__":
    main()
