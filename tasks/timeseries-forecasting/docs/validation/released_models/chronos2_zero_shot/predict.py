#!/usr/bin/env python3
"""Chronos-2 zero-shot reference package for timeseries-forecast/v1.

Replays the leaderboard row `Chronos-2` of GIFT-Eval, produced by gift-eval
`notebooks/chronos-2.ipynb`:

  * `BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda",
    torch_dtype="float32")` -> `Chronos2Pipeline` (model context 8192, native
    horizon 64 output patches x 16 = 1024 steps, so every GIFT-Eval horizon
    (<= 720) is produced in one forward pass);
  * `predict_quantiles(inputs=[{"target": ...}], prediction_length=H,
    batch_size=100, quantile_levels=[0.1 .. 0.9], predict_batches_jointly=True)`.
    `predict_batches_jointly` is the deprecated spelling of `cross_learning`
    in chronos-forecasting >= 2.2; this package passes `cross_learning=True`,
    which is the same code path (all items of a batch get group id 0, so the
    model attends across them -- in-context learning across series);
  * because cross-learning makes a forecast depend on the other members of its
    batch, the notebook predicts **one rolling window at a time**
    (`itertools.islice(test_data.input, window_idx, None, n_windows)`) to avoid
    leakage between windows of the same series. The staged case files are
    ordered item-major / window-minor exactly like `test_data.input`, so
    selecting `window == k` in file order reproduces the notebook's batches
    item for item;
  * multivariate configs are passed to the model **natively** as (D, T)
    targets (`use_multivariate_data=True` in the notebook), not flattened;
  * the point forecast is the 0.5 quantile -- the notebook builds a gluonts
    `QuantileForecast`, whose `mean` falls back to the median;
  * NaNs in the history are passed straight through; Chronos-2 handles missing
    values natively.

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

MODEL_REPO = "amazon/chronos-2"
MODEL_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
MODEL_LICENSE = "Apache-2.0"
CHRONOS_VERSION = "2.3.2"  # pip install chronos-forecasting==2.3.2 (Chronos-2 needs >= 2.1)
DEVICE_MAP = os.environ.get("CHRONOS_DEVICE", "cuda")
BATCH_SIZE = int(os.environ.get("CHRONOS_BATCH_SIZE", "100"))  # the notebook's batch_size
CROSS_LEARNING = True  # notebook: predict_batches_jointly=True
QUANTILE_LEVELS = list(ts_io.QUANTILE_LEVELS)


def pretrained_root() -> Path:
    """Where maintainers stage external-target weights: $TIMESERIES_PRETRAINED_DIR, else /data/pretrained."""
    return Path(os.environ.get("TIMESERIES_PRETRAINED_DIR", "/data/pretrained"))


def model_dir() -> str:
    """Download `MODEL_REPO` at `MODEL_REVISION` (cached) and return the local path."""
    local = os.environ.get("CHRONOS2_DIR") or str(pretrained_root() / "chronos-2")
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=MODEL_REPO, revision=MODEL_REVISION, local_dir=local)


def load_pipeline():
    from chronos import BaseChronosPipeline, Chronos2Pipeline

    pipe = BaseChronosPipeline.from_pretrained(model_dir(), device_map=DEVICE_MAP, torch_dtype="float32")
    assert isinstance(pipe, Chronos2Pipeline), f"expected Chronos2Pipeline, got {type(pipe)}"
    return pipe


def run_config(pipe, case: ts_io.CaseFile, out_path: Path) -> dict:
    n, d, h = case.n, case.n_variates, case.prediction_length
    ctx_len = pipe.model_context_length  # 8192; the pipeline would truncate anyway
    quantiles = np.empty((n, d, len(QUANTILE_LEVELS), h), np.float32)
    mean = np.empty((n, d, h), np.float32)
    windows = np.asarray(case.window)
    bs_used = BATCH_SIZE
    for w in np.unique(windows):
        idx = np.flatnonzero(windows == w)  # file order == gluonts item order
        inputs = []
        for i in idx:
            ctx = np.ascontiguousarray(case.context(int(i))[:, -ctx_len:])
            inputs.append({"target": ctx if d > 1 else ctx[0]})
        bs = bs_used
        while True:
            try:
                q, m = pipe.predict_quantiles(
                    inputs=inputs,
                    prediction_length=h,
                    batch_size=bs,
                    quantile_levels=QUANTILE_LEVELS,
                    cross_learning=CROSS_LEARNING,
                )
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                print(f"    OOM -> batch_size {bs}", file=sys.stderr, flush=True)
        bs_used = bs
        assert len(q) == len(idx)
        for j, i in enumerate(idx):
            qj = q[j].float().cpu().numpy()  # (D, H, 9)
            quantiles[i] = qj.transpose(0, 2, 1)
            mean[i] = m[j].float().cpu().numpy()
    if not (np.isfinite(quantiles).all() and np.isfinite(mean).all()):
        raise RuntimeError(f"{case.meta['config_id']}: non-finite forecast")
    ts_io.write_prediction_file(out_path, mean, quantiles, {"model": "chronos-2"})
    return {"n_windows": int(len(np.unique(windows))), "batch_size": bs_used}


def manifest() -> dict:
    return {
        "name": "chronos2_zero_shot",
        "contract": ts_io.CONTRACT,
        "parameters": 119477664,
        "pretrained_assets": [{"repo": MODEL_REPO, "revision": MODEL_REVISION, "license": MODEL_LICENSE}],
        "sources_used": [],
        "settings": {
            "library": f"chronos-forecasting=={CHRONOS_VERSION}",
            "pipeline": "chronos.Chronos2Pipeline via BaseChronosPipeline.from_pretrained",
            "device_map": DEVICE_MAP,
            "torch_dtype": "float32",
            "call": "predict_quantiles(inputs=[{'target': ...}], prediction_length=H, batch_size=100, "
                    "quantile_levels=[0.1..0.9], cross_learning=True)",
            "quantile_levels": QUANTILE_LEVELS,
            "point_forecast": "0.5 quantile (what gluonts uses as the mean of a QuantileForecast)",
            "model_context_length": 8192,
            "model_prediction_length": 1024,
            "cross_learning": CROSS_LEARNING,
            "cross_learning_note": "one rolling window at a time, as in the notebook, so windows of the same "
                                   "series never share a batch",
            "multivariate": "native: (D, T) target passed to the model",
            "nan_handling": "passed through to the model as missing observations",
            "batch_size": BATCH_SIZE,
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
              f"in {info['n_windows']} window(s), bs={info['batch_size']}, "
              f"{time.time() - t:.1f}s, peak {peak:.2f} GiB", file=sys.stderr, flush=True)
    man = manifest()
    man["runtime_seconds"] = round(time.time() - t0, 1)
    if torch.cuda.is_available():
        man["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    (outp / "model_manifest.json").write_text(json.dumps(man, indent=1) + "\n")


if __name__ == "__main__":
    main()
