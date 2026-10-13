#!/usr/bin/env python3
"""Contract inference (timeseries-forecast/v1) for a TimesFM-3-architecture model trained from scratch.

    bash infer.sh INPUT_DIR OUTPUT_DIR

The package holds `config.json` + `model.safetensors` (our weights, saved with
TimesFM3Torch.save_pretrained) and the vendored TimesFM-3 PyTorch code. Inference is the
released TimesFM-3 forecaster (TimesFM3Evaluator) with the GIFT-Eval notebook's flags, run
univariate (the model was trained on univariate windows) and with the context capped at the
training context. No pretrained weights are used.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
for cand in (HERE, Path("/workspace/evaluation"), HERE.parents[1] / "evaluation"):
    if (cand / "ts_io.py").exists():
        sys.path.insert(0, str(cand))
        break
import ts_io  # noqa: E402

PREDICT_FLAGS = dict(return_quantiles=True, use_symmetric_averaging=True, make_positive=True, sort_quantiles=True,
                     use_znorm=False, padding_mode="none", univariate=True)
BATCH = int(os.environ.get("TFM3_BATCH", 256))


def load_forecaster(checkpoint: Path):
    from timesfm3.torch import timesfm3_forecaster as tf
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    meta = json.loads((checkpoint / "training.json").read_text()) if (checkpoint / "training.json").exists() else {}
    tf._MAX_CONTEXT_LENGTH = int(meta.get("context", 2048))  # RoPE positions beyond training are untrained
    cfg = ModelConfig(checkpoint_path=str(checkpoint), input_patch_length=32, output_patch_length=64,
                      per_core_batch_size=BATCH, use_variate_attention=False, use_sdpa=True)
    return TimesFM3Evaluator(cfg)


def predict_dir(forecaster, input_dir: Path, output_dir: Path, configs: list[str] | None = None, verbose=True):
    import torch
    index = json.loads((input_dir / "index.json").read_text())
    (output_dir / "cases").mkdir(parents=True, exist_ok=True)
    total = 0
    for cfg in index["configs"]:
        cid = cfg["config_id"]
        if configs is not None and cid not in configs:
            continue
        case = ts_io.CaseFile(input_dir / "cases" / f"{cid}.h5")
        H, D, N = case.prediction_length, case.n_variates, case.n
        mean = np.zeros((N, D, H), np.float32)
        quant = np.zeros((N, D, 9, H), np.float32)
        step = max(1, BATCH // max(D, 1))
        for start in range(0, N, step):
            idx = range(start, min(start + step, N))
            raw = [case.context(i) for i in idx]
            contexts = [c[0] if D == 1 else c for c in raw]
            with torch.no_grad():
                outs = list(forecaster.predict_batch(contexts=contexts, horizon=H, **PREDICT_FLAGS))
            for k, i in enumerate(idx):
                q = np.asarray(outs[k].quantiles, dtype=np.float32)
                m = np.asarray(outs[k].forecast, dtype=np.float32)
                if q.ndim == 2:
                    q, m = q[None], m[None]
                q = np.transpose(q[:, :H, :], (0, 2, 1))
                m = m[:, :H]
                if not np.isfinite(q).all() or not np.isfinite(m).all():
                    fin = raw[k][np.isfinite(raw[k])]
                    fb = float(fin[-1]) if fin.size else 0.0
                    q = np.where(np.isfinite(q), q, fb)
                    m = np.where(np.isfinite(m), m, q[:, 4, :])
                quant[i], mean[i] = q, m
        case.close()
        tmp = output_dir / "cases" / f"{cid}.h5.part"
        ts_io.write_prediction_file(tmp, mean, quant)
        tmp.replace(output_dir / "cases" / f"{cid}.h5")
        total += N
        if verbose:
            print(f"{cid}: {N} instances", flush=True)
    return total


def main():
    input_dir, output_dir = Path(sys.argv[1]), Path(sys.argv[2])
    t0 = time.time()
    n = predict_dir(load_forecaster(HERE), input_dir, output_dir)
    manifest = json.loads((HERE / "model_manifest.json").read_text())
    manifest["runtime"] = {"wall_clock_s": round(time.time() - t0, 1), "instances": n}
    (output_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
