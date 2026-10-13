#!/usr/bin/env python3
"""TimesFM-3.0 zero-shot reference package for timeseries-forecast/v1.

Replays the GIFT-Eval leaderboard row `TimesFM-3` (gift-eval
`notebooks/timesfm3.ipynb`): the released checkpoint
`google/timesfm-3.0-pytorch` run zero-shot through
`timesfm3.TimesFM3Evaluator` with cross-variate attention.  Multivariate
configs are passed to the model as multivariate series (`to_univariate=False`
in the notebook), which TimesFM-3 supports natively.

    bash infer.sh INPUT_DIR OUTPUT_DIR

Environment overrides (all optional):
    TIMESFM3_CHECKPOINT  local checkpoint dir (default: TIMESERIES_PRETRAINED_DIR/timesfm-3.0-pytorch)
    TIMESERIES_PRETRAINED_DIR    where the checkpoint is downloaded (default: /data/pretrained)
    TIMESFM3_BATCH_UNI / TIMESFM3_BATCH_MULTI   batch sizes (defaults 64 / 16, the notebook values)
"""
from __future__ import annotations

import json
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

REPO_ID = "google/timesfm-3.0-pytorch"
REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"
TIMESFM_GIT = "https://github.com/google-research/timesfm"
TIMESFM_COMMIT = "e31dadd84cb26bd5153fde6687502b8312e918fb"  # timesfm 3.0.2

# --- notebook settings (gift-eval/notebooks/timesfm3.ipynb) --------------------
INPUT_PATCH_LENGTH = 32
OUTPUT_PATCH_LENGTH = 64
PER_CORE_BATCH_SIZE = 64
USE_VARIATE_ATTENTION = True
USE_SDPA = True
PREDICT_FLAGS = dict(
    return_quantiles=True,
    use_symmetric_averaging=True,
    make_positive=True,
    sort_quantiles=True,
)
BATCH_UNI = int(os.environ.get("TIMESFM3_BATCH_UNI", 64))
BATCH_MULTI = int(os.environ.get("TIMESFM3_BATCH_MULTI", 16))
# Memory guard: with a 15360-step context a forward pass of 64 series does not fit
# in the GPU budget of this task, so the *inner* per-core batch is additionally
# capped at TOKEN_BUDGET / (context length * variates).  The outer batch, and with
# it the grouping the model sees, stays at the notebook's 64 / 16.
TOKEN_BUDGET = int(os.environ.get("TIMESFM3_TOKEN_BUDGET", 245760))


def pretrained_root() -> Path:
    """Where maintainers stage external-target weights: $TIMESERIES_PRETRAINED_DIR, else /data/pretrained."""
    return Path(os.environ.get("TIMESERIES_PRETRAINED_DIR", "/data/pretrained"))


def checkpoint_dir() -> Path:
    env = os.environ.get("TIMESFM3_CHECKPOINT")
    if env:
        return Path(env)
    root = pretrained_root()
    dst = root / "timesfm-3.0-pytorch"
    if not (dst / "model.safetensors").exists():
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=REPO_ID, revision=REVISION, local_dir=str(dst))
    return dst


def load_model():
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    cfg = ModelConfig(
        checkpoint_path=str(checkpoint_dir()),
        input_patch_length=INPUT_PATCH_LENGTH,
        output_patch_length=OUTPUT_PATCH_LENGTH,
        per_core_batch_size=PER_CORE_BATCH_SIZE,
        use_variate_attention=USE_VARIATE_ATTENTION,
        use_sdpa=USE_SDPA,
    )
    f = TimesFM3Evaluator(cfg)
    print(f"[timesfm-3.0] loaded on {f.device}", flush=True)
    return f


def predict_batch(forecaster, contexts, horizon, batch):
    import torch

    while True:
        try:
            forecaster.config.per_core_batch_size = batch
            return list(forecaster.predict_batch(
                contexts=contexts, horizon=horizon,
                past_only_covariates=[None] * len(contexts), **PREDICT_FLAGS))
        except torch.cuda.OutOfMemoryError:
            if batch <= 1:
                raise
            batch = max(1, batch // 2)
            torch.cuda.empty_cache()
            print(f"  [oom] retrying with per_core_batch_size={batch}", flush=True)


def run_config(forecaster, case: ts_io.CaseFile, out_path: Path):
    H, D, N = case.prediction_length, case.n_variates, case.n
    batch = BATCH_MULTI if D > 1 else BATCH_UNI
    mean = np.zeros((N, D, H), np.float32)
    quant = np.zeros((N, D, 9, H), np.float32)
    for start in range(0, N, batch):
        idx = range(start, min(start + batch, N))
        raw = [case.context(i) for i in idx]
        max_ctx = max(c.shape[1] for c in raw) * D
        inner = int(np.clip(TOKEN_BUDGET // max(max_ctx, 1), 1, batch))
        # the notebook passes `np.array(entry["target"])`: 1-D for a univariate
        # config, (D, T) for a multivariate one
        contexts = [c[0] if D == 1 else c for c in raw]
        outs = predict_batch(forecaster, contexts, H, inner)
        for k, i in enumerate(idx):
            q = np.asarray(outs[k].quantiles, dtype=np.float32)   # (H, 9) or (D, H, 9)
            m = np.asarray(outs[k].forecast, dtype=np.float32)    # (H,)   or (D, H)
            if q.ndim == 2:
                q, m = q[None], m[None]
            q = np.transpose(q[:, :H, :], (0, 2, 1))              # (D, 9, H)
            m = m[:, :H]
            if not np.isfinite(q).all() or not np.isfinite(m).all():
                fin = raw[k][np.isfinite(raw[k])]
                fb = float(fin[-1]) if fin.size else 0.0
                q = np.where(np.isfinite(q), q, fb)
                m = np.where(np.isfinite(m), m, q[:, 4, :])
            quant[i] = q
            mean[i] = m
    ts_io.write_prediction_file(out_path, mean, quant)
    return N


def main():
    input_dir, output_dir = Path(sys.argv[1]), Path(sys.argv[2])
    index = json.loads((input_dir / "index.json").read_text())
    (output_dir / "cases").mkdir(parents=True, exist_ok=True)
    forecaster = load_model()
    t_all = time.time()
    total = 0
    for k, cfg in enumerate(index["configs"], 1):
        cid = cfg["config_id"]
        out_path = output_dir / "cases" / f"{cid}.h5"
        if out_path.exists() and os.environ.get("TIMESERIES_RESUME", "1") == "1":
            print(f"[{k}/{len(index['configs'])}] {cid}: already done", flush=True)
            continue
        t0 = time.time()
        case = ts_io.CaseFile(input_dir / "cases" / f"{cid}.h5")
        tmp = out_path.with_suffix(".h5.part")
        n = run_config(forecaster, case, tmp)
        case.close()
        tmp.replace(out_path)
        total += n
        print(f"[{k}/{len(index['configs'])}] {cid}: {n} instances (D={cfg['n_variates']}) in {time.time()-t0:.1f}s", flush=True)
    manifest = json.loads((HERE / "model_manifest.json").read_text())
    manifest["runtime"] = {"wall_clock_s": round(time.time() - t_all, 1), "instances": total}
    (output_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"total {time.time()-t_all:.1f}s for {total} instances", flush=True)


if __name__ == "__main__":
    main()
