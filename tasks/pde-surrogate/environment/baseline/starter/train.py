#!/usr/bin/env python3
"""Train the channel-shared U-Net starter on the manifest's training pools.

    python3 train.py --data DATA_DIR --manifest split.json --out RUN_DIR [--steps 2000 --batch 8 --width 32 --crop 128]
Writes RUN_DIR/{weights.pt, normalization.json, config.json, log.jsonl}.
"""
import argparse
import json
import sys
import random
import time
from pathlib import Path

import numpy as np
import torch

from data import Pools
from model import UNet


def crop(x, y, size, rng):
    if size and x.shape[-1] > size and x.shape[-2] > size:
        i, j = rng.randrange(x.shape[-2] - size + 1), rng.randrange(x.shape[-1] - size + 1)
        return x[..., i:i + size, j:j + size], y[..., i:i + size, j:j + size]
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--width", type=int, default=32); ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--datasets", default="", help="comma-separated subset of dataset names")
    ap.add_argument("--max-windows-per-traj", type=int, default=0)
    ap.add_argument("--allow-missing-pools", action="store_true",
                    help="train even if some manifest pools are absent on disk (they are recorded in config.json)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed); rng = random.Random(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pools = Pools(args.data, args.manifest, "train", datasets=set(args.datasets.split(",")) if args.datasets else None,
                  max_windows_per_traj=args.max_windows_per_traj or None, allow_missing=args.allow_missing_pools)
    if pools.missing:
        print(f"WARNING: {len(pools.missing)} pool file(s) absent, training without them: {pools.missing[:5]}", file=sys.stderr)
    stats = pools.normalization(seed=args.seed)
    (out / "normalization.json").write_text(json.dumps(stats, indent=1))
    model = UNet(context=4, width=args.width).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    log = open(out / "log.jsonl", "a"); t0 = time.time()
    print(f"datasets: {len(pools.datasets)}; windows: {sum(len(v) for v in pools.by_dataset.values())}; params: {sum(p.numel() for p in model.parameters())}")
    for step in range(1, args.steps + 1):
        ds = rng.choice(pools.datasets)
        x, y = pools.sample(ds, args.batch, rng)
        m, s = (np.array(stats[ds][k], dtype=np.float32)[None, :, None, None] for k in ("mean", "std"))
        x = (x - m[:, None]) / s[:, None]; y = (y - m) / s
        x, y = crop(x, y, args.crop, rng)
        n, t, c, h, w = x.shape
        xt = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 2, 1, 3, 4).reshape(n * c, t, h, w))).to(args.device)
        yt = torch.from_numpy(np.ascontiguousarray(y.reshape(n * c, 1, h, w))).to(args.device)
        loss = torch.nn.functional.mse_loss(model(xt), yt)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % 25 == 0 or step == 1:
            rec = {"step": step, "dataset": ds, "loss": float(loss), "elapsed_s": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec)
    torch.save(model.state_dict(), out / "weights.pt")
    (out / "config.json").write_text(json.dumps({
        "context": 4, "width": args.width, "steps": args.steps, "batch": args.batch, "crop": args.crop, "lr": args.lr,
        "seed": args.seed, "manifest": str(Path(args.manifest).resolve()), "max_windows_per_traj": args.max_windows_per_traj,
        "datasets": pools.datasets, "windows": {d: len(v) for d, v in pools.by_dataset.items()},
        "missing_pools": pools.missing, "command": " ".join(sys.argv), "seconds": round(time.time() - t0, 1)}, indent=1))
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
