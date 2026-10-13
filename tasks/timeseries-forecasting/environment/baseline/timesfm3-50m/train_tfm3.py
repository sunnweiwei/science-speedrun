#!/usr/bin/env python3
"""Train the TimesFM-3 architecture from scratch (random init) on the packed corpus, data-parallel.

    torchrun --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 29512 --nproc_per_node 8 \
        train_tfm3.py --packed /workspace/research/packed --out /workspace/research/runs/tfm3 \
        --steps 200000 --batch 1024 --context 2048 --dims 768 --layers 14 --heads 12 \
        --synthetic-fraction 0.1 --valid /workspace/data/valid

Model: `timesfm3.torch.TimesFM3Torch` (vendored, Apache-2.0) at a reduced width/depth, input patch 32,
output patch 64, nine quantiles, RevIN, rotary positions, univariate (no variate attention).
Objective: every input patch forecasts the next 64 points; pinball loss on the nine quantiles in
the patch's running-normalized units. Contexts have random length (left-masked, as the sampler
pads with NaN). `--batch` is the global batch. Every `--eval-every` steps rank 0 saves the model with
`save_pretrained`, forecasts the stratified validation configs with the released TimesFM-3
forecaster (`predict.py`) and scores them with the task grader; the best is kept in `best/`.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
sys.path.insert(0, str(HERE))
STARTER = next(p for p in (HERE.parent / "starter", Path("/workspace/baseline/starter")) if (p / "data.py").exists())
sys.path.insert(0, str(STARTER))
from data import WindowSampler  # noqa: E402
from timesfm3.torch import configs  # noqa: E402
from timesfm3.torch.model import TimesFM3Torch  # noqa: E402

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PATCH, OUT_PATCH = 32, 64


def build(dims, layers, heads):
    tc = configs.StackedTransformersConfig(num_layers=layers, transformer=configs.TransformerConfig(
        model_dims=dims, hidden_dims=dims, num_heads=heads, attention_norm="rms", feedforward_norm="rms", qk_norm="rms",
        use_rope_seq=True, use_rope_var=False, use_bias=False, ff_activation="relu", deterministic=True))
    rb = configs.ResidualBlockConfig(hidden_dims=dims, output_dims=dims, use_bias=False, activation="relu")
    return TimesFM3Torch(input_patch_len=PATCH, output_patch_len=OUT_PATCH, quantiles=QUANTILES,
                         residual_block_config=rb, transformer_config=tc, use_variate_attention=False)


def loss_fn(model, ctx, tgt, clip):
    """ctx (B, C) NaN-left-padded, tgt (B, 64). Returns pinball loss in normalized units."""
    b, c = ctx.shape
    n = c // PATCH
    seq = torch.cat([ctx, tgt], dim=1)                                    # (B, C + 64)
    x = ctx.view(b, 1, n, PATCH)
    masks = torch.isnan(x)
    out = model({"values": torch.nan_to_num(x), "masks": masks, "patch_is_target": torch.ones(b, 1, n, dtype=torch.bool, device=ctx.device)})
    mean, std = out["revin_stats"]                                        # (B, 1, n)
    mean, std = mean[:, 0, :, None], std[:, 0, :, None]
    std = torch.where(std > 1e-8, std, torch.ones_like(std))
    idx = (torch.arange(n, device=ctx.device)[:, None] + 1) * PATCH + torch.arange(OUT_PATCH, device=ctx.device)[None]
    targets = seq[:, idx]                                                  # (B, n, 64)
    z_t = ((targets - mean) / std).clamp(-clip, clip)
    z_p = (out["logits"][:, 0].float() - mean[..., None]) / std[..., None]    # (B, n, 64, 9)
    seen = (~masks[:, 0].all(-1)).cumsum(-1) > 0                          # patches with some context so far
    valid = torch.isfinite(z_t) & seen[..., None]
    q = torch.tensor(QUANTILES, device=ctx.device)
    err = torch.nan_to_num(z_t)[..., None] - z_p
    pin = torch.maximum(q * err, (q - 1) * err).mean(-1)                  # (B, n, 64)
    return (pin * valid).sum() / valid.sum().clamp(min=1)


def evaluate(model, out_dir: Path, valid_dir: Path, configs_: list[str], baseline: Path, context: int):
    sys.path.insert(0, str(HERE))
    from predict import load_forecaster, predict_dir
    sys.path.insert(0, str(STARTER.parents[1] / "evaluation"))
    from score import score
    ck = out_dir / "eval_ckpt"
    shutil.rmtree(ck, ignore_errors=True)
    model.save_pretrained(ck)
    (ck / "training.json").write_text(json.dumps({"context": context}))
    f = load_forecaster(ck)
    pred = out_dir / "valid_pred"
    shutil.rmtree(pred, ignore_errors=True)
    predict_dir(f, valid_dir / "input", pred, configs=configs_, verbose=False)
    base = json.loads(baseline.read_text())["configs"] if baseline.exists() else None
    rep = score(valid_dir / "truth", pred, base, allow_missing_baseline=base is None, configs=configs_)
    del f
    torch.cuda.empty_cache()
    return ck, {"skill": rep["skill_score"], "rel_mase": rep["rel_mase_gmean"], "rel_wql": rep["rel_wql_gmean"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200000); ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--dims", type=int, default=768); ap.add_argument("--layers", type=int, default=14); ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--weight-decay", type=float, default=0.05); ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--synthetic-fraction", type=float, default=0.1)
    ap.add_argument("--synthetic-bank", type=int, default=4096); ap.add_argument("--synthetic-workers", type=int, default=8)
    ap.add_argument("--valid", default="/workspace/data/valid"); ap.add_argument("--valid-configs", default="")
    ap.add_argument("--valid-baseline", default=str(STARTER.parents[1] / "evaluation" / "seasonal_naive_reference_valid.json"))
    ap.add_argument("--eval-every", type=int, default=5000); ap.add_argument("--target-clip", type=float, default=20.0)
    ap.add_argument("--max-hours", type=float, default=0, help="stop (and evaluate) after this many hours of training")
    ap.add_argument("--sampling", default="mixture", choices=["mixture", "series"],
                    help="mixture: source-balanced (mixture.py); series: uniform over series (the starter's sampler)")
    args = ap.parse_args()
    if args.context % PATCH:
        raise SystemExit("--context must be a multiple of 32")
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    lead = rank == 0
    if world > 1:
        import torch.distributed as dist
        local = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local)
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2), device_id=torch.device("cuda", local))
        device = torch.device("cuda", local)
    else:
        dist, device = None, torch.device("cuda")
    if args.batch % world:
        raise SystemExit("--batch must divide by the number of ranks")
    local_batch = args.batch // world
    torch.manual_seed(args.seed); np.random.seed(args.seed + rank)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    synthetic = None
    if args.synthetic_fraction > 0:
        from synth import BankSampler, make_bank
        bank_file = out / "synthetic_bank.npy"
        if lead and not bank_file.exists():
            np.save(bank_file, make_bank(args.synthetic_bank, args.context + OUT_PATCH, seed=args.seed, workers=args.synthetic_workers))
        if dist:
            dist.barrier()
        synthetic = BankSampler(np.load(bank_file))
    if args.sampling == "mixture":
        from mixture import MixtureSampler
        sampler = MixtureSampler(Path(args.packed), args.context, OUT_PATCH, seed=args.seed + rank, synthetic=synthetic,
                                 synthetic_fraction=args.synthetic_fraction)
    else:
        sampler = WindowSampler(Path(args.packed), args.context, OUT_PATCH, seed=args.seed + rank, synthetic=synthetic,
                                synthetic_fraction=args.synthetic_fraction)
    model = build(args.dims, args.layers, args.heads).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index]) if dist else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    sched = lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))  # noqa: E731
    valid_dir = Path(args.valid)
    valid_configs = [c for c in args.valid_configs.split(",") if c]
    if not valid_configs and (valid_dir / "input" / "index.json").exists():
        idx = json.loads((valid_dir / "input" / "index.json").read_text())
        small = [c for c in idx["configs"] if c["n_instances"] <= 1000]
        valid_configs = [c["config_id"] for term in ("short", "medium", "long") for c in [x for x in small if x["term"] == term][:5]]
    if lead:
        if hasattr(sampler, "group_share"):
            print(f"sampling shares (before synthetic): {json.dumps({k: round(v, 3) for k, v in sampler.group_share.items()})}", file=sys.stderr)
        print(f"parameters {nparams:,}  series {len(sampler.candidates):,}  ranks {world}  batch {args.batch} ({local_batch}/rank)  "
              f"context {args.context}", file=sys.stderr, flush=True)
        (out / "config.json").write_text(json.dumps({**vars(args), "parameters": nparams, "world": world}, indent=1))
    log = open(out / "log.jsonl", "a") if lead else None
    step, best, t0, run_loss = 0, -1e9, time.time(), 0.0
    stop = torch.zeros(1, device=device)
    ddp.train()
    while step < args.steps:
        for g in opt.param_groups:
            g["lr"] = args.lr * sched(step)
        ctx, tgt = sampler.sample(local_batch)
        ctx = torch.from_numpy(ctx).to(device, non_blocking=True); tgt = torch.from_numpy(tgt).to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(ddp, ctx, tgt, args.target_clip)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
        step += 1
        run_loss += loss.item()
        if step % 100 == 0 and lead:
            print(f"step {step} loss {run_loss / 100:.4f} lr {opt.param_groups[0]['lr']:.2e} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
            run_loss = 0.0
        if args.max_hours and lead and time.time() - t0 > args.max_hours * 3600:
            stop.fill_(1)
        last = step == args.steps
        if step % args.eval_every == 0 or last:
            if dist:
                dist.broadcast(stop, 0)
            if lead:
                model.eval()
                ck, rep = evaluate(model, out, valid_dir, valid_configs, Path(args.valid_baseline), args.context)
                model.train()
                rep.update({"step": step, "seconds": round(time.time() - t0), "loss": loss.item()})
                log.write(json.dumps(rep) + "\n"); log.flush()
                print(f"eval step {step}: skill {rep['skill']:.4f} rel_mase {rep['rel_mase']:.4f} rel_wql {rep['rel_wql']:.4f}", file=sys.stderr, flush=True)
                if rep["skill"] > best:
                    best = rep["skill"]
                    shutil.rmtree(out / "best", ignore_errors=True)
                    shutil.copytree(ck, out / "best")
                    (out / "best" / "training.json").write_text(json.dumps({"context": args.context, "step": step, "valid_skill": best}))
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best}, out / "last.pt")
            if dist:
                dist.barrier()
            if stop.item() > 0:
                break
    if dist:
        dist.destroy_process_group()
    if lead:
        print(f"done at step {step}: best validation skill {best:.4f}; {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
