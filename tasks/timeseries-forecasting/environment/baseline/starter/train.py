#!/usr/bin/env python3
"""Train the starter from scratch on the packed corpus.

    python3 train.py --packed /workspace/research/packed --out /workspace/research/runs/starter --steps 20000 \
        [--valid /workspace/data/valid --valid-configs m4_weekly__W__short,hospital__M__short,...] [--synthetic-fraction 0.1]

Multi-GPU: `torchrun --nproc_per_node 8 train.py ...` trains data-parallel. `--batch` is the global
batch, split evenly over ranks; steps, learning rate and schedule are unchanged, so the recipe is the
same as on one GPU. Each rank samples its own windows (seed + rank); gradients are averaged with an
all-reduce. Rank 0 evaluates and writes checkpoints.

Every `--eval-every` steps the current weights forecast the chosen validation
configs through predict.py's code path and are scored with the task grader
against the validation Seasonal Naive reference; the checkpoint with the best
validation skill is kept as `best.pt` (plus `last.pt`). A JSON line per
evaluation goes to `log.jsonl`.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import WindowSampler  # noqa: E402
from model import PatchForecaster, QUANTILES, count_parameters, quantile_loss  # noqa: E402


def make_targets(seq: torch.Tensor, patch: int, horizon: int, positions: int):
    """seq (B, context + horizon) standardized with NaN -> targets (B, positions, horizon): the next `horizon` values after each patch."""
    idx = torch.arange(horizon, device=seq.device)[None, :] + (torch.arange(positions, device=seq.device)[:, None] + 1) * patch
    return seq[:, idx]  # (B, positions, horizon)


def evaluate(model, device, valid_dir: Path, configs: list[str], work: Path, baseline: Path, batch: int):
    from predict import run as predict_run  # noqa: E402
    sys.path.insert(0, str(HERE.parents[1] / "evaluation"))
    from score import score  # noqa: E402

    out = work / "valid_pred"
    predict_run(model, device, valid_dir / "input", out, configs=configs, batch=batch)
    base = json.loads(baseline.read_text())["configs"] if baseline.exists() else None
    rep = score(valid_dir / "truth", out, base, allow_missing_baseline=base is None, configs=configs)
    rep_sub = {"reward": rep["reward"], "skill": rep["skill_score"], "rel_mase": rep["rel_mase_gmean"], "rel_wql": rep["rel_wql_gmean"],
               "per_config": {c["config_id"]: c["relative"] for c in rep["configs"]}}
    return rep_sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--context", type=int, default=1024); ap.add_argument("--horizon", type=int, default=128); ap.add_argument("--patch", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=256); ap.add_argument("--layers", type=int, default=6); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--max-context", type=int, default=0, help="inference context; 0 = the training context (every positional slot is then trained)")
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--weight-decay", type=float, default=0.05); ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--synthetic-fraction", type=float, default=0.0)
    ap.add_argument("--synthetic-bank", type=int, default=4096, help="pre-generated synthetic series (KernelSynth is ~1 s per series)")
    ap.add_argument("--synthetic-workers", type=int, default=8)
    ap.add_argument("--valid", default="/workspace/data/valid", help="staged validation split (input/, truth/); '' disables evaluation"); ap.add_argument("--valid-configs", default="")
    ap.add_argument("--valid-baseline", default=str(HERE.parents[1] / "evaluation" / "seasonal_naive_reference_valid.json"))
    ap.add_argument("--eval-every", type=int, default=2000); ap.add_argument("--eval-batch", type=int, default=512)
    ap.add_argument("--resume", default=""); ap.add_argument("--target-clip", type=float, default=20.0)
    args = ap.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    dist = None
    if world > 1:
        import torch.distributed as dist
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2), device_id=torch.device("cuda", local))
        device = torch.device("cuda", local)
        if args.batch % world:
            raise SystemExit(f"--batch {args.batch} is the global batch and must divide by {world} ranks")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lead = rank == 0
    local_batch = args.batch // world
    torch.manual_seed(args.seed); np.random.seed(args.seed + rank)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    synthetic = None
    if args.synthetic_fraction > 0:
        from synth import BankSampler, make_bank
        bank_file = out / "synthetic_bank.npy"
        if lead:  # one bank for all ranks, as on one GPU
            t_bank = time.time()
            bank = make_bank(args.synthetic_bank, args.context + args.horizon, seed=args.seed, workers=args.synthetic_workers)
            np.save(bank_file, bank)
            print(f"synthetic bank: {bank.shape} in {time.time() - t_bank:.0f}s", file=sys.stderr)
        if dist:
            dist.barrier()
        bank = np.load(bank_file)
        synthetic = BankSampler(bank)
    sampler = WindowSampler(Path(args.packed), args.context, args.horizon, seed=args.seed + rank, synthetic=synthetic,
                            synthetic_fraction=args.synthetic_fraction)
    max_context = args.max_context or args.context
    model = PatchForecaster(args.patch, args.horizon, args.d_model, args.layers, args.heads, max_context).to(device)
    cfg = {**{k: getattr(args, k) for k in ("patch", "horizon", "d_model", "layers", "heads")}, "max_context": max_context}
    if lead:
        print(f"parameters: {count_parameters(model):,}  series: {len(sampler.candidates):,}  ranks: {world}  "
              f"batch {args.batch} ({local_batch} per rank)", file=sys.stderr)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    sched = lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))  # noqa: E731
    step, best = 0, -1e9
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]; best = ck.get("best", best)
    valid_dir = Path(args.valid)
    if args.valid and not (valid_dir / "input" / "index.json").exists():
        raise SystemExit(f"--valid {valid_dir} has no input/index.json: stage the validation split there "
                         "(environment/evaluation/README.md) or pass --valid '' to train with no evaluation "
                         "(best.pt then equals last.pt and nothing selects a checkpoint)")
    valid_configs = [c for c in args.valid_configs.split(",") if c]
    if not valid_configs and (valid_dir / "input" / "index.json").exists():
        # default: small configs, stratified over terms, so long horizons are represented (they dominate the test reward)
        idx = json.loads((valid_dir / "input" / "index.json").read_text())
        small = [c for c in idx["configs"] if c["n_instances"] <= 1000]
        valid_configs = [c["config_id"] for term in ("short", "medium", "long") for c in [x for x in small if x["term"] == term][:5]]
    log = open(out / "log.jsonl", "a") if lead else None
    positions = args.context // args.patch
    t0, run_loss, skipped = time.time(), 0.0, 0
    model.train()
    while step < args.steps:
        for g in opt.param_groups:
            g["lr"] = args.lr * sched(step)
        ctx, tgt = sampler.sample(local_batch)
        ctx = torch.from_numpy(ctx).to(device); tgt = torch.from_numpy(tgt).to(device)
        z, mask, mu, sigma = model.standardize(ctx)
        seq = torch.cat([torch.where(mask > 0, z, torch.full_like(z, float("nan"))), (tgt - mu) / sigma], dim=1)
        targets = make_targets(seq, args.patch, args.horizon, positions).clamp(-args.target_clip, args.target_clip)  # (B, P, H), NaN = missing
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outp = model.forward_tokens(z, mask)  # (B, P, n_out, H)
        outp = outp.float()
        ql = quantile_loss(outp[:, :, :len(QUANTILES)], targets)
        finite = torch.isfinite(targets)
        mse = (torch.nn.functional.huber_loss(outp[:, :, len(QUANTILES)], torch.nan_to_num(targets), reduction="none", delta=1.0) * finite).sum() / finite.sum().clamp(min=1)
        loss = ql + 0.5 * mse
        finite_loss = torch.isfinite(loss).float()
        if dist:  # every rank skips together
            dist.all_reduce(finite_loss, op=dist.ReduceOp.MIN)
        if finite_loss.item() < 1:  # a pathological batch must not poison the weights
            skipped += 1
            if skipped % 10 == 1 and lead:
                print(f"step {step}: non-finite loss, batch skipped ({skipped} so far)", file=sys.stderr)
            step += 1
            continue
        opt.zero_grad(set_to_none=True); loss.backward()
        if dist:  # average gradients over ranks (same update as one GPU on the global batch)
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            flat = torch.cat([g.reshape(-1) for g in grads])
            dist.all_reduce(flat)
            flat /= world
            offset = 0
            for g in grads:
                g.copy_(flat[offset:offset + g.numel()].view_as(g)); offset += g.numel()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            skipped += 1; step += 1
            continue
        opt.step()
        step += 1; run_loss += loss.item()
        if step % 100 == 0 and lead:
            print(f"step {step} loss {run_loss/100:.4f} ql {ql.item():.4f} mse {mse.item():.4f} lr {opt.param_groups[0]['lr']:.2e} {time.time()-t0:.0f}s", file=sys.stderr)
            run_loss = 0.0
        if (step % args.eval_every == 0 or step == args.steps) and not lead:
            dist.barrier()  # wait while rank 0 evaluates and saves
        elif step % args.eval_every == 0 or step == args.steps:
            ck = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "config": cfg, "best": best}
            if valid_configs and (valid_dir / "input").exists():
                model.eval()
                rep = evaluate(model, device, valid_dir, valid_configs, out, Path(args.valid_baseline), args.eval_batch)
                model.train()
                rep.update({"step": step, "seconds": round(time.time() - t0), "loss": loss.item(), "skipped_batches": skipped})
                log.write(json.dumps(rep) + "\n"); log.flush()
                print(f"eval step {step}: skill {rep['skill']:.4f} rel_mase {rep['rel_mase']:.4f} rel_wql {rep['rel_wql']:.4f}", file=sys.stderr)
                if rep["skill"] > best:
                    best = rep["skill"]; ck["best"] = best
                    torch.save(ck, out / "best.pt")
            else:
                torch.save(ck, out / "best.pt")
            ck["best"] = best
            torch.save(ck, out / "last.pt")   # written after the evaluation so --resume restores the real best
            if dist:
                dist.barrier()
    if dist:
        dist.destroy_process_group()
    if not lead:
        return
    log.close()
    print(f"done: best validation skill {best:.4f}; {skipped} batches skipped for non-finite loss; checkpoints in {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
