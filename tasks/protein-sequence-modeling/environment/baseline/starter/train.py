#!/usr/bin/env python3
"""Runnable starter: pretrain an ESM-2 encoder, fine-tune it on all eleven
tasks, and write a submission that satisfies ../contract.md.

This is a working starting point, not a strong model. It demonstrates the parts
that are tedious to get right — the MLM objective, length-bucketed token-budget
batching, joint multi-task training, model selection on `tune` with the
grader's own metric, and packaging — so the research can start from something
that already runs end to end.

Defaults are sized for a quick smoke pass. Raise --pretrain-hours and
--finetune-hours to use the real budget, and see `esm2.CONFIGS` for larger
sizes.

    python3 train.py --data /workspace/data --out /workspace/research/seed-0/submission \\
        --model esm2_t6_8M --pretrain-hours 8 --finetune-hours 3
    python3 publish.py /workspace/research/seed-0/submission --to /workspace/submission
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import esm2  # noqa: E402

# The grader's own metric implementation, so model selection optimizes exactly
# what the verifier will measure. Copied trainers (research forks under
# /workspace/research/...) no longer sit beside evaluation/, so fall back to
# the canonical mount point.
_metrics_dirs = [HERE.parents[1] / "evaluation", Path("/workspace/evaluation")]
sys.path.insert(0, str(next((d for d in _metrics_dirs if (d / "metrics.py").is_file()),
                            _metrics_dirs[0])))
import metrics  # noqa: E402

TASKS = metrics.TASKS
CLASSES = esm2.Model.CLASSES
PAIR = esm2.Model.PAIR
MAX_LEN = 1024


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_jsonl(path, limit=None):
    rows = []
    with open(path) as handle:
        for line in handle:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def read_fasta(path, limit=None):
    sequences, current = [], []
    with open(path) as handle:
        for line in handle:
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
                    if limit and len(sequences) >= limit:
                        return sequences
            else:
                current.append(line.rstrip())
    if current:
        sequences.append("".join(current))
    return sequences


def build_batches(lengths, token_budget, rng):
    """Length-bucketed token-budget batches.

    Worth understanding before changing: over this corpus (mean length 265,
    p99 1659) batching a fixed *number* of sequences pads every batch to its
    longest member and wastes 3.4x of the compute. Sorting by length first
    recovers essentially all of it. ESM-2 batches by token count for the same
    reason.
    """
    order = sorted(range(len(lengths)), key=lengths.__getitem__)
    batches, current, width = [], [], 0
    for index in order:
        length = min(lengths[index], MAX_LEN - 2) + 2
        candidate = max(width, length)
        if current and candidate * (len(current) + 1) > token_budget:
            batches.append(current)
            current, width = [index], length
        else:
            current.append(index)
            width = candidate
    if current:
        batches.append(current)
    rng.shuffle(batches)
    return batches


def save_checkpoint(path, model, optimizer, phase, elapsed, extra=None):
    """Write a resumable checkpoint atomically.

    A research loop resumes constantly, and a checkpoint half-written when a job
    is preempted is worse than none: it looks loadable and is not. Write beside
    the destination and rename.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(),
               "optimizer": optimizer.state_dict() if optimizer else None,
               "phase": phase, "elapsed": elapsed, "extra": extra or {}}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer=None, device=None):
    """Load a checkpoint, transferring only the tensors that actually fit.

    Resuming a checkpoint from a *different* architecture is a normal thing to
    try mid-loop — pretrain at 8M, then see whether those weights help a 35M
    run. `strict=False` does not cover it: it forgives missing and unexpected
    keys but still raises on every shape mismatch, so the useful signal drowns
    in pages of errors. Transfer what fits, say plainly what did not, and let
    the caller decide.
    """
    state = torch.load(path, map_location=device or "cpu", weights_only=False)
    current = model.state_dict()
    fitting, mismatched = {}, []
    for key, value in state["model"].items():
        if key in current and current[key].shape == value.shape:
            fitting[key] = value
        elif key in current:
            mismatched.append(key)
    missing, unexpected = model.load_state_dict(fitting, strict=False)
    if mismatched or missing or unexpected:
        log(f"  partial load: {len(fitting)} tensors transferred, "
            f"{len(mismatched)} shape-mismatched, {len(missing)} left at their "
            f"initialization, {len(unexpected)} ignored")
        if mismatched:
            log(f"    the checkpoint is a different architecture "
                f"(e.g. {mismatched[0]}); optimizer state will not be reused")
    if optimizer is not None and state.get("optimizer") and not mismatched:
        optimizer.load_state_dict(state["optimizer"])
    return state


def schedule(elapsed, budget, warmup_fraction, peak):
    """Warmup then cosine decay, driven by *elapsed wall clock*, not a step count.

    Both training phases stop on a time budget, so a schedule keyed to a guessed
    step count desynchronizes from it: overshoot the guess and `min(progress, 1)`
    pins the cosine at pi, making the learning rate exactly zero while the loop
    keeps burning compute on updates that change nothing. Keying the schedule to
    the same clock that ends the phase makes that impossible.
    """
    progress = min(elapsed / max(budget, 1e-9), 1.0)
    if progress < warmup_fraction:
        return peak * progress / warmup_fraction
    tail = (progress - warmup_fraction) / max(1.0 - warmup_fraction, 1e-9)
    return peak * 0.5 * (1 + math.cos(math.pi * min(tail, 1.0)))


def pretrain(model, sequences, device, token_budget, peak_lr, max_hours,
             log_every, checkpoint=None, resume_state=None, save_every=2000,
             seed=0):
    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=peak_lr,
                                  weight_decay=0.01, betas=(0.9, 0.98))
    rng = random.Random(seed)
    batches = build_batches([len(s) for s in sequences], token_budget, rng)
    log(f"  {len(batches):,} batches")
    model.train()
    tokens_seen = step = epochs = 0
    prior_elapsed = 0.0
    if resume_state is not None:
        if resume_state.get("optimizer"):
            optimizer.load_state_dict(resume_state["optimizer"])
        extra = resume_state.get("extra", {})
        tokens_seen = extra.get("tokens_seen", 0)
        step = extra.get("step", 0)
        epochs = extra.get("epochs", 0)
        prior_elapsed = resume_state.get("elapsed", 0.0)
        log(f"  resumed at step {step:,}, {tokens_seen:,} tokens, "
            f"{prior_elapsed/3600:.2f}h already spent")
        if prior_elapsed >= max_hours * 3600:
            log("  pretraining budget already spent in the resumed run")
            return tokens_seen, prior_elapsed, step
    start = time.time() - prior_elapsed
    last_save = time.time()
    while True:
        for batch in batches:
            step += 1
            if (time.time() - start) > max_hours * 3600:
                elapsed = time.time() - start
                log(f"  budget reached: {step:,} steps, {tokens_seen:,} tokens, "
                    f"{epochs + step/len(batches):.2f} epochs")
                # Always leave something to resume from. A phase that ends
                # cleanly is exactly the one worth reusing for a different
                # downstream recipe, so never let the periodic-save cadence
                # decide whether that is possible.
                if checkpoint:
                    save_checkpoint(checkpoint, model, optimizer, "pretrain",
                                    elapsed, {"tokens_seen": tokens_seen,
                                              "step": step, "epochs": epochs,
                                              "complete": True})
                return tokens_seen, elapsed, step
            elapsed = time.time() - start
            for group in optimizer.param_groups:
                group["lr"] = schedule(elapsed, max_hours * 3600, 0.01, peak_lr)
            tokens = esm2.pad_batch(
                [esm2.encode(sequences[i], MAX_LEN) for i in batch], device)
            target = tokens.clone()
            residue = (tokens >= esm2.FIRST_RESIDUE) & (tokens <= esm2.LAST_RESIDUE)
            selected = (torch.rand_like(tokens, dtype=torch.float) < 0.15) & residue
            if not bool(selected.any()):
                continue
            roll = torch.rand_like(tokens, dtype=torch.float)
            tokens = torch.where(selected & (roll < 0.8),
                                 torch.full_like(tokens, esm2.MASK), tokens)
            noise = torch.randint_like(tokens, esm2.FIRST_RESIDUE, esm2.LAST_RESIDUE + 1)
            tokens = torch.where(selected & (roll >= 0.8) & (roll < 0.9), noise, tokens)
            tokens_seen += int(tokens.ne(esm2.PAD).sum())
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                hidden, _ = model.encoder(tokens)
                loss = F.cross_entropy(
                    model.encoder.lm_logits(hidden)[selected].float(), target[selected])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
            optimizer.step()
            # Save on a step cadence, but also on a wall-clock cadence: on
            # slow hardware save_every steps can take longer than the process
            # is allowed to live, and a kill would then lose the whole phase.
            if checkpoint and (step % save_every == 0
                               or time.time() - last_save > 600):
                save_checkpoint(checkpoint, model, optimizer, "pretrain",
                                time.time() - start,
                                {"tokens_seen": tokens_seen, "step": step,
                                 "epochs": epochs})
                last_save = time.time()
            if step % log_every == 0 or step == 1:
                elapsed = time.time() - start
                log(f"  mlm {step:,} loss {loss.item():.4f} "
                    f"ppl {math.exp(min(loss.item(), 20)):.2f} "
                    f"{tokens_seen/elapsed/1e3:.0f}K tok/s "
                    f"({elapsed/3600:.2f}/{max_hours}h)")
        epochs += 1
        rng.shuffle(batches)


@torch.no_grad()
def predict_split(model, rows, task, device, token_budget=16384):
    model.eval()
    out = {}
    if task in PAIR:
        lengths = [max(len(r["features"]["sequence_a"]),
                       len(r["features"]["sequence_b"])) for r in rows]
    else:
        lengths = [len(r["features"]["sequence"]) for r in rows]
    for batch in build_batches(lengths, token_budget, random.Random(0)):
        chunk = [rows[i] for i in batch]
        features = [r["features"] for r in chunk]
        with torch.autocast(device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            if task == "secondary_structure":
                tokens = esm2.pad_batch(
                    [esm2.encode(f["sequence"], MAX_LEN) for f in features], device)
                hidden, _ = model.encoder(tokens)
                logits = model.heads[task](hidden).float().softmax(-1).cpu()
                for row, record in enumerate(chunk):
                    length = len(record["features"]["sequence"])
                    used = min(length, MAX_LEN - 2)
                    scores = [[float(x) for x in logits[row, 1 + i]] for i in range(used)]
                    scores += [[1.0, 1.0, 1.0]] * (length - used)
                    out[record["example_id"]] = {"ss3_scores": scores}
                continue
            representation = model.represent(task, features, device, MAX_LEN)
            output = model.heads[task](representation).float().cpu()
        for row, record in enumerate(chunk):
            if task in CLASSES:
                probabilities = output[row].softmax(-1).clamp(min=1e-9)
                out[record["example_id"]] = {"scores": [float(x) for x in probabilities]}
            else:
                out[record["example_id"]] = {"value": float(output[row, 0])}
    model.train()
    return out


def evaluate(model, tune, device, std):
    rewards = {}
    for task in TASKS:
        rows = tune[task]
        truth = {r["example_id"]: r["label"] for r in rows}
        found = predict_split(model, rows, task, device)
        try:
            rewards[task] = metrics.score_task(task, truth, found, std)["reward"]
        except (ValueError, TypeError, KeyError) as error:
            log(f"    {task} scoring failed: {error}")
            rewards[task] = 0.0
    return sum(rewards.values()) / len(TASKS), rewards


def finetune(model, data, tune, device, peak_lr, std, max_hours,
             eval_every, log_every, batch_size=32, checkpoint=None,
             resume_state=None, save_every=2000, seed=0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)
    rng = random.Random(seed + 1)
    # sqrt(rows): uniform round-robin starves the large tasks, raw proportion
    # starves the small ones.
    weights = [math.sqrt(len(data[t])) for t in TASKS]
    total = sum(weights)
    weights = [w / total for w in weights]
    order = {}
    for task in TASKS:
        rows = data[task]
        key = ((lambda r: max(len(r["features"]["sequence_a"]),
                              len(r["features"]["sequence_b"])))
               if task in PAIR else (lambda r: len(r["features"]["sequence"])))
        order[task] = sorted(range(len(rows)), key=lambda i: key(rows[i]))
    best_reward, best_state, history = -1.0, None, []
    step = 0
    prior_elapsed = 0.0
    if resume_state is not None:
        if resume_state.get("optimizer"):
            optimizer.load_state_dict(resume_state["optimizer"])
        extra = resume_state.get("extra", {})
        step = extra.get("step", 0)
        best_reward = extra.get("best_reward", -1.0)
        history = extra.get("history", [])
        prior_elapsed = resume_state.get("elapsed", 0.0)
        log(f"  resumed at step {step:,}, best tune reward {best_reward:.4f}, "
            f"{prior_elapsed/3600:.2f}h already spent")
    start = time.time() - prior_elapsed
    last_save = time.time()
    while (time.time() - start) <= max_hours * 3600:
        step += 1
        for group in optimizer.param_groups:
            group["lr"] = schedule(time.time() - start, max_hours * 3600,
                                   0.02, peak_lr)
        task = rng.choices(TASKS, weights=weights)[0]
        rows, indices = data[task], order[task]
        begin = rng.randrange(max(1, len(indices) - batch_size))
        batch = [rows[i] for i in indices[begin:begin + batch_size]]
        with torch.autocast(device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            if task == "secondary_structure":
                tokens = esm2.pad_batch(
                    [esm2.encode(x["features"]["sequence"], MAX_LEN) for x in batch],
                    device)
                hidden, _ = model.encoder(tokens)
                logits = model.heads[task](hidden)
                flat_logits, flat_target = [], []
                for row, item in enumerate(batch):
                    labels = item["label"]["ss3"][:MAX_LEN - 2]
                    valid = item["label"]["valid_mask"][:MAX_LEN - 2]
                    keep = [i for i, v in enumerate(valid) if v]
                    if not keep:
                        continue
                    index = torch.tensor(keep, device=device) + 1
                    flat_logits.append(logits[row, index])
                    flat_target.append(
                        torch.tensor([labels[i] for i in keep], device=device))
                if not flat_logits:
                    continue
                loss = F.cross_entropy(torch.cat(flat_logits).float(),
                                       torch.cat(flat_target))
            else:
                representation = model.represent(
                    task, [x["features"] for x in batch], device, MAX_LEN)
                output = model.heads[task](representation)
                if task in CLASSES:
                    target = torch.tensor([x["label"] for x in batch], device=device)
                    loss = F.cross_entropy(output.float(), target)
                else:
                    target = torch.tensor([float(x["label"]) for x in batch],
                                          device=device, dtype=torch.float)
                    loss = F.smooth_l1_loss(output.float().squeeze(-1), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % log_every == 0 or step == 1:
            log(f"  ft {step:,} [{task}] loss {loss.item():.4f} "
                f"({(time.time()-start)/3600:.2f}/{max_hours}h)")
        if step % eval_every == 0:
            reward, per_task = evaluate(model, tune, device, std)
            history.append({"step": step, "tune_reward": reward, "tasks": per_task})
            marker = ""
            if reward > best_reward:
                best_reward = reward
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
                marker = "  <- best"
            log(f"  [eval] step {step:,} tune reward {reward:.4f}{marker}")
        if checkpoint and (step % save_every == 0
                           or time.time() - last_save > 600):
            save_checkpoint(checkpoint, model, optimizer, "finetune",
                            time.time() - start,
                            {"step": step, "best_reward": best_reward,
                             "history": history})
            last_save = time.time()
    reward, per_task = evaluate(model, tune, device, std)
    history.append({"step": step, "tune_reward": reward, "tasks": per_task})
    if reward > best_reward:
        best_reward = reward
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    log(f"  [eval] final step {step:,} tune reward {reward:.4f}")
    if checkpoint:
        save_checkpoint(checkpoint, model, optimizer, "finetune",
                        time.time() - start,
                        {"step": step, "best_reward": best_reward,
                         "history": history, "complete": True})
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_reward, history, step


def export(model, config, out, notes):
    """Write a package that satisfies ../contract.md."""
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "esm2.py").write_text((HERE / "esm2.py").read_text())
    (out / "predict.py").write_text((HERE / "predict.py").read_text())
    (out / "infer.sh").write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\n'
        'exec python3 "$(dirname "$0")/predict.py" "$1" "$2"\n')
    (out / "infer.sh").chmod(0o755)
    backbone = sum(p.numel() for p in model.encoder.parameters())
    total = sum(p.numel() for p in model.parameters())
    (out / "README.md").write_text(f"""# Submission

ESM-2 architecture (`{config['name']}`) trained from random initialization in
this session; no external pretrained weights.

- **Shared backbone**: {config['layers']} layers, d={config['dim']},
  {config['heads']} heads, rotary position embeddings, tied LM head.
  {backbone:,} parameters, {total:,} including heads. Every task reads this
  same parameter set.
- **Pretraining**: masked language modelling on the provided UniRef50 corpus —
  {notes['pretrain_tokens']:,} tokens in {notes['pretrain_hours']:.2f} h.
- **Fine-tuning**: eleven tasks jointly, sampled proportional to sqrt(rows).
  Heads are small MLPs on the pooled representation; pair tasks concatenate two
  pooled sequences; `secondary_structure` reads per-residue states.
- **Model selection**: best of periodic `tune`-split scores, at
  {notes['best_tune_reward']:.4f}.
- **Limitations**: sequences truncated to {MAX_LEN} tokens, with
  `secondary_structure` beyond that filled by a uniform score vector.
  Single training run, seed {notes['seed']}.

Run with `bash infer.sh INPUT_ROOT OUTPUT_ROOT`.
""")
    return backbone, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/workspace/data"))
    parser.add_argument("--out", type=Path, default=Path("/workspace/research/run/submission"),
                        help="where the trained package is written; publish it with publish.py --to /workspace/submission")
    parser.add_argument("--model", default="esm2_t6_8M", choices=list(esm2.CONFIGS))
    parser.add_argument("--token-budget", type=int, default=32768)
    parser.add_argument("--pretrain-hours", type=float, default=0.05)
    parser.add_argument("--finetune-hours", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--corpus-limit", type=int, default=None,
                        help="read only the first N corpus sequences (for quick runs)")
    parser.add_argument("--rows-per-task", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="write resumable checkpoints here (recommended: "
                             "somewhere under /workspace/research)")
    parser.add_argument("--resume", type=Path,
                        help="resume from a checkpoint written by --checkpoint-dir")
    parser.add_argument("--seed", type=int, default=0,
                        help="seeds parameter init, MLM masking and batch order; "
                             "recorded in the package")
    parser.add_argument("--skip-pretrain", action="store_true",
                        help="go straight to fine-tuning, e.g. when resuming a "
                             "pretrained encoder to try a different recipe")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device {device}")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    model, config = esm2.build(args.model)
    model = model.to(device)
    log(f"{config['name']}: backbone "
        f"{sum(p.numel() for p in model.encoder.parameters()):,}")

    resume_state = resume_phase = None
    if args.resume:
        resume_state = load_checkpoint(args.resume, model, device=device)
        resume_phase = resume_state.get("phase")
        log(f"loaded {args.resume} (phase={resume_phase})")
    pretrain_ckpt = args.checkpoint_dir / "pretrain.pt" if args.checkpoint_dir else None
    finetune_ckpt = args.checkpoint_dir / "finetune.pt" if args.checkpoint_dir else None

    if args.skip_pretrain or resume_phase == "finetune":
        log("skipping pretraining (resuming a fine-tune, or --skip-pretrain)")
        tokens_seen, seconds, mlm_steps = 0, 0.0, 0
    else:
        log("loading pretraining corpus")
        sequences = read_fasta(args.data / "pretrain" / "uniref50_subset.fasta",
                               args.corpus_limit)
        log(f"  {len(sequences):,} sequences")
        log("MLM pretraining")
        tokens_seen, seconds, mlm_steps = pretrain(
            model, sequences, device, args.token_budget, 4e-4,
            args.pretrain_hours, args.log_every, pretrain_ckpt,
            resume_state if resume_phase == "pretrain" else None,
            seed=args.seed)
        del sequences

    log("loading supervised splits")
    data = {t: read_jsonl(args.data / "fit" / f"{t}.jsonl", args.rows_per_task)
            for t in TASKS}
    tune = {t: read_jsonl(args.data / "tune" / f"{t}.jsonl") for t in TASKS}
    std = json.loads((args.data / "stats.json").read_text())["ppi_affinity"]["std"]

    log("multi-task fine-tuning with tune-split model selection")
    ft_start = time.time()
    best, history, ft_steps = finetune(
        model, data, tune, device, 1e-4, std, args.finetune_hours,
        args.eval_every, args.log_every, checkpoint=finetune_ckpt,
        resume_state=resume_state if resume_phase == "finetune" else None,
        seed=args.seed)
    notes = {"seed": args.seed,
             "pretrain_tokens": tokens_seen, "pretrain_hours": seconds / 3600,
             "resumed_from": str(args.resume) if args.resume else None,
             "ft_steps": ft_steps, "ft_hours": (time.time() - ft_start) / 3600,
             "best_tune_reward": best}
    backbone, total = export(model, config, args.out, notes)
    (args.out / "training_history.json").write_text(json.dumps(
        {"config": config, "notes": notes, "history": history,
         "mlm_steps": mlm_steps,
         "parameters": {"backbone": backbone, "total": total}}, indent=2))
    log(f"wrote submission to {args.out} (best tune reward {best:.4f})")


if __name__ == "__main__":
    main()
