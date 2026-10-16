"""Submit official Large training and its dependent validation on configured-gpu-partition."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument(
        "--pretrain-source", choices=["shard0000", "two-shards"], default="shard0000"
    )
    p.add_argument("--seed", type=int, default=12536)
    p.add_argument("--data-seed", type=int, default=3622)
    p.add_argument("--lr", type=float, default=1.25e-5)
    p.add_argument("--global-batch", type=int, choices=[64, 512], default=64)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument(
        "--evaluate-step",
        type=int,
        help="Retain and evaluate this checkpoint after the full training schedule",
    )
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--gpus", type=int, choices=[1, 2], default=1)
    p.add_argument(
        "--evaluation-gpu", choices=["rtx_pro_6000", "a5000"], default="rtx_pro_6000"
    )
    p.add_argument("--evaluation-root", type=Path)
    p.add_argument(
        "--shard-inputs",
        type=Path,
        help="Use fixed-order A5000 parallel validation after CPU export",
    )
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", a.name):
        p.error("name must be a plain identifier without path separators")
    if not 0 < a.warmup < a.steps or a.steps % 250 or a.lr <= 0:
        p.error("require 0 < warmup < steps, steps divisible by 250, lr > 0")
    evaluation_step = a.steps if a.evaluate_step is None else a.evaluate_step
    if not 0 < evaluation_step <= a.steps or evaluation_step % 250:
        p.error("evaluate-step must be a positive multiple of250, no later than steps")
    root, upstream = a.data.resolve(), a.upstream.resolve()
    evaluation_root = a.evaluation_root.resolve() if a.evaluation_root else root
    if a.shard_inputs and not (a.shard_inputs / "manifest.json").is_file():
        p.error("shard-inputs requires a prepared full-release manifest")
    if a.shard_inputs:
        manifest = json.loads((a.shard_inputs / "manifest.json").read_text())
        if (
            not manifest["full_release"]
            or manifest["start"] != 0
            or manifest["stop"] != manifest["source_rows"]
            or manifest["batch_size"] != 16
            or Path(manifest["source"]).resolve()
            != (evaluation_root / "eval/lfmc/inputs.h5").resolve()
            or not manifest["shards"]
            or not all(
                (a.shard_inputs / s["file"]).is_file() for s in manifest["shards"]
            )
        ):
            p.error("require complete batch16 LFMC shards for the evaluation root")
    if (evaluation_root / "execution-context.json").exists() and not a.shard_inputs:
        p.error("fixed-context evaluation requires --shard-inputs")
    if a.evaluation_root and not all(
        (evaluation_root / name).is_dir() for name in ["eval", "packages"]
    ):
        p.error(
            "evaluation-root must contain eval and packages directories (or shared-data symlinks)"
        )
    track = Path(__file__).resolve().parents[2]
    run = root / a.name
    name = f"{a.name}-step{evaluation_step}"
    receipt = root / "launches" / f"{a.name}.json"
    count = 9923 if a.pretrain_source == "shard0000" else 19837
    if not (root / str(count)).is_dir():
        p.error(f"prepare the filtered {a.pretrain_source} release ({count}) first")
    if not (upstream / ".venv/bin/python").is_file():
        p.error("upstream runtime is missing")
    if run.exists() or receipt.exists() or (root / "packages" / name).exists():
        p.error("run, launch receipt, or output package already exists; use a new name")
    settings = {
        "OLMO_DATA": str(root),
        "OLMO_ROOT": str(evaluation_root),
        "OLMO_UPSTREAM": str(upstream),
        "OLMO_TRACK": str(track),
        "OLMO_RUN": str(run),
        "OLMO_VARIANT": "large",
        "OLMO_ACTIVATION_CHECKPOINT": "1",
        "OLMO_GPUS": str(a.gpus),
        "OLMO_SEED": str(a.seed),
        "OLMO_DATA_SEED": str(a.data_seed),
        "OLMO_PRETRAIN_DATA": str(root / str(count)),
        "OLMO_LR": str(a.lr),
        "OLMO_STEPS": str(a.steps),
        "OLMO_WARMUP": str(a.warmup),
        "OLMO_WEIGHT_DECAY": "0.002",
        "OLMO_BATCH": str(a.global_batch),
        "OLMO_MICROBATCH": "32",
        "OLMO_REPEATS": "32",
        "OLMO_SAVE_INTERVAL": "1000",
        "OLMO_EPHEMERAL_INTERVAL": "250"
        if a.steps % 500 or evaluation_step % 500
        else "500",
        "OLMO_FIXED_CHECKPOINT_STEP": "",
        "OLMO_KEEP_CHECKPOINTS": "",
        "OLMO_NAME": name,
        "OLMO_CHECKPOINT": str(run / f"step{evaluation_step}" / "model_and_optim"),
        "OLMO_TASKS": "awf lfmc",
        "OLMO_USE_EXISTING_PACKAGE": "0",
    }
    if evaluation_step != a.steps:
        # Make the chosen midpoint permanent and retain it until the dependent
        # evaluator runs. This does not shorten the planned cosine schedule.
        settings["OLMO_FIXED_CHECKPOINT_STEP"] = str(evaluation_step)
        settings["OLMO_KEEP_CHECKPOINTS"] = str(a.steps // 1000 + 3)
    if a.gpus == 2:
        # The allocated RTX Pro6000 nodes stalled in the NCCL P2P path.
        # Host-mediated communication passed the two-rank training/export check.
        settings["NCCL_P2P_DISABLE"] = "1"
    base = [
        "sbatch",
        "--parsable",
        "--partition=" + os.environ.get("EARTH_SLURM_PARTITION", "gpu"),
        "--export=ALL",
    ]
    train = base + [
        f"--gres=gpu:rtx_pro_6000:{a.gpus}",
        f"--job-name={a.name}",
        f"--cpus-per-task={8 * a.gpus}",
        f"--mem={64 if a.gpus == 1 else 96}G",
        "--time=08:00:00",
        f"--output={root}/train-%j.log",
        str(track / "docs/slurm/train.sbatch"),
    ]
    validate = base + [
        f"--gres=gpu:{a.evaluation_gpu}:1",
        f"--job-name={a.name}-eval",
        "--cpus-per-task=4",
        "--mem=32G",
        f"--output={root}/validate-%j.log",
        "--dependency=afterok:TRAIN_JOB",
        str(track / "docs/slurm/validate.sbatch"),
    ]
    if a.shard_inputs:
        settings["OLMO_SHARD_INPUTS"] = str(a.shard_inputs.resolve())
        validate = base + [
            f"--job-name={a.name}-dispatch",
            f"--output={evaluation_root}/dispatch-%j.log",
            "--dependency=afterok:TRAIN_JOB",
            str(track / "docs/slurm/dispatch_sharded_eval.sbatch"),
        ]
    plan = {"settings": settings, "training": train, "validation": validate}
    if a.shard_inputs:
        plan["validation_mode"] = "fixed-order-sharded"
        plan["validation_receipt"] = str(
            evaluation_root / "launches" / f"sharded-{name}.json"
        )
    if a.dry_run:
        print(json.dumps(plan, indent=2))
        return
    receipt.parent.mkdir(parents=True, exist_ok=True)
    # Persist a receipt before submission; partial submissions remain reviewable.
    plan["submitted_at"] = datetime.now(timezone.utc).isoformat()
    with receipt.open("x") as f:
        json.dump(plan, f, indent=2)
    env = dict(os.environ, **settings)
    train_job = subprocess.check_output(train, env=env, text=True).strip().split(";")[0]
    if not train_job.isdigit():
        raise RuntimeError(f"Unexpected sbatch response: {train_job}")
    plan["training_job"] = int(train_job)
    receipt.write_text(json.dumps(plan, indent=2))
    validate = [arg.replace("TRAIN_JOB", train_job) for arg in validate]
    plan["validation"] = validate
    eval_job = (
        subprocess.check_output(validate, env=env, text=True).strip().split(";")[0]
    )
    if not eval_job.isdigit():
        raise RuntimeError(f"Unexpected sbatch response: {eval_job}")
    plan["dispatch_job" if a.shard_inputs else "validation_job"] = int(eval_job)
    receipt.write_text(json.dumps(plan, indent=2))
    print(
        f"Training {train_job}; dependent {'dispatch' if a.shard_inputs else 'full validation'} "
        f"{eval_job}; receipt {receipt}"
    )
    if a.shard_inputs:
        print(
            "Dispatch submits validation jobs; completion is recorded separately in "
            + plan["validation_receipt"]
        )
    print(shlex.join(["squeue", "-j", f"{train_job},{eval_job}"]))


if __name__ == "__main__":
    main()
