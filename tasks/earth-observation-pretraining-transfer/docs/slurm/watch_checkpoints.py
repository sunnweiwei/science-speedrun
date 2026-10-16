"""Submit validation after async checkpoints finish, before ephemeral removal."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipts", type=Path, nargs="+", required=True)
    p.add_argument("--steps", type=int, nargs="+", default=[500, 1000])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeout-seconds", type=int, default=21600)
    p.add_argument("--evaluation-gpu", choices=["rtx_pro_6000", "a5000"])
    p.add_argument("--evaluation-root", type=Path)
    a = p.parse_args()
    plans = [json.loads(path.read_text()) for path in a.receipts]
    if any(plan.get("validation_mode") == "fixed-order-sharded" for plan in plans):
        p.error(
            "sharded training launches already schedule checkpoint export/dispatch; "
            "use submit_sharded_eval.py for a separately exported additional midpoint"
        )
    records = json.loads(a.output.read_text()) if a.output.exists() else []
    submitted = {(r["training_job"], r["step"]) for r in records}
    pending = [
        (plan, step)
        for plan in plans
        for step in a.steps
        if (plan["training_job"], step) not in submitted
    ]
    deadline = time.monotonic() + a.timeout_seconds
    while pending:
        if time.monotonic() > deadline:
            raise TimeoutError(
                "Checkpoints did not become ready within the watcher deadline"
            )
        for plan, step in pending.copy():
            settings = plan["settings"]
            root = Path(settings["OLMO_DATA"])
            log = root / f"train-{plan['training_job']}.log"
            if (
                not log.exists()
                or f"Checkpoint for step {step} saved successfully"
                not in log.read_text()
            ):
                continue
            run = Path(settings["OLMO_RUN"])
            checkpoint = run / f"step{step}" / "model_and_optim"
            if not checkpoint.is_dir():
                raise FileNotFoundError(
                    f"Completed checkpoint was already removed: {checkpoint}"
                )
            name = f"{run.name}-step{step}"
            command = [
                arg for arg in plan["validation"] if not arg.startswith("--dependency=")
            ]
            command = [
                f"--job-name={name}-eval" if arg.startswith("--job-name=") else arg
                for arg in command
            ]
            if a.evaluation_gpu:
                command = [
                    f"--gres=gpu:{a.evaluation_gpu}:1"
                    if arg.startswith("--gres=")
                    else arg
                    for arg in command
                ]
            env = dict(os.environ, **settings)
            if a.evaluation_root:
                if not all(
                    (a.evaluation_root / item).is_dir() for item in ["eval", "packages"]
                ):
                    raise ValueError(
                        "evaluation-root requires shared eval and packages directories"
                    )
                env["OLMO_ROOT"] = str(a.evaluation_root.resolve())
            # Export on CPU before waiting for a GPU allocation. An ephemeral
            # checkpoint can be deleted while its validation job is still queued.
            package = root / "packages" / name
            provenance = package / "provenance.json"
            if provenance.exists():
                existing = json.loads(provenance.read_text())
                if (
                    existing.get("checkpoint") != str(checkpoint)
                    or existing.get("training_validity") == "invalid"
                ):
                    raise ValueError("Existing package is not this valid checkpoint")
                if not all(
                    (package / f).is_file()
                    for f in [
                        "encoder.pt",
                        "config.json",
                        "infer.py",
                        "LICENSE-OlmoEarth",
                    ]
                ):
                    raise ValueError("Existing package is incomplete")
            else:
                subprocess.run(
                    [
                        sys.executable,
                        str(
                            Path(__file__).resolve().parents[2]
                            / "environment/baseline/starter/export.py"
                        ),
                        "--upstream",
                        settings["OLMO_UPSTREAM"],
                        "--variant",
                        settings["OLMO_VARIANT"],
                        "--seed",
                        settings["OLMO_SEED"],
                        "--checkpoint",
                        str(checkpoint),
                        "--output",
                        str(package),
                    ],
                    env=dict(env, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4"),
                    check=True,
                )
            env.update(
                OLMO_NAME=name,
                OLMO_CHECKPOINT=str(checkpoint),
                OLMO_USE_EXISTING_PACKAGE="1",
            )
            result = (
                subprocess.check_output(command, env=env, text=True)
                .strip()
                .split(";")[0]
            )
            if not result.isdigit():
                raise RuntimeError(f"Unexpected sbatch receipt: {result}")
            row = {
                "training_job": plan["training_job"],
                "step": step,
                "evaluation_job": int(result),
                "name": name,
                "package_exported_before_queue": True,
                "evaluation_root": env["OLMO_ROOT"],
                "gpu_request": next(
                    arg for arg in command if arg.startswith("--gres=")
                ),
            }
            records.append(row)
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(row), flush=True)
            pending.remove((plan, step))
        if pending:
            ids = sorted({str(plan["training_job"]) for plan, _ in pending})
            states = subprocess.check_output(
                [
                    "sacct",
                    "-n",
                    "-X",
                    "-j",
                    ",".join(ids),
                    "--format=JobID,State",
                    "-P",
                ],
                text=True,
            )
            for line in states.splitlines():
                job, state = line.split("|")[:2]
                if state.split()[0].rstrip("+") in {
                    "FAILED",
                    "CANCELLED",
                    "OUT_OF_MEMORY",
                    "TIMEOUT",
                    "NODE_FAIL",
                }:
                    raise RuntimeError(f"Training {job} stopped: {state}")
            time.sleep(20)


if __name__ == "__main__":
    main()
