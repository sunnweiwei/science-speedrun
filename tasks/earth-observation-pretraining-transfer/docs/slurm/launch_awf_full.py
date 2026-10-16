"""Launch the full official AWF schedule only after the paired integration gate."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/analysis"))
from verify_awf_pair import verify

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--receipt", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--dry-run", action="store_true")
a = p.parse_args()
if a.output.exists():
    raise ValueError("Use a new launch receipt")
receipt = json.loads(a.receipt.read_text())
gate = verify(a.root, receipt)
track = Path(__file__).resolve().parents[2]
for source, expected in receipt["source_sha256"].items():
    if source in ("slurm/awf_finetune.sbatch", "docs/slurm/awf_finetune.sbatch"):
        continue
    if hashlib.sha256((track / source).read_bytes()).hexdigest() != expected:
        raise ValueError(f"Source changed after pilot: {source}")
plans = []
for kind in ["trained", "random"]:
    package = (
        Path(os.environ.get("EARTH_DATA_ROOT", "/data/earth"))
        / "olmoearth/packages"
        / receipt["package_names"][kind]
    )
    out = a.root / f"full-{kind}-v1"
    if out.exists():
        raise ValueError(f"Existing training output {out}")
    command = [
        "sbatch",
        "--parsable",
        "--partition=" + os.environ.get("EARTH_SLURM_PARTITION", "gpu"),
        "--gres=gpu:1",
        "--time=48:00:00",
        f"--job-name=awf-full-{kind}",
        str(track / "docs/slurm/awf_finetune.sbatch"),
    ]
    plans.append(
        {"kind": kind, "package": str(package), "output": str(out), "command": command}
    )
report = {
    "recorded_utc": datetime.now(timezone.utc).isoformat(),
    "gate": gate,
    "planned_epochs": 100,
    "unfreeze_epoch": 20,
    "train_rows": 1115,
    "val_rows": 344,
    "full_batches": True,
    "head_and_loader_seed": 8123,
    "validation_rng_seed": 73491,
    "checkpoint_selection": "highest validation accuracy",
    "test_used": False,
    "scope": "Development pair only; no independent generalization or multi-seed claim",
    "plans": plans,
    "dry_run": a.dry_run,
}
a.output.write_text(json.dumps(report, indent=2) + "\n")
if not a.dry_run:
    for plan in plans:
        env = dict(
            os.environ,
            OLMO_PACKAGE=plan["package"],
            OLMO_OUTPUT=plan["output"],
            OLMO_PILOT="0",
        )
        job = subprocess.check_output(plan["command"], env=env, text=True).strip()
        plan["job"] = int(job.split(";")[0])
        a.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
