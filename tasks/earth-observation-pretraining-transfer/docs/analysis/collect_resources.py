"""Snapshot Slurm resources for explicitly listed runs, excluding unrelated jobs."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipts", type=Path, nargs="+", required=True)
    p.add_argument("--midpoints", type=Path, nargs="+", default=[])
    p.add_argument("--extra-jobs", type=int, nargs="+", default=[])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    plans = [json.loads(path.read_text()) for path in a.receipts]
    ids = {
        int(plan[key])
        for plan in plans
        for key in ("training_job", "validation_job", "dispatch_job")
        if key in plan
    }
    pending_validation_receipts = []
    for plan in plans:
        if "validation_receipt" not in plan:
            continue
        path = Path(plan["validation_receipt"])
        if not path.exists():
            pending_validation_receipts.append(str(path))
            continue
        validation = json.loads(path.read_text())
        ids.update(
            int(validation[key])
            for key in ("array_job", "awf_job", "validation_job")
            if key in validation
        )
    ids.update(a.extra_jobs)
    for midpoint_file in a.midpoints:
        ids.update(
            int(row["evaluation_job"]) for row in json.loads(midpoint_file.read_text())
        )
    raw = subprocess.check_output(
        [
            "sacct",
            "-n",
            "-P",
            "-j",
            ",".join(map(str, sorted(ids))),
            "--format=JobID,JobName%100,Partition,State,ElapsedRaw,AllocTRES%200,MaxRSS,ExitCode",
        ],
        text=True,
    )
    records = []
    fields = [
        "job_id",
        "name",
        "partition",
        "state",
        "elapsed_seconds",
        "allocated_tres",
        "max_rss",
        "exit_code",
    ]
    for line in raw.splitlines():
        parts = line.split("|")
        if len(parts) < len(fields):
            raise ValueError(f"Malformed Slurm accounting row: {line}")
        row = dict(zip(fields, parts))
        row["elapsed_seconds"] = int(row["elapsed_seconds"])
        if "." not in row["job_id"]:
            if row["partition"] != "configured-gpu-partition":
                raise ValueError("GPU experiment used an unexpected partition")
            match = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", row["allocated_tres"])
            row["allocated_gpu_seconds"] = (
                int(match.group(1)) * row["elapsed_seconds"] if match else 0
            )
        records.append(row)
    memory = {}
    for plan in plans:
        job = plan["training_job"]
        log = Path(plan["settings"]["OLMO_DATA"]) / f"train-{job}.log"
        values = (
            re.findall(r"GPU reserved mem \(GiB\)=([\d.]+)", log.read_text())
            if log.exists()
            else []
        )
        memory[str(job)] = max(map(float, values)) if values else None
    report = {
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Only supplied launch receipts, midpoint lists and explicit extra jobs; not total development cost",
        "accounting": records,
        "pending_validation_receipts": pending_validation_receipts,
        "rank0_maximum_logged_pytorch_reserved_gib": memory,
        "limitations": [
            "Running jobs have partial elapsed time; pending jobs have no allocation yet.",
            "GPU seconds measure allocated resources, not utilization or FLOPs.",
            "Logged rank0 memory is sampled and excludes non-PyTorch allocations; not a full-device peak or minimum capacity.",
        ],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(a.output)


if __name__ == "__main__":
    main()
