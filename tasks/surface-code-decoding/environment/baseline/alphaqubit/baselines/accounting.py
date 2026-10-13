"""Snapshot Slurm accounting for explicitly registered campaign jobs only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from .cache import atomic_json


def allocation_ceiling(jobs: list[dict[str, Any]], records: list[dict[str, Any]]) -> float:
    """Keep original submission caps; replace retired caps with settled usage.

    A pivot can release unused allocation only after every reported task in the
    retired job is terminal. Missing accounting is not evidence of zero usage.
    """
    terminal = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY",
                "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
    ceiling = 0.0
    for job in jobs:
        if job.get("retired", False):
            rows = [row for row in records if row["job_id"].split("_", 1)[0] == str(job["id"])]
            if not rows or any(row["state"].split()[0].rstrip("+") not in terminal for row in rows):
                raise ValueError(f"Retired job {job['id']} lacks terminal accounting")
            ceiling += sum(row["gpu_hours"] for row in rows)
        elif not job.get("cancelled_before_allocation", False):
            ceiling += job["gpu_count"] * job["array_tasks"] * job["max_hours_per_task"]
    return ceiling


def snapshot(asset_root: Path) -> dict[str, Any]:
    ledger = json.loads((asset_root / "campaign.json").read_text())
    jobs = ledger["jobs"]
    identifiers = [str(job["id"]) for job in jobs]
    if len(set(identifiers)) != len(identifiers) or not all(re.fullmatch(r"\d+", x) for x in identifiers):
        raise ValueError("Campaign job IDs must be unique Slurm integer IDs")
    output = subprocess.check_output([
        "sacct", "-X", "-n", "-P", "-j", ",".join(identifiers),
        "--format=JobID%60,State%30,ElapsedRaw,AllocTRES%200,Start,End,NodeList%100",
    ], text=True)
    records: list[dict[str, Any]] = []
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) < 7:
            raise ValueError(f"Unexpected Slurm accounting record: {line}")
        job_id, state, elapsed, resources, start, end, nodes = fields[:7]
        if job_id.split("_", 1)[0] not in identifiers:
            raise ValueError("Slurm returned a job outside the campaign")
        allocations = dict(item.split("=", 1) for item in resources.split(",") if "=" in item)
        # The untyped total already includes typed GPUs. Do not count both.
        gpu_count = int(allocations.get("gres/gpu", "0"))
        seconds = int(elapsed or 0)
        records.append({
            "job_id": job_id, "state": state, "elapsed_seconds": seconds,
            "allocated_gpus": gpu_count, "allocated_cpus": int(allocations.get("cpu", "0")),
            "gpu_hours": gpu_count * seconds / 3600,
            "allocated_cpu_hours": int(allocations.get("cpu", "0")) * seconds / 3600,
            "start": start, "end": end, "nodes": nodes, "allocated_tres": resources,
        })
    actual = sum(record["gpu_hours"] for record in records)
    active = sum(record["allocated_gpus"] for record in records
                 if record["state"] in {"RUNNING", "COMPLETING"})
    declared = allocation_ceiling(jobs, records)
    report = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "only campaign.json job IDs; elapsed allocated time, including setup and idle allocation",
        "actual_allocated_gpu_hours": actual,
        "actual_allocated_cpu_hours": sum(record["allocated_cpu_hours"] for record in records),
        "active_allocated_gpus": active,
        "declared_allocation_ceiling_gpu_hours": declared,
        "allocation_ceiling_definition": "settled actual use for retired jobs plus original hard caps for other jobs",
        "max_actual_allocated_gpu_hours": ledger["max_actual_allocated_gpu_hours"],
        "max_concurrent_gpus": ledger["max_concurrent_gpus"],
        "remaining_actual_gpu_hours": ledger["max_actual_allocated_gpu_hours"] - actual,
        "records": records,
        "accounting_caveat": "Slurm accounting may lag; typed GPU card-hours are not H200-equivalent hours",
    }
    atomic_json(asset_root / "accounting.json", report)
    if actual > ledger["max_actual_allocated_gpu_hours"] or declared > ledger["max_actual_allocated_gpu_hours"]:
        raise RuntimeError("Campaign actual use or declared allocation ceiling exceeds the GPU-hour budget")
    if active > ledger["max_concurrent_gpus"]:
        raise RuntimeError("Campaign exceeds the concurrent GPU limit")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()
    report = snapshot(args.asset_root)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}))


if __name__ == "__main__":
    main()
