"""Retry only authoritative terminal failures into a separate output root."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from acquire_external_lfmc_cohort import task_inputs


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--original-array", required=True)
    parser.add_argument("--official-config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    old_job = f"{args.original_array}_{index}"
    accounting = subprocess.check_output(
        ["sacct", "-j", old_job, "-n", "-P", "--format=JobID%40,State,ExitCode"],
        text=True,
    )
    states = [
        line.split("|")
        for line in accounting.splitlines()
        if line.split("|")[0] == old_job
    ]
    if len(states) != 1 or states[0][1] not in {
        "FAILED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
    }:
        raise ValueError(
            f"Prior task must be an authoritative terminal failure: {accounting}"
        )
    old = args.original_root / f"{index:04d}"
    if not (old / "failure.json").exists() or (old / "completion.json").exists():
        raise ValueError("Expected preserved unsuccessful prior acquisition")
    parent_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    child, grid = task_inputs(json.loads(args.manifest.read_text()), index)
    child.update(parent_manifest_sha256=parent_hash, parent_task_index=index)
    if json.loads((old / "manifest.json").read_text()) != child:
        raise ValueError("Retry changed fixed manifest row")
    out = args.root / f"{index:04d}"
    out.mkdir(parents=True, exist_ok=False)
    write(out / "manifest.json", child)
    write(out / "grid.json", grid)
    source = Path(__file__).parent
    names = [
        Path(__file__).name,
        "materialize_external_lfmc_repaired.py",
        "acquire_external_lfmc_cohort.py",
        "materialize_external_lfmc.py",
        "verify_external_lfmc_mosaic.py",
        "prepare_external_lfmc_input.py",
        "external_lfmc_input.py",
        "lfmc_window.py",
        "external_lfmc_availability.py",
    ]
    receipt = {
        "task_index": index,
        "site_id": grid["site_id"],
        "date": grid["date"],
        "parent_manifest_sha256": parent_hash,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "target_used": False,
        "model_used": False,
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in names
        },
        "preserved_failure_directory": str(old),
        "prior_job_accounting": states[0],
        "explicit_deviation": "Repair invalid catalog footprints before official projection; preserve all original failures and repaired WKB audit.",
    }
    write(out / "launch.json", receipt)
    env = dict(os.environ, LFMC_GEOMETRY_REPAIR_LOG=str(out / "geometry-repairs.jsonl"))
    steps = [
        [
            "materialize_external_lfmc_repaired.py",
            "--manifest",
            str(out / "manifest.json"),
            "--probe",
            str(out / "grid.json"),
            "--official-config",
            str(args.official_config),
            "--output",
            str(out / "dataset"),
        ],
        [
            "verify_external_lfmc_mosaic.py",
            "--dataset",
            str(out / "dataset"),
            "--output",
            str(out / "verification.json"),
        ],
        [
            "prepare_external_lfmc_input.py",
            "--dataset",
            str(out / "dataset"),
            "--verification",
            str(out / "verification.json"),
            "--manifest",
            str(out / "manifest.json"),
            "--output",
            str(out / "input"),
        ],
    ]
    for step in steps:
        result = subprocess.run(
            [sys.executable, str(source / step[0]), *step[1:]], env=env
        )
        if result.returncode:
            write(
                out / "failure.json", {"step": step[0], "exit_code": result.returncode}
            )
            raise SystemExit(result.returncode)
    receipt["input_sha256"] = hashlib.sha256(
        (out / "input/input.npz").read_bytes()
    ).hexdigest()
    write(out / "completion.json", receipt)


if __name__ == "__main__":
    main()
