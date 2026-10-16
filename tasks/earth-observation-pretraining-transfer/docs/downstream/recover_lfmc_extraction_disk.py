"""After-terminal extraction gate with preserved failures and local archive staging."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from extract_trainval import extract


def copy_to_disk(source, destination):
    digest = hashlib.sha256()
    total = 0
    with source.open("rb") as src, destination.open("xb") as dst:
        while chunk := src.read(16 * 1024**2):
            dst.write(chunk)
            digest.update(chunk)
            total += len(chunk)
            if total % (256 * 1024**2) == 0:
                dst.flush()
                os.fsync(dst.fileno())
                os.posix_fadvise(dst.fileno(), 0, total, os.POSIX_FADV_DONTNEED)
                os.posix_fadvise(src.fileno(), 0, total, os.POSIX_FADV_DONTNEED)
            if total % (1024**3) == 0:
                print(json.dumps({"staged_bytes": total}), flush=True)
        dst.flush()
        os.fsync(dst.fileno())
    return {"bytes": total, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-job", required=True)
    parser.add_argument("--staging-root", type=Path, default=Path("/var/tmp"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = subprocess.run(
        ["sacct", "-X", "-n", "-j", args.previous_job, "--format=JobIDRaw,State", "-P"],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = [r.split("|") for r in result.stdout.splitlines() if r.strip()]
    states = [r[1].split()[0] for r in rows if r[0] == args.previous_job]
    terminal = {
        "COMPLETED",
        "FAILED",
        "TIMEOUT",
        "CANCELLED",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
    }
    if len(states) != 1 or states[0] not in terminal:
        raise RuntimeError("Previous job is not confirmed terminal; do not restart")
    report = args.output / "extraction.json"
    if report.exists():
        existing = json.loads(report.read_text())
        if existing["window_counts"] != {"train": 31175, "val": 8262, "test": 4585}:
            raise ValueError("Invalid prior coverage")
        if (
            existing["extracted_splits"] != ["train", "val"]
            or existing["heldout_window_payloads_read"]
        ):
            raise ValueError("Invalid prior extraction policy")
        print("Previous extraction complete; reuse without rerun", flush=True)
        return
    staging_root = args.staging_root
    filesystem = subprocess.check_output(
        ["/usr/bin/stat", "-f", "-c", "%T", str(staging_root)], text=True
    ).strip()
    if filesystem not in {"ext2/ext3", "xfs", "btrfs"}:
        raise RuntimeError(f"Require real node-local disk, got {filesystem}")
    required = args.archive.stat().st_size + 5 * 1024**3
    if shutil.disk_usage(staging_root).free < required:
        raise RuntimeError(
            "Insufficient node-local staging space; preserve previous output"
        )
    preserved = args.output.with_name(args.output.name + "-failed-" + args.previous_job)
    if preserved.exists():
        raise ValueError("Failure preservation path already exists")
    with tempfile.TemporaryDirectory(
        prefix="lfmc-archive-", dir=staging_root
    ) as temporary:
        local_archive = Path(temporary) / "lfmc.tar"
        print("Staging existing archive to node-local disk", flush=True)
        copy_report = copy_to_disk(args.archive, local_archive)
        if local_archive.stat().st_size != args.archive.stat().st_size:
            raise ValueError("Incomplete staged archive")
        if args.output.exists():
            args.output.rename(preserved)
        print("Local archive staged; starting train/validation extraction", flush=True)
        result = extract(
            local_archive, args.output, {"train": 31175, "val": 8262, "test": 4585}
        )
        result.update(
            archive=str(args.archive),
            node_local_archive_staging=True,
            staging_filesystem=filesystem,
            staged_archive_copy=copy_report,
            previous_job=args.previous_job,
            previous_state=states[0],
            previous_output_preserved=str(preserved),
            recovery_job=os.environ.get("SLURM_JOB_ID"),
        )
        report.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
