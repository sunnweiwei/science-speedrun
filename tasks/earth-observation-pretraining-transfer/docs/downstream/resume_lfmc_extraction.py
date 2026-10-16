"""Resume a terminal extractor using its logged closed-file milestone.

The accepted prefix relies on the pinned original copier's successful-close log,
not a new byte audit of those files. Suffix files are compared bytewise; incomplete
ones are quarantined. All future suffix writes use atomic rename and checkpoints.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import os
import shutil
import subprocess
import tarfile

EXPECTED = {"train": 31175, "val": 8262, "test": 4585}


def logged_prefix(path):
    values = []
    for line in path.read_text().splitlines():
        if not line.startswith("{'files':"):
            continue
        row = ast.literal_eval(line)
        if row["windows"] != EXPECTED or row["files"] % 100000:
            raise ValueError("Invalid closed-file milestone")
        values.append(row["files"])
    if not values or values != sorted(set(values)):
        raise ValueError("Missing or inconsistent extraction progress")
    return values[-1]


def equal_stream(source, existing):
    with existing.open("rb") as target:
        while chunk := source.read(1024**2):
            if target.read(len(chunk)) != chunk:
                return False
        return target.read(1) == b""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-job", required=True)
    parser.add_argument("--previous-log", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    args = parser.parse_args()
    state = subprocess.check_output(
        ["sacct", "-X", "-n", "-j", args.previous_job, "--format=JobIDRaw,State", "-P"],
        text=True,
    )
    states = [
        r.split("|")[1]
        for r in state.splitlines()
        if r.startswith(args.previous_job + "|")
    ]
    if len(states) != 1 or states[0] not in {"TIMEOUT", "FAILED", "OUT_OF_MEMORY"}:
        raise ValueError("Previous writer is not an authoritative terminal failure")
    if not args.output.is_dir() or (args.output / "extraction.json").exists():
        raise ValueError("Require existing incomplete extraction")
    original = Path(__file__).with_name("extract_trainval.py")
    source_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    if source_hash != "8892eff586c111f2820c8a8ae1aae0ccdab007c8eba070aef89f7908f6d94f0f":
        raise ValueError("Original copier differs from the audited milestone implementation")
    prefix = logged_prefix(args.previous_log)
    receipt = {
        "previous_job": args.previous_job,
        "previous_state": states[0],
        "previous_log_sha256": hashlib.sha256(
            args.previous_log.read_bytes()
        ).hexdigest(),
        "original_extractor_sha256": source_hash,
        "accepted_closed_file_prefix": prefix,
        "prefix_evidence": "Original copier logs only after destination close; prefix not independently rehashed.",
        "archive": str(args.archive),
        "archive_bytes": args.archive.stat().st_size,
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    with (args.output / "resume-launch.json").open("x") as handle:
        json.dump(receipt, handle, indent=2)
    print(json.dumps(receipt), flush=True)
    decisions, counts = {}, {k: 0 for k in EXPECTED}
    with tarfile.open(args.archive, "r:") as archive:
        for member in archive:
            archive.members.clear()
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or "windows" not in parts:
                continue
            i = parts.index("windows")
            if parts[i + 3 :] != ("metadata.json",):
                continue
            meta = json.load(archive.extractfile(member))
            key = "/".join(parts[i + 1 : i + 3])
            split = meta["options"]["split"]
            if key in decisions or split not in counts:
                raise ValueError("Invalid split metadata")
            decisions[key] = split
            counts[split] += 1
    if counts != EXPECTED:
        raise ValueError("Wrong archive window population")
    files = bytes_written = existing_verified = quarantined = 0
    with tarfile.open(args.archive, "r:") as archive:
        for member in archive:
            archive.members.clear()
            parts = PurePosixPath(member.name).parts
            if (
                not parts
                or ".." in parts
                or member.name.startswith("/")
                or member.issym()
                or member.islnk()
            ):
                raise ValueError("Unsafe archive member")
            if not member.isfile():
                continue
            relative = Path(*parts[1:])
            if "windows" in parts:
                i = parts.index("windows")
                key = "/".join(parts[i + 1 : i + 3])
                if decisions[key] == "test":
                    continue
            elif relative.as_posix() != "config.json":
                continue
            files += 1
            bytes_written += member.size
            if files <= prefix:
                if files == prefix:
                    print(json.dumps({"accepted_prefix_reached": files}), flush=True)
                continue
            target = args.output / relative
            if target.exists():
                if target.stat().st_size == member.size and equal_stream(
                    archive.extractfile(member), target
                ):
                    existing_verified += 1
                    continue
                saved = args.quarantine / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                if saved.exists():
                    raise ValueError("Quarantine file already exists")
                target.rename(saved)
                quarantined += 1
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".resume-tmp")
            with archive.extractfile(member) as source, temporary.open("xb") as dest:
                shutil.copyfileobj(source, dest)
            temporary.rename(target)
            if files % 10000 == 0:
                checkpoint = {
                    "closed_files": files,
                    "payload_bytes": bytes_written,
                    "archive_offset": archive.offset,
                }
                tmp = args.output / "resume-checkpoint.tmp"
                tmp.write_text(json.dumps(checkpoint) + "\n")
                tmp.replace(args.output / "resume-checkpoint.json")
                print(json.dumps(checkpoint), flush=True)
    if files < prefix or not (args.output / "config.json").exists():
        raise ValueError("Prefix/config inconsistent with archive")
    report = {
        **receipt,
        "window_counts": counts,
        "extracted_splits": ["train", "val"],
        "heldout_window_payloads_read": False,
        "files": files,
        "payload_bytes_written": bytes_written,
        "suffix_existing_files_byte_verified": existing_verified,
        "quarantined_incomplete_files": quarantined,
    }
    with (args.output / "extraction.json").open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
