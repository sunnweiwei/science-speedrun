#!/usr/bin/env python3
"""Run input-only generation and publish a sealed output snapshot."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import shutil
import signal
import subprocess
import time

from scoring.contract import load_requests, validate_submission


MAX_OUTPUT_BYTES = 20 * 1024 * 1024 * 1024
MAX_OUTPUT_FILES = 20_100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_final_suite(input_root: Path, allow_small: bool = False) -> dict:
    requests = load_requests(input_root)
    manifest = json.loads((input_root / "suite_manifest.json").read_text())
    if manifest.get("schema") != "crystalgen-final-suite/v1":
        raise ValueError("invalid final-suite manifest")
    if manifest.get("requests_sha256") != sha256(input_root / "requests.jsonl"):
        raise ValueError("final request checksum mismatch")
    expected_manifest_fields = {"schema", "requests_sha256", "unconditional_samples"}
    if set(manifest) != expected_manifest_fields:
        raise ValueError("invalid final-suite manifest fields")
    unconditional = sum(item["n_samples"] for item in requests.values())
    if not allow_small and (len(requests) != 1 or unconditional != 2500):
        raise ValueError("production suite requires one 2500-sample unconditional request")
    if manifest["unconditional_samples"] != unconditional:
        raise ValueError("final-suite counts do not match requests")
    return requests


def kill_agent_processes(uid: int) -> None:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            status = (entry / "status").read_text()
            process_uid = int(next(line for line in status.splitlines() if line.startswith("Uid:")).split()[1])
            if process_uid == uid:
                os.kill(int(entry.name), signal.SIGKILL)
        except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
            pass


def freeze_submission(source: Path, destination: Path) -> Path:
    """Copy the validated package into a root-owned read-only replay tree."""
    if destination.exists():
        raise FileExistsError("frozen submission destination already exists")
    shutil.copytree(source, destination, symlinks=False)
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise ValueError("submission symlinks are not allowed")
        if path.is_file():
            path.chmod(0o444 | (path.stat().st_mode & 0o111))
    for path in sorted((item for item in destination.rglob("*") if item.is_dir()), reverse=True):
        path.chmod(0o555)
    destination.chmod(0o555)
    return destination


def run_submission(submission: Path, input_root: Path, work: Path, timeout: int) -> Path:
    agent = pwd.getpwnam("agent")
    output = work / "untrusted-output"
    output.mkdir()
    os.chown(output, agent.pw_uid, agent.pw_gid)
    log = work / "generation.log"
    env = {
        "PATH": "/opt/conda/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME": agent.pw_dir,
        "PYTHONUNBUFFERED": "1",
        "PIP_NO_INDEX": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"),
    }

    def demote() -> None:
        os.setgid(agent.pw_gid)
        os.setuid(agent.pw_uid)

    process = None
    try:
        with log.open("wb") as stream:
            process = subprocess.Popen(
                ["bash", str(submission / "generate.sh"), str(input_root), str(output)],
                cwd=submission,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=demote,
            )
            code = process.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, process.args)
    except subprocess.TimeoutExpired:
        if process:
            os.killpg(process.pid, signal.SIGKILL)
        raise
    finally:
        if process:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        kill_agent_processes(agent.pw_uid)
    return output


def seal_output(untrusted: Path, sealed: Path, request_path: Path) -> None:
    if sealed.exists():
        raise FileExistsError("sealed destination already exists")
    temporary = sealed.with_name(sealed.name + ".tmp")
    temporary.mkdir(parents=True)
    hashes = {}
    total_size = 0
    file_count = 0
    for source in sorted(untrusted.rglob("*")):
        if source.is_symlink():
            raise ValueError("output symlinks are not allowed")
        if not source.is_file():
            continue
        relative = source.relative_to(untrusted)
        destination = temporary / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_count += 1
        total_size += source.stat().st_size
        if file_count > MAX_OUTPUT_FILES or total_size > MAX_OUTPUT_BYTES:
            raise ValueError("generated output exceeds file-count or byte limit")
        shutil.copyfile(source, destination, follow_symlinks=False)
        hashes[relative.as_posix()] = sha256(destination)
        destination.chmod(0o444)
    snapshot = {
        "schema": "crystalgen-output-snapshot/v0",
        "requests_sha256": sha256(request_path),
        "files": hashes,
    }
    snapshot_path = temporary / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    snapshot_path.chmod(0o444)
    for directory in sorted((path for path in temporary.rglob("*") if path.is_dir()), reverse=True):
        directory.chmod(0o555)
    os.replace(temporary, sealed)
    sealed.chmod(0o555)
    (sealed.parent / "READY").write_text(sha256(sealed / "snapshot.json") + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--input", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--sealed", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--output", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--timeout", type=int, default=86400)
    parser.add_argument("--grader-wait", type=int, default=259200)
    parser.add_argument("--allow-small-suite", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.sealed.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(args.output, 0o777)
    validate_final_suite(args.input, args.allow_small_suite)
    work = args.output / "main"
    work.mkdir()
    try:
        validate_submission(args.submission)
        frozen_submission = freeze_submission(args.submission, work / "frozen-submission")
        validate_submission(frozen_submission)
        untrusted = run_submission(frozen_submission, args.input, work, args.timeout)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        # An invalid package, a failed generate.sh or a timeout scores 0 (TASK_STANDARD.md). The
        # INVALID marker tells the grader there is nothing to score.
        reason = f"{type(error).__name__}: {error}"
        (args.output / "reward.json").write_text(json.dumps(
            {"reward": 0.0, "unconditional_sun_metasun": 0.0, "invalid_submission": 1.0},
            indent=2, sort_keys=True) + "\n")
        (args.output / "report.json").write_text(json.dumps(
            {"status": "invalid_submission", "error": reason[:4000]}, indent=2) + "\n")
        (args.sealed.parent / "INVALID").write_text(reason + "\n")
        print(f"invalid submission: {reason}")
        return
    seal_output(untrusted, args.sealed, args.input / "requests.jsonl")
    if not args.no_wait:
        trusted_output = args.sealed.parent / "results"
        deadline = time.monotonic() + args.grader_wait
        while not (trusted_output / "reward.json").is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError("trusted grader did not publish reward.json")
            time.sleep(1)
        for name in ("report.json", "reward.json"):
            shutil.copyfile(trusted_output / name, args.output / name)


if __name__ == "__main__":
    main()
