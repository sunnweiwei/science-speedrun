#!/usr/bin/env python3
"""Label-free inference supervisor; trusted scoring runs in the grader service."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import pwd
import resource
import shutil
import signal
import socket
import stat
import subprocess
import time
import uuid

from common import sha256, write_json


def invalid(output: Path, status: str, error: Exception) -> None:
    write_json(output / "report.json", {
        "schema": "natural-disease-progression-report/v3",
        "status": status,
        "model_quality_measured": False,
        "error": f"{type(error).__name__}: {error}",
    })
    write_json(output / "reward.json", {"reward": 0.0})


def kill_uid_processes(uid: int) -> None:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text()
            owner = int(next(row for row in status.splitlines()
                             if row.startswith("Uid:")).split()[1])
            if owner == uid:
                os.kill(int(entry.name), signal.SIGKILL)
        except (OSError, StopIteration):
            pass


def validate_submission(root: Path) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Submission must be an ordinary directory")
    total = 0
    for path in root.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not (stat.S_ISREG(info.st_mode)
                                              or stat.S_ISDIR(info.st_mode)):
            raise ValueError("Submission may contain only ordinary files and directories")
        if stat.S_ISREG(info.st_mode):
            # Hard links are accepted: the verifier reads its own copy of the package, so a second
            # name for a file cannot change what is scored (atomic publishers often use `cp -al`).
            total += info.st_size
    if total > 50 * 1024**3:
        raise ValueError("Submission exceeds 50 GiB")
    inference = root / "infer.sh"
    if not inference.is_file():
        raise FileNotFoundError("submission/infer.sh is required")
    if not (root / "README.md").is_file():
        raise FileNotFoundError("submission/README.md is required for architecture review")
    # Model/training/configuration/weight filenames are intentionally not guessed
    # here. Shallow extension checks cannot establish that infer.sh actually uses
    # a jointly trained neural model; reviewers inspect the frozen package path.
    return inference


def run_submission(submission: Path, input_dir: Path, work: Path,
                   timeout: int) -> Path:
    agent = pwd.getpwnam("agent")
    frozen = work / "submission"
    shutil.copytree(submission, frozen)
    for path in frozen.rglob("*"):
        os.chown(path, 0, 0)
        path.chmod(0o555 if path.is_dir() or path.stat().st_mode & 0o111 else 0o444)
    frozen.chmod(0o555)
    output = work / "output"
    output.mkdir()
    os.chown(output, agent.pw_uid, agent.pw_gid)
    environment = {
        "PATH": "/opt/conda/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME": agent.pw_dir,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PIP_NO_INDEX": "1",
        "HF_HUB_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    }

    def demote() -> None:
        os.setgroups([])
        os.setgid(agent.pw_gid)
        os.setuid(agent.pw_uid)
        resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**3, 8 * 1024**3))
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(38, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")

    process = None
    try:
        with (work / "inference.log").open("wb") as stream:
            process = subprocess.Popen(
                ["bash", str(frozen / "infer.sh"), str(input_dir), str(output)],
                cwd=frozen, env=environment, stdout=stream,
                stderr=subprocess.STDOUT, preexec_fn=demote,
                start_new_session=True)
            code = process.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, process.args)
    finally:
        if process:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        kill_uid_processes(agent.pw_uid)
    return output


def seal_prediction(source_dir: Path, sealed: Path, input_dir: Path,
                    run_id: str) -> None:
    source = source_dir / "predictions.parquet"
    if not source.exists() and not source.is_symlink():
        raise ValueError("Missing predictions.parquet")
    info = source.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 8 * 1024**3:
        raise ValueError("predictions.parquet must be a regular file no larger than 8 GiB")
    temporary = sealed.parent / f".snapshot-{run_id}"
    temporary.mkdir(mode=0o700)
    destination = temporary / "predictions.parquet"
    with os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW), "rb") as reader:
        with destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
    destination.chmod(0o444)
    write_json(temporary / "snapshot.json", {
        "schema": "natural-disease-progression-prediction-snapshot/v1",
        "run_id": run_id,
        "input_manifest_sha256": sha256(input_dir / "manifest.json"),
        "predictions_sha256": sha256(destination),
    })
    (temporary / "snapshot.json").chmod(0o444)
    temporary.chmod(0o555)
    os.replace(temporary, sealed)
    write_json(sealed.parent / "READY", {
        "run_id": run_id,
        "snapshot_sha256": sha256(sealed / "snapshot.json"),
    })


def evaluate(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    args.sealed.parent.mkdir(parents=True, exist_ok=True)
    args.sealed.parent.chmod(0o700)
    for name in ("reward.json", "report.json"):
        (args.output / name).unlink(missing_ok=True)
    if any(name != "lo" for _, name in socket.if_nameindex()):
        raise RuntimeError("Verifier inference requires a network_mode:none namespace")
    if args.sealed.exists() or (args.sealed.parent / "READY").exists():
        raise RuntimeError("Stale sealed output; use a fresh verifier volume")
    manifest_path = args.input / "manifest.json"
    if not manifest_path.is_file() or not (args.input / "events.parquet").is_file():
        raise FileNotFoundError("Public test input is incomplete")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("split") != "test":
        raise ValueError("Verifier main accepts only the public test split")
    try:
        validate_submission(args.submission)
    except (OSError, ValueError) as error:
        invalid(args.output, "invalid_submission", error)
        return
    run_id = uuid.uuid4().hex
    work = args.output / f"inference-{run_id}"
    work.mkdir()
    try:
        untrusted = run_submission(args.submission, args.input, work, args.timeout)
    except subprocess.TimeoutExpired as error:
        invalid(args.output, "inference_timeout", error)
        return
    except subprocess.CalledProcessError as error:
        invalid(args.output, "inference_failed", error)
        return
    try:
        seal_prediction(untrusted, args.sealed, args.input, run_id)
    except ValueError as error:
        invalid(args.output, "invalid_prediction", error)
        return
    result_root = args.sealed.parent / "results" / run_id
    deadline = time.monotonic() + args.grader_timeout
    while not (result_root / "DONE").is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("Trusted grader did not publish a result")
        time.sleep(0.25)
    report = json.loads((result_root / "report.json").read_text())
    if report.get("run_id") != run_id or report.get("status") == "infrastructure_error":
        raise RuntimeError(f"Trusted grader infrastructure error: {report}")
    for name in ("report.json", "reward.json"):
        shutil.copyfile(result_root / name, args.output / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/input"))
    parser.add_argument("--submission", type=Path, default=Path("/workspace/submission"))
    parser.add_argument("--sealed", type=Path, default=Path("/sealed/run"))
    parser.add_argument("--output", type=Path, default=Path("/logs/verifier"))
    parser.add_argument("--timeout", type=int, default=42000)
    parser.add_argument("--grader-timeout", type=int, default=600)
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("Inference supervisor must run as root before dropping to uid 10001")
    try:
        evaluate(args)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "reward.json").unlink(missing_ok=True)
        write_json(args.output / "report.json", {
            "schema": "natural-disease-progression-report/v3",
            "status": "infrastructure_error",
            "model_quality_measured": False,
            "error": f"{type(error).__name__}: {error}",
        })
        raise


if __name__ == "__main__":
    main()
