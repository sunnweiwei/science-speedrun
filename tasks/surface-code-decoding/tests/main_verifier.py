#!/usr/bin/env python3
"""Trusted label-free inference supervisor; labels exist only in grader service."""
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
import sys
import time
import uuid
from typing import Any

PROJECT = Path(__file__).resolve().parents[1] / "environment" / "baseline" / "alphaqubit" if (Path(__file__).resolve().parents[1] / "environment" / "baseline" / "alphaqubit").is_dir() else Path(__file__).resolve().parents[1]  # the baseline project: environment/baseline/alphaqubit here, /opt/task in the verifier image
sys.path.insert(0, str(PROJECT))
from qecbench.data import contained, load_manifest, sha256, write_json


def validate_submission(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Submission must be a directory")
    size = 0
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("Submission must contain ordinary files/directories only")
        if path.is_file():
            # Hard links are accepted: the verifier scores an immutable relocated copy of the package.
            size += path.stat().st_size
    if size > 50 * 1024 ** 3:
        raise ValueError("Submission exceeds 50 GiB")
    for name in ("infer.sh", "README.md", "model_manifest.json", "requirements.lock"):
        contained(root, name)
    manifest = json.loads((root / "model_manifest.json").read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != "surface-code-decoding-model/v1":
        raise ValueError("Unsupported submission manifest")
    records = manifest.get("checkpoints")
    if not isinstance(records, list) or not records or len(records) > 1000:
        raise ValueError("Submit a trained model/checkpoint or serialized decoder state")
    paths = set()
    for record in records:
        path = contained(root, record["path"])
        if record["path"] in paths or path.stat().st_size == 0 or sha256(path) != record["sha256"]:
            raise ValueError("Duplicate/empty/corrupt checkpoint")
        paths.add(record["path"])
    for field in ("model_config", "training_provenance"):
        if not isinstance(json.loads(contained(root, manifest[field]).read_text()), dict):
            raise ValueError(f"{field} must be a JSON object")
    return manifest


def kill_uid_processes(uid: int) -> None:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text()
            owner = int(next(row for row in status.splitlines() if row.startswith("Uid:")).split()[1])
            if owner == uid:
                os.kill(int(entry.name), signal.SIGKILL)
        except (OSError, StopIteration):
            pass


def run_submission(submission: Path, inputs: Path, work: Path, timeout: int) -> Path:
    identity = pwd.getpwnam("agent")
    frozen = work / "submission"
    shutil.copytree(submission, frozen)
    for path in frozen.rglob("*"):
        os.chown(path, 0, 0)
        path.chmod(0o555 if path.is_dir() or (path.stat().st_mode & 0o111) else 0o444)
    frozen.chmod(0o555)
    output = work / "output"
    output.mkdir()
    os.chown(output, identity.pw_uid, identity.pw_gid)
    environment = {"PATH": "/opt/qec/.venv/bin:/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp/agent-home",
                   "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                   "PIP_NO_INDEX": "1", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                   "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")}

    def demote() -> None:
        os.setgroups([])
        os.setgid(identity.pw_gid)
        os.setuid(identity.pw_uid)
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 ** 3, 1024 ** 3))
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(38, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")

    process = None
    try:
        with (work / "inference.log").open("wb") as log:
            process = subprocess.Popen(["bash", str(frozen / "infer.sh"), str(inputs), str(output)], cwd=frozen,
                env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=demote)
            code = process.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, process.args)
    finally:
        if process:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        kill_uid_processes(identity.pw_uid)
    return output


def seal_output(output: Path, sealed: Path, inputs: Path, manifest: dict[str, Any], run_id: str) -> None:
    temporary = sealed.parent / f".snapshot-{run_id}"
    temporary.mkdir(mode=0o700)
    files = {}
    for row in manifest["conditions"]:
        name = f"{row['name']}.npz"
        path = output / name
        # Missing output is sealed as missing and classified by the grader.
        if not path.exists() and not path.is_symlink():
            continue
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > row["shots"] * 12 + 65536:
            raise ValueError(f"Invalid prediction file: {name}")
        destination = temporary / name
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as source:
            with destination.open("xb") as target:
                shutil.copyfileobj(source, target)
        destination.chmod(0o444)
        files[name] = sha256(destination)
    write_json(temporary / "snapshot.json", {"schema": "surface-code-decoding-snapshot/v1", "run_id": run_id,
        "input_manifest_sha256": sha256(inputs / "manifest.json"), "files": files})
    (temporary / "snapshot.json").chmod(0o444)
    temporary.chmod(0o555)
    os.replace(temporary, sealed)
    write_json(sealed.parent / "READY", {"run_id": run_id, "snapshot_sha256": sha256(sealed / "snapshot.json")})


def invalid(output: Path, status: str, error: Exception) -> None:
    write_json(output / "report.json", {"schema": "surface-code-decoding-report/v1", "status": status,
        "model_quality_measured": False, "score": 0.0, "score_unclipped": None, "performance": None,
        "error": f"{type(error).__name__}: {error}"})
    write_json(output / "reward.json", {"reward": 0.0})


def evaluate(args: argparse.Namespace) -> None:
    for path, mode in ((args.output, 0o755), (args.sealed.parent, 0o700)):
        path.mkdir(parents=True, exist_ok=True)
        os.chown(path, 0, 0)
        path.chmod(mode)
    for name in ("reward.json", "reward.txt", "report.json"):
        (args.output / name).unlink(missing_ok=True)
    if any(name != "lo" for _, name in socket.if_nameindex()):
        raise RuntimeError("Verifier inference requires a network_mode:none namespace")
    if args.sealed.exists():
        raise RuntimeError("Stale sealed output; create a fresh verifier volume")
    manifest = load_manifest(args.input)
    try:
        validate_submission(args.submission)
    except (OSError, ValueError, KeyError, TypeError) as error:
        invalid(args.output, "invalid_submission", error)
        return
    work = args.output / "inference"
    work.mkdir()
    try:
        output = run_submission(args.submission, args.input, work, args.timeout)
    except subprocess.TimeoutExpired as error:
        invalid(args.output, "inference_timeout", error)
        return
    except subprocess.CalledProcessError as error:
        invalid(args.output, "inference_failed", error)
        return
    run_id = uuid.uuid4().hex
    try:
        seal_output(output, args.sealed, args.input, manifest, run_id)
    except ValueError as error:
        invalid(args.output, "invalid_prediction", error)
        return
    deadline = time.monotonic() + 300
    result_root = args.sealed.parent / "results" / run_id
    while not (result_root / "DONE").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Trusted grader did not publish a result")
        time.sleep(0.25)
    result = json.loads((result_root / "report.json").read_text())
    if result.get("run_id") != run_id or result.get("status") == "infrastructure_error":
        raise RuntimeError(f"Trusted grader infrastructure error: {result}")
    for name in ("report.json", "reward.json"):
        shutil.copyfile(result_root / name, args.output / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=Path("/workspace/submission"))
    parser.add_argument("--input", type=Path, default=Path("/benchmark"))
    parser.add_argument("--sealed", type=Path, default=Path("/sealed/run"))
    parser.add_argument("--output", type=Path, default=Path("/logs/verifier"))
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("Trusted supervisor requires root to isolate inference UID 10001")
    if not 1 <= args.timeout <= 3600:
        parser.error("timeout must be in 1..3600")
    try:
        evaluate(args)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        for name in ("reward.json", "reward.txt"):
            (args.output / name).unlink(missing_ok=True)
        write_json(args.output / "report.json", {"schema": "surface-code-decoding-report/v1",
            "status": "infrastructure_error", "model_quality_measured": False,
            "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
