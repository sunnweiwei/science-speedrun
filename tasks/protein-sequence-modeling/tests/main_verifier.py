#!/usr/bin/env python3
"""Run an untrusted offline submission and seal only bounded predictions."""

import argparse
import json
import os
from pathlib import Path
import pwd
import shutil
import signal
import stat
import subprocess
import time

from common import SNAPSHOT_SCHEMA, sha256, validate_input, write_json
from metrics import TASKS

MAX_OUTPUT_BYTES = 512 * 1024 * 1024


def validate_submission(root):
    root = root.resolve()
    for required in ("infer.sh", "README.md"):
        if not (root / required).is_file() or (root / required).is_symlink():
            raise FileNotFoundError(f"submission/{required} is required")
    for path in root.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            raise ValueError(f"submission symlink escapes package: {path.relative_to(root)}")


def kill_agent_processes(uid):
    proc = Path("/proc")
    if not proc.is_dir():
        return
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            status = (entry / "status").read_text()
            process_uid = int(next(line for line in status.splitlines()
                                   if line.startswith("Uid:")).split()[1])
            if process_uid == uid:
                os.kill(int(entry.name), signal.SIGKILL)
        except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
            pass


def run_submission(submission, input_root, work, timeout):
    agent = pwd.getpwnam("agent")
    if os.geteuid() != 0:
        raise PermissionError(
            "the verifier must run as root inside its own container so it can "
            f"drop to the unprivileged 'agent' user before executing the "
            f"submission; running as uid {os.geteuid()} cannot isolate untrusted "
            "code. Run tests/Dockerfile under a normal Docker daemon.")
    output = work / "untrusted-output"
    output.mkdir()
    os.chown(output, agent.pw_uid, agent.pw_gid)
    env = {
        "PATH": "/opt/conda/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME": agent.pw_dir,
        "PYTHONUNBUFFERED": "1",
        "PIP_NO_INDEX": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    }

    def demote():
        os.setgid(agent.pw_gid)
        os.setgroups([agent.pw_gid])
        os.setuid(agent.pw_uid)
        # Never take the drop on faith. Some sandboxes (apptainer's LD_PRELOAD
        # fakeroot, for one) let setuid report success while the process keeps
        # its original privileges — which would run an untrusted submission as
        # root while looking perfectly healthy. Verify, and refuse to run if the
        # drop is not real and irrevocable. Raising here fails the Popen.
        if os.getuid() != agent.pw_uid or os.geteuid() != agent.pw_uid:
            raise RuntimeError("privilege drop did not take effect")
        try:
            os.setuid(0)
        except OSError:
            pass
        else:
            raise RuntimeError("root could be regained after the privilege drop")

    process = None
    try:
        with (work / "inference.log").open("wb") as log:
            process = subprocess.Popen(
                ["bash", str(submission / "infer.sh"), str(input_root), str(output)],
                cwd=submission, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, preexec_fn=demote,
            )
            code = process.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, process.args)
    except subprocess.TimeoutExpired:
        if process:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        raise
    finally:
        if process:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                process.wait(timeout=5)
        kill_agent_processes(agent.pw_uid)
    return output


def validate_output_tree(root):
    total = 0
    for directory, names, files in os.walk(root, followlinks=False):
        for name in names + files:
            path = Path(directory) / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                raise ValueError("untrusted output contains a symlink or special file")
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
                if total > MAX_OUTPUT_BYTES:
                    raise ValueError("prediction output exceeds 512 MiB")


def seal_predictions(untrusted, sealed, input_manifest):
    if sealed.exists():
        raise FileExistsError("sealed destination already exists")
    validate_output_tree(untrusted)
    temporary = sealed.with_name(sealed.name + f".tmp-{os.getpid()}")
    temporary.mkdir(parents=True)
    hashes = {}
    for task in TASKS:
        source = untrusted / "predictions" / f"{task}.jsonl"
        if not source.is_file() or source.is_symlink():
            continue
        if not source.resolve().is_relative_to(untrusted.resolve()):
            raise ValueError("prediction path escapes output root")
        destination = temporary / f"{task}.jsonl"
        shutil.copyfile(source, destination, follow_symlinks=False)
        hashes[destination.name] = sha256(destination)
        destination.chmod(0o444)
    snapshot = {
        "schema": SNAPSHOT_SCHEMA,
        "input_manifest_sha256": sha256(input_manifest),
        "files": hashes,
    }
    write_json(temporary / "snapshot.json", snapshot)
    (temporary / "snapshot.json").chmod(0o444)
    temporary.chmod(0o555)
    os.replace(temporary, sealed)
    (sealed.parent / "READY").write_text(sha256(sealed / "snapshot.json") + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--input", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--sealed", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--output", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--timeout", type=int, default=85800)
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.chmod(args.output, 0o777)
    args.sealed.parent.mkdir(parents=True, exist_ok=True)
    validate_input(args.input)
    try:
        validate_submission(args.submission)
        work = args.output / "main"
        work.mkdir()
        untrusted = run_submission(args.submission, args.input, work, args.timeout)
        seal_predictions(untrusted, args.sealed, args.input / "manifest.json")
    except (FileNotFoundError, ValueError, subprocess.SubprocessError) as error:
        # An invalid package, a failed infer.sh or a timeout scores 0 (TASK_STANDARD.md); Harbor needs reward.json.
        reason = f"{type(error).__name__}: {error}"
        write_json(args.output / "report.json", {"status": "invalid_submission", "error": reason[:4000], "reward": 0.0})
        write_json(args.output / "reward.json", {"reward": 0.0, "invalid_submission": 1.0})
        (args.sealed.parent / "INVALID").write_text(reason + "\n")
        print(f"invalid submission: {reason}")
        return
    if not args.no_wait:
        results = args.sealed.parent / "results"
        deadline = time.monotonic() + 600
        while not (results / "reward.json").is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError("trusted grader did not publish reward.json")
            time.sleep(1)
        for name in ("report.json", "reward.json"):
            shutil.copyfile(results / name, args.output / name)


if __name__ == "__main__":
    main()
