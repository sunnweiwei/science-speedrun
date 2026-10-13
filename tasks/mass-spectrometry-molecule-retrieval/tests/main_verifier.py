#!/usr/bin/env python3
"""Replay an untrusted model on input-only data and seal its ranking output."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import pwd
import shutil
import signal
import subprocess
import time
import sys

# In the verifier image these shared files are copied beside this module.
# In a checkout import the same source used by development validation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment/evaluation"))
from benchmark_score import normalize

from common import contained, sha256, validate_input, write_json


def validate_submission(root: Path):
    root = root.resolve()
    for required in ("infer.sh", "model_manifest.json", "README.md"):
        path = root / required
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"submission/{required} is required and must not be a symlink"
            )
    for path in root.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            raise ValueError(f"submission symlink escapes package: {path.relative_to(root)}")
    manifest = json.loads((root / "model_manifest.json").read_text())
    if set(manifest) != {"schema", "checkpoint", "model_config", "training_provenance"}:
        raise ValueError("invalid model manifest fields")
    if manifest["schema"] != "massspecgym-retrieval-model/v1":
        raise ValueError("invalid model manifest schema")
    checkpoint = manifest["checkpoint"]
    if (
        set(checkpoint) != {"path", "sha256", "format"}
        or checkpoint["format"] not in {"pytorch", "safetensors"}
        or not isinstance(checkpoint["sha256"], str)
        or len(checkpoint["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint["sha256"])
    ):
        raise ValueError("invalid checkpoint declaration")
    checkpoint_path = contained(root, checkpoint["path"])
    if sha256(checkpoint_path) != checkpoint["sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    config = contained(root, manifest["model_config"])
    if not isinstance(json.loads(config.read_text()), dict):
        raise ValueError("model config must contain a JSON object")
    provenance = manifest["training_provenance"]
    if (
        not isinstance(provenance, dict)
        or set(provenance) != {"data", "pretrained_assets"}
        or not isinstance(provenance["data"], str)
        or not isinstance(provenance["pretrained_assets"], list)
    ):
        raise ValueError("invalid training provenance")
    if provenance['pretrained_assets']:
        raise ValueError('this task release supplies no authorized pretrained assets')
    return manifest


def kill_uid_processes(uid: int):
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            status = (entry / "status").read_text()
            process_uid = int(
                next(line for line in status.splitlines() if line.startswith("Uid:")).split()[1]
            )
            if process_uid == uid:
                os.kill(int(entry.name), signal.SIGKILL)
        except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
            pass


def run_submission(submission: Path, input_root: Path, work: Path, timeout: int):
    agent = pwd.getpwnam("agent")
    output = work / "untrusted-output"
    output.mkdir()
    os.chown(output, agent.pw_uid, agent.pw_gid)
    log = work / "inference.log"
    environment = {
        "PATH": "/opt/conda/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME": agent.pw_dir,
        "PYTHONUNBUFFERED": "1",
        "PIP_NO_INDEX": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    }

    def demote():
        os.setgroups([])
        os.setgid(agent.pw_gid)
        os.setuid(agent.pw_uid)
        # Prevent setuid binaries or file capabilities from recovering privilege.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
            raise OSError(ctypes.get_errno(), 'PR_SET_NO_NEW_PRIVS failed')

    process = None
    try:
        with log.open("wb") as stream:
            process = subprocess.Popen(
                ["bash", str(submission / "infer.sh"), str(input_root), str(output)],
                cwd=submission,
                env=environment,
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
        kill_uid_processes(agent.pw_uid)
    return output


def seal_output(untrusted: Path, sealed: Path, input_manifest: Path):
    if sealed.exists():
        raise FileExistsError(sealed)
    temporary = sealed.with_name(sealed.name + ".tmp")
    temporary.mkdir(parents=True)
    files = {}
    source = untrusted / "rankings.jsonl"
    if source.is_file() and not source.is_symlink():
        if source.stat().st_size > 512 * 1024 * 1024:
            raise ValueError("ranking output exceeds 512 MiB")
        if not source.resolve().is_relative_to(untrusted.resolve()):
            raise ValueError("ranking output escapes output root")
        destination = temporary / "rankings.jsonl"
        shutil.copyfile(source, destination, follow_symlinks=False)
        destination.chmod(0o444)
        files["rankings.jsonl"] = sha256(destination)
    snapshot = {
        "schema": "massspecgym-retrieval-ranking-snapshot/v1",
        "input_manifest_sha256": sha256(input_manifest),
        "files": files,
    }
    write_json(temporary / "snapshot.json", snapshot)
    (temporary / "snapshot.json").chmod(0o444)
    temporary.chmod(0o555)
    os.replace(temporary, sealed)
    ready = sealed.parent / "READY.tmp"
    ready.write_text(sha256(sealed / "snapshot.json") + "\n")
    os.replace(ready, sealed.parent / "READY")


def prepare_trusted_directories(output: Path, sealed: Path):
    for path, mode in ((output, 0o755), (sealed.parent, 0o700)):
        path.mkdir(parents=True, exist_ok=True)
        os.chown(path, os.geteuid(), os.getegid())
        path.chmod(mode)
    # Harbor also accepts reward.txt, even if test.sh returns nonzero. Neither
    # stale results nor files written by the inference UID may become rewards.
    for name in ("reward.json", "reward.txt", "report.json"):
        (output / name).unlink(missing_ok=True)


def failed_submission(output: Path, queries: int, status: str, error: Exception):
    metrics = {"hit_rate_at_1": 0.0, "hit_rate_at_5": 0.0,
               "hit_rate_at_20": 0.0, "mces_at_1": 100.0}
    write_json(output / "report.json", {
        "schema": "massspecgym-retrieval-report/v1", "status": status,
        "model_quality_measured": False, "queries": queries,
        "valid_rankings": 0, "invalid_or_missing_rankings": queries,
        "global_error": str(error), "metrics": metrics,
        "benchmark_score": normalize(metrics["hit_rate_at_1"]),
        "metric_directions": {name: "minimize" if name == "mces_at_1" else "maximize"
                              for name in metrics},
    })
    write_json(output / "reward.json", {"reward": metrics["hit_rate_at_1"]})


def evaluate(args):
    started = time.monotonic()
    prepare_trusted_directories(args.output, args.sealed)
    _, queries, _ = validate_input(args.input)
    try:
        validate_submission(args.submission)
    except (OSError, ValueError, TypeError, KeyError) as error:
        failed_submission(args.output, len(queries), "invalid_submission", error)
        return
    work = args.output / "main"
    work.mkdir()
    try:
        untrusted = run_submission(args.submission, args.input, work, args.timeout)
    except subprocess.TimeoutExpired as error:
        failed_submission(args.output, len(queries), "inference_timeout", error)
        return
    except subprocess.CalledProcessError as error:
        failed_submission(args.output, len(queries), "inference_failed", error)
        return
    try:
        seal_output(untrusted, args.sealed, args.input / "manifest.json")
    except ValueError as error:
        failed_submission(args.output, len(queries), "invalid_output", error)
        return
    if not args.no_wait:
        results = args.sealed.parent / "results"
        deadline = min(time.monotonic() + 3600, started + 21600)
        while not (results / "reward.json").is_file():
            if (results / "error.json").is_file():
                raise RuntimeError(f"trusted grader failed: {(results / 'error.json').read_text()}")
            if time.monotonic() >= deadline:
                raise TimeoutError("trusted grader did not publish reward.json")
            time.sleep(1)
        for name in ("report.json", "reward.json"):
            shutil.copyfile(results / name, args.output / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--input", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--sealed", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--output", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--timeout", type=int, default=18000)
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("the trusted verifier must run as root to isolate inference UID 10001")
    if not 0 < args.timeout <= 18000:
        parser.error("inference timeout must be in 1..18000 seconds")
    try:
        evaluate(args)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        for name in ("reward.json", "reward.txt"):
            (args.output / name).unlink(missing_ok=True)
        write_json(args.output / "report.json", {
            "schema": "massspecgym-retrieval-report/v1",
            "status": "infrastructure_error", "model_quality_measured": False,
            "global_error": f"{type(error).__name__}: {error}",
        })
        raise


if __name__ == "__main__":
    main()
