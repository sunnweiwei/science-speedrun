#!/usr/bin/env python3
"""Replay a snapshot without label access, then score every frozen test target."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile

# Local invocation; the image puts these trusted modules under /tests.
ENVIRONMENT = Path(__file__).resolve().parents[1] / "environment"
if ENVIRONMENT.is_dir():
    sys.path.insert(0, str(ENVIRONMENT))
    sys.path.insert(0, str(ENVIRONMENT / "evaluation"))
from input_view import stage_inputs
from publish import validate_package
from score import evaluate
from aggregation import REPORT_SCHEMA, SCORING_PROTOCOL

INFERENCE_UID = 65534


def require_private_boundary(root: Path) -> None:
    if os.geteuid() != 0:
        raise RuntimeError("verifier requires root to demote candidate inference")
    for parent in (root, *root.parents):
        info = parent.stat()
        if info.st_uid == 0 and info.st_mode & 0o077 == 0:
            return
    raise RuntimeError("private benchmark must be below a root-owned mode-0700 directory")


def readonly_tree(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def remove_runtime(root: Path) -> None:
    # Also works for root inside a user namespace without DAC override.
    for directory, subdirs, _ in os.walk(root, followlinks=False):
        Path(directory).chmod(0o700)
        for name in subdirs:
            child = Path(directory) / name
            if not child.is_symlink():
                child.chmod(0o700)
    shutil.rmtree(root)


def demote() -> None:
    os.setgroups([])
    os.setgid(INFERENCE_UID)
    os.setuid(INFERENCE_UID)
    os.umask(0o077)


def replay(package: Path, inputs: Path, output: Path, scratch: Path,
           log: Path, timeout: float) -> None:
    env = {name: os.environ[name] for name in (
        "PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES") if name in os.environ}
    env.update(HOME=str(scratch), TMPDIR=str(scratch), PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
    with log.open("wb") as stream:
        process = subprocess.Popen(
            ["bash", str(package / "infer.sh"), str(inputs), str(output)],
            cwd=package, env=env, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True, preexec_fn=demote)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise ValueError(f"inference exceeded {timeout:g} seconds") from None
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        if code:
            raise ValueError(f"inference exited with status {code}; see inference.log")


def grade(submission: Path, data_root: Path, output: Path, timeout: float = 79200.0,
          *, permutation_seed_file: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    output.chmod(0o700)
    reward_path = output / "reward.json"
    details_path = output / "details.json"
    report_path = output / "report.json"
    report_path.unlink(missing_ok=True)
    reward_path.unlink(missing_ok=True)
    details_path.unlink(missing_ok=True)
    # Preparation errors are infrastructure failures: no reward is written.
    require_private_boundary(data_root.resolve())
    seed = None
    if permutation_seed_file is not None:
        require_private_boundary(permutation_seed_file.resolve())
        seed = permutation_seed_file.read_bytes()
    runtime = Path(tempfile.mkdtemp(prefix="drugclip-verifier-"))
    runtime.chmod(0o755)
    try:
        inputs = runtime / "inputs"
        # Both the seed and original indices remain below root-only output.
        # Keep them for controller reproducibility; never under runtime/inputs.
        mapping = output / "private-order"
        if mapping.exists():
            shutil.rmtree(mapping)
        stage_inputs(data_root, inputs, mapping, full_suite=True, seed=seed)
        readonly_tree(inputs)
        predictions = runtime / "predictions"
        predictions.mkdir()
        os.chown(predictions, INFERENCE_UID, INFERENCE_UID)
        predictions.chmod(0o700)
        scratch = runtime / "scratch"
        scratch.mkdir()
        os.chown(scratch, INFERENCE_UID, INFERENCE_UID)
        scratch.chmod(0o700)
        try:
            validate_package(submission)
            if not (submission / "README.md").is_file():
                raise ValueError("package must include README.md")
            package = runtime / "model"
            shutil.copytree(submission, package)
            readonly_tree(package)
            replay(package, inputs, predictions, scratch, output / "inference.log", timeout)
            csv_path = predictions / "scores.csv"
            if csv_path.is_symlink() or not csv_path.is_file():
                raise ValueError("inference did not produce a regular scores.csv")
            metrics = evaluate(data_root, csv_path, mapping, require_full_suite=True)
            metrics["permutation_manifest_sha256"] = hashlib.sha256(
                (mapping / "manifest.json").read_bytes()).hexdigest()
            result = {"reward": metrics["dataset_balanced_bedroc"], "valid": True, **metrics}
        except (ValueError, KeyError, TypeError, OSError) as error:
            result = {"reward": 0.0, "valid": False, "error": str(error),
                      "schema": REPORT_SCHEMA, "scoring_protocol": SCORING_PROTOCOL}
        # The shared runner collects report.json. Keep details.json for existing readers.
        report = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        for path in (report_path, details_path):
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(report)
            temporary.replace(path)
        scores = {"reward": result["reward"]}
        if result["valid"]:
            scores.update((k, v) for k, v in result.items() if k.startswith("overall_mean_") or k == "dataset_balanced_bedroc")
        reward_temporary = reward_path.with_suffix(".json.tmp")
        reward_temporary.write_text(json.dumps(scores, allow_nan=False) + "\n")
        reward_temporary.replace(reward_path)
        return result
    finally:
        remove_runtime(runtime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=Path("/workspace/submission"))
    parser.add_argument("--data-root", type=Path, default=Path("/private/benchmark"))
    parser.add_argument("--output", type=Path, default=Path("/logs/verifier"))
    parser.add_argument("--inference-timeout", type=float, default=79200.0)
    parser.add_argument("--permutation-seed-file", type=Path)
    args = parser.parse_args()
    if args.inference_timeout <= 0:
        parser.error("inference timeout must be positive")
    grade(args.submission, args.data_root, args.output, args.inference_timeout,
          permutation_seed_file=args.permutation_seed_file)
