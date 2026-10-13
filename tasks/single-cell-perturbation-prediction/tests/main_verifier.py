#!/usr/bin/env python3
"""Replay untrusted inference on the public test inputs (one call per context) and seal the
predictions for the trusted grader. Never sees private responses."""

from __future__ import annotations

import argparse
import json
import os
import pwd
import shutil
import signal
import subprocess
import time
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

from cellscore import read_requests, sha256


def validate_submission(root: Path) -> None:
    root = root.resolve()
    if not (root / "infer.sh").is_file():
        raise FileNotFoundError("submission/infer.sh is required")
    for path in root.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            raise ValueError(f"submission symlink escapes package: {path.relative_to(root)}")


def kill_uid(uid: int) -> None:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            status = (entry / "status").read_text()
            process_uid = int(next(x for x in status.splitlines() if x.startswith("Uid:")).split()[1])
            if process_uid == uid:
                os.kill(int(entry.name), signal.SIGKILL)
        except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
            pass


def validate_output(path: Path, basal_path: Path, request_path: Path) -> dict:
    expected = read_requests(request_path)
    basal = ad.read_h5ad(basal_path, backed="r")
    predicted = ad.read_h5ad(path)
    if "request_id" not in predicted.obs or predicted.var_names.has_duplicates:
        raise ValueError("predictions need obs.request_id and unique genes")
    if list(map(str, predicted.var_names)) != list(map(str, basal.var_names)):
        raise ValueError("prediction genes/order differ from input basal.h5ad")
    ids = predicted.obs["request_id"].astype(str)
    counts = ids.value_counts().to_dict()
    if counts != {row["request_id"]: row["n_cells"] for row in expected}:
        raise ValueError("prediction request IDs or per-request cell counts differ from requests.jsonl")
    values = predicted.X.data if sparse.issparse(predicted.X) else np.asarray(predicted.X)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("predictions contain negative or non-finite values")
    return {"n_cells": int(predicted.n_obs), "n_genes": int(predicted.n_vars)}


def run_inference(submission: Path, input_dir: Path, output_dir: Path, log_path: Path, timeout: int, gpu: str) -> None:
    agent = pwd.getpwnam("agent")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chown(output_dir, agent.pw_uid, agent.pw_gid)
    env = {"PATH": "/opt/conda/bin:/usr/local/bin:/usr/bin:/bin", "HOME": agent.pw_dir, "PYTHONUNBUFFERED": "1",
           "PIP_NO_INDEX": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": gpu,
           "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "16")}

    def demote():
        os.setgid(agent.pw_gid)
        os.setuid(agent.pw_uid)

    process = None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(["bash", str(submission / "infer.sh"), str(input_dir), str(output_dir)],
                                       cwd=submission, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, preexec_fn=demote)
            code = process.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, process.args)
    finally:
        if process:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        kill_uid(agent.pw_uid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True, type=lambda x: Path(x).resolve())
    ap.add_argument("--benchmark", required=True, type=lambda x: Path(x).resolve(), help="dir with one sub-dir per context")
    ap.add_argument("--sealed", required=True, type=lambda x: Path(x).resolve())
    ap.add_argument("--output", required=True, type=lambda x: Path(x).resolve())
    ap.add_argument("--timeout-per-context", type=int, default=3600)
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        validate_submission(args.submission)
    except (FileNotFoundError, ValueError) as error:
        # An invalid package scores 0 (TASK_STANDARD.md); Harbor needs reward.json.
        reason = f"{type(error).__name__}: {error}"
        (args.output / "report.json").write_text(json.dumps(
            {"status": "invalid_submission", "error": reason[:4000], "reward": 0.0}, indent=2) + "\n")
        (args.output / "reward.json").write_text(json.dumps({"reward": 0.0, "invalid_submission": 1.0}, indent=2) + "\n")
        print(f"invalid submission: {reason}")
        return
    # $TASK_DATA_DIR/test/<ctx>/public/{basal.h5ad, requests.jsonl, context.json}
    contexts = sorted(p for p in args.benchmark.iterdir() if (p / "public" / "requests.jsonl").exists())
    if not contexts:
        raise FileNotFoundError(f"no contexts under {args.benchmark} (expected <ctx>/public/requests.jsonl)")
    work = args.output / "main"
    work.mkdir(exist_ok=True)
    temporary = args.sealed.with_name(args.sealed.name + ".tmp")
    temporary.mkdir(parents=True)
    snapshot = {"schema": "virtual-cell-zeroshot-snapshot/v1", "contexts": {}}
    status = {}
    for ctx_dir in contexts:
        ctx = ctx_dir / "public"
        ctx_name = ctx_dir.name
        untrusted = work / ctx_name / "untrusted-output"
        t0 = time.time()
        try:
            run_inference(args.submission, ctx, untrusted, work / ctx_name / "inference.log", args.timeout_per_context,
                          os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
            prediction = untrusted / "predictions.h5ad"
            if prediction.stat().st_size > 16 * 1024 ** 3:
                raise ValueError("prediction file exceeds 16 GiB")
            info = validate_output(prediction, ctx / "basal.h5ad", ctx / "requests.jsonl")
            (temporary / ctx_name).mkdir()
            shutil.copyfile(prediction, temporary / ctx_name / "predictions.h5ad", follow_symlinks=False)
            snapshot["contexts"][ctx_name] = {"predictions_sha256": sha256(temporary / ctx_name / "predictions.h5ad"),
                                              "requests_sha256": sha256(ctx / "requests.jsonl"),
                                              "basal_sha256": sha256(ctx / "basal.h5ad"), **info,
                                              "seconds": round(time.time() - t0, 1)}
            status[ctx_name] = {"status": "ok", "seconds": round(time.time() - t0, 1)}
        except Exception as exc:  # a failed context scores 0 but must not hide the others
            status[ctx_name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "seconds": round(time.time() - t0, 1)}
            snapshot["contexts"][ctx_name] = {"failed": True, "error": status[ctx_name]["error"]}
        finally:
            shutil.rmtree(untrusted, ignore_errors=True)
    (temporary / "snapshot.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    for path in temporary.rglob("*"):
        path.chmod(0o444 if path.is_file() else 0o555)
    temporary.chmod(0o555)
    os.replace(temporary, args.sealed)
    (args.output / "inference_status.json").write_text(json.dumps(status, indent=2) + "\n")
    (args.sealed.parent / "READY").write_text(sha256(args.sealed / "snapshot.json") + "\n")
    if not args.no_wait:
        result = args.sealed.parent / "results"
        deadline = time.monotonic() + 7200
        while not (result / "reward.json").is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError("grader did not publish rewards")
            time.sleep(2)
        for name in ("reward.json", "report.json", "agent_feedback.json"):
            if (result / name).is_file():
                shutil.copyfile(result / name, args.output / name)


if __name__ == "__main__":
    main()
