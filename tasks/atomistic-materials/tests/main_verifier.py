#!/usr/bin/env python3
"""Run the submitted calculator through every test suite as uid 10001 and seal the predictions.

All suites run at the same time, each with its own workers spread over every visible GPU, inside one
shared time budget: physical (WBM discovery, MDR phonons, MP elasticity) with run_local.py, and
thermal conductivity, MOF bulk modulus, amorphous AM26, cleavage and Li-migration NEB with extra.py.
The long-tailed suites (MOF, NEB) get the most workers and start their most expensive items first. The evaluator code
drives every relaxation, displacement and strain; the submission only supplies the calculator.
At the shared time limit (3 h by default) every suite stops: the cases it finished are kept and the
rest are scored as failures. A suite that crashes is sealed as it is: the grader scores its missing
cases as failures (extra suites) or marks the suite invalid (physical suites).
"""
from __future__ import annotations

import argparse
import json
import os
import pwd
import shutil
import signal
import stat
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from verifier_common import sha256, tree_sha256, write_json

EVALUATION = Path("/evaluation") if Path("/evaluation").is_dir() else Path(__file__).resolve().parents[1] / "environment/evaluation"
EXTRA_SUITES = ("kappa", "mof", "amorphous", "cleavage", "neb")
SUITES = ("physical",) + EXTRA_SUITES
SCHEMA = "atomistic-materials-prediction-snapshot/v2"
MAX_SNAPSHOT_BYTES = 8 * 1024 ** 3
# Workers per GPU for each suite when they all run at once. MOF (EOS of large cells) and NEB (relaxed end
# points, then a climbing-image band) dominate the wall time, so they get the most workers; relaxations
# and phono3py leave a single worker's GPU mostly idle.
WORKERS_PER_GPU = {"physical": 2, "kappa": 2, "mof": 4, "amorphous": 1, "cleavage": 1, "neb": 3}


def harbor_reward(values):
    """Harbor parses reward.json as a flat object of finite numbers; strings, None and nested details
    belong in report.json / status.json."""
    import math as _math
    return {k: v for k, v in values.items() if isinstance(v, (int, float)) and _math.isfinite(v)}


def validate_submission(root: Path):
    root = root.resolve()
    entry = root / "calculator.py"
    if not entry.is_file() or entry.is_symlink():
        raise FileNotFoundError("submission/calculator.py required")
    for path in root.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            raise ValueError(f"submission symlink escapes package: {path.relative_to(root)}")


def validate_input(root: Path):
    for suite in SUITES:
        inputs = root / suite / "inputs"
        if not inputs.is_dir():
            raise FileNotFoundError(f"missing verifier input: {inputs}")
        if any(p.name.startswith(("references", "targets")) for p in (root / suite).rglob("*")):
            raise ValueError(f"untrusted input contains references: {suite}")


def gpu_ids():
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return os.environ["CUDA_VISIBLE_DEVICES"]
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    return ",".join(str(i) for i in range(max(1, sum(line.startswith("GPU ") for line in out.splitlines()))))


def command(suite, inputs, submission, output, gpus):
    py = "/opt/venv/bin/python"
    if suite == "physical":
        return [py, str(EVALUATION / "run_local.py"), "--inputs", str(inputs / "cases.jsonl"), "--factory", "submission",
                "--artifact", str(submission), "--output", str(output), "--gpus", gpus]
    return [py, str(EVALUATION / "extra.py"), "run", "--suite", suite, "--inputs", str(inputs), "--factory", "submission",
            "--artifact", str(submission), "--output", str(output), "--gpus", gpus]


def _agent_processes(uid):
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text()
            real_uid = int(next(line for line in status.splitlines() if line.startswith("Uid:")).split()[1])
            state = next(line for line in status.splitlines() if line.startswith("State:")).split()[1]
        except (FileNotFoundError, PermissionError, StopIteration, ValueError):
            continue
        if real_uid == uid and state != "Z":
            result.append(int(entry.name))
    return result


def _stop_untrusted(uid):
    # SIGKILLed GPU workers can take tens of seconds to exit while the driver tears down their CUDA
    # contexts (seen at the time limit, with ~100 workers killed at once).
    deadline = time.monotonic() + 180
    while True:
        for pid in _agent_processes(uid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        remaining = _agent_processes(uid)
        if not remaining:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"untrusted processes survived cleanup: {remaining}")
        time.sleep(0.05)


def run_suite(suite, args, work, gpus, timeout):
    agent = pwd.getpwnam("agent")
    output = work / suite
    env = {"PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin", "HOME": agent.pw_dir, "PYTHONUNBUFFERED": "1",
           "PIP_NO_INDEX": "1", "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "2",
           # Many workers share each GPU; fewer fragmented blocks means fewer out-of-memory retries.
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    def demote():
        os.setgid(agent.pw_gid)
        os.setuid(agent.pw_uid)
    started = time.monotonic()
    status = "complete"
    with (work / f"{suite}.log").open("wb") as log:
        process = subprocess.Popen(command(suite, args.input / suite / "inputs", args.submission, output, gpus),
                                   cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, preexec_fn=demote)
        try:
            code = process.wait(timeout=max(1, timeout))
            if code:
                status = f"exit {code}"
        except subprocess.TimeoutExpired:
            status = "timeout"
        finally:
            # Only this suite's process group: the other suites are still running as the same user.
            # Anything that left the group is killed by the user-wide sweep after the last suite.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                process.wait()
    return {"suite": suite, "status": status, "seconds": round(time.monotonic() - started, 1)}


def close_unfinished(suite, args, work):
    """After the shared time limit: keep what a suite finished and mark the rest as failed cases.

    Physical: merge the finished chunks; the cases of unfinished chunks become failed records.
    Extra suites: write the run record the driver would have written; items without a file are
    scored as failures by score_extra."""
    import sys
    sys.path.insert(0, str(EVALUATION))
    from common import digest, read_rows
    reason = "verifier time limit reached"
    output = work / suite
    if suite == "physical":
        from distributed import chunk_plan, merge
        if (output / "run.json").exists():
            return
        inputs = args.input / suite / "inputs" / "cases.jsonl"
        merge(inputs, work / "physical.chunks", chunk_plan(read_rows(inputs)), output, digest(args.submission),
              unfinished_reason=reason)
        return
    if (output / "run.json").exists():
        return
    import hashlib
    done = len(list((output / "items").glob("*.json"))) if (output / "items").is_dir() else 0
    manifest = args.input / suite / "inputs" / "manifest.json"
    write_json(output / "run.json", {
        "protocol": "atomistic-extra-suites-v1", "suite": suite,
        "inputs_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "checkpoint_sha256": digest(args.submission), "factory": "submission",
        "completed": done, "complete": False, "unfinished_reason": reason})


def _copy_snapshot(source: Path, destination: Path):
    keep = ["run.json", "predictions.jsonl"] if (source / "predictions.jsonl").exists() else None
    destination.mkdir(parents=True)
    if not source.exists():
        return
    for path in source.rglob("*"):
        mode = path.lstat().st_mode
        if path.is_symlink() or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(f"prediction output contains non-regular entry: {path.relative_to(source)}")
    if keep:
        for name in keep:
            shutil.copyfile(source / name, destination / name)
    else:
        for name in ("run.json",):
            if (source / name).is_file():
                shutil.copyfile(source / name, destination / name)
        if (source / "items").is_dir():
            shutil.copytree(source / "items", destination / "items")


def _lock(root: Path):
    for path in sorted(root.rglob("*"), reverse=True):
        os.chown(path, 0, 0, follow_symlinks=False)
        os.chmod(path, 0o500 if path.is_dir() else 0o400, follow_symlinks=False)
    os.chown(root, 0, 0)
    os.chmod(root, 0o500)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--submission", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--sealed", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeout", type=int, default=10800, help="shared limit for all suites (s)")
    p.add_argument("--grader-timeout", type=int, default=7200)
    p.add_argument("--suites", default=",".join(SUITES))
    p.add_argument("--workers-per-gpu", type=int, default=0, help="same for every suite (default: WORKERS_PER_GPU)")
    p.add_argument("--no-wait", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    try:
        validate_submission(a.submission)
        validate_input(a.input)
    except (FileNotFoundError, ValueError) as error:
        # The package, not the verifier, failed: every metric is unavailable and the run scores 0.
        from grader import expanded_score, flatten
        aggregate = expanded_score({}, valid=False)
        write_json(a.output / "report.json", {"status": "invalid_submission", "error": str(error)[:2000],
                                              "aggregate": aggregate})
        write_json(a.output / "reward.json", harbor_reward({"reward": 0.0, "crystal_potential_score": 0.0,
                                                            "invalid_submission": True, **flatten({})}))
        return
    work = a.output / "main"
    work.mkdir()
    agent = pwd.getpwnam("agent")
    os.chown(work, agent.pw_uid, agent.pw_gid)
    os.chmod(work, 0o700)
    # The untrusted user reads a private copy of the package, whatever the snapshot's ownership.
    package = work / "submission"
    shutil.copytree(a.submission, package, symlinks=True)
    for path in [package, *package.rglob("*")]:
        os.chown(path, agent.pw_uid, agent.pw_gid, follow_symlinks=False)
    a.submission = package
    # Label-free inputs, likewise copied so their host permissions cannot block the untrusted user.
    inputs = work / "inputs"
    shutil.copytree(a.input, inputs)
    for path in [inputs, *inputs.rglob("*")]:
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    original_input, a.input = a.input, inputs
    devices = gpu_ids().split(",")
    deadline = time.monotonic() + a.timeout
    suites = a.suites.split(",")

    def launch(suite):
        per_gpu = a.workers_per_gpu or WORKERS_PER_GPU[suite]
        run = run_suite(suite, a, work, ",".join(devices * per_gpu), int(deadline - time.monotonic()))
        print(json.dumps({"event": "suite_done", "workers": per_gpu * len(devices), **run}), flush=True)
        return run
    with ThreadPoolExecutor(max_workers=len(suites)) as pool:
        runs = list(pool.map(launch, suites))
    _stop_untrusted(agent.pw_uid)
    for run in runs:
        if run["status"] == "timeout":
            close_unfinished(run["suite"], a, work)
    temporary = a.sealed.with_name(a.sealed.name + f".tmp-{os.getpid()}")
    for suite in a.suites.split(","):
        _copy_snapshot(work / suite, temporary / suite)
    size = sum(f.stat().st_size for f in temporary.rglob("*") if f.is_file())
    if size > MAX_SNAPSHOT_BYTES:
        raise ValueError("prediction snapshot exceeds 8 GiB")
    snapshot = {"schema": SCHEMA, "runs": runs,
                "inputs": {s: tree_sha256(original_input / s / "inputs") for s in a.suites.split(",")},
                "predictions": {s: tree_sha256(temporary / s) for s in a.suites.split(",")}}
    write_json(temporary / "snapshot.json", snapshot)
    _lock(temporary)
    os.replace(temporary, a.sealed)
    (a.sealed.parent / "READY").write_text(sha256(a.sealed / "snapshot.json") + "\n")
    if not a.no_wait:
        results = a.sealed.parent / "results"
        end = time.monotonic() + a.grader_timeout
        while not (results / "reward.json").is_file():
            if time.monotonic() >= end:
                raise TimeoutError("grader did not publish reward")
            time.sleep(2)
        for name in ("report.json", "reward.json"):
            shutil.copyfile(results / name, a.output / name)


if __name__ == "__main__":
    main()
