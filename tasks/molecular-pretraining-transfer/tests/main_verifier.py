#!/usr/bin/env python3
"""Verifier main: fine-tune the frozen encoder package on every task and seal the test predictions.

Runs as root in a networkless container that holds the fine-tune data (/benchmark/downstream) and the test
molecules without labels (/benchmark/test-inputs). Package code only ever runs as the unprivileged user `agent`:
the encoder check (protocol/check_encoder.py) and the 24 fixed-protocol fine-tunes (protocol/run_suite.py). The
predictions are validated, copied into a root-owned sealed snapshot with hashes and a READY marker; the separate
grader scores them against the labels. An invalid package, a failed encoder check or a fine-tune that fails twice
makes the submission invalid (reward.json: valid = 0).
"""
import argparse
import csv
import hashlib
import json
import math
import os
import pwd
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol"
sys.path.insert(0, str(PROTOCOL))
from tasks import NUM_TARGETS, TASKS  # noqa: E402

MAX_PACKAGE_BYTES = 4 * 1024 ** 3
MAX_PACKAGE_FILES = 2000
ALLOWED_TOP = {"encoder.json", "checkpoint.pt", "README.md"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def validate_inputs(bench: Path) -> dict:
    manifest = json.loads((bench / "downstream" / "manifest.json").read_text())
    if sorted(manifest.get("tasks", {})) != sorted(TASKS):
        raise RuntimeError("downstream manifest does not list the 24 suite tasks")
    for t in TASKS:
        for p in (bench / "downstream" / t / "train.lmdb", bench / "downstream" / t / "valid.lmdb",
                  bench / "test-inputs" / t / "test.lmdb"):
            if not p.is_file():
                raise RuntimeError(f"missing verifier input {p}")
    if not (bench / "downstream" / "task_metainfo.json").is_file():
        raise RuntimeError("missing task_metainfo.json")
    return manifest


def data_view(bench: Path, work: Path) -> Path:
    """DATA/<task>/{train,valid,test}.lmdb as the fine-tune expects it, by symlinks into the read-only mounts."""
    view = work / "data"
    for t in TASKS:
        (view / t).mkdir(parents=True)
        for split in ("train", "valid"):
            (view / t / f"{split}.lmdb").symlink_to(bench / "downstream" / t / f"{split}.lmdb")
        (view / t / "test.lmdb").symlink_to(bench / "test-inputs" / t / "test.lmdb")
    (view / "task_metainfo.json").symlink_to(bench / "downstream" / "task_metainfo.json")
    for p in [view, *view.rglob("*")]:
        if p.is_dir():
            p.chmod(0o755)
    return view


def validate_package(pkg: Path) -> None:
    if not pkg.is_dir():
        raise ValueError("no package at /workspace/submission")
    total = count = 0
    for p in pkg.rglob("*"):
        if p.is_symlink():
            raise ValueError(f"symlinks are not allowed: {p.relative_to(pkg)}")
        if p.is_file():
            count += 1; total += p.stat().st_size
    if count > MAX_PACKAGE_FILES or total > MAX_PACKAGE_BYTES:
        raise ValueError("package exceeds 4 GiB or 2,000 files")
    for name in ("encoder.json", "checkpoint.pt"):
        if not (pkg / name).is_file():
            raise ValueError(f"package has no {name}")
    spec = json.loads((pkg / "encoder.json").read_text())
    if not isinstance(spec, dict) or set(spec) - {"arch", "user_dir", "description"}:
        raise ValueError("encoder.json must be an object with keys among arch, user_dir, description")
    top = {p.name for p in pkg.iterdir()} - ALLOWED_TOP - ({spec["user_dir"]} if spec.get("user_dir") else set())
    if top:
        raise ValueError(f"unexpected top-level entries: {sorted(top)}")


def freeze(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst, symlinks=False)
    for p in dst.rglob("*"):
        if p.is_file():
            p.chmod(0o444 | (p.stat().st_mode & 0o111))
    for p in sorted((d for d in dst.rglob("*") if d.is_dir()), reverse=True) + [dst]:
        p.chmod(0o555)
    return dst


def as_agent(cmd: list[str], log: Path, timeout: int, cwd: Path) -> int:
    agent = pwd.getpwnam("agent")
    env = {"PATH": "/opt/conda/bin:/usr/local/bin:/usr/bin:/bin", "HOME": agent.pw_dir, "PYTHONUNBUFFERED": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}

    def demote():
        os.setgid(agent.pw_gid); os.setuid(agent.pw_uid)
    with log.open("ab") as out:
        p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=cwd, preexec_fn=demote,
                             start_new_session=True)
        try:
            return p.wait(timeout=timeout)
        finally:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def kill_agent_processes() -> None:
    uid = pwd.getpwnam("agent").pw_uid
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text()
            if int(next(l for l in status.splitlines() if l.startswith("Uid:")).split()[1]) == uid:
                os.kill(int(entry.name), signal.SIGKILL)
        except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
            pass


def read_predictions(path: Path, task: str, limit: int) -> list[list[str]]:
    k = NUM_TARGETS[task]
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] != ["smi"] + [f"predict_{j}" for j in range(k)]:
        raise ValueError(f"{task}: bad predictions header")
    if not 0 < len(rows) - 1 <= limit:
        raise ValueError(f"{task}: {len(rows) - 1} predictions for at most {limit} test molecules")
    for r in rows[1:]:
        if len(r) != k + 1 or not all(math.isfinite(float(v)) for v in r[1:]):
            raise ValueError(f"{task}: malformed or non-finite prediction row")
    return rows


def seal(runs: Path, work: Path, sealed: Path, manifest: dict) -> None:
    tmp = sealed.with_name(sealed.name + ".tmp")
    tmp.mkdir(parents=True)
    files = {}
    for t in TASKS:
        rows = read_predictions(runs / t / "predictions.csv", t, manifest["tasks"][t]["test_molecules"])
        with (tmp / f"{t}.csv").open("w", newline="") as f:
            csv.writer(f).writerows(rows)
        files[f"{t}.csv"] = sha256(tmp / f"{t}.csv")
    for name in ("encoder_check.json", "suite.json"):
        shutil.copyfile(work / name, tmp / name); files[name] = sha256(tmp / name)
    (tmp / "snapshot.json").write_text(json.dumps({"schema": "mpt-v2-predictions/v1", "files": files}, indent=1) + "\n")
    for p in tmp.iterdir():
        p.chmod(0o444)
    tmp.chmod(0o555)
    os.replace(tmp, sealed)
    (sealed.parent / "READY").write_text(sha256(sealed / "snapshot.json") + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=Path, default=Path("/workspace/submission"))
    ap.add_argument("--benchmark", type=Path, default=Path("/benchmark"))
    ap.add_argument("--sealed", type=Path, default=Path("/sealed/run"))
    ap.add_argument("--output", type=Path, default=Path("/logs/verifier"))
    ap.add_argument("--work", type=Path, default=Path("/var/tmp/mpt-verifier"))
    ap.add_argument("--timeout", type=int, default=3 * 3600, help="per fine-tune")
    ap.add_argument("--grader-wait", type=int, default=3600)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    a.sealed.parent.mkdir(parents=True, exist_ok=True)
    if a.work.exists():
        shutil.rmtree(a.work)
    a.work.mkdir(parents=True)
    a.work.chmod(0o755)
    manifest = validate_inputs(a.benchmark)  # infrastructure errors propagate: no reward
    data = data_view(a.benchmark, a.work)
    runs = a.work / "runs"; runs.mkdir(); os.chown(runs, pwd.getpwnam("agent").pw_uid, pwd.getpwnam("agent").pw_gid)
    t0 = time.time()
    try:
        validate_package(a.submission)
        pkg = freeze(a.submission, a.work / "package")
        code = as_agent([sys.executable, str(PROTOCOL / "check_encoder.py"), str(pkg), str(data), str(runs / "encoder_check.json")],
                        a.work / "encoder_check.log", 1800, runs)
        if code != 0 or not (runs / "encoder_check.json").is_file():
            tail = (a.work / "encoder_check.log").read_text(errors="replace")[-1500:]
            raise ValueError(f"encoder check failed (exit {code}): {tail}")
        shutil.copyfile(runs / "encoder_check.json", a.work / "encoder_check.json")
        suite = subprocess.run([sys.executable, str(PROTOCOL / "run_suite.py"), "--encoder", str(pkg), "--data", str(data),
                                "--out", str(runs / "suite"), "--user", "agent", "--timeout", str(a.timeout), "--retries", "1"],
                               stdout=(a.work / "suite.log").open("wb"), stderr=subprocess.STDOUT)
        kill_agent_processes()
        (a.output / "main").mkdir(exist_ok=True)
        for t in TASKS:
            if (runs / "suite" / t / "train.log").is_file():
                shutil.copyfile(runs / "suite" / t / "train.log", a.output / "main" / f"{t}.log")
        shutil.copyfile(a.work / "suite.log", a.output / "main" / "suite.log")
        if suite.returncode != 0 or not (runs / "suite" / "suite.json").is_file():
            raise ValueError("fine-tune failed: " + (a.work / "suite.log").read_text(errors="replace")[-800:])
        shutil.copyfile(runs / "suite" / "suite.json", a.work / "suite.json")
        seal(runs / "suite", a.work, a.sealed, manifest)
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        kill_agent_processes()
        reason = f"{type(error).__name__}: {error}"
        (a.output / "reward.json").write_text(json.dumps({"valid": 0.0, "invalid_submission": 1.0}, indent=1) + "\n")
        (a.output / "report.json").write_text(json.dumps({"status": "invalid_submission", "error": reason[:4000]}, indent=1) + "\n")
        (a.sealed.parent / "INVALID").write_text(reason + "\n")
        print(f"invalid submission: {reason}")
        return
    print(f"fine-tunes and sealing took {(time.time() - t0) / 60:.1f} min; waiting for the grader")
    results = a.sealed.parent / "results"
    deadline = time.monotonic() + a.grader_wait
    while not (results / "reward.json").is_file():
        if time.monotonic() > deadline:
            raise TimeoutError("grader did not publish reward.json")
        time.sleep(2)
    for name in ("reward.json", "report.json"):
        shutil.copyfile(results / name, a.output / name)
    print((a.output / "reward.json").read_text())


if __name__ == "__main__":
    main()
