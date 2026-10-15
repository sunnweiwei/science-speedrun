#!/usr/bin/env python3
"""Run the fixed fine-tune protocol (finetune.sh) of one encoder on many tasks, spread over the visible GPUs.

usage: run_suite.py --encoder PKG|none --data DIR --out DIR [--tasks T ...] [--gpus 0,1,...] [--slots 3]
                    [--user NAME] [--timeout SEC] [--retries 1]
Each task writes OUT/<task>/{train.log, predictions.csv}; OUT/suite.json lists status, GPU and wall time per task.
With --user (the verifier, running as root) every fine-tune runs as that unprivileged user with a clean
environment. A task that fails is retried --retries times; the run fails if any task still has no predictions.
"""
import argparse
import json
import os
import pwd
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tasks import TASKS  # noqa: E402

# Longest first (QM9 has the largest test sets, COMPAS the largest training molecules), so the tail is short.
ORDER = {t: i for i, t in enumerate([t for t in TASKS if t.startswith("qm9")] + [t for t in TASKS if t.startswith("compas")]
                                    + [t for t in TASKS if t.startswith("exrx")] + [t for t in TASKS if t.startswith("asap")])}


def visible_gpus() -> list[str]:
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return [g for g in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if g]
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    return [str(i) for i, line in enumerate(l for l in out.splitlines() if l.startswith("GPU "))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--gpus")
    ap.add_argument("--slots", type=int, default=3, help="concurrent fine-tunes per GPU")
    ap.add_argument("--user")
    ap.add_argument("--timeout", type=int, default=3 * 3600, help="per fine-tune, seconds")
    ap.add_argument("--retries", type=int, default=1)
    a = ap.parse_args()
    unknown = [t for t in a.tasks if t not in TASKS]
    if unknown:
        sys.exit(f"unknown tasks: {unknown}")
    gpus = a.gpus.split(",") if a.gpus else visible_gpus()
    a.out.mkdir(parents=True, exist_ok=True)
    user = pwd.getpwnam(a.user) if a.user else None
    queue = sorted(a.tasks, key=ORDER.get)
    lock = threading.Lock()
    status: dict[str, dict] = {}

    def run(task: str, gpu: str) -> bool:
        out = a.out / task
        out.mkdir(exist_ok=True)
        (out / "predictions.csv").unlink(missing_ok=True)
        env = {"PATH": "/opt/conda/bin:/usr/local/bin:/usr/bin:/bin", "HOME": user.pw_dir if user else os.environ.get("HOME", "/tmp"),
               "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": gpu,
               "FT_EPOCHS": os.environ.get("FT_EPOCHS", "15") if not user else "15"}
        preexec = None
        if user:
            os.chown(out, user.pw_uid, user.pw_gid)
            def preexec():  # noqa: E306
                os.setgid(user.pw_gid); os.setuid(user.pw_uid)
        start = time.time()
        with open(out / "train.log", "ab") as log:
            p = subprocess.Popen(["bash", str(HERE / "finetune.sh"), task, str(a.data), a.encoder, str(out)],
                                 stdout=log, stderr=subprocess.STDOUT, env=env, preexec_fn=preexec, start_new_session=True)
            try:
                code = p.wait(timeout=a.timeout)
            except subprocess.TimeoutExpired:
                code = "timeout"
            finally:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        ok = code == 0 and (out / "predictions.csv").is_file()
        with lock:
            s = status.setdefault(task, {"attempts": 0})
            s.update(attempts=s["attempts"] + 1, exit=code, ok=ok, gpu=gpu, minutes=round((time.time() - start) / 60, 2))
            print(f"{time.strftime('%H:%M:%S')} {task} gpu={gpu} exit={code} ok={ok} {s['minutes']} min", flush=True)
        return ok

    def worker(gpu: str) -> None:
        while True:
            with lock:
                if not queue:
                    return
                task = queue.pop(0)
            for _ in range(1 + a.retries):
                if run(task, gpu):
                    break

    t0 = time.time()
    # Slot-major order, started one by one, so the longest tasks (front of the queue) land on different GPUs.
    threads = [threading.Thread(target=worker, args=(g,)) for _ in range(a.slots) for g in gpus]
    for t in threads:
        t.start(); time.sleep(0.2)
    for t in threads:
        t.join()
    summary = {"encoder": a.encoder, "gpus": gpus, "slots": a.slots, "minutes": round((time.time() - t0) / 60, 2),
               "tasks": status, "failed": sorted(t for t in a.tasks if not status.get(t, {}).get("ok"))}
    (a.out / "suite.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("minutes", "failed")}))
    if summary["failed"]:
        sys.exit(f"fine-tunes failed: {summary['failed']}")


if __name__ == "__main__":
    main()
