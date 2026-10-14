"""Collect privileged-PID demonstration episodes to disk (observations, actions, descriptors).

Used to fit the magnetic-descriptor estimator of the deployable PID package and to
seed behaviour cloning. Usage:
  python starter/collect_demos.py --output DIR --episodes-per-worker 4 --workers 24 --solver linear
"""

import argparse
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from tokagym import CASES  # noqa: E402
from workers import worker_main  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--cases", nargs="*", default=sorted(CASES))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--episodes-per-worker", type=int, default=4)
    ap.add_argument("--solver", default="linear", choices=["linear", "nonlinear"])
    ap.add_argument("--randomization", default="visible")
    ap.add_argument("--max-minutes", type=float, default=120.0)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=4 * args.workers)
    cfg = {"randomization": args.randomization, "solver": args.solver, "horizon": None,
           "explore_noise": 0.0, "random_scale": 0.0, "demo_episodes": args.episodes_per_worker}
    procs = []
    for w in range(args.workers):
        cases = [args.cases[(w + k) % len(args.cases)] for k in range(len(args.cases))]
        p = ctx.Process(target=worker_main, args=(w, cases, queue, out / "no-actor" / "actor.npz", cfg), daemon=True)
        p.start(); procs.append(p)
    want = args.workers * args.episodes_per_worker
    got, started = 0, time.time()
    O, A, DS, EP = [], [], [], []
    log = open(out / "collect.log", "a")
    while got < want and time.time() - started < args.max_minutes * 60:
        try:
            msg = queue.get(timeout=60)
        except Exception:  # noqa: BLE001
            continue
        if "error" in msg:
            log.write(f"worker {msg['worker']} error: {msg['error']}\n"); log.flush(); continue
        if not msg.get("demo"):
            continue
        T = len(msg["act"])
        O.append(msg["obs"][:T]); A.append(msg["act"]); DS.append(msg["desc"])
        EP.append({"case": msg["case"], "seed": msg["seed"], "steps": T, "feasible": msg["summary"]["feasible"],
                   "termination": msg["summary"]["termination"],
                   "shape_cm": None if msg["summary"]["shape_rms_m_last_half"] is None
                   else 100 * msg["summary"]["shape_rms_m_last_half"]})
        got += 1
        log.write(f"{got}/{want} {EP[-1]}\n"); log.flush()
        if got % 20 == 0:
            np.savez(out / "demos.npz", obs=np.concatenate(O), act=np.concatenate(A), desc=np.concatenate(DS),
                     episode_index=np.repeat(np.arange(len(EP)), [e["steps"] for e in EP]))
    np.savez(out / "demos.npz", obs=np.concatenate(O), act=np.concatenate(A), desc=np.concatenate(DS),
             episode_index=np.repeat(np.arange(len(EP)), [e["steps"] for e in EP]))
    import json
    (out / "episodes.json").write_text(json.dumps(EP, indent=1))
    for p in procs:
        p.terminate()
    log.write("done\n"); log.close()


if __name__ == "__main__":
    main()
