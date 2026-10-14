"""Evaluate frozen actor snapshots of a training run on the sealed suite -> learning curve.

Usage: python evaluation/curve.py --run RUN_DIR --output OUT --randomization FILE [--seeds ...]
       [--workers 6] [--watch]   (watch: keep polling until RUN_DIR/DONE exists)
Each snapshot is packaged exactly like a publication and replayed by harbor_verify.py;
results accumulate in OUT/curve.jsonl and OUT/curve.md.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "starter"))
from publish import publish  # noqa: E402

CASES_ORDER = ["mastu_diverted_hold", "mastu_p4_shape_hold", "mastu_high_ip_hold",
               "mastu_ip_ramp", "mastu_vertical_shift", "mastu_pressure_drop"]


def evaluate(snapshot, out_dir, randomization, seeds, workers):
    pkg = out_dir / "packages" / snapshot.name
    if not (pkg / "infer.sh").exists():
        publish(snapshot, pkg, {"snapshot": snapshot.name})
    replay = out_dir / "replays" / snapshot.name
    if (replay / "reward.json").exists():
        return json.loads((replay / "reward.json").read_text()), "cached"
    cmd = [sys.executable, str(HERE / "harbor_verify.py"), "--submission", str(pkg), "--output", str(replay),
           "--randomization", randomization, "--workers", str(workers)]
    if seeds:
        cmd += ["--seeds"] + [str(s) for s in seeds]
    log = out_dir / "logs" / f"{snapshot.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if (replay / "reward.json").exists():
        return json.loads((replay / "reward.json").read_text()), f"exit={code}"
    return None, f"exit={code}"


def write_table(out_dir):
    rows = [json.loads(l) for l in (out_dir / "curve.jsonl").read_text().splitlines() if l.strip()]
    lines = ["| episodes | steps | " + " | ".join(f"{c.replace('mastu_', '')} feas / shape cm" for c in CASES_ORDER) + " | mean feas | mean shape cm |",
             "|" + "---|" * (len(CASES_ORDER) + 4)]
    for r in rows:
        rw = r["reward"] or {}
        cells, feas, shapes = [], [], []
        for c in CASES_ORDER:
            f = rw.get(f"{c}/feasible_fraction"); sh = rw.get(f"{c}/shape_rms_cm")
            feas.append(f if f is not None else 0.0)
            if sh is not None:
                shapes.append(sh)
            cells.append(f"{'-' if f is None else f'{f:.2f}'} / {'-' if sh is None else f'{sh:.2f}'}")
        mf = sum(feas) / len(feas)
        ms = sum(shapes) / len(shapes) if shapes else None
        lines.append(f"| {r['episodes']} | {r['steps']} | " + " | ".join(cells) +
                     f" | {mf:.2f} | {'-' if ms is None else f'{ms:.2f}'} |")
    (out_dir / "curve.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--randomization", required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--every", type=int, default=1, help="evaluate every k-th snapshot")
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    done_names = set()
    curve = out / "curve.jsonl"
    if curve.exists():
        done_names = {json.loads(l)["name"] for l in curve.read_text().splitlines() if l.strip()}
    while True:
        index = run / "snapshots" / "index.jsonl"
        entries = [json.loads(l) for l in index.read_text().splitlines() if l.strip()] if index.exists() else []
        pending = [e for i, e in enumerate(entries) if i % args.every == 0 and Path(e["dir"]).name not in done_names]
        for e in pending:
            snap = Path(e["dir"])
            t0 = time.time()
            reward, status = evaluate(snap, out, args.randomization, args.seeds, args.workers)
            rec = {"name": snap.name, "episodes": e["episodes"], "steps": e["steps"], "updates": e.get("updates"),
                   "train_time_sec": e.get("time_sec"), "reward": reward, "status": status,
                   "eval_wall_sec": time.time() - t0}
            with open(curve, "a") as f:
                f.write(json.dumps(rec) + "\n")
            done_names.add(snap.name)
            write_table(out)
            print(json.dumps({k: rec[k] for k in ("name", "status", "eval_wall_sec")}), flush=True)
        if not args.watch or ((run / "DONE").exists() and not pending):
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
