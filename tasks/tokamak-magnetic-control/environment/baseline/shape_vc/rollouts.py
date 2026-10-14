"""Parallel rollouts of the shape/virtual-circuit controller (privileged, estimator or built package).

python rollouts.py --out OUT --mode truth|est|pkg [--pkg DIR] [--rz dev_randomization|visible|FILE]
                   [--cases ...] [--seeds ...] [--params '{...}'] [--horizon N] [--record]
--record saves per-step observations, noise-free probes and boundary labels (estimator training data).
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_ENVS = {}


def worker(job):
    import numpy as np
    import common as lab
    import controller as K
    case, seed, a = job
    try:
        if case not in _ENVS:
            _ENVS[case] = lab.TokamakMagneticEnv(case, randomization=lab.load_rz(a["rz"]), solver="nonlinear")
        env = _ENVS[case]
        model = None if a["mode"] == "pkg" else pickle.loads(Path(a["model"]).read_bytes())
        if a["mode"] == "pkg":
            c = lab.PkgAdapter(a["pkg"])
        elif a["mode"] == "truth":
            meas = K.TruthMeas()
        else:
            import estimator as est
            meas = est.EstMeas(a["est"])
        if a["mode"] != "pkg":
            c = K.ShapeController(model, meas, a["params"])
        s = lab.run_episode(env, c, seed, horizon=a.get("horizon"), record=a.get("record", False))
        rec = s.pop("_record", None)
        if rec is not None:
            keys = rec[0].keys()
            np.savez_compressed(Path(a["out"]) / f"rec_{case}_{seed}.npz",
                                **{k: np.array([r[k] for r in rec]) for k in keys})
    except Exception as error:  # noqa: BLE001
        import traceback
        s = {"seed": seed, "outcome": "error", "steps": 0, "wall": 0.0, "termination": repr(error)[:300],
             "tb": traceback.format_exc()[-2000:]}
    s["case"] = case
    (Path(a["out"]) / f"ep_{case}_{seed}.json").write_text(json.dumps(s, default=float))
    import common as lab
    print(f"[{case}] " + lab.short(s), flush=True)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="truth")
    ap.add_argument("--rz", default="dev_randomization")
    ap.add_argument("--model", default=str(HERE / "model.pkl"), help="jacobians.py output")
    ap.add_argument("--est", default=str(HERE / "estimator.npz"))
    ap.add_argument("--cases", nargs="*", default=["mastu_diverted_hold", "mastu_high_ip_hold", "mastu_ip_ramp",
                                                    "mastu_p4_shape_hold", "mastu_pressure_drop", "mastu_sequence",
                                                    "mastu_vertical_shift"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[5001, 5002, 5003])
    ap.add_argument("--params", default="{}")
    ap.add_argument("--pkg", default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    a = {"pkg": args.pkg, "rz": args.rz, "model": args.model, "est": args.est, "mode": args.mode, "params": json.loads(args.params),
         "horizon": args.horizon, "out": str(out), "record": args.record}
    (out / "config.json").write_text(json.dumps(a))
    jobs = [(c, s, a) for c in args.cases for s in args.seeds]
    # group by case so a worker reuses its environment
    t0 = time.monotonic()
    with mp.get_context("fork").Pool(min(args.workers, len(jobs)), maxtasksperchild=None) as pool:
        res = pool.map(worker, jobs, chunksize=1)
    rows = {}
    for s in res:
        rows.setdefault(s["case"], []).append(s)
    summ = {}
    for c, eps in rows.items():
        def mean(k):
            v = [e[k] for e in eps if e.get("feasible") and e.get(k) is not None]
            return round(100 * float(sum(v) / len(v)), 2) if v else None
        summ[c] = {"n": len(eps), "success": sum(e.get("outcome") == "success" for e in eps),
                   "kept": sum(bool(e.get("feasible")) for e in eps),
                   "outcomes": [e.get("outcome") for e in eps],
                   "shape_cm": mean("shape_rms_steady"), "rin_cm": mean("rin_error_steady"),
                   "rout_cm": mean("rout_error_steady"), "zx_cm": mean("zx_error_steady"),
                   "ip_pct": mean("ip_error_rel_steady")}
    tot = sum(v["n"] for v in summ.values())
    summ["_total"] = {"success": sum(v["success"] for v in summ.values()), "kept": sum(v["kept"] for v in summ.values()),
                      "n": tot, "wall_s": round(time.monotonic() - t0)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    for c, v in summ.items():
        print(c, json.dumps(v), flush=True)


if __name__ == "__main__":
    main()
