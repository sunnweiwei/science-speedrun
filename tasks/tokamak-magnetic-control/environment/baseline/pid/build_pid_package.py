"""Build a deployable (protocol-conforming) PID/virtual-circuit package from demonstrations.

Components stored in the package (NumPy arrays + JSON, no simulator access at inference):
  * a ridge-regression estimator from the 515 real-time magnetic/current signals to the
    PCS descriptors [Rin, Rout, Rx, Zx, Zcur]  (the 'real-time reconstruction'),
  * virtual circuits computed once from the base reference equilibrium,
  * coil resistances / inductances / limits of the machine,
  * the FPDT PID gains.
Setpoints are read from the observation's target points, target X-point and target Ip,
so one package serves every case without a case lookup.

Virtual circuits, set-point offsets and the reference current centroid are stored for every
distinct reference shape of the case set; at run time the package picks the stored reference
whose target boundary points are nearest to the observation's targets (gain scheduling on
the requested shape, no case lookup) and ramps its set-points from the first estimate to
the targets over ``ramp_steps``.

Usage: python baseline/pid/build_pid_package.py --demos DEMOS.npz [DEMOS2.npz ...] --output PKG_DIR
"""

import argparse
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from tokagym import TokamakMagneticEnv  # noqa: E402
from tokagym import cases as C  # noqa: E402
import pid_reference as P  # noqa: E402

N_RT = 102 + 398 + 12 + 1     # real-time signals: flux loops, pickups, coil currents, Ip


def load_demos(paths):
    """Concatenate demonstration files; episode indices are made unique across files."""
    obs, desc, ep, offset = [], [], [], 0
    for path in paths:
        d = np.load(path)
        obs.append(d["obs"][:, :N_RT]); desc.append(d["desc"]); ep.append(d["episode_index"] + offset)
        offset += int(d["episode_index"].max()) + 1
    return {"obs": np.concatenate(obs), "desc": np.concatenate(desc), "episode_index": np.concatenate(ep)}


def fit_estimator(demos, ridge=1e-3, holdout=0.2, xpoint_window=0.25):
    """Ridge regression per descriptor. X-point columns are fitted only on samples whose
    labelled lower X-point lies within ``xpoint_window`` of the median (the label jumps to a
    far-away saddle point when the divertor configuration is lost)."""
    obs, desc, ep = demos["obs"][:, :N_RT].astype(np.float64), demos["desc"].astype(np.float64), demos["episode_index"]
    ok = np.isfinite(desc).all(1) & np.isfinite(obs).all(1)
    obs, desc, ep = obs[ok], desc[ok], ep[ok]
    episodes = np.unique(ep)
    rng = np.random.default_rng(0)
    test_eps = set(rng.choice(episodes, max(1, int(holdout * len(episodes))), replace=False).tolist())
    te = np.array([e in test_eps for e in ep]); tr = ~te
    mu, sd = obs[tr].mean(0), obs[tr].std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Xb = np.hstack([(obs - mu) / sd, np.ones((len(obs), 1))])
    xref = np.median(desc[tr][:, 2:4], axis=0)
    near = np.all(np.abs(desc[:, 2:4] - xref) < xpoint_window, axis=1)
    W = np.zeros((Xb.shape[1], desc.shape[1]))
    rmse_tr, rmse_te = np.zeros(desc.shape[1]), np.zeros(desc.shape[1])
    for j in range(desc.shape[1]):
        mask_tr = tr & (near if j in (2, 3) else True)
        mask_te = te & (near if j in (2, 3) else True)
        A = Xb[mask_tr].T @ Xb[mask_tr] + ridge * mask_tr.sum() * np.eye(Xb.shape[1])
        W[:, j] = np.linalg.solve(A, Xb[mask_tr].T @ desc[mask_tr, j])
        pred = Xb @ W[:, j]
        rmse_tr[j] = np.sqrt(((pred[mask_tr] - desc[mask_tr, j]) ** 2).mean())
        rmse_te[j] = np.sqrt(((pred[mask_te] - desc[mask_te, j]) ** 2).mean())
    return {"mu": mu, "sd": sd, "W": W, "rmse_train": rmse_tr, "rmse_test": rmse_te,
            "n_train": int(tr.sum()), "n_test": int(te.sum()), "xpoint_fraction": float(near.mean())}


def machine_arrays(env, targets):
    tok = env.tokamak
    n = len(P.CTRL_COILS)
    lower = env.target_xpoints[np.argmin(env.target_xpoints[:, 1])]
    box = [(lower[0] - 0.3, lower[1] - 0.3), (lower[0] + 0.3, lower[1] + 0.3)]
    descriptors = P.make_descriptor_fn(box)
    from freegsnke import GSstaticsolver
    from freegsnke.control_loop.vc_provider import VCGenerator
    solver = GSstaticsolver.NKGSsolver(env.reference_eq)
    vcg = VCGenerator(solver=solver, target_calculator=descriptors, target_names=targets,
                      targets_ctrl=[t for t in targets if t != "Zx"], targets_calc=targets,
                      coils_calc=P.CTRL_COILS[1:], coils=P.CTRL_COILS, n_vc_workers=1)
    with contextlib.redirect_stdout(io.StringIO()):
        circuits = vcg.generate_fixed_schedule(times=[0.0], eq_list=[env.reference_eq],
                                               profile_list=[env.reference_profiles],
                                               tikhonov_lambda=1e-12 * np.ones(n - 1), verbose=False)
    vcs = {t: np.asarray(circuits[t]["vals"][0], dtype=float).tolist() for t in targets if t in circuits}
    idx = [list(env.spec.active_coils).index(c) for c in P.CTRL_COILS]
    lims = env.spec.current_limits
    # Setpoint calibration: the observation carries psi_N=0.995 contour points, the PID
    # descriptors are separatrix values at Z=0; store the reference offsets.
    pts = env.target_points
    z_mid = float(np.median(pts[:, 1]))
    near = np.abs(pts[:, 1] - z_mid) < 0.25
    rin_c, rout_c = float(pts[near, 0].min()), float(pts[near, 0].max())
    rin_s, rout_s = env.reference_eq.innerOuterSeparatrix()
    return {
        "case": env.case.name,
        "target_points": np.asarray(pts, dtype=float).tolist(),
        "reference_ip": float(env.reference_ip),
        "reference_zcur": float(env.reference_eq.Zcurrent()),
        "setpoint_offsets": {"Rin": float(rin_s) - rin_c, "Rout": float(rout_s) - rout_c},
        "targets": targets, "virtual_circuits": vcs,
        "coil_resist": np.asarray(tok.coil_resist[:n], dtype=float).tolist(),
        "coil_self_ind": np.asarray(tok.coil_self_ind[:n, :n], dtype=float).tolist(),
        "active_coil_resist": np.asarray(tok.coil_resist[: env.n_active], dtype=float).tolist(),
        "min_coil_curr": lims[idx, 0].tolist(), "max_coil_curr": lims[idx, 1].tolist(),
        "coil_voltage_lims": env.vmax[idx].tolist(),
        "gains": {"Rin": 120.0, "Rout": 170.0, "Rx": 120.0, "Zx": 120.0},
        "ip_gains": {"k_prop": -20.0, "k_int": -100.0}, "vertical_gains": {"k_prop": 0.025, "k_deriv": 5e-7},
        "reference_axis_z": float(env.reference_axis[1]),
        "reference_descriptors": np.asarray(descriptors(env.reference_eq), dtype=float).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True, nargs="+")
    ap.add_argument("--output", required=True)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--cases", nargs="*", default=sorted(C.CASES), help="reference shapes to schedule over")
    ap.add_argument("--ramp-steps", type=int, default=120, help="set-point ramp from first estimate to targets")
    ap.add_argument("--filter-alpha", type=float, nargs=5, default=[0.15, 0.15, 0.15, 0.15, 0.35])
    ap.add_argument("--vertical-gains", type=float, nargs=2, default=[0.025, 5e-7], metavar=("K_PROP", "K_DERIV"),
                    help="vertical (P6) controller gains on z*Ip and d(z*Ip)/dt")
    args = ap.parse_args()
    demos = load_demos(args.demos)
    est = fit_estimator(demos, args.ridge)
    print("estimator RMSE (cm) train:", np.round(100 * est["rmse_train"], 2), "test:", np.round(100 * est["rmse_test"], 2),
          f"(n_train={est['n_train']}, n_test={est['n_test']})", flush=True)
    targets = ["Rin", "Rout", "Rx", "Zx"]
    refs = []
    for case_name in args.cases:
        env = TokamakMagneticEnv(case_name, randomization="none")
        a = machine_arrays(env, targets)
        dup = [r for r in refs if np.allclose(r["target_points"], a["target_points"], atol=1e-4)
               and abs(r["reference_ip"] - a["reference_ip"]) < 1.0]
        if dup:
            print(f"{case_name}: same reference shape as {dup[0]['case']}; skipped", flush=True)
            continue
        refs.append(a)
        print(f"{case_name}: reference stored (offsets {a['setpoint_offsets']}, Ip {a['reference_ip']:.0f})", flush=True)
    arrays = dict(refs[0])
    arrays["references"] = [{k: r[k] for k in ("case", "target_points", "reference_ip", "reference_zcur",
                                               "setpoint_offsets", "virtual_circuits", "reference_axis_z",
                                               "reference_descriptors")} for r in refs]
    arrays["ramp_steps"] = args.ramp_steps
    arrays["vertical_gains"] = {"k_prop": args.vertical_gains[0], "k_deriv": args.vertical_gains[1]}
    arrays["filter_alpha"] = args.filter_alpha
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "estimator.npz", mu=est["mu"], sd=est["sd"], W=est["W"])
    (out / "machine.json").write_text(json.dumps(arrays))
    (out / "estimator_report.json").write_text(json.dumps(
        {"rmse_train_m": est["rmse_train"].tolist(), "rmse_test_m": est["rmse_test"].tolist(),
         "n_train": est["n_train"], "n_test": est["n_test"], "descriptors": ["Rin", "Rout", "Rx", "Zx", "Zcur"]}, indent=1))
    shutil.copy2(HERE / "pid_package_policy.py", out / "policy.py")
    (out / "infer.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(dirname \"$(readlink -f \"$0\")\")\"\nexec python3 policy.py\n")
    (out / "infer.sh").chmod(0o755)
    (out / "README.md").write_text(
        "# Deployable PID / virtual-circuit reference (tokamak-policy/v1)\n\n"
        "Classical controller with a ridge-regression estimator from real-time magnetics to\n"
        "[Rin, Rout, Rx, Zx, Zcur]; virtual circuits from the base reference equilibrium; FPDT gains.\n"
        f"Estimator held-out RMSE [cm]: {np.round(100 * est['rmse_test'], 2).tolist()} (Rin, Rout, Rx, Zx, Zcur).\n"
        "Set-points are derived from the observation's targets, ramped from the first estimate over\n"
        f"{args.ramp_steps} steps; virtual circuits are scheduled on the nearest stored reference shape\n"
        f"({len(refs)} references: {[r['case'] for r in refs]}). No learned RL component; no case lookup.\n")
    print("package written to", out, flush=True)


if __name__ == "__main__":
    main()
