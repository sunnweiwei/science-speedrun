"""Privileged classical reference: FreeGSNKE pulse-design-tool PCS (PID + virtual circuits).

The controller reads the true plasma descriptors (Rin, Rout, X-point, vertical position,
Ip) from the simulator, i.e. it has a perfect real-time reconstruction. It is a
reference number for the tables, not a protocol-conforming submission. Gains and
structure follow FreeGSNKE example11 (MAST-U flat-top); virtual circuits are
regenerated per case from the nominal reference equilibrium. Since v0.2 the lower X-point
height is feedback-controlled as on MAST-U (Lvovskiy et al. 2025); without it
(--no-zx-feedback, the v0.1 reference) the X-point drifts by several cm under the
randomized plant.

Usage: python baseline/pid/pid_reference.py --output OUT [--cases ...] [--seeds ...]
       [--randomization visible|none|FILE] [--workers N]
"""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from tokagym import CASES, TokamakMagneticEnv  # noqa: E402
from tokagym import cases as C  # noqa: E402
from tokagym import metrics as M  # noqa: E402
sys.path.insert(0, str(HERE.parents[1] / "evaluation"))
import supervise  # noqa: E402

CTRL_COILS = ["Solenoid", "PX", "D1", "D2", "D3", "Dp", "D5", "D6", "D7", "P4", "P5"]
VERTICAL_COILS = ["P6"]
GAINS = {"Rin": 120.0, "Rout": 170.0, "Rx": 120.0, "Zx": 120.0}


def make_descriptor_fn(xpoint_box):
    def plasma_descriptors(eq):
        rin, rout = eq.innerOuterSeparatrix(Z=float(np.asarray(eq.opt)[0, 1]))   # gaps at the axis height
        if xpoint_box is None:
            return np.array([rin, rout])
        (r0, z0), (r1, z1) = xpoint_box
        xpt = np.asarray(eq.xpt)
        if xpt.size:
            m = (xpt[:, 0] >= r0) & (xpt[:, 0] <= r1) & (xpt[:, 1] >= z0) & (xpt[:, 1] <= z1)
            cand = xpt[m, :2]
        else:
            cand = np.zeros((0, 2))
        if len(cand) == 0:
            cand = xpt[:, :2] if xpt.size else np.array([[np.nan, np.nan]])
        opt = np.asarray(eq.opt)[0, :2]
        rx, zx = cand[np.argmin(np.linalg.norm(cand - opt, axis=1))]
        return np.array([rin, rout, rx, zx])
    return plasma_descriptors


def waveform(times, vals):
    return {"times": np.asarray(times, dtype=float), "vals": np.asarray(vals, dtype=float)}


def build_pcs(env, targets, descriptors, tmax, zx_feedback=True):
    from freegsnke import GSstaticsolver
    from freegsnke.control_loop.pcs import PlasmaControlSystem
    from freegsnke.control_loop.vc_provider import VCGenerator

    tok = env.tokamak
    wf = env.case.waveforms
    ip0 = env.reference_ip
    ip_times = np.array(wf.ip_times, dtype=float)
    plasma_data = {
        "ip_ref": waveform(ip_times, ip0 * np.array(wf.ip_scale)),
        "vloop_ff": waveform([0.0, tmax], [0.0, 0.0]),
        "ip_blend": waveform([0.0, tmax], [1.0, 1.0]),
        "k_prop": waveform([0.0], [-20.0]), "k_int": waveform([0.0], [-100.0]),
        "k_deriv": waveform([0.0], [0.0]), "M_solenoid": waveform([0.0], [1.0]),
    }
    z_times = np.array(wf.z_times, dtype=float)
    # set-points follow the environment's (possibly shifted and blended) targets every 5 ms
    ref_times = np.arange(0.0, tmax + 5e-3, 5e-3)
    refs = {"Rin": [], "Rout": [], "Rx": [], "Zx": []}
    for t in ref_times:
        gaps, zx = env.gap_targets_at(t)
        _, xpts, _ = env.targets_at(t)
        refs["Rin"].append(gaps[0]); refs["Rout"].append(gaps[1])
        if len(xpts):
            refs["Rx"].append(xpts[0][0]); refs["Zx"].append(zx)
    shape_data = {}
    for i, name in enumerate(targets):
        refwf = waveform(ref_times, refs[name])
        shape_data[name] = {"ff": waveform([0.0, tmax], [0.0, 0.0]), "ref": refwf,
                            # Zx (lower X-point height) is feedback-controlled unless --no-zx-feedback (v0.1)
                            "blend": waveform([0.0, tmax], [0.0 if name == "Zx" and not zx_feedback else 1.0] * 2),
                            "k_prop": waveform([0.0], [GAINS[name]]), "k_int": waveform([0.0], [0.0]),
                            "k_deriv": waveform([0.0], [0.0])}
    solver = GSstaticsolver.NKGSsolver(env.reference_eq)
    targets_ctrl = [t for t in targets if t != "Zx" or zx_feedback]
    vcg = VCGenerator(solver=solver, target_calculator=descriptors, target_names=targets,
                      targets_ctrl=targets_ctrl, targets_calc=targets, coils_calc=CTRL_COILS[1:],
                      coils=CTRL_COILS, n_vc_workers=1)
    with contextlib.redirect_stdout(io.StringIO()):
        circuits_data = vcg.generate_fixed_schedule(
            times=[0.0], eq_list=[env.reference_eq], profile_list=[env.reference_profiles],
            tikhonov_lambda=1e-12 * np.ones(len(CTRL_COILS) - 1), verbose=False)
    circuits_data["plasma"] = {"times": np.array([0.0]), "vals": [np.array([1.0] + [0.0] * (len(CTRL_COILS) - 1))]}
    circuits_data.setdefault("coil_order", CTRL_COILS)
    zeros = waveform([0.0, tmax], [0.0, 0.0])
    for coil in CTRL_COILS:
        circuits_data[coil + "_ref"] = zeros
    lims = env.spec.current_limits
    idx = [list(env.spec.active_coils).index(c) for c in CTRL_COILS]
    systems_data = {
        "min_coil_curr_lims": {"times": [0.0], "vals": [lims[idx, 0]]},
        "max_coil_curr_lims": {"times": [0.0], "vals": [lims[idx, 1]]},
        "max_coil_curr_ramp_lims": {"times": [0.0], "vals": [1e12 * np.ones(len(CTRL_COILS))]},
    }
    for coil in CTRL_COILS:
        systems_data[coil + "_pert"] = zeros
    n = len(CTRL_COILS)
    pf_data = {
        "R_matrix": {"times": [0.0], "vals": [np.asarray(tok.coil_resist[:n])]},
        "M_FF_matrix": {"times": [0.0], "vals": [np.asarray(tok.coil_self_ind[:n, :n])]},
        "M_FB_matrix": {"times": [0.0], "vals": [np.asarray(tok.coil_self_ind[:n, :n])]},
        "coil_gains": {"times": [0.0], "vals": [0.015 * np.ones(n)]},
        "coil_voltage_lims": {"times": np.array([0.0]), "vals": [list(env.vmax[idx])]},
        "coil_voltage_slew_lims": {"times": [0.0], "vals": [1e12 * np.ones(n)]},
    }
    # vertical set-point: the current-centroid height of the (blended) target equilibrium
    z_ref = np.array([env.vertical_target_at(t) for t in ref_times])
    vertical_data = {"z_ref": waveform(ref_times, z_ref), "k_prop": waveform([0.0], [0.025]),
                     "k_deriv": waveform([0.0], [5e-7])}
    coil_activation_data = {c + "_activation": waveform([0.0], [1.0]) for c in env.spec.active_coils}
    pcs = PlasmaControlSystem(
        plasma_data=plasma_data, shape_data=shape_data, shape_control_mode="PID",
        circuits_data=circuits_data, systems_data=systems_data, pf_data=pf_data,
        vertical_data=vertical_data, coil_activation_data=coil_activation_data,
        active_coils=list(env.spec.active_coils), ctrl_coils=CTRL_COILS,
        vertical_coils=VERTICAL_COILS, ctrl_targets=targets, plasma_target=["plasma"])
    return pcs


def run_episode(env, pcs, descriptors, seed, queue=None):
    obs, info = env.reset(seed=seed)
    tok = env.tokamak
    dt = env.stepping.dt_step
    n = len(CTRL_COILS)
    ip_hist = ip_err = 0.0
    T_err = np.zeros(len(pcs.ShapeController.data) if hasattr(pcs, "ShapeController") else 4)
    T_hist = np.zeros_like(T_err)
    I_prev = env._coil_currents()[:n]
    V_prev = np.zeros(n)
    z_prev = env.stepping.eq1.Zcurrent()
    zipv = 0.0
    t = 0.0
    terminated = truncated = False
    resist = np.asarray(tok.coil_resist[: env.n_active], dtype=float)
    while not (terminated or truncated):
        eq = env.stepping.eq1
        ip = env._plasma_current()
        z = eq.Zcurrent()
        T_meas = descriptors(eq)
        with contextlib.redirect_stdout(io.StringIO()):
            V, ip_hist, ip_err, T_err, T_hist, I_appr, _ = pcs.calculate_ctrl_voltages(
                t=t, dt=dt, dt_simulator=dt, ip_meas=ip, ip_hist_prev=ip_hist, ip_err_prev=ip_err,
                T_meas=np.asarray(T_meas, dtype=float), T_err_prev=T_err, T_hist_prev=T_hist,
                I_approved_prev=I_prev, I_meas=env._coil_currents()[:n], V_approved_prev=V_prev,
                zip_meas=z * ip, zipv_meas=zipv, active_coil_resists=resist.copy(), verbose=False)
        V = np.asarray(V, dtype=float).ravel()
        V = np.clip(V, -env.vmax, env.vmax)
        obs, r, terminated, truncated, step_info = env.step(V)
        if queue is not None:
            queue.put({"kind": "heartbeat", "shape_rms": step_info["shape_rms"],
                       "ip_error_rel": step_info["ip_error_rel"]})
        zipv = ((eq.Zcurrent() - z_prev) / dt) * ip
        z_prev = z
        I_prev = np.asarray(I_appr, dtype=float).ravel()[:n]
        V_prev = V[:n]
        t += dt
    s = env.episode_summary()
    s["seed"] = seed
    return s


def run_case_stream(args, seeds, queue):
    case_name, randomization, out_dir, horizon, zx_feedback = args
    case = CASES[case_name]
    env = TokamakMagneticEnv(case_name, randomization=randomization, solver="nonlinear", horizon=horizon)
    if case.diverted:
        lower = env.target_xpoints[np.argmin(env.target_xpoints[:, 1])]
        box = [(lower[0] - 0.3, lower[1] - 0.3), (lower[0] + 0.3, lower[1] + 0.3)]
        targets = ["Rin", "Rout", "Rx", "Zx"]
    else:
        box, targets = None, ["Rin", "Rout"]
    descriptors = make_descriptor_fn(box)
    setup = None
    for seed in seeds:
        t_ep = time.monotonic()
        try:
            t0 = time.monotonic()
            pcs = build_pcs(env, targets, descriptors, tmax=env.horizon * C.DT,   # fresh controller state per episode
                            zx_feedback=zx_feedback)
            setup = time.monotonic() - t0
            e = run_episode(env, pcs, descriptors, seed, queue)
        except Exception as error:  # noqa: BLE001
            e = {"seed": seed, "feasible": False, "complete": False,
                 "termination": f"reference_error: {type(error).__name__}: {error}"[:300]}
        e["kind"] = "episode"; e["wall_time_sec"] = time.monotonic() - t_ep
        print(f"[pid {case_name}] seed={seed} outcome={e.get('outcome')} steps={e.get('steps')} "
              f"shape_cm={None if e.get('shape_rms_steady') is None else round(100 * e['shape_rms_steady'], 2)} "
              f"term={e.get('termination')} wall={e['wall_time_sec']:.0f}s", file=sys.stderr, flush=True)
        queue.put(e)
    queue.put({"kind": "case", "targets": targets, "pcs_setup_sec": setup})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--cases", nargs="*", default=sorted(CASES))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--randomization", default="visible")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=None, help="development only; shorter episodes")
    ap.add_argument("--no-zx-feedback", dest="zx_feedback", action="store_false",
                    help="no feedback on the lower X-point height (the v0.1 reference); MAST-U controls it")
    args = ap.parse_args()
    rz = args.randomization
    if rz not in ("visible", "none"):
        rz = json.loads(Path(rz).read_text())["randomization"]
    jobs = [(c, rz, args.output, args.horizon, args.zx_feedback) for c in args.cases]
    timeout = float(os.environ.get("TOKAGYM_EPISODE_TIMEOUT", "5400"))   # as the verifier: the sequence case is 1200 steps
    raw = supervise.run_supervised(run_case_stream, jobs, list(args.seeds), timeout, max(1, args.workers),
                                   silence_timeout=float(os.environ.get("TOKAGYM_STEP_SILENCE", "600")))
    results = []
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    for i, job in enumerate(jobs):
        r = raw[i]
        result = {"case": job[0], "targets": (r["case"] or {}).get("targets"),
                  "pcs_setup_sec": (r["case"] or {}).get("pcs_setup_sec"),
                  "episodes": [{k: v for k, v in e.items() if k != "kind"} for e in r["episodes"]],
                  "supervisor_notes": r["errors"]}
        (out / f"{job[0]}.json").write_text(json.dumps(result, indent=1, default=float))
        results.append(result)
    rows = []
    for r in results:
        for e in r["episodes"]:   # supervisor-killed episodes are classified as the verifier does
            e["outcome"] = e.get("outcome") or ("solver_void" if M.void_candidate(e.get("last_states", []))
                                                else "solver_failure")
        scored = [e for e in r["episodes"] if e["outcome"] != "solver_void"]
        feas = [e for e in scored if e.get("feasible")]

        def steady(key, scale=100.0):
            vals = [e[key] for e in feas if e.get(key) is not None]
            return scale * float(np.mean(vals)) if vals else None

        rows.append({
            "case": r["case"], "n": len(r["episodes"]), "void": len(r["episodes"]) - len(scored),
            "success": sum(e["outcome"] == "success" for e in scored), "feasible": len(feas),
            "shape_rms_cm": steady("shape_rms_steady"), "xpoint_cm": steady("xpoint_error_steady"),
            "rin_cm": steady("rin_error_steady"), "rout_cm": steady("rout_error_steady"),
            "zx_cm": steady("zx_error_steady"), "ip_pct": steady("ip_error_rel_steady"),
            "outcomes": [e["outcome"] for e in r["episodes"]],
            "terminations": [e.get("termination") for e in r["episodes"]],
        })
    Path(args.output, "summary.json").write_text(json.dumps(rows, indent=1))
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
