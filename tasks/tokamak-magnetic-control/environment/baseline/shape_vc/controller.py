"""Full-shape magnetic controller in NumPy: least-squares virtual circuits on boundary rays.

Pure NumPy at run time. The plasma state comes from a measurement provider: ``TruthMeas``
(privileged, reads the simulator) or ``EstMeas`` (deployable, linear reconstruction from the
real-time magnetics).
"""
import numpy as np

from common import ANG, CENTER, CTRL, N_COIL, N_PROBE, N_RT, NA, interp_periodic, polar

DT = 5e-4
DEFAULTS = dict(
    k_shape=300.0,         # 1/s: dI/dt = k_shape * least-squares current correction
    lam=3e-3,              # Tikhonov weight on (dI / I_scale)^2 relative to the mean-square boundary error
    w_mid=1.0,             # extra weight of the outboard/inboard midplane points (indices 0, 16)
    w_gap=0.0,             # weight of the separatrix inner/outer gap errors (rows NA+4, NA+5 of J)
    w_x=0.5,               # weight of the X-point errors relative to one boundary point (x32 points)
    k_ip=-20.0, ki_ip=-100.0,
    kz_p=0.003, kz_d=-5e-6, kz_i=0.05,  # delay-robust PID (linear design over 39 plants x delay 0-2)
    tau_c=0.015,           # current-tracking time constant of the coil loop [s]
    margin=0.06,           # current-limit margin (fraction of the full range)
    dmax=400.0,            # max |I_ref - I_meas| [A] (anti-windup)
    ramp=60,               # steps over which the shape gain ramps up
    filt=1.0,              # exponential filter on measurements (1 = none)
    filt_z=1.0,
    x_gate=0.1,            # ignore an X-point measurement this far [m] from its target (lost X-point)
    e_clip=0.05,           # clip each boundary error [m]
    k_zt=0.0,              # 1/s: vertical set-point outer loop from the P6 column of the least squares
    excite=0.0,            # data collection only: amplitude [m] of random low-frequency target offsets
)


class TruthMeas:
    def reset(self, env, info):
        pass

    def __call__(self, obs, env):
        from common import truth
        t = truth(env)
        gaps = np.asarray(env._gaps(env.stepping.eq1, env.gap_height_at(env.t)), float)
        return {"rho": t["rho"], "xlow": t["xlow"], "xup": t["xup"], "zcur": t["zcur"], "gaps": gaps,
                "ip": float(env._plasma_current()), "I": env._coil_currents().copy()}


class ShapeController:
    def __init__(self, model, meas, params=None):
        """model: dict with machine arrays and per-reference Jacobians (see build_model)."""
        self.m = model
        self.meas = meas
        self.p = dict(DEFAULTS, **(params or {}))
        self.R = np.asarray(model["R"], float)
        self.L = np.asarray(model["L"], float)
        self.lim = np.asarray(model["lim"], float)
        self.vmax = np.asarray(model["vmax"], float)
        self.iscale = np.max(np.abs(self.lim), axis=1)

    def select_reference(self, pts, ip_t):
        def dist(r):
            return float(np.sqrt(((pts - r["pts"]) ** 2).sum(1).mean())) + 0.5 * abs(ip_t - r["ip"]) / r["ip"]
        return min(self.m["refs"], key=dist)

    def reset(self, env=None, info=None):
        self.meas.reset(env, info)
        self.k = 0
        self.I_ref = None
        self.ip_int = 0.0
        self.z_int = 0.0
        self.z_prev = None
        self.z_t = None
        self.ref = None
        self.f = None
        if self.p["excite"] > 0:
            rng = np.random.default_rng()
            self.ex = {"a": rng.uniform(0.3, 1.0, 5) * self.p["excite"], "m": np.arange(5),
                       "ph": rng.uniform(0, 2 * np.pi, 5), "f": rng.uniform(3, 30, 6), "ps": rng.uniform(0, 2 * np.pi, 6)}

    def act(self, obs, env=None):
        p = self.p
        obs = np.asarray(obs, float)
        base = N_RT
        pts = obs[base: base + 64].reshape(32, 2)
        xp = obs[base + 64: base + 68].reshape(2, 2)
        flags = obs[base + 68: base + 70]
        ip_t = float(obs[base + 70])
        if self.ref is None:
            self.ref = self.select_reference(pts, ip_t)
        ref = self.ref
        ms = self.meas(obs, env)
        # optional filtering of the reconstruction
        if self.f is None:
            self.f = {k: np.array(v, float) for k, v in ms.items() if k in ("rho", "xlow", "xup", "zcur", "gaps")}
        else:
            for k in [k_ for k_ in ("rho", "xlow", "xup", "gaps") if k_ in self.f and k_ in ms]:
                a = p["filt"]
                self.f[k] = np.where(np.isfinite(ms[k]), self.f[k] + a * (np.asarray(ms[k]) - self.f[k]), self.f[k]) \
                    if np.all(np.isfinite(self.f[k])) else np.asarray(ms[k], float)
            self.f["zcur"] = self.f["zcur"] + p["filt_z"] * (ms["zcur"] - self.f["zcur"])
        rho, xlow, xup, z = self.f["rho"], self.f["xlow"], self.f["xup"], float(self.f["zcur"])
        I = np.asarray(ms["I"], float)
        ip = float(ms["ip"])
        n = len(CTRL)
        if self.I_ref is None:
            self.I_ref = I[:n].copy()
        # ---- shape: weighted least squares over boundary rays and X-points
        th, r_t = polar(pts)
        rho_k = interp_periodic(th, rho)
        J = ref["J"]
        Jk = interp_periodic(th, J[:NA])
        e = r_t - rho_k
        dz_ex = 0.0
        if p["excite"] > 0:
            t = self.k * DT; ex = self.ex
            amp = ex["a"] * np.sin(2 * np.pi * ex["f"][:5] * t + ex["ps"][:5])
            e = e + (amp[None, :] * np.cos(ex["m"][None, :] * th[:, None] + ex["ph"][None, :])).sum(1)
            dz_ex = 0.5 * p["excite"] * np.sin(2 * np.pi * ex["f"][5] * t + ex["ps"][5])
        w = np.where(np.isfinite(e), 1.0 / 32, 0.0)
        w[0] *= p["w_mid"]; w[16] *= p["w_mid"]
        e = np.clip(np.where(np.isfinite(e), e, 0.0), -p["e_clip"], p["e_clip"])
        rows, errs, wts = [Jk], [e], [w]
        if flags.sum() > 0:
            xs = xp[flags > 0.5]
            order = np.argsort(xs[:, 1])
            lo_t = xs[order[0]]
            for meas_x, Jx, tgt in ((xlow, J[NA:NA + 2], lo_t),) + (
                    ((xup, J[NA + 2:NA + 4], xs[order[-1]]),) if len(xs) > 1 else ()):
                ex = tgt - meas_x
                ok = np.all(np.isfinite(ex)) and float(np.max(np.abs(ex))) < p["x_gate"]
                rows.append(Jx); errs.append(np.where(ok, ex, 0.0)); wts.append(np.full(2, p["w_x"] / 32 if ok else 0.0))
        if p["w_gap"] > 0 and "gaps" in self.f and J.shape[0] >= NA + 6:
            g_t = ref["gaps"] + np.array([pts[16, 0] - ref["pts"][16, 0], pts[0, 0] - ref["pts"][0, 0]])
            eg = g_t - self.f["gaps"]
            okg = np.isfinite(eg) & (np.abs(eg) < p["x_gate"])
            rows.append(J[NA + 4:NA + 6]); errs.append(np.where(okg, eg, 0.0)); wts.append(np.where(okg, p["w_gap"] / 32, 0.0))
        A = np.vstack(rows)[:, 1:]                # shape coils and P6 (no solenoid)
        b = np.concatenate(errs); wv = np.concatenate(wts)
        S = self.iscale[1:A.shape[1] + 1]
        As = A * S                                # columns in units of the coil scale
        H = As.T @ (wv[:, None] * As) + p["lam"] * np.eye(A.shape[1])
        x = np.linalg.solve(H, As.T @ (wv * b)) * S
        g = p["k_shape"] * min(1.0, (self.k + 1) / max(1, p["ramp"]))
        dIdt = np.zeros(n)
        dIdt[1:] = g * x[:n - 1]
        dz_shape = float(J[NA + 6, n] * x[n - 1]) if A.shape[1] == n else 0.0   # P6 part -> vertical set-point
        # ---- plasma current through the solenoid
        ip_err = ip_t - ip
        self.ip_int += ip_err * DT
        dIdt[0] = p["k_ip"] * ip_err + p["ki_ip"] * self.ip_int
        # ---- currents -> voltages
        I_new = self.I_ref + dIdt * DT
        span = self.lim[:n, 1] - self.lim[:n, 0]
        lo = self.lim[:n, 0] + p["margin"] * span
        hi = self.lim[:n, 1] - p["margin"] * span
        I_new = np.clip(I_new, lo, hi)
        I_new = np.clip(I_new, I[:n] - p["dmax"], I[:n] + p["dmax"])
        dIdt_eff = (I_new - self.I_ref) / DT
        self.I_ref = I_new
        Lc = self.L[:n, :n]
        V = self.R[:n] * I[:n] + Lc @ dIdt_eff + Lc @ ((self.I_ref - I[:n]) / p["tau_c"])
        # ---- vertical (P6)
        if self.z_t is None:
            self.z_t = ref["zcur"] + float(np.median(pts[:, 1]) - np.median(ref["pts"][:, 1]))
        # outer loop: the vertical set-point follows the boundary error's vertical (P6) component
        self.z_t += p["k_zt"] * min(1.0, (self.k + 1) / max(1, p["ramp"])) * dz_shape * DT
        z_t = self.z_t + dz_ex
        zip_err = (z_t - z) * ip
        zipv = 0.0 if self.z_prev is None else (z - self.z_prev) / DT * ip
        self.z_prev = z
        self.z_int += zip_err * DT
        v6 = p["kz_p"] * zip_err + p["kz_d"] * zipv + p["kz_i"] * self.z_int
        out = np.concatenate([V, [v6]])
        self.k += 1
        out = np.where(np.isfinite(out), out, 0.0)
        return np.clip(out, -self.vmax, self.vmax)


def build_model(case_names, rz="none", cache_dir=None):
    """Machine arrays and one Jacobian per distinct reference shape."""
    from pathlib import Path
    from common import TokamakMagneticEnv, static_jacobian
    refs = []
    model = None
    for name in case_names:
        env = TokamakMagneticEnv(name, randomization="none")
        if model is None:
            tok = env.tokamak
            model = {"R": np.asarray(tok.coil_resist[:12], float), "L": np.asarray(tok.coil_self_ind[:12, :12], float),
                     "lim": np.asarray(env.spec.current_limits, float), "vmax": np.asarray(env.vmax, float), "refs": []}
        pts = np.asarray(env.target_points, float)
        if any(np.allclose(r["pts"], pts, atol=1e-4) and abs(r["ip"] - env.reference_ip) < 1.0 for r in refs):
            continue
        cache = None if cache_dir is None else str(Path(cache_dir) / f"jac_{name}.npz")
        jac = static_jacobian(env, cache)
        refs.append({"case": name, "pts": pts, "ip": float(env.reference_ip), "J": jac["J"], "v0": jac["v0"],
                     "gaps": np.asarray(env.reference_gaps, float),
                     "zcur": float(env.reference_eq.Zcurrent()), "xpts": np.asarray(env.target_xpoints, float)})
    model["refs"] = refs
    return model
