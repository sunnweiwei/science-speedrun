"""Resident tokamak-policy/v1 server: linear magnetic reconstruction + full-shape virtual circuits.

NumPy only. Per step: (1) a linear estimator maps the 513 real-time signals (flux loops,
pickups, coil currents, Ip) to the psi_N=0.995 boundary radius on 64 rays, the two X-points
and the current centroid height; channels reading exactly 0 at the first step are dropped by
re-solving the stored normal equations; (2) a weighted least-squares virtual circuit turns
the boundary/X-point errors at the observed target points into coil-current rates of change
(integral action in current space), the solenoid holds Ip with a PI loop, and the coil
currents follow the integrated references through an inductive voltage law with a
current-limit governor; (3) a delay-robust PD on the current-centroid height drives P6.
"""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL = "tokamak-policy/v1"
DT = 5e-4
N_PROBE, N_COIL = 500, 12
N_RT = N_PROBE + N_COIL + 1
NA = 64
CENTER = np.array([0.95, 0.0])
N_CTRL = 11   # Solenoid .. P5; P6 is the vertical coil


def polar(points):
    d = np.asarray(points, float) - CENTER
    return np.arctan2(d[:, 1], d[:, 0]), np.hypot(d[:, 0], d[:, 1])


def interp_periodic(theta, g):
    x = (np.asarray(theta) + np.pi) / (2 * np.pi) * NA
    f = np.floor(x)
    i0 = f.astype(int) % NA
    i1 = (i0 + 1) % NA
    w = x - f
    if g.ndim == 1:
        return (1 - w) * g[i0] + w * g[i1]
    return (1 - w)[:, None] * g[i0] + w[:, None] * g[i1]


class Estimator:
    """Linear reconstruction, one set of normal equations per stored reference shape; the set is
    chosen per step from the observation's targets (gain scheduling, as the virtual circuits)."""
    def __init__(self, path):
        d = np.load(path)
        self.sx, self.var, self.lam = d["sx"], d["var"], float(d["lam"])
        self.n = int(d["n_groups"]) if "n_groups" in d.files else 1
        if "G" in d.files:   # v0 single-group file
            self.groups = [{"mu_x": d["mu_x"], "mu_y": d["mu_y"], "G": d["G"], "C": d["C"]}]
        else:
            self.groups = [{"mu_x": d[f"mu_x{i}"], "mu_y": d[f"mu_y{i}"], "G": d[f"G{i}"], "C": d[f"C{i}"]}
                           for i in range(self.n)]
        self.ref_pts = d["ref_pts"] if "ref_pts" in d.files else None
        self.ref_ip = d["ref_ip"] if "ref_ip" in d.files else None
        for g in self.groups:
            g["W_all"] = self._solve(g, np.ones(N_RT, bool))
        self.reset()

    def _solve(self, g, keep):
        G = g["G"][np.ix_(keep, keep)] + np.diag(self.var[keep] + self.lam)
        return np.linalg.solve(G, g["C"][keep])

    def reset(self):
        self.keep = None
        self.W = [None] * len(self.groups)
        self.last_t = None
        self.gi = 0

    def group(self, obs):
        if self.ref_pts is None or len(obs) < N_RT + 71:
            return 0
        t = obs[N_RT: N_RT + 71]
        if self.last_t is None or not np.array_equal(t, self.last_t):
            pts = t[:64].reshape(32, 2); ipt = t[70]
            d = np.sqrt(((pts[None] - self.ref_pts) ** 2).sum(2).mean(1)) + 0.5 * np.abs(ipt - self.ref_ip) / self.ref_ip
            self.gi = int(np.argmin(d)); self.last_t = t.copy()
        return self.gi

    def predict(self, obs):
        obs = np.asarray(obs, float)
        x = obs[:N_RT]
        if self.keep is None:
            keep = np.ones(N_RT, bool)
            keep[:N_PROBE] = x[:N_PROBE] != 0.0
            self.keep = keep
        gi = self.group(obs)
        g = self.groups[gi]
        if self.W[gi] is None:
            self.W[gi] = g["W_all"] if self.keep.all() else self._solve(g, self.keep)
        xs = ((x - g["mu_x"]) / self.sx)[self.keep]
        return g["mu_y"] + xs @ self.W[gi]


class Controller:
    """Same control law as the reference implementation, organised for speed: while the targets
    are unchanged, the estimator columns needed at the target rays (interpolation folded into the
    weights) and the least-squares operator are cached, so a step is one 513 x 37 product plus a
    few small vector operations."""

    def __init__(self):
        m = np.load(HERE / "model.npz")
        self.R, self.L, self.lim, self.vmax = m["R"], m["L"], m["lim"], m["vmax"]
        self.refs = [{"pts": m[f"pts{i}"], "ip": float(m["ip"][i]), "J": m[f"J{i}"], "zcur": float(m["zcur"][i])}
                     for i in range(int(m["n_refs"]))]
        self.p = p = json.loads((HERE / "params.json").read_text())
        if p.get("filt", 1.0) != 1.0 or p.get("filt_z", 1.0) != 1.0 or p.get("w_gap", 0.0) != 0.0:
            raise SystemExit("this implementation assumes filt = filt_z = 1 and w_gap = 0")
        self.est = Estimator(HERE / "estimator.npz")
        self.iscale = np.max(np.abs(self.lim), axis=1)
        span = self.lim[:N_CTRL, 1] - self.lim[:N_CTRL, 0]
        self.lo = self.lim[:N_CTRL, 0] + p["margin"] * span
        self.hi = self.lim[:N_CTRL, 1] - p["margin"] * span
        self.Lc = self.L[:N_CTRL, :N_CTRL]
        self.Rc = self.R[:N_CTRL].copy()
        self.ncol = min(N_COIL, self.refs[0]["J"].shape[1]) - 1     # shaping coils (+ P6 if in the Jacobian)
        self.lam_eye = p["lam"] * np.eye(self.ncol)
        self.S = self.iscale[1:1 + self.ncol]
        self.w_pts = np.full(32, 1.0 / 32)
        self.w_pts[[0, 16]] *= p["w_mid"]
        self.out = np.empty(N_COIL)

    def reset(self, low, high):
        self.low, self.high = np.asarray(low, float), np.asarray(high, float)
        self.est.reset()
        self.k = 0
        self.I_ref = None
        self.ip_int = 0.0
        self.z_prev = None
        self.z_int = 0.0
        self.ref = None
        self.tkey = None
        self.z_t = None

    def _targets(self, obs, t):
        """Rebuild the per-target cache (targets changed)."""
        p = self.p
        pts = t[:64].reshape(32, 2)
        xp = t[64:68].reshape(2, 2)
        flags = t[68:70]
        if self.ref is None:
            ip_t = float(t[70])
            def dist(r):
                return float(np.sqrt(((pts - r["pts"]) ** 2).sum(1).mean())) + 0.5 * abs(ip_t - r["ip"]) / r["ip"]
            self.ref = min(self.refs, key=dist)
        est = self.est
        if est.keep is None:
            keep = np.ones(N_RT, bool)
            keep[:N_PROBE] = obs[:N_PROBE] != 0.0
            est.keep = keep
        gi = est.group(obs)
        g = est.groups[gi]
        if est.W[gi] is None:
            est.W[gi] = g["W_all"] if est.keep.all() else est._solve(g, est.keep)
        W = est.W[gi]
        th, r_t = polar(pts)
        x = (th + np.pi) / (2 * np.pi) * NA
        f = np.floor(x)
        i0 = f.astype(int) % NA
        i1 = (i0 + 1) % NA
        w = x - f
        mu_y = g["mu_y"]
        cols = [(1 - w) * W[:, i0] + w * W[:, i1], W[:, NA:NA + 4], W[:, -1:]]
        self.Wb = np.hstack(cols)
        self.mub = np.concatenate([(1 - w) * mu_y[i0] + w * mu_y[i1], mu_y[NA:NA + 4], mu_y[-1:]])
        self.mu_x = g["mu_x"][est.keep]
        self.isx = 1.0 / est.sx[est.keep]
        self.keep_idx = np.flatnonzero(est.keep)
        J = self.ref["J"]
        Jk = (1 - w)[:, None] * J[i0] + w[:, None] * J[i1]
        self.r_t = r_t
        xs = xp[flags > 0.5]
        order = np.argsort(xs[:, 1]) if len(xs) else []
        self.nx = min(len(xs), 2)
        self.xt = np.zeros(4)
        if self.nx >= 1:
            self.xt[:2] = xs[order[0]]
        if self.nx == 2:
            self.xt[2:] = xs[order[-1]]
        self.A_all = np.vstack([Jk, J[NA:NA + 4]])[:, 1:1 + self.ncol] * self.S     # 36 x (10 or 11)
        self.dz_dI6 = float(J[NA + 6, N_CTRL]) if self.ncol == N_CTRL else 0.0
        self.K = {}
        if self.z_t is None:
            self.z_t = self.ref["zcur"] + float(np.median(pts[:, 1]) - np.median(self.ref["pts"][:, 1]))

    def _operator(self, okl, oku):
        key = (okl, oku)
        K = self.K.get(key)
        if K is None:
            p = self.p
            wv = np.concatenate([self.w_pts, np.full(2, p["w_x"] / 32 if okl else 0.0),
                                 np.full(2, p["w_x"] / 32 if oku else 0.0)])
            A = self.A_all
            H = A.T @ (wv[:, None] * A) + self.lam_eye
            K = np.linalg.solve(H, A.T * wv) * self.S[:, None]        # 10 x 36
            self.K[key] = K
        return K

    def act(self, obs):
        p = self.p
        obs = np.asarray(obs, float)
        t = obs[N_RT: N_RT + 71]
        if self.tkey is None or not np.array_equal(t, self.tkey):
            self._targets(obs, t)
            self.tkey = t.copy()
        xs = (obs[self.keep_idx] - self.mu_x) * self.isx
        y = self.mub + xs @ self.Wb
        e = np.clip(self.r_t - y[:32], -p["e_clip"], p["e_clip"])
        b = np.empty(36)
        b[:32] = e
        okl = oku = False
        if self.nx >= 1:
            ex = self.xt[:2] - y[32:34]
            okl = abs(ex[0]) < p["x_gate"] and abs(ex[1]) < p["x_gate"]
            b[32:34] = ex if okl else 0.0
        else:
            b[32:34] = 0.0
        if self.nx == 2:
            ex = self.xt[2:] - y[34:36]
            oku = abs(ex[0]) < p["x_gate"] and abs(ex[1]) < p["x_gate"]
            b[34:36] = ex if oku else 0.0
        else:
            b[34:36] = 0.0
        x = self._operator(okl, oku) @ b
        z = float(y[36])
        I = obs[N_PROBE:N_PROBE + N_CTRL]
        ip = float(obs[N_PROBE + N_COIL])
        if self.I_ref is None:
            self.I_ref = I.copy()
        ramp = min(1.0, (self.k + 1) / max(1, p["ramp"]))
        g = p["k_shape"] * ramp
        dIdt = np.empty(N_CTRL)
        dIdt[1:] = g * x[:N_CTRL - 1]
        if self.ncol == N_CTRL:   # outer loop: the P6 part of the correction moves the vertical set-point
            self.z_t += p.get("k_zt", 0.0) * ramp * self.dz_dI6 * x[N_CTRL - 1] * DT
        ip_err = float(t[70]) - ip
        self.ip_int += ip_err * DT
        dIdt[0] = p["k_ip"] * ip_err + p["ki_ip"] * self.ip_int
        I_new = np.clip(self.I_ref + dIdt * DT, self.lo, self.hi)
        I_new = np.clip(I_new, I - p["dmax"], I + p["dmax"])
        dIdt_eff = (I_new - self.I_ref) / DT
        self.I_ref = I_new
        out = self.out
        out[:N_CTRL] = self.Rc * I + self.Lc @ (dIdt_eff + (I_new - I) / p["tau_c"])
        zipv = 0.0 if self.z_prev is None else (z - self.z_prev) / DT * ip
        self.z_prev = z
        zip_err = (self.z_t - z) * ip
        self.z_int += zip_err * DT
        out[N_CTRL] = p["kz_p"] * zip_err + p["kz_d"] * zipv + p["kz_i"] * self.z_int
        self.k += 1
        res = np.where(np.isfinite(out), out, 0.0)
        return np.clip(res, self.low, self.high)


def main():
    ctl = Controller()
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("protocol") != PROTOCOL:
            raise SystemExit("unsupported protocol")
        reply = {"request_id": req["request_id"]}
        if req["op"] == "reset":
            ctl.reset(req["action_low"], req["action_high"])
            reply["ready"] = True
        elif req["op"] == "act":
            reply["action"] = ctl.act(req["observation"]).tolist()
        elif req["op"] == "close":
            reply["closed"] = True
            print(json.dumps(reply), flush=True)
            return
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
