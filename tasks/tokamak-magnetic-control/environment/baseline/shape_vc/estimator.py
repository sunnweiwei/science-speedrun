"""Linear real-time reconstruction: magnetics + coil currents + Ip -> boundary rays, X-points, Zcur.

Fit on noise-free probe signals with the sensor-error covariance added to the Gram matrix
(equivalent to training on infinitely many noisy copies). Channels that read exactly 0 at the
first step of an episode (dead) are dropped by re-solving the normal equations on the rest.

python estimator.py fit --recs 'DATA/rec_*.npz' --out estimator.npz --model model.pkl --lin LIN.npz
"""
import argparse
import glob
import json

import numpy as np

N_PROBE, N_COIL = 500, 12
N_X = N_PROBE + N_COIL + 1
NA = 64
N_Y = NA + 7          # rho(64), xlow(2), xup(2), gaps(2), zcur


WITH_GAPS = False


def label_matrix(r, xref):
    if WITH_GAPS:
        gaps = r["gaps"] if "gaps" in r else np.full((len(r["rho"]), 2), np.nan)
        y = np.hstack([r["rho"], r["xlow"], r["xup"], gaps, r["zcur"][:, None]])
    else:
        y = np.hstack([r["rho"], r["xlow"], r["xup"], r["zcur"][:, None]])
    ok = np.isfinite(y).all(1)
    ok &= np.all(np.abs(r["xlow"] - xref[0]) < 0.25, axis=1) & np.all(np.abs(r["xup"] - xref[1]) < 0.25, axis=1)
    return y, ok


def ref_index(obs_rows, refs):
    """Index of the stored reference nearest to each frame's targets (as ShapeController.select_reference)."""
    pts = obs_rows[:, N_X: N_X + 64].reshape(-1, 32, 2)
    ipt = obs_rows[:, N_X + 70]
    d = np.stack([np.sqrt(((pts - r["pts"]) ** 2).sum(2).mean(1)) + 0.5 * np.abs(ipt - r["ip"]) / r["ip"] for r in refs], 1)
    return d.argmin(1)


def load(files, xref, clean=True, refs=None):
    X, Y, E = [], [], []
    for i, f in enumerate(files):
        r = dict(np.load(f))
        if len(r["obs"]) < 5:
            continue
        x = r["obs"][:, :N_X].copy()
        if clean:
            x[:, :N_PROBE] = r["clean"]
        y, ok = label_matrix(r, xref)
        g = ref_index(r["obs"], refs) if refs is not None else np.zeros(len(x), int)
        X.append(x[ok]); Y.append(y[ok]); E.append(g[ok])
    return np.vstack(X), np.vstack(Y), np.concatenate(E)


def fit(args):
    files = sorted(glob.glob(args.recs))
    rng = np.random.default_rng(0)
    test = set(rng.choice(len(files), max(1, len(files) // 6), replace=False).tolist())
    tr_files = [f for i, f in enumerate(files) if i not in test]
    te_files = [f for i, f in enumerate(files) if i in test]
    lin = np.load(args.lin)
    pscale = lin["probe_scale"]
    lim = np.array([55000, 7000, 9000, 9000, 9000, 9000, 9000, 9000, 9000, 10000, 10000, 5000.0])
    sx = np.concatenate([pscale, lim, [6.2e5]])
    xref = np.array([[0.61, -1.12], [0.61, 1.12]])
    import pickle
    refs = pickle.load(open(args.model, "rb"))["refs"] if args.grouped else None
    X, Y, grp = load(tr_files, xref, refs=refs)
    var = np.concatenate([np.full(N_PROBE, args.probe_var), np.full(N_COIL, args.current_var), [args.ip_var]])
    out = dict(sx=sx, var=var, lam=np.array(args.lam), n_groups=np.array(1 if refs is None else len(refs)))
    if refs is not None:
        out["ref_pts"] = np.stack([r["pts"] for r in refs]); out["ref_ip"] = np.array([r["ip"] for r in refs])
    for gi in range(int(out["n_groups"])):
        m = grp == gi
        Xg, Yg = X[m], Y[m]
        mu_x, mu_y = Xg.mean(0), Yg.mean(0)
        Xs = (Xg - mu_x) / sx
        out[f"mu_x{gi}"], out[f"mu_y{gi}"] = mu_x, mu_y
        out[f"G{gi}"] = Xs.T @ Xs / len(Xg)
        out[f"C{gi}"] = Xs.T @ (Yg - mu_y) / len(Xg)
        print(f"group {gi}: {m.sum()} frames", flush=True)
    np.savez(args.out, **out)
    print(f"fit on {len(tr_files)} episodes / {len(X)} frames; test {len(te_files)} episodes", flush=True)
    # evaluation on the recorded (really noisy, offset, dead-channel) observations of held-out episodes
    est = Estimator(args.out)
    errs = []
    for f in te_files:
        r = dict(np.load(f))
        if len(r["obs"]) < 5:
            continue
        y, ok = label_matrix(r, xref)
        est.reset()
        pred = np.array([est.predict(o) for o in r["obs"]])
        errs.append((pred - y)[ok])
    err = np.vstack(errs)
    rms = np.sqrt(np.nanmean(err ** 2, 0))
    rep = {"rho_rms_cm_mean": float(100 * rms[:NA].mean()), "rho_rms_cm_max": float(100 * rms[:NA].max()),
           "xlow_cm": (100 * rms[NA:NA + 2]).round(2).tolist(), "xup_cm": (100 * rms[NA + 2:NA + 4]).round(2).tolist(),
           "zcur_cm": float(100 * rms[-1]), "gaps_cm": (100 * rms[NA + 4:NA + 6]).round(2).tolist() if len(rms) > NA + 5 else None, "rho_rms_cm_by_angle": (100 * rms[:NA]).round(2).tolist()}
    print(json.dumps(rep), flush=True)


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
            g["W_all"] = self._solve(g, np.ones(N_X, bool))
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
        if self.ref_pts is None or len(obs) < N_X + 71:
            return 0
        t = obs[N_X: N_X + 71]
        if self.last_t is None or not np.array_equal(t, self.last_t):
            pts = t[:64].reshape(32, 2); ipt = t[70]
            d = np.sqrt(((pts[None] - self.ref_pts) ** 2).sum(2).mean(1)) + 0.5 * np.abs(ipt - self.ref_ip) / self.ref_ip
            self.gi = int(np.argmin(d)); self.last_t = t.copy()
        return self.gi

    def predict(self, obs):
        obs = np.asarray(obs, float)
        x = obs[:N_X]
        if self.keep is None:
            keep = np.ones(N_X, bool)
            keep[:N_PROBE] = x[:N_PROBE] != 0.0
            self.keep = keep
        gi = self.group(obs)
        g = self.groups[gi]
        if self.W[gi] is None:
            self.W[gi] = g["W_all"] if self.keep.all() else self._solve(g, self.keep)
        xs = ((x - g["mu_x"]) / self.sx)[self.keep]
        return g["mu_y"] + xs @ self.W[gi]


class EstMeas:
    """Measurement provider for controller.ShapeController from the observation only."""
    def __init__(self, path):
        self.est = Estimator(path)

    def reset(self, env=None, info=None):
        self.est.reset()

    def __call__(self, obs, env=None):
        y = self.est.predict(obs)
        out = {"rho": y[:NA], "xlow": y[NA:NA + 2], "xup": y[NA + 2:NA + 4], "zcur": float(y[-1]),
                "ip": float(obs[N_PROBE + N_COIL]), "I": np.asarray(obs[N_PROBE:N_PROBE + N_COIL], float)}
        if len(y) == NA + 7:
            out["gaps"] = y[NA + 4:NA + 6]
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--recs", required=True)
    ap.add_argument("--lin", required=True, help="any file with the probe_scale array (vertical_design.py extract output)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe-var", type=float, default=0.002)
    ap.add_argument("--current-var", type=float, default=1e-4)
    ap.add_argument("--ip-var", type=float, default=1e-6)
    ap.add_argument("--lam", type=float, default=1e-6)
    ap.add_argument("--model", required=True)
    ap.add_argument("--grouped", type=int, default=1)
    args = ap.parse_args()
    if args.cmd == "fit":
        fit(args)
