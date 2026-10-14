"""Delay-robust vertical (P6) gains from FreeGSNKE's linearised circuit model.

1. extract: reset the plant for many seeds and save the linear model (M x' + x = B V) together with
   the current-centroid height map dZcur/dx built from dIy/dI:
       python vertical_design.py extract --case mastu_diverted_hold --seeds 5001-5040 --rz dev_randomization --out LINDIR
2. design: discretise (ZOH, 0.5 ms), close the loop with a PID on Zcur*Ip into P6, a supply delay of
   0-2 steps and current-holding loops on the other coils, and print the worst spectral radius:
       python vertical_design.py design --lin LINDIR
The baseline uses kz_p=0.003, kz_d=5e-6 (on dZ/dt, sign as in controller.py), kz_i=0.05: stable for
every extracted plant (growth times 2-150 ms) and delays 0, 1 and 2. FreeGSNKE's PCS default
(0.025, 5e-7) is unstable for most plants once the supply delays by one or two steps.
"""
import argparse
import glob
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DT = 5e-4


def _extract(a):
    import common as lab
    case, seed, rz, out = a
    env = lab.TokamakMagneticEnv(case, randomization=lab.load_rz(rz), solver="nonlinear")
    obs, info = env.reset(seed=seed)
    st = env.stepping; ls = st.linearised_sol
    B = np.zeros((ls.Mmatrix.shape[0], 12)); B[:-1] = ls.Pm1Rm1[:, :12]
    lh = st.limiter_handler; eq = st.eq1
    one = lh.Iy_from_jtor(np.ones_like(eq.Z)); zy = lh.Iy_from_jtor(eq.Z) / one; ry = lh.Iy_from_jtor(eq.R) / one
    Iy = st.Iy; ipy = Iy.sum(); z0 = (Iy * zy).sum() / ipy; r0 = (Iy * ry).sum() / ipy
    Cz = ((zy - z0) @ st.dIydI) / ipy; Cr = ((ry - r0) @ st.dIydI) / ipy
    np.savez(Path(out) / f"lin_{case}_{seed}.npz", M=ls.Mmatrix, B=B, dRZdI=np.vstack([Cr, Cz]),
             tau=np.real(ls.instability_timescale), delay=info["supply_delay_steps"], ip=env._plasma_current(),
             probe_scale=env.probe_scale, L=np.asarray(env.tokamak.coil_self_ind[:12, :12]))
    return seed, np.round(np.real(ls.instability_timescale), 4).tolist()


def plant(f):
    from scipy.linalg import expm
    d = np.load(f)
    M, B, C = d["M"], d["B"], d["dRZdI"]
    n = M.shape[0]
    Mi = np.linalg.inv(M)
    big = np.zeros((n + 12, n + 12)); big[:n, :n] = -Mi * DT; big[:n, n:] = Mi @ B * DT
    E = expm(big)
    return {"Ad": E[:n, :n], "Bd": E[:n, n:], "Cz": C[1], "n": n, "ip": float(d["ip"]), "L": np.diag(d["L"])}


def radius(P, kp, kd, ki, delay, tau_c=0.015):
    """Spectral radius of the closed loop: V6 = -Ip (kp z + kd dz/dt + ki int z), others hold current."""
    n = P["n"]; Ad, Bd, Cz, ip = P["Ad"], P["Bd"], P["Cz"], P["ip"]
    Kx = np.zeros((12, n)); Kz = np.zeros(12); Ki = np.zeros(12)
    for j in range(11):
        Kx[j, j] = -P["L"][j] / tau_c
    Kx[11] = -ip * (kp + kd / DT) * Cz; Kz[11] = ip * kd / DT; Ki[11] = -ip * ki
    m = n + 2 + 12 * delay
    T = np.zeros((m, m)); zi, ii = n, n + 1
    if delay == 0:
        T[:n, :n] = Ad + Bd @ Kx; T[:n, zi] = Bd @ Kz; T[:n, ii] = Bd @ Ki
    else:
        T[:n, :n] = Ad; T[:n, n + 2:n + 14] = Bd
        for s in range(delay - 1):
            a = n + 2 + 12 * s; T[a:a + 12, a + 12:a + 24] = np.eye(12)
        last = n + 2 + 12 * (delay - 1)
        T[last:last + 12, :n] = Kx; T[last:last + 12, zi] = Kz; T[last:last + 12, ii] = Ki
    T[zi, :n] = Cz
    T[ii, :n] = DT * Cz; T[ii, ii] = 1.0
    return float(np.max(np.abs(np.linalg.eigvals(T))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "design"])
    ap.add_argument("--case", default="mastu_diverted_hold")
    ap.add_argument("--seeds", default="5001-5040")
    ap.add_argument("--rz", default="dev_randomization")
    ap.add_argument("--out", default="lin")
    ap.add_argument("--lin", default="lin")
    a = ap.parse_args()
    if a.cmd == "extract":
        lo, hi = map(int, a.seeds.split("-"))
        Path(a.out).mkdir(parents=True, exist_ok=True)
        with mp.get_context("fork").Pool(min(64, hi - lo + 1)) as pool:
            for r in pool.imap(_extract, [(a.case, s, a.rz, a.out) for s in range(lo, hi + 1)]):
                print(*r, flush=True)
        return
    plants = [plant(f) for f in sorted(glob.glob(str(Path(a.lin) / "lin_*.npz")))]
    print(len(plants), "plants")
    for kp, kd, ki in [(0.025, -5e-7, 0.0), (0.002, 5e-6, 0.0), (0.003, 5e-6, 0.05), (0.003, 5e-6, 0.1)]:
        r = np.array([[radius(P, kp, kd, ki, d) for d in (0, 1, 2)] for P in plants])
        print(f"kp={kp} kd={kd} ki={ki}: worst radius by delay {np.round(r.max(0), 5).tolist()}, unstable {(r > 1).sum(0).tolist()}")


if __name__ == "__main__":
    main()
