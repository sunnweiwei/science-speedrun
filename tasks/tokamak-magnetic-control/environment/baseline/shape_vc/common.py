"""Shared geometry, privileged measurements and the episode loop of the shape/virtual-circuit baseline.

Geometry: the controlled boundary is the psi_N=0.995 contour described by its radius rho(theta)
along NA fixed rays from CENTER; targets are converted to the same representation.
"""
import copy
import io
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # environment/ (or /workspace)
from tokagym import TokamakMagneticEnv  # noqa: E402
from tokagym import cases as C  # noqa: E402

HERE = Path(__file__).resolve().parent
CENTER = np.array([0.95, 0.0])
NA = 64
ANG = np.linspace(-np.pi, np.pi, NA, endpoint=False)
CTRL = ["Solenoid", "PX", "D1", "D2", "D3", "Dp", "D5", "D6", "D7", "P4", "P5"]
ACTIVE = CTRL + ["P6"]
N_PROBE, N_COIL = 500, 12
N_RT = N_PROBE + N_COIL + 1


def load_rz(name):
    if name in ("visible", "none"):
        return name
    p = Path(name)
    if not p.exists():
        p = HERE / f"{name}.json"
    if not p.exists():
        p = HERE / "dev_randomization.json"
    return json.loads(p.read_text())["randomization"]


def ray_radii(poly, center=CENTER, ang=ANG):
    """Largest intersection distance of each ray from ``center`` with the closed polyline."""
    a = np.asarray(poly, float) - center
    b = np.roll(a, -1, 0)
    e = b - a
    dx, dy = np.cos(ang)[:, None], np.sin(ang)[:, None]
    den = dx * e[None, :, 1] - dy * e[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (a[None, :, 0] * e[None, :, 1] - a[None, :, 1] * e[None, :, 0]) / den
        u = (a[None, :, 0] * dy - a[None, :, 1] * dx) / den
    ok = (u >= 0) & (u <= 1) & (s > 0) & np.isfinite(s)
    s = np.where(ok, s, -np.inf).max(1)
    return np.where(np.isfinite(s), s, np.nan)


def polar(points, center=CENTER):
    d = np.asarray(points, float) - center
    return np.arctan2(d[:, 1], d[:, 0]), np.hypot(d[:, 0], d[:, 1])


def interp_periodic(theta, grid_vals):
    """Linear interpolation of values on ANG (axis 0) at angles theta."""
    x = (np.asarray(theta) + np.pi) / (2 * np.pi) * NA
    i0 = np.floor(x).astype(int) % NA
    w = (x - np.floor(x))
    i1 = (i0 + 1) % NA
    g = np.asarray(grid_vals)
    if g.ndim == 1:
        return (1 - w) * g[i0] + w * g[i1]
    return (1 - w)[:, None] * g[i0] + w[:, None] * g[i1]


def xpoints_near(eq, refs):
    """For each reference X-point, the nearest candidate X-point of eq (NaN if none)."""
    cand = C.candidate_xpoints(eq)
    out = np.full((len(refs), 2), np.nan)
    if len(cand):
        for i, r in enumerate(refs):
            out[i] = cand[np.argmin(np.linalg.norm(cand - r, axis=1))]
    return out


def truth(env, eq=None):
    """Privileged plasma quantities: rho grid, lower/upper X-points, Zcur, Ip, Rin/Rout."""
    eq = env.stepping.eq1 if eq is None else eq
    sep = C.boundary_contour(eq)
    rho = ray_radii(sep) if len(sep) > 2 else np.full(NA, np.nan)
    xr = env.target_xpoints if len(env.target_xpoints) else np.zeros((0, 2))
    xp = xpoints_near(eq, xr) if len(xr) else np.full((2, 2), np.nan)
    if len(xp) < 2:
        xp = np.vstack([xp, np.full((2 - len(xp), 2), np.nan)])
    try:
        zc = float(eq.Zcurrent()); rc = float(eq.Rcurrent())
    except Exception:  # noqa: BLE001
        zc = rc = math.nan
    return {"rho": rho, "xlow": xp[0], "xup": xp[1], "zcur": zc, "rcur": rc}


def static_jacobian(env, cache=None, rel=0.01):
    """d(rho, lower/upper X-point, gaps, Zcur)/dI for the 11 controlled coils and P6 at the case reference (central FD)."""
    if cache and Path(cache).exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    lims = env.spec.current_limits
    base = dict(env.reference_currents)
    rows = []
    t0 = truth(env, env.reference_eq)

    def vec(t, eq):
        return np.concatenate([t["rho"], t["xlow"], t["xup"], np.asarray(env._gaps(eq, float(np.asarray(eq.opt)[0, 1]))),
                               [float(eq.Zcurrent())]])
    v0 = vec(t0, env.reference_eq)
    J = np.zeros((len(v0), len(CTRL) + 1))
    for j, coil in enumerate(CTRL + ["P6"]):
        k = list(env.spec.active_coils).index(coil)
        h = rel * float(np.max(np.abs(lims[k])))
        vs = []
        for sgn in (+1, -1):
            cur = dict(base); cur[coil] = base[coil] + sgn * h
            eq = env.reference_eq.create_auxiliary_equilibrium()
            prof = copy.deepcopy(env.reference_profiles)
            env._solve_static(eq, prof, cur)
            vs.append(vec(truth(env, eq), eq))
        J[:, j] = (vs[0] - vs[1]) / (2 * h)
        rows.append(coil)
    for name, value in base.items():   # restore the shared machine's currents
        env.tokamak.set_coil_current(coil_label=name, current_value=value)
    out = {"J": J, "v0": v0}
    if cache:
        np.savez(cache, **out)
    return out


def clean_probes(env):
    eq = env.stepping.eq1
    fl = np.asarray(env.tokamak.probes.calculate_fluxloop_value(eq), dtype=float)
    pu = np.asarray(env.tokamak.probes.calculate_pickup_value(eq), dtype=float)
    return np.concatenate([fl, pu])


def run_episode(env, ctl, seed, record=False, horizon=None, verbose=False):
    """Roll one episode; ``ctl.act(obs, env)`` returns 12 voltages (env given for privileged mode)."""
    obs, info = env.reset(seed=seed)
    ctl.reset(env, info)
    term = trunc = False
    rec = []
    t0 = time.monotonic()
    k = 0
    lat = []
    while not (term or trunc):
        ta = time.perf_counter()
        v = ctl.act(obs, env)
        lat.append(time.perf_counter() - ta)
        v = np.clip(np.asarray(v, float), -env.vmax, env.vmax)
        if record:
            tr = truth(env)
            gaps = env._gaps(env.stepping.eq1, env.gap_height_at(env.t))
            rec.append({"obs": obs.copy(), "clean": clean_probes(env), "rho": tr["rho"], "xlow": tr["xlow"], "gaps": np.asarray(gaps),
                        "xup": tr["xup"], "zcur": tr["zcur"], "rcur": tr["rcur"], "v": v})
        obs, r, term, trunc, inf = env.step(v)
        k += 1
        if verbose and k % 50 == 0:
            print(f"  k={k} shape={inf.get('shape_rms', float('nan')):.4f} rin={inf.get('rin_error', 0):.4f} "
                  f"rout={inf.get('rout_error', 0):.4f} zx={inf.get('zx_error', 0):.4f} ip={inf.get('ip_error_rel', 0):.4f}",
                  flush=True)
        if horizon and k >= horizon:
            break
    s = env.episode_summary()
    s["seed"] = seed
    s["wall"] = time.monotonic() - t0
    s["lat_ms_median"] = 1e3 * float(np.median(lat)) if lat else None
    s["info"] = {k_: info[k_] for k_ in ("profile_constraint", "handover_steps", "supply_delay_steps",
                                          "growth_rate", "dead_channels", "controllability_redraws")}
    s["trace"] = [{kk: (float(r_[kk]) if np.isscalar(r_[kk]) else None) for kk in
                   ("shape_rms", "rin_error", "rout_error", "zx_error", "ip_error_rel")} for r_ in env.records]
    if record:
        s["_record"] = rec
    return s


def short(s):
    def cm(x):
        return None if x is None else round(100 * x, 2)
    return (f"seed={s['seed']} out={s.get('outcome')} steps={s['steps']} shape={cm(s.get('shape_rms_steady'))} "
            f"rin={cm(s.get('rin_error_steady'))} rout={cm(s.get('rout_error_steady'))} zx={cm(s.get('zx_error_steady'))} "
            f"ip%={None if s.get('ip_error_rel_steady') is None else round(100 * s['ip_error_rel_steady'], 2)} "
            f"term={s.get('termination')} wall={s['wall']:.0f}s lat={s.get('lat_ms_median') and round(s['lat_ms_median'], 3)}ms")


class PkgAdapter:
    """Runs a built package's Controller in-process (same code path as the protocol server)."""
    def __init__(self, pkg):
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"pkgpolicy_{abs(hash(str(pkg)))}", str(Path(pkg) / "policy.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        self.c = mod.Controller()

    def reset(self, env, info):
        self.c.reset(-env.vmax, env.vmax)

    def act(self, obs, env=None):
        return self.c.act(obs)
