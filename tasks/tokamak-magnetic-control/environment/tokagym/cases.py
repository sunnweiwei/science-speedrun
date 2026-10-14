"""Frozen control cases: reference equilibrium, targets and reference waveforms.

Every case starts from a converged MAST-U-like equilibrium and asks the controller to
hold (or move) the plasma boundary, X-point and plasma current for a fixed horizon.
Targets are geometric: control points on the reference last-closed flux surface,
the reference X-point(s) and a plasma-current waveform.
"""

from dataclasses import dataclass, field
import pickle
from pathlib import Path

import numpy as np

DT = 5e-4                # simulator/control step [s]
DEFAULT_HORIZON = 700    # 0.35 s: at least 100 ms of steady state after the 100 ms settling time
N_CONTROL_POINTS = 32


@dataclass
class Waveforms:
    """Reference changes relative to the reference equilibrium (piecewise linear in time)."""
    ip_times: tuple = (0.0,)
    ip_scale: tuple = (1.0,)          # multiplicative on the reference Ip
    z_times: tuple = (0.0,)
    z_shift: tuple = (0.0,)           # current-centroid shift [m]; targets blend toward the same plasma held
                                      # that much higher by the vertical coil (env._shifted), not a rigid shift
    shape_times: tuple = (0.0,)
    shape_mix: tuple = (0.0,)         # 0: reference shape, 1: the case's second shape (linear blend)

    def ip_factor(self, t):
        return float(np.interp(t, self.ip_times, self.ip_scale))

    def z_offset(self, t):
        return float(np.interp(t, self.z_times, self.z_shift))

    def shape_weight(self, t):
        return float(np.interp(t, self.shape_times, self.shape_mix))

    def last_change_time(self):
        """End of the last segment in which a reference value changes (0 if none does)."""
        end = 0.0
        for times, vals in ((self.ip_times, self.ip_scale), (self.z_times, self.z_shift),
                            (self.shape_times, self.shape_mix)):
            for k in range(1, len(times)):
                if vals[k] != vals[k - 1]:
                    end = max(end, float(times[k]))
        return end


@dataclass
class Disturbance:
    """Unannounced plasma-profile change during the episode (not part of the targets)."""
    time: float = None
    paxis_scale: float = 1.0
    alpha_scale: float = 1.0


@dataclass
class Case:
    """A case is built by continuation from the base diverted equilibrium.

    ``construction`` is a tuple of steps applied in order, each re-solved from the
    previous converged equilibrium: ("currents_scale", {coil: factor}),
    ("currents_file", pickle_name, n_substeps) or ("profile", {param: value}).
    ``second_shape`` (optional) holds further steps from the reference equilibrium whose
    boundary, X-points and gaps the targets blend into following ``waveforms.shape_mix``.
    """
    name: str
    family: str
    description: str
    profile: str = "paxis"             # "paxis" (ConstrainPaxisIp) or "lao85"
    profile_params: dict = None
    construction: tuple = ()
    second_shape: tuple = ()
    waveforms: Waveforms = field(default_factory=Waveforms)
    disturbance: Disturbance = field(default_factory=Disturbance)
    horizon: int = DEFAULT_HORIZON
    diverted: bool = True

    def __post_init__(self):
        if self.profile_params is None:
            self.profile_params = dict(BASE_PROFILE)

    def last_change_time(self):
        """Time of the last announced or unannounced change: waveform end or disturbance."""
        return max(self.waveforms.last_change_time(), self.disturbance.time or 0.0)


# FPDT Lao85 double-null (759 kA): kept for reference; excluded from the suite because the
# nonlinear evolutive step did not converge in reasonable time with this profile class.
BASE_CURRENTS_FILE = "simple_diverted_currents_PaxisIp.pk"   # FreeGSNKE example01a inverse solve
BASE_PROFILE = dict(paxis=8.1e3, Ip=6.2e5, fvac=0.5, alpha_m=1.8, alpha_n=1.2)

CASES = {
    "mastu_diverted_hold": Case(
        name="mastu_diverted_hold", family="hold",
        description="Hold the base double-null diverted 620 kA plasma at its reference shape."),
    "mastu_high_ip_hold": Case(
        name="mastu_high_ip_hold", family="hold",
        description="Hold a 750 kA plasma obtained by similarity scaling (coil currents and Ip x1.21, pressure x1.46).",
        profile_params=dict(paxis=8.1e3 * 1.21 ** 2, Ip=6.2e5 * 1.21, fvac=0.5, alpha_m=1.8, alpha_n=1.2),
        construction=(("scale_all", 1.1), ("scale_all", 1.1))),
    "mastu_p4_shape_hold": Case(
        name="mastu_p4_shape_hold", family="hold",
        description="Hold the shape obtained with the P4 coil current scaled by 1.2 (cf. FreeGSNKE example02).",
        construction=(("currents_scale", {"P4": 1.2}),)),
    "mastu_ip_ramp": Case(
        name="mastu_ip_ramp", family="track",
        description="Ramp plasma current +15% between 50 and 150 ms while holding the diverted shape.",
        waveforms=Waveforms(ip_times=(0.0, 0.05, 0.15, 0.25), ip_scale=(1.0, 1.0, 1.15, 1.15))),
    "mastu_vertical_shift": Case(
        name="mastu_vertical_shift", family="track",
        description="Move the diverted plasma up by 2 cm with the vertical coil (ramp 50-100 ms) and hold "
                    "the shape the machine reaches there.",
        waveforms=Waveforms(z_times=(0.0, 0.05, 0.10, 0.25), z_shift=(0.0, 0.0, 0.02, 0.02))),
    "mastu_pressure_drop": Case(
        name="mastu_pressure_drop", family="disturbance",
        description="Reject an unannounced 30% axis-pressure drop at 100 ms on the diverted plasma.",
        disturbance=Disturbance(time=0.10, paxis_scale=0.7)),
    # Modelled on the fundamental-capability discharge of Degrave et al. (Nature 2022): current
    # ramp, vertical shift, shape change with X-point motion, ramp-down, all in one episode.
    "mastu_sequence": Case(
        name="mastu_sequence", family="sequence",
        description="Ramp Ip +15% (100-150 ms), move the plasma up 10 cm (170-220 ms), change to the "
                    "P4-scaled shape (250-330 ms), ramp Ip back down (350-400 ms) and hold.",
        second_shape=(("currents_scale", {"P4": 1.2}),),
        waveforms=Waveforms(ip_times=(0.0, 0.10, 0.15, 0.35, 0.40), ip_scale=(1.0, 1.0, 1.15, 1.15, 1.0),
                            z_times=(0.0, 0.17, 0.22), z_shift=(0.0, 0.0, 0.10),
                            shape_times=(0.0, 0.25, 0.33), shape_mix=(0.0, 0.0, 1.0)),
        horizon=1200),
}


def get_case(name):
    if name not in CASES:
        raise KeyError(f"Unknown case {name!r}; known: {sorted(CASES)}")
    return CASES[name]


def load_currents_file(name, data_dir):
    with open(Path(data_dir) / "MAST-U" / name, "rb") as f:
        return {k: float(v) for k, v in pickle.load(f).items()}


def make_profiles(case, eq, params=None):
    """Instantiate the FreeGSNKE profile object for a case (optionally perturbed params)."""
    from freegsnke.jtor_update import ConstrainPaxisIp, Lao85

    p = dict(case.profile_params if params is None else params)
    if case.profile == "paxis":
        return ConstrainPaxisIp(eq=eq, **p)
    if case.profile == "lao85":
        return Lao85(eq=eq, **p)
    raise ValueError(case.profile)


def perturb_profile_params(case, rng, scales):
    """Return perturbed profile parameters; ``scales`` gives half-widths of log-uniform factors."""
    p = dict(case.profile_params)
    if case.profile == "paxis":
        p["paxis"] = p["paxis"] * float(np.exp(rng.uniform(-scales["pressure"], scales["pressure"])))
        p["alpha_m"] = p["alpha_m"] + float(rng.uniform(-scales["alpha"], scales["alpha"]))
        p["alpha_n"] = p["alpha_n"] + float(rng.uniform(-scales["alpha"], scales["alpha"]))
        p["Ip"] = p["Ip"] * float(np.exp(rng.uniform(-scales["ip"], scales["ip"])))
    else:
        f = float(np.exp(rng.uniform(-scales["pressure"], scales["pressure"])))
        p["alpha"] = [a * f for a in p["alpha"]]
        p["Ip"] = p["Ip"] * float(np.exp(rng.uniform(-scales["ip"], scales["ip"])))
    return p


BOUNDARY_PSI_NORM = 0.995   # closed flux surface just inside the separatrix (avoids the X-point)


def boundary_contour(eq, psi_norm=BOUNDARY_PSI_NORM):
    """Closed (R, Z) polyline of the normalized-flux surface ``psi_norm`` enclosing the axis.

    Marching squares on the equilibrium grid: fast and robust near X-points, unlike
    ray-based separatrix tracing. Returns an empty array if no closed contour encloses
    the magnetic axis.
    """
    from matplotlib.path import Path as MplPath
    from skimage import measure

    opt = np.asarray(eq.opt)
    if opt.size == 0:
        return np.zeros((0, 2))
    r_axis, z_axis, psi_axis = opt[0, :3]
    psi_b = float(eq._profiles.psi_bndry)
    if not np.isfinite(psi_b) or abs(psi_b - psi_axis) < 1e-12:
        return np.zeros((0, 2))
    psi_n = (np.asarray(eq.psi()) - psi_axis) / (psi_b - psi_axis)
    R, Z = np.asarray(eq.R), np.asarray(eq.Z)
    r0, dr = R[0, 0], R[1, 0] - R[0, 0]
    z0, dz = Z[0, 0], Z[0, 1] - Z[0, 0]
    best = None
    for c in measure.find_contours(psi_n, psi_norm):
        if len(c) < 8 or np.linalg.norm(c[0] - c[-1]) > 1e-9:
            continue
        pts = np.stack([r0 + c[:, 0] * dr, z0 + c[:, 1] * dz], axis=1)
        if MplPath(pts).contains_point((r_axis, z_axis)):
            if best is None or len(pts) > len(best):
                best = pts
    return best if best is not None else np.zeros((0, 2))


def control_points(eq, n=N_CONTROL_POINTS):
    """``n`` control points equally spaced in arc length along the reference boundary.

    The first point is the outboard midplane (largest R near the axis height) and the
    points run counter-clockwise, so two shapes' points correspond index by index.
    """
    pts = boundary_contour(eq)
    if len(pts) < n:
        raise RuntimeError("Reference boundary contour too short")
    if np.linalg.norm(pts[0] - pts[-1]) < 1e-9:
        pts = pts[:-1]
    area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    if area < 0:   # clockwise in (R, Z): reverse
        pts = pts[::-1]
    z_axis = float(np.asarray(eq.opt)[0, 1])
    near = np.abs(pts[:, 1] - z_axis) < 0.05 * np.ptp(pts[:, 1])
    start = int(np.argmax(np.where(near, pts[:, 0], -np.inf)))
    pts = np.concatenate([np.roll(pts, -start, axis=0), np.roll(pts, -start, axis=0)[:1]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0, cum[-1], n, endpoint=False)
    out = np.empty((n, 2))
    for k, t in enumerate(targets):
        j = int(np.searchsorted(cum, t, side="right") - 1)
        j = min(max(j, 0), len(pts) - 2)
        w = (t - cum[j]) / max(cum[j + 1] - cum[j], 1e-12)
        out[k] = (1 - w) * pts[j] + w * pts[j + 1]
    return out


XPOINT_PSIN_TOL = 0.1   # X-points with |psi_N - 1| <= this count as on (or next to) the boundary


def _psi_n(eq, psi):
    """Normalized flux (0 on the axis, 1 on the boundary) of flux values ``psi``."""
    prof = getattr(eq, "_profiles", None)
    pa = getattr(prof, "psi_axis", None) if prof is not None else None
    pb = getattr(prof, "psi_bndry", None) if prof is not None else None
    if pa is None or pb is None:
        pa, pb = eq.psi_axis, eq.psi_bndry
    return (np.asarray(psi, dtype=float) - float(pa)) / (float(pb) - float(pa))


def boundary_xpoints(eq, tol=XPOINT_PSIN_TOL):
    """X-points with normalized flux within ``tol`` of the boundary (|psi_N - 1| <= tol), closest first.

    The tolerance is on psi_N, not relative to psi_bndry: the boundary flux drifts with resistive flux
    consumption during an episode, which made a fixed relative tolerance drop a nearby X-point.
    """
    xpt = np.asarray(eq.xpt)
    if xpt.size == 0:
        return np.zeros((0, 2))
    gap = np.abs(_psi_n(eq, xpt[:, 2]) - 1.0)
    order = np.argsort(gap)
    keep = gap[order] <= tol
    return xpt[order][keep, :2]


def candidate_xpoints(eq, tol=XPOINT_PSIN_TOL):
    """Boundary X-points if any survive the tolerance, else every X-point in the domain."""
    near = boundary_xpoints(eq, tol)
    if len(near):
        return near
    xpt = np.asarray(eq.xpt)
    return xpt[:, :2] if xpt.size else np.zeros((0, 2))
