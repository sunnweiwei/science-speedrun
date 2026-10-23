"""Physical control metrics. Units: metres, amperes, seconds; directions documented in METRICS.md."""

import math

import numpy as np


def point_polyline_distance(points, polyline):
    """Minimum Euclidean distance from each point to a closed polyline (N,2)."""
    a = np.asarray(polyline, dtype=float)
    if len(a) < 2:
        return np.full(len(points), np.inf)
    b = np.roll(a, -1, axis=0)
    ab = b - a
    ab2 = np.maximum((ab ** 2).sum(axis=1), 1e-12)
    out = np.empty(len(points))
    for i, p in enumerate(np.asarray(points, dtype=float)):
        t = np.clip(((p - a) * ab).sum(axis=1) / ab2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        out[i] = np.sqrt(((proj - p) ** 2).sum(axis=1).min())
    return out


def shape_rms(separatrix, targets):
    d = point_polyline_distance(targets, separatrix)
    return float(np.sqrt(np.mean(d ** 2)))


XPOINT_CAP_M = 0.5


def xpoint_error(xpoints, target_xpoints, cap=XPOINT_CAP_M):
    """Mean distance from each target X-point to the closest candidate X-point, capped.

    A lost X-point (no candidate) counts as the cap. The cap keeps a transiently lost
    divertor configuration from dominating the episode mean.
    """
    if len(target_xpoints) == 0:
        return 0.0
    if len(xpoints) == 0:
        return float(cap)
    d = [min(np.min(np.linalg.norm(np.asarray(xpoints) - t, axis=1)), cap) for t in target_xpoints]
    return float(np.mean(d))


# Success criterion of tokamak-shape-metrics/v2. The MAST-U team asks magnetic controllers for a
# "small steady-state error (about 1 cm or less)" after a "fast settling response time (about
# 100 ms)" and did not request a small transient error (Lvovskiy et al., PPCF 67, 2025); no
# MAST-U plasma-current tolerance is published, so the 3% of Degrave et al. (Nature 602, 2022)
# is kept.
SUCCESS_SHAPE_M = 0.01      # steady-state boundary RMS error
SUCCESS_GAP_M = 0.01        # steady-state inner/outer gap and X-point height errors
SUCCESS_IP_REL = 0.03       # steady-state relative plasma-current error
SETTLE_S = 0.10             # the steady state starts this long after the last target change
MIN_WINDOW_S = 0.05         # shortest steady-state window if the horizon ends sooner
VOID_LOOKBACK_STEPS = 10    # steps before a solver failure checked for within_tolerance_at_failure
STEADY_KEYS = ("shape_rms", "xpoint_error", "ip_error_rel", "rin_error", "rout_error", "zx_error")


def summarize_episode(records, horizon, steady_start=None):
    """Aggregate per-step records into the fixed reporting quantities.

    ``records`` is a list of dicts with keys shape_rms, xpoint_error, ip_error_rel,
    coil_violation (0/1) and, from v2, rin_error, rout_error, zx_error; ``records[j]``
    is the state after step j+1. The steady-state window is ``records[steady_start:]``
    of a complete episode (default: the last half); the last-half means are kept as
    diagnostics. A steady-state value is None if the window holds a non-finite error.
    """
    n = len(records)
    complete = n == horizon
    half = records[horizon // 2:] if complete else records[max(0, n // 2):]
    start = horizon // 2 if steady_start is None else int(steady_start)
    steady = records[start:] if complete else []

    def mean(key, rows):
        vals = [r[key] for r in rows if key in r and math.isfinite(r[key])]
        return float(np.mean(vals)) if vals else None

    def strict_mean(key, rows):
        vals = [r.get(key, math.inf) for r in rows]
        return float(np.mean(vals)) if vals and all(math.isfinite(v) for v in vals) else None

    summary = {
        "steps": n,
        "complete": complete,
        "steady_start_step": start,
        "shape_rms_m_full": mean("shape_rms", records),
        "shape_rms_m_last_half": mean("shape_rms", half),
        "xpoint_error_m_last_half": mean("xpoint_error", half),
        "ip_error_rel_last_half": mean("ip_error_rel", half),
        "coil_violation_fraction": float(np.mean([r["coil_violation"] for r in records])) if n else None,
        "max_shape_rms_m": float(max(r["shape_rms"] for r in records)) if n else None,
    }
    for key in STEADY_KEYS:
        summary[f"{key}_steady"] = strict_mean(key, steady)
    # Continuous companion of the success flag: the share of the steady-state window (from
    # ``start`` to the horizon) whose steps are within every success tolerance. Steps an episode
    # did not reach count as outside, so early termination lowers it.
    window = max(1, horizon - start)
    summary["tolerance_time_fraction"] = sum(within_tolerance(r) for r in records[start:horizon]) / window
    return summary


def within_tolerance(record):
    """True if one step's errors are all within the success tolerances."""
    limits = (("shape_rms", SUCCESS_SHAPE_M), ("rin_error", SUCCESS_GAP_M), ("rout_error", SUCCESS_GAP_M),
              ("zx_error", SUCCESS_GAP_M), ("ip_error_rel", SUCCESS_IP_REL))
    return all(math.isfinite(record.get(k, math.inf)) and record.get(k, math.inf) <= v for k, v in limits)


def meets_success(summary):
    """True if a complete episode's steady-state errors are all within the v2 tolerances."""
    limits = {"shape_rms_steady": SUCCESS_SHAPE_M, "rin_error_steady": SUCCESS_GAP_M,
              "rout_error_steady": SUCCESS_GAP_M, "zx_error_steady": SUCCESS_GAP_M,
              "ip_error_rel_steady": SUCCESS_IP_REL}
    return bool(summary.get("complete")) and all(
        summary.get(k) is not None and summary[k] <= v for k, v in limits.items())


def void_candidate(states, lookback=VOID_LOOKBACK_STEPS):
    """True if the plasma was within the success tolerances just before a solver failure.

    ``states`` are the last per-step records (or heartbeats) before the failure, each with
    shape_rms and ip_error_rel. Reported with the episode (``within_tolerance_at_failure``);
    from tokamak-shape-metrics/v4 such an episode still counts as a solver failure.
    """
    recent = [s for s in list(states)[-lookback:]
              if math.isfinite(s.get("shape_rms", math.inf)) and math.isfinite(s.get("ip_error_rel", math.inf))]
    if len(recent) < lookback:
        return False
    return (float(np.mean([s["shape_rms"] for s in recent])) <= SUCCESS_SHAPE_M
            and float(np.mean([s["ip_error_rel"] for s in recent])) <= SUCCESS_IP_REL)
