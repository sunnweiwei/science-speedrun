"""Derive data-grounded randomization statistics from the FAIR-MAST EFIT traces (FIX_PROPOSAL fix 3).

Reads raw/shots_summary.json and raw/efit/*.nc (see README.md) and writes derived_ranges.json:
  - spread of beta_p and li across flat-top states (within a shot, and between double-null shots),
  - slow drift rates of beta_p and li within the flat-top,
  - rate and depth of fast beta_p crashes between consecutive EFIT slices,
  - pickup-probe and flux-loop noise, offset and dead-channel levels from EFIT fit residuals.
The output is derived from FAIR-MAST data (CC BY-SA 4.0, UKAEA) and carries the same licence.

Usage: .venv/bin/python derive_ranges.py [--max-shots N] [--workers 8]
"""

import argparse
import glob
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FLAT_TOP_FRACTION = 0.9     # flat-top: |Ip| within 10% of the shot's peak
MIN_FLAT_TOP_SLICES = 10    # 50 ms of 5 ms EFIT slices
SMOOTH_SLICES = 5           # 25 ms running median against slice-to-slice EFIT jitter
DRIFT_WINDOW_SLICES = 20    # 100 ms
CRASH_DROP = 0.10           # a fast crash: beta_p falls by more than 10% within one slice ...
CRASH_HOLD_SLICES = 3       # ... and the 15 ms after stay that far below the 15 ms before


def running_median(x, k=SMOOTH_SLICES):
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.array([np.median(xp[i:i + k]) for i in range(len(x))])


def shot_stats(path):
    import xarray as xr

    with xr.open_dataset(path) as ds:
        if "beta_pol" not in ds or "ip" not in ds:
            return None
        ip = np.abs(ds["ip"].values)
        bp, li, t = ds["beta_pol"].values, ds["li"].values, ds["time"].values
        ok = np.isfinite(ip) & np.isfinite(bp) & np.isfinite(li) & (bp > 0) & (li > 0)
        if ok.sum() < MIN_FLAT_TOP_SLICES:
            return None
        flat = ok & (ip >= FLAT_TOP_FRACTION * np.nanmax(ip[ok]))
        idx = np.where(flat)[0]
        if len(idx) < MIN_FLAT_TOP_SLICES:
            return None
        # longest contiguous flat-top run
        runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        run = max(runs, key=len)
        if len(run) < MIN_FLAT_TOP_SLICES:
            return None
        raw_lbp, tt = np.log(bp[run]), t[run]
        lbp, lli = running_median(raw_lbp), running_median(li[run])
        out = {"shot": int(ds.attrs.get("shot_id", Path(path).stem)), "ip_peak": float(np.nanmax(ip[ok])),
               "beta_pol_median": float(np.exp(np.median(lbp))), "li_median": float(np.median(lli)),
               "flat_top_s": float(tt[-1] - tt[0]),
               "log_beta_pol_dev": (lbp - np.median(lbp)).tolist(), "li_dev": (lli - np.median(lli)).tolist()}
        w = DRIFT_WINDOW_SLICES
        if len(run) > w:
            dt = tt[w:] - tt[:-w]
            out["log_beta_pol_rate"] = (np.abs(lbp[w:] - lbp[:-w]) / dt).tolist()
            out["li_rate"] = (np.abs(lli[w:] - lli[:-w]) / dt).tolist()
        # persistent drops on the unsmoothed trace: one-slice fall, and the next 15 ms stay below the previous 15 ms
        h, depths = CRASH_HOLD_SLICES, []
        k = h
        while k < len(raw_lbp) - h:
            before, after = np.median(raw_lbp[k - h:k]), np.median(raw_lbp[k:k + h])
            if raw_lbp[k] - raw_lbp[k - 1] < np.log(1.0 - CRASH_DROP) and after - before < np.log(1.0 - CRASH_DROP):
                depths.append(float(1.0 - np.exp(after - before)))
                k += h
            else:
                k += 1
        out["crashes"] = len(depths)
        out["crash_depths"] = depths

        sensors = {}
        for kind, meas, recon in (("pickup", "b_field_pol_probe_measured", "b_field_pol_probe_reconstructed"),
                                  ("flux_loop", "flux_loop_measured", "flux_loop_reconstructed")):
            if meas not in ds or recon not in ds:
                continue
            m, r = ds[meas].values[run], ds[recon].values[run]
            used = np.isfinite(m).all(axis=0) & (np.abs(m) > 0).any(axis=0)   # channels EFIT used
            if used.sum() < 3:
                continue
            m, r = m[:, used], r[:, used]
            typical = np.median(np.abs(m), axis=0)
            scale = np.maximum(typical, 0.1 * np.median(typical) + 1e-12)   # as the env's probe_scale
            res = m - r
            sensors[kind] = {"noise_rel": (np.std(res, axis=0) / scale).tolist(),
                             "offset_rel": (np.abs(np.mean(res, axis=0)) / scale).tolist(),
                             "n_used": int(used.sum()), "n_total": int(len(used))}
        out["sensors"] = sensors
        return out


def resistance_stats(path):
    """Flat-top plasma resistance R_p = loop voltage / Ip from the `vloop` stage (raw/vloop/<shot>.nc)."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        if "vloop_dynamic" not in ds or "ip" not in ds:
            return None
        ip, v, t = np.abs(ds["ip"].values), ds["vloop_dynamic"].values, ds["time"].values
        ok = np.isfinite(ip) & np.isfinite(v) & (v > 0)
        if ok.sum() < MIN_FLAT_TOP_SLICES:
            return None
        idx = np.where(ok & (ip >= FLAT_TOP_FRACTION * np.nanmax(ip[ok])))[0]
        if len(idx) < MIN_FLAT_TOP_SLICES:
            return None
        run = max(np.split(idx, np.where(np.diff(idx) > 1)[0] + 1), key=len)
        if len(run) < MIN_FLAT_TOP_SLICES:
            return None
        lr = running_median(np.log10(v[run] / ip[run]))
        out = {"shot": int(Path(path).stem), "ip_peak": float(np.nanmax(ip[ok])),
               "rp_median": float(10 ** np.median(lr)), "loop_voltage_median": float(np.median(v[run])),
               "log10_rp_dev": (lr - np.median(lr)).tolist()}
        w = DRIFT_WINDOW_SLICES
        if len(run) > w:
            out["log10_rp_rate"] = (np.abs(lr[w:] - lr[:-w]) / (t[run][w:] - t[run][:-w])).tolist()
        return out


def pct(values, qs=(5, 25, 50, 75, 95)):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return {f"p{q}": float(np.percentile(v, q)) for q in qs} if len(v) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", default=str(HERE / "derived_ranges.json"))
    args = ap.parse_args()
    files = sorted(glob.glob(str(HERE / "raw" / "efit" / "*.nc")))
    if args.max_shots:
        files = files[:: max(1, len(files) // args.max_shots)]
    vloop_files = sorted(glob.glob(str(HERE / "raw" / "vloop" / "*.nc")))
    with mp.get_context("spawn").Pool(args.workers) as pool:
        stats = [s for s in pool.map(shot_stats, files, chunksize=16) if s]
        rstats = [s for s in pool.map(resistance_stats, vloop_files, chunksize=16) if s]
    summary = {int(r["shot_id"]): r for r in json.loads((HERE / "raw" / "shots_summary.json").read_text())}

    def double_null(rows):
        return [s for s in rows if summary.get(s["shot"], {}).get("plasma_shape") == "Connected Double Null"
                and 5.5e5 <= s["ip_peak"] <= 8.5e5]

    cdn, rcdn = double_null(stats), double_null(rstats)
    log_rp = np.log10([s["rp_median"] for s in rcdn]) if rcdn else np.zeros(0)

    def cat(key, rows):
        return [x for s in rows for x in s.get(key, [])]

    def sensor(kind, key):
        return [x for s in stats if kind in s["sensors"] for x in s["sensors"][kind][key]]

    def dead(kind):
        return [s["sensors"][kind]["n_total"] - s["sensors"][kind]["n_used"] for s in stats if kind in s["sensors"]]

    log_bp_between = np.log([s["beta_pol_median"] for s in cdn])
    li_between = np.array([s["li_median"] for s in cdn])
    flat_time = sum(s["flat_top_s"] for s in stats)
    result = {
        "source": "FAIR-MAST level-2 EFIT (UKAEA), CC BY-SA 4.0; derived by environment/data/fair_mast/derive_ranges.py",
        "n_shots_with_flat_top": len(stats),
        "n_double_null_550_850kA": len(cdn),
        "flat_top_definition": f"longest run with |Ip| >= {FLAT_TOP_FRACTION} x peak, >= {MIN_FLAT_TOP_SLICES} EFIT slices",
        "within_shot": {"log_beta_pol_dev": pct(cat("log_beta_pol_dev", stats)),
                        "li_dev": pct(cat("li_dev", stats))},
        "between_double_null_shots": {
            "beta_pol": pct(np.exp(log_bp_between)), "li": pct(li_between),
            "log_beta_pol_dev": pct(log_bp_between - np.median(log_bp_between)),
            "li_dev": pct(li_between - np.median(li_between))},
        "drift_per_s_over_100ms": {"log_beta_pol": pct(cat("log_beta_pol_rate", stats), (25, 50, 75, 90, 95)),
                                   "li": pct(cat("li_rate", stats), (25, 50, 75, 90, 95))},
        "smoothing": f"{SMOOTH_SLICES}-slice running median (25 ms) for spreads and drifts",
        "fast_crashes": {"drop_threshold": CRASH_DROP, "between_slices_s": 0.005,
                         "persistence_s": 0.005 * CRASH_HOLD_SLICES,
                         "rate_per_s": sum(s["crashes"] for s in stats) / max(flat_time, 1e-9),
                         "depth": pct(cat("crash_depths", stats), (25, 50, 75, 95)),
                         "flat_top_seconds": flat_time},
        "plasma_resistance_double_null": {
            "n_shots": len(rcdn),
            "definition": "flat-top median of vloop_dynamic / |Ip| (EFIT), double-null shots with 550-850 kA peak",
            "loop_voltage_V": pct([s["loop_voltage_median"] for s in rcdn]),
            "rp_uohm": pct([1e6 * s["rp_median"] for s in rcdn]),
            "log10_rp_dev": pct(log_rp - np.median(log_rp)) if len(log_rp) else None,
            "within_shot_log10_rp_dev": pct(cat("log10_rp_dev", rcdn)),
            "log10_rp_drift_per_s_over_100ms": pct(cat("log10_rp_rate", rcdn), (25, 50, 75, 90))},
        "sensors": {kind: {"noise_rel": pct(sensor(kind, "noise_rel")), "offset_rel": pct(sensor(kind, "offset_rel")),
                           "unused_channels_per_shot": pct(dead(kind))}
                    for kind in ("pickup", "flux_loop")},
        "sensor_note": "EFIT fit residuals (measured - reconstructed) over the flat-top, relative to each channel's "
                       "median |measured| (floored at 0.1 x the median over channels); they include EFIT model error, "
                       "so they bound the diagnostic noise from above. Unused channels are those EFIT did not fit.",
    }
    Path(args.output).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
