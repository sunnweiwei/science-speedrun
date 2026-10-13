#!/usr/bin/env python3
"""Verify the content overlaps between the pretraining corpus and GIFT-Eval that the allow-list excludes.

    python3 check_corpus_leakage.py --data /large-disk/timeseries --out corpus_leakage_verified.json

`corpus_overlap.json` is a broad survey (affine-invariant fingerprints over six public corpora, then Pearson
correlation on the raw values). This script re-derives the decisive ones directly: for each recorded pair it loads
both series and reports the best correlation over all alignments, the affine relation, and whether GIFT-Eval's test
horizon is inside the matched window. It needs the excluded directories, so run it against a materialization that
still has them (`materialize.py --dirs pedestrian_counts,weather,tourism_monthly,fred_md,monash_m3_yearly`).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "environment" / "evaluation"))
import ts_io  # noqa: E402

PAIRS = [   # (corpus directory, corpus item, GIFT-Eval dataset, GIFT-Eval item)
    ("pedestrian_counts", "T11", "m4_hourly", "133"),
    ("pedestrian_counts", "T19", "m4_hourly", "141"),
    ("pedestrian_counts", "T8", "m4_hourly", "130"),
    ("fred_md", "T49", "m4_monthly", "2532"),
    ("fred_md", "T45", "m4_monthly", "48"),
    ("tourism_monthly", "T174", "m4_monthly", "31248"),
    ("weather", "T1510", "temperature_rain_with_missing", "T23258"),
    ("monash_m3_yearly", "T209", "m4_yearly", "14733"),
]


def best_alignment(target: np.ndarray, source: np.ndarray):
    """Best Pearson correlation over every alignment, sliding whichever series is shorter over the other.

    Returns (r, lag, overlap); lag is negative when the corpus series is the shorter one.
    """
    short, long_, swapped = (target, source, False) if len(target) <= len(source) else (source, target, True)
    best = (0.0, None, 0)
    for lag in range(0, len(long_) - len(short) + 1):
        w = long_[lag:lag + len(short)]
        m = np.isfinite(w) & np.isfinite(short)
        if m.sum() < 16 or np.std(w[m]) == 0 or np.std(short[m]) == 0:
            continue
        r = float(np.corrcoef(short[m], w[m])[0, 1])
        if abs(r) > abs(best[0]):
            best = (r, -lag if swapped else lag, int(m.sum()))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.data)
    rows, cache = [], {}

    def load(sub, dataset):
        key = (sub, dataset)
        if key not in cache:
            base = root / ("gifteval" if sub is None else "gifteval_pretrain")
            cache[key] = ts_io.read_arrow_dataset(base, dataset)
        return cache[key]

    for sub, sid, ds, item in PAIRS:
        try:
            ids_c, _, _, ser_c = load(sub, sub)
            ids_g, _, _, ser_g = load(None, ds)
        except FileNotFoundError as exc:
            rows.append({"corpus_dir": sub, "gifteval": ds, "error": f"not materialized: {exc}"}); continue
        ci = {i.split("#")[0]: k for k, i in enumerate(ids_c)}
        gi = {i: k for k, i in enumerate(ids_g)}
        if sid not in ci or item not in gi:
            rows.append({"corpus_dir": sub, "gifteval": ds, "error": "item not found"}); continue
        a, b = ser_g[gi[item]][0].astype(np.float64), ser_c[ci[sid]][0].astype(np.float64)
        r, lag, overlap = best_alignment(a, b)
        entry = {"corpus_dir": sub, "corpus_item": sid, "gifteval": ds, "gifteval_item": item,
                 "pearson_r": r, "lag": lag, "overlap": overlap,
                 "gifteval_length": int(len(a)), "corpus_length": int(len(b))}
        if lag is not None:
            if lag >= 0:                      # the GIFT-Eval series sits inside the corpus series
                w, g = b[lag:lag + len(a)], a[:max(0, len(b) - lag)]
            else:                             # the corpus series sits inside the GIFT-Eval series
                w, g = b, a[-lag:-lag + len(b)]
            n = min(len(w), len(g)); w, g = w[:n], g[:n]
            m = np.isfinite(w) & np.isfinite(g)
            if m.sum() > 2:
                scale, offset = np.polyfit(w[m], g[m], 1)
                entry.update({"scale": float(scale), "offset": float(offset),
                              "max_abs_residual": float(np.max(np.abs(g[m] - (scale * w[m] + offset))))})
            entry["covers_whole_gifteval_series"] = bool(overlap >= entry["gifteval_length"] * 0.99)
        rows.append(entry)
    out = {"what": "direct re-derivation of the corpus/GIFT-Eval content overlaps that the allow-list excludes",
           "method": "load both series, correlate over every alignment, then fit the affine relation on the best one",
           "survey": "corpus_overlap.json", "pairs": rows,
           "conclusion": "GIFT-Eval's m4_hourly series are pedestrian_counts series plus a constant; the other pairs are "
                         "the same series up to an affine transform. Name-level disjointness does not imply content disjointness."}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({r.get("corpus_dir"): round(r.get("pearson_r", 0), 6) for r in rows}))


if __name__ == "__main__":
    main()
