"""Content check: do the extra GiftEvalPretrain directories contain GIFT-Eval test-region series?

Affine-invariant fingerprints: every 32-point window (stride 1) of the last 20 % of each GIFT-Eval
series is z-normalized and rounded to 0.1. A corpus series "hits" a GIFT-Eval series when at least
32 consecutive window starts match (>= 63 aligned points; windows need >= 10 distinct levels), the signature of a copied series up to
scale and offset (how pedestrian_counts/m4_hourly was found). Corpus series are sampled.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "environment" / "baseline" / "starter"))
sys.path.insert(0, "/workspace/baseline/starter")
from data import iter_arrow_series  # noqa: E402

W, RUN, MIN_LEVELS = 32, 32, 10


def windows(x):
    x = x[np.isfinite(x)].astype(np.float64)
    if len(x) < W + RUN:
        return None
    s = np.lib.stride_tricks.sliding_window_view(x, W)
    sd = s.std(1)
    z = (s - s.mean(1, keepdims=True)) / np.where(sd > 1e-9, sd, 1)[:, None]
    q = np.round(z * 10).astype(np.int16)
    srt = np.sort(q, axis=1)
    levels = 1 + (np.diff(srt, axis=1) != 0).sum(1)
    ok = (sd > 1e-9) & (levels >= MIN_LEVELS)  # skip flat, step and near-constant windows
    return [hash(r.tobytes()) if k else None for r, k in zip(q, ok)]


gift = Path("/data/gifteval")
index = defaultdict(set)
n_g = 0
for ds in sorted(p for p in gift.rglob("*") if p.is_dir() and any(p.glob("data-*.arrow"))):
    for i, (_, x) in enumerate(iter_arrow_series(ds)):
        x = x[int(len(x) * 0.8):]
        for k, h in enumerate(windows(x) or []):
            if h is not None:
                index[h].add((str(ds.relative_to(gift)), i))
        n_g += 1
print(json.dumps({"gift_series": n_g, "fingerprints": len(index)}), flush=True)

rng = np.random.default_rng(0)
frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
report = {}
for d in sorted(Path("/extra").iterdir()):
    if not d.is_dir():
        continue
    seen = hits = 0
    examples = []
    for _, x in iter_arrow_series(d):
        if rng.random() > frac:
            continue
        seen += 1
        hs = windows(x) or []
        run, prev = 0, None
        for h in hs:
            m = index.get(h) if h is not None else None
            run = run + 1 if m else 0
            if run >= RUN:
                hits += 1
                if len(examples) < 5:
                    examples.append(sorted(m)[:3])
                break
    report[d.name] = {"sampled_series": seen, "series_with_copied_window": hits, "examples": examples}
    print(json.dumps({d.name: report[d.name]}), flush=True)
Path("/out/extra_leak_check.json").write_text(json.dumps({"window": W, "run": RUN, "sample_fraction": frac,
                                                           "gift_series": n_g, "dirs": report}, indent=1))
