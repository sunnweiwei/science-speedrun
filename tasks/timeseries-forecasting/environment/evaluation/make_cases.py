#!/usr/bin/env python3
"""Stage timeseries-forecast/v1 cases (INPUT_DIR) and hidden labels (TRUTH_DIR) from the pinned GIFT-Eval data.

    python3 make_cases.py --data /data --manifest ../data/split.json --split test \
        --input /work/input --truth /work/truth [--limit N] [--configs a,b]

Validation: a series appears in up to eight configs (three terms x several aggregated frequencies of the same
meter), and their test regions begin at different times. Staging each config's validation window just before its
own test region would put the short-term window, and the long context behind it, inside the medium- and long-term
test regions of the same series. Validation is therefore staged against the earliest test-region start over every
config that contains the series, so no agent-visible observation is ever part of a held-out window.

Cohorts: rolling windows overlap, so window k's label is the tail of window
k+1's context, and a dataset's short-term contexts reach into its medium- and
long-term test regions. The verifier therefore never runs a package on all
instances at once: every instance gets a cohort id and one call receives one
cohort. Call c holds, for each timeline, the instances at its c-th cutoff
timestamp; a timeline is a dataset directory (all items, frequencies and terms
staged from it) or directories that record one system on one clock
(RELATED_DIRECTORIES). Within a timeline every context of a call ends where
every label starts. The M4 directories carry placeholder timestamps, so there
series are coloured by content instead: series whose values hold another's test
label verbatim, and copies of one series, never share a call
(content_conflicts). The number of calls is the largest number of cutoffs on
one timeline (82 on the test split), not the number of configs or instances.

Windows follow gift-eval `Dataset.test_data` exactly: for a series of length T
with horizon L and W rolling windows, cutoff k = T - L*W + k*L (k = 0..W-1) and
the label is the next L observations. The `valid` split is the single window
before the test region (gift-eval `validation_dataset`). Each instance carries
its own context, capped at the last CONTEXT_CAP observations, so no instance
reveals another's label. The truth file also stores gluonts' seasonal error,
computed on the full uncapped history, which is all the grader needs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ts_io  # noqa: E402


def test_region_starts(data_root: Path, configs):
    """{(dataset directory, item): the earliest timestamp at which any config's test region begins for that series}.

    A series stored verbatim under two directories (a handful of M4 series, at different frequencies) is one series:
    its copies have the same values, so the boundary is taken in index space over every copy (the earliest test
    region start of any copy) and converted to each copy's own timestamps."""
    import pandas as pd

    earliest, rows, first_idx = {}, [], {}
    for cfg in configs:
        directory = cfg["dataset"].split("/")[0]
        item_ids, starts, freq, lengths, fingerprints = ts_io.read_arrow_index(data_root / cfg["gifteval_subdir"], cfg["dataset"])
        L, W = cfg["prediction_length"], cfg["windows"]
        for i, length in enumerate(lengths):
            idx = int(length) - L * W
            rows.append(((directory, item_ids[i]), starts[i], freq, idx))
            first_idx[fingerprints[i]] = min(first_idx.get(fingerprints[i], idx), idx)
            rows[-1] += (fingerprints[i],)
    for key, start, freq, idx, fp in rows:
        when = pd.Timestamp(ts_io.timestamp_after(start, freq, min(idx, first_idx[fp])))
        if key not in earliest or when < earliest[key]:
            earliest[key] = when
    return earliest


def valid_cut(cfg, item_id, start, freq, length, boundaries):
    """Index at which this config's validation window starts: the last horizon that ends at or before the boundary."""
    directory = cfg["dataset"].split("/")[0]
    boundary = boundaries[(directory, item_id)]
    cut = ts_io.steps_between(start, freq, str(boundary)) - cfg["prediction_length"]
    return min(cut, ts_io.valid_cutoff(length, cfg["prediction_length"], cfg["windows"]))


# Directories whose timestamps are placeholders (the M4 competition series are anonymised and carry no real start
# date), so their series are unrelated and "absolute time" orders nothing: they keep one cohort rank per window
# of each item. Everywhere else the timestamps are real and correlated series (air-quality stations, meters, roads)
# must not see each other's futures, so the cutoffs are ranked across the whole directory.
SYNTHETIC_TIME_DIRECTORIES = ("m4_",)


def synthetic_time(directory: str) -> bool:
    return directory.startswith(SYNTHETIC_TIME_DIRECTORIES)


# Directories that record the same system on one clock share a timeline: bizitobs_service holds the 21 services of
# the Robot-Shop application that bizitobs_application measures, sampled one 10-second step later, so ranking them
# separately put the service contexts one step past the application labels in the same call. Every other pair of
# directories with real timestamps measures unrelated phenomena (docs/validation/cohort_leakage.json lists them).
RELATED_DIRECTORIES = {"bizitobs_service": "bizitobs_application"}


def timeline(directory: str) -> str:
    return RELATED_DIRECTORIES.get(directory, directory)


def content_conflicts(data_root: Path, configs):
    """Verbatim copies among the placeholder-time (M4) series, which absolute time cannot order.

    Returns (edges, containers): `edges` pairs two series that must not share an infer.sh call, because one's test
    label lies verbatim in the other's context (533 labels on the test split; a lookup of the last few context values
    recovers most of them) or because they are the same series stored under two directories; `containers` are the
    series whose agent-visible validation region (everything before their own test region) holds another series'
    test label, so their validation window is not staged. Labels with fewer than 3 distinct values are skipped: a
    constant run is predictable from any context and matches everywhere."""
    series, labels = {}, {}
    for cfg in configs:
        d = cfg["dataset"].split("/")[0]
        if not synthetic_time(d):
            continue
        item_ids, _, _, values = ts_io.read_arrow_dataset(data_root / cfg["gifteval_subdir"], cfg["dataset"])
        L, W = cfg["prediction_length"], cfg["windows"]
        for item, s in zip(item_ids, values):
            x = np.asarray(s[0], np.float32)
            T = len(x)
            series[(d, item)] = x[: T - L * W]           # the region before the test windows: contexts and validation
            for c in ts_io.test_cutoffs(T, L, W):
                lab = x[c:c + L]
                if len(np.unique(lab[np.isfinite(lab)])) >= 3 and np.isfinite(lab).all():
                    labels.setdefault((d, item), []).append(lab)
    # candidate positions: the first two values of a label, as one 64-bit key over the float32 bit patterns
    def keys(a):
        b = a.view(np.uint32).astype(np.uint64)
        return (b[:-1] << np.uint64(32)) | b[1:]
    by_key = {}
    for owner, labs in labels.items():
        for lab in labs:
            by_key.setdefault(int(keys(lab[:2])[0]), []).append((owner, lab))
    wanted = np.fromiter(by_key.keys(), dtype=np.uint64)
    edges, containers = set(), set()
    fingerprint = {}
    # one pass over every region at once: concatenated with a NaN between series, so no key spans two of them
    names = list(series)
    parts = [np.append(series[n], np.float32(np.nan)) for n in names]
    starts = np.cumsum([0] + [len(p) for p in parts[:-1]])
    flat = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    if len(flat) >= 2:
        k = keys(flat)
        for pos in np.nonzero(np.isin(k, wanted))[0]:
            j = int(np.searchsorted(starts, pos, side="right") - 1)
            key, x, off = names[j], series[names[j]], int(pos - starts[j])
            for owner, lab in by_key[int(k[pos])]:
                if owner != key and off + len(lab) <= len(x) and np.array_equal(x[off:off + len(lab)], lab):
                    edges.add(frozenset((owner, key))); containers.add(key)
    # the same series under two directories (identical values, different frequency): one timeline, never one call
    for cfg in configs:
        d = cfg["dataset"].split("/")[0]
        if not synthetic_time(d):
            continue
        item_ids, _, _, _, fps = ts_io.read_arrow_index(data_root / cfg["gifteval_subdir"], cfg["dataset"])
        for item, fp in zip(item_ids, fps):
            other = fingerprint.setdefault(fp, (d, item))
            if other != (d, item):
                edges.add(frozenset((other, (d, item))))
    return edges, containers


def color(nodes, edges):
    """Greedy colouring, highest degree first: {node: colour}, adjacent nodes never share one."""
    adj = {}
    for e in edges:
        a, b = tuple(e)
        adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
    out = {}
    for n in sorted(nodes, key=lambda n: (-len(adj.get(n, ())), str(n))):
        used = {out[m] for m in adj.get(n, ()) if m in out}
        c = 0
        while c in used:
            c += 1
        out[n] = c
    return out


def cohort_table(data_root: Path, configs, split: str, boundaries=None):
    """{dataset directory: {item_id: {forecast_start: cohort rank}}} over every window of every config of that directory.

    Call c holds, for every timeline (a dataset directory, or directories recording one system on one clock:
    RELATED_DIRECTORIES), the instances at that timeline's c-th cutoff timestamp, whichever item, frequency or term
    they come from (electricity/15T, /H, /D, /W are aggregates of the same meters; kdd_cup_2018's stations are
    correlated). Within a timeline every context of a call ends where every label starts, so no context reaches into
    a related series' label; different timelines in one call measure unrelated things. Directories with placeholder
    timestamps (`synthetic_time`, M4) have no order to rank; their series are coloured instead so that series whose
    values contain another's test label, or copies of one series, never share a call (content_conflicts).
    """
    times = {}
    for cfg in configs:
        d = cfg["dataset"].split("/")[0]
        item_ids, starts, freq, lengths, _ = ts_io.read_arrow_index(data_root / cfg["gifteval_subdir"], cfg["dataset"])
        L, W = cfg["prediction_length"], cfg["windows"]
        for i, T in enumerate(int(x) for x in lengths):
            cuts = ts_io.test_cutoffs(T, L, W) if split == "test" else [valid_cut(cfg, item_ids[i], starts[i], freq, T, boundaries)]
            times.setdefault(d, {}).setdefault(item_ids[i], set()).update(ts_io.timestamp_after(starts[i], freq, c) for c in cuts)
    colours = None
    if any(synthetic_time(d) for d in times):
        edges, _ = content_conflicts(data_root, [c for c in configs if synthetic_time(c["dataset"].split("/")[0])])
        colours = color([(d, item) for d in times if synthetic_time(d) for item in times[d]], edges)
    return rank_cohorts(times, colours)


def rank_cohorts(times, colours=None):
    """{directory: {item: set of cutoff timestamps}} -> {directory: {item: {timestamp: cohort id}}}.

    Real-time directories: the rank of the timestamp on the directory's timeline (shared by RELATED_DIRECTORIES).
    Placeholder-time directories: the item's own rank r of its cutoffs, spread by its colour (colour * W + r, W the
    most windows any such item has), so conflicting series land in different calls."""
    table, shared = {}, {}
    for d, items in times.items():
        if not synthetic_time(d):
            shared.setdefault(timeline(d), set()).update(*items.values())
    rank_of = {tl: {t: r for r, t in enumerate(sorted(ts))} for tl, ts in shared.items()}
    width = max((len(v) for d, items in times.items() if synthetic_time(d) for v in items.values()), default=1)
    for d, items in times.items():
        if synthetic_time(d):
            table[d] = {item: {t: (colours or {}).get((d, item), 0) * width + r for r, t in enumerate(sorted(v))}
                        for item, v in items.items()}
        else:
            table[d] = {item: {t: rank_of[timeline(d)][t] for t in v} for item, v in items.items()}
    return table


def stage_config(data_root: Path, cfg: dict, split: str, input_dir: Path, truth_dir: Path, context_cap: int, limit_series: int = 0,
                 cohort_of=None, boundaries=None, containers=None):
    item_ids, starts, freq, series = ts_io.read_arrow_dataset(data_root / cfg["gifteval_subdir"], cfg["dataset"])
    if freq != cfg["freq"]:
        raise RuntimeError(f"{cfg['dataset']}: frequency {freq!r} on disk, manifest says {cfg['freq']!r}")
    if len(series) != cfg["n_series"]:
        raise RuntimeError(f"{cfg['dataset']}: {len(series)} series on disk, manifest says {cfg['n_series']}")
    L, W, m = cfg["prediction_length"], cfg["windows"], cfg["seasonality"]
    contexts, labels, se, ids, wins, fstarts, hist, cohorts = [], [], [], [], [], [], [], []
    if split == "valid" and boundaries is None:
        boundaries = test_region_starts(data_root, [cfg])
    cohort_of = (cohort_of or cohort_table(data_root, [cfg], split, boundaries))[cfg["dataset"].split("/")[0]]
    directory = cfg["dataset"].split("/")[0]
    if split == "valid" and containers is None and synthetic_time(directory):
        containers = content_conflicts(data_root, [cfg])[1]
    skipped = dropped = 0
    for i, s in enumerate(series):
        if limit_series and i >= limit_series:
            break
        T = s.shape[1]
        cutoffs = ts_io.test_cutoffs(T, L, W) if split == "test" else [valid_cut(cfg, item_ids[i], starts[i], freq, T, boundaries)]
        for k, cut in enumerate(cutoffs):
            if split == "valid" and cut < ts_io.VALID_MIN_CONTEXT:
                skipped += 1      # the series' earliest test region leaves no room for a validation window
                continue
            if split == "valid" and (directory, item_ids[i]) in (containers or ()):
                dropped += 1      # its visible values hold another series' test label verbatim (content_conflicts)
                continue
            if cut <= 0:
                raise RuntimeError(f"{cfg['dataset']} item {item_ids[i]}: cutoff {cut} <= 0 (T={T}, L={L}, W={W})")
            history = s[:, :cut]
            contexts.append(np.ascontiguousarray(history[:, -context_cap:]))
            labels.append(s[:, cut:cut + L])
            se.append(ts_io.seasonal_error(history, m))
            ids.append(item_ids[i]); wins.append(k); hist.append(cut)
            fstarts.append(ts_io.timestamp_after(starts[i], freq, cut))
            cohorts.append(cohort_of[item_ids[i]][fstarts[-1]])
    if not contexts:
        raise RuntimeError(f"{cfg['config_id']}: no {split} window fits before the earliest held-out region of these series")
    meta = {k: cfg[k] for k in ("dataset", "freq", "freq_unit", "term", "config_id", "leaderboard_key", "prediction_length",
                                "seasonality", "windows", "n_variates", "domain")}
    meta["split"] = split
    meta["context_cap"] = context_cap
    ts_io.write_case_file(input_dir / "cases" / f"{cfg['config_id']}.h5", meta, contexts, ids, wins, fstarts, hist, cohorts)
    ts_io.write_truth_file(truth_dir / "cases" / f"{cfg['config_id']}.h5", np.stack(labels), np.stack(se), meta)
    out = {"config_id": cfg["config_id"], "leaderboard_key": cfg["leaderboard_key"], "dataset": cfg["dataset"], "term": cfg["term"],
           "freq": cfg["freq"], "prediction_length": L, "seasonality": m, "n_variates": cfg["n_variates"],
           "n_instances": len(contexts), "domain": cfg["domain"], "n_cohorts": len(set(cohorts))}
    if skipped:
        out["skipped_series"] = skipped
    if dropped:
        out["dropped_containers"] = dropped
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="materialized data root (contains gifteval/)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test", choices=["test", "valid"])
    ap.add_argument("--input", required=True); ap.add_argument("--truth", required=True)
    ap.add_argument("--limit", type=int, default=0, help="stage only the first N configs")
    ap.add_argument("--limit-series", type=int, default=0, help="stage only the first N series per config (smoke)")
    ap.add_argument("--configs", default="", help="comma-separated config ids to stage")
    ap.add_argument("--context-cap", type=int, default=ts_io.CONTEXT_CAP)
    args = ap.parse_args()
    man = json.loads(Path(args.manifest).read_text())
    configs = man["configs"]
    if args.configs:
        wanted = set(args.configs.split(","))
        configs = [c for c in configs if c["config_id"] in wanted]
    if args.limit:
        configs = configs[:args.limit]
    inp, truth = Path(args.input), Path(args.truth)
    inp.mkdir(parents=True, exist_ok=True); truth.mkdir(parents=True, exist_ok=True)
    boundaries = test_region_starts(Path(args.data), json.loads(Path(args.manifest).read_text())["configs"]) if args.split == "valid" else None
    cohort_of = cohort_table(Path(args.data), configs, args.split, boundaries)
    containers = None
    if args.split == "valid":
        every = json.loads(Path(args.manifest).read_text())["configs"]
        containers = content_conflicts(Path(args.data), [c for c in every if synthetic_time(c["dataset"].split("/")[0])])[1]
    staged = []
    for cfg in configs:
        staged.append(stage_config(Path(args.data), cfg, args.split, inp, truth, args.context_cap, args.limit_series, cohort_of,
                                   boundaries, containers))
        print(f"  staged {cfg['config_id']:48s} {staged[-1]['n_instances']:7d} instances", file=sys.stderr)
    n_cohorts = 1 + max((max(r.values()) for items in cohort_of.values() for r in items.values()), default=0)
    index = {"contract": ts_io.CONTRACT, "split": args.split, "context_cap": args.context_cap, "n_cohorts": n_cohorts,
             "quantile_levels": list(ts_io.QUANTILE_LEVELS), "configs": staged}
    (inp / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    (truth / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(f"staged {len(staged)} configs, {sum(s['n_instances'] for s in staged)} instances", file=sys.stderr)


if __name__ == "__main__":
    main()
