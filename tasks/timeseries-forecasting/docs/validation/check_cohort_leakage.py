#!/usr/bin/env python3
"""Measure how much of the held-out labels a package could read out of other instances' contexts.

    python3 check_cohort_leakage.py --input STAGED_INPUT_DIR --out cohort_leakage.json

GIFT-Eval's test windows roll, so window k's labels are the tail of window k+1's context, and a dataset's
short-term contexts run into its medium- and long-term test regions. This script quantifies that in absolute
time: for every forecast instance it enumerates the timestamps of its label window and of every instance's
context interval, per (dataset directory, item), and reports the share of label points that fall inside some
other instance's context

  * when the whole split is staged into one `infer.sh` call (what a naive verifier would do), and
  * when instances are grouped into the cutoff cohorts the verifier actually uses (`evaluation/make_cases.py`).

Stage the input with `make_cases.py --limit-series N` for a quick representative run, or without it for the
full split.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "environment" / "evaluation"))
import ts_io  # noqa: E402
from make_cases import content_conflicts, synthetic_time, timeline  # noqa: E402


def collect(input_dir: Path):
    """(dataset directory, item) -> list of instances (cohort, context start, cutoff, label timestamps, label points)."""
    index = json.loads((input_dir / "index.json").read_text())
    groups = collections.defaultdict(list)
    for cfg in index["configs"]:
        case = ts_io.CaseFile(input_dir / "cases" / f"{cfg['config_id']}.h5")
        directory = cfg["dataset"].split("/")[0]
        freq = ts_io._pandas_freq(case.meta["freq"])
        for i in range(case.n):
            cutoff = pd.Timestamp(case.forecast_start[i])
            labels = pd.date_range(cutoff, periods=case.prediction_length, freq=freq)
            context_start = pd.date_range(end=cutoff, periods=int(case.context_length[i]) + 1, freq=freq)[0]
            groups[(directory, case.item_id[i])].append(
                (int(case.cohort[i]), context_start, cutoff, labels, case.prediction_length * case.n_variates))
        case.close()
    return index, groups


def _union(spans):
    """Sorted disjoint intervals (ns since epoch) covering the union of half-open spans."""
    if not len(spans):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    order = np.argsort(spans[:, 0], kind="stable"); spans = spans[order]
    lo, hi = [spans[0, 0]], [spans[0, 1]]
    for a, b in spans[1:]:
        if a <= hi[-1]:
            hi[-1] = max(hi[-1], b)
        else:
            lo.append(a); hi.append(b)
    return np.array(lo), np.array(hi)


def exposure(groups, within_cohort: bool, across_items: bool = False, skip=lambda directory: False, pool=None, tally=None):
    """Share of label points that lie inside another instance's context, over instances visible in the same call.

    `across_items` compares an instance with every instance of its dataset directory, not only of its own item
    (correlated series such as kdd_cup_2018's stations); `skip` drops directories from the tally. Instances of
    the same cutoff never count against each other (their contexts end where the labels start)."""
    total = covered = 0.0
    pools = collections.defaultdict(list)
    for (directory, item), items in groups.items():
        if skip(directory):
            continue
        key = pool(directory) if pool else (timeline(directory) if across_items else (directory, item))
        pools[key].extend((directory, *inst) for inst in items)
    for members in pools.values():
        by_cohort = collections.defaultdict(list)
        for directory, cohort, s, t, labels, points in members:
            by_cohort[cohort if within_cohort else 0].append((s.value, t.value, labels.asi8, points, directory))
        for insts in by_cohort.values():
            spans = np.array([(s, t) for s, t, _, _, _ in insts], dtype=np.int64).reshape(-1, 2)
            cut = spans[:, 1]
            unions = {c: _union(spans[cut != c]) for c in np.unique(cut)}
            for _, t, lab, points, directory in insts:
                lo, hi = unions[t]
                got = 0.0
                if len(lo) and len(lab):
                    k = np.searchsorted(lo, lab, side="right") - 1
                    hit = (k >= 0) & (lab < hi[np.maximum(k, 0)])
                    got = float(hit.mean()) * points
                covered += got; total += points
                if tally is not None:
                    tally[directory][0] += got; tally[directory][1] += points
    return covered / total if total else 0.0, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="", help="materialized data root: also check M4 content conflicts (make_cases.content_conflicts)")
    ap.add_argument("--manifest", default="", help="split manifest, with --data")
    args = ap.parse_args()
    index, groups = collect(Path(args.input))
    one, total = exposure(groups, within_cohort=False)
    per_cohort, _ = exposure(groups, within_cohort=True)
    one_dir, _ = exposure(groups, within_cohort=False, across_items=True)
    per_cohort_dir, _ = exposure(groups, within_cohort=True, across_items=True)
    per_cohort_dir_real, real_total = exposure(groups, within_cohort=True, across_items=True, skip=synthetic_time)
    per_directory = {}
    for directory in sorted({d for d, _ in groups}):
        sub = {k: v for k, v in groups.items() if k[0] == directory}
        per_directory[directory] = {"exposed_cohort_wise_any_item": exposure(sub, True, True)[0], "synthetic_time": synthetic_time(directory)}
    # different timelines in one call: which pairs of timelines, and how close their cutoffs come (documentation of
    # the "unrelated" judgement behind RELATED_DIRECTORIES)
    call_cut = collections.defaultdict(dict)
    for (directory, _), items in groups.items():
        if not synthetic_time(directory):
            for cohort, _, cutoff, _, _ in items:
                call_cut[cohort][timeline(directory)] = cutoff
    cross = {}
    for cohort, tl in call_cut.items():
        names = sorted(tl)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if tl[a] != tl[b]:
                    gap = abs(tl[a] - tl[b]); key = f"{a} | {b}"
                    cross[key] = min(cross.get(key, gap), gap)
    closest = [{"timelines": k, "closest_cutoffs_apart": str(v)} for k, v in sorted(cross.items(), key=lambda kv: kv[1])[:12]]
    # the residual: label points of real-timestamp series inside any other real-timestamp series' context in the same
    # call, whatever its timeline (different timelines measure unrelated things; this says how much time they share)
    tally = collections.defaultdict(lambda: [0.0, 0.0])
    across, _ = exposure(groups, within_cohort=True, skip=synthetic_time, pool=lambda d: "all", tally=tally)
    across_by_dir = {d: round(c / t, 6) for d, (c, t) in sorted(tally.items()) if c > 0}
    content = None
    if args.data and args.manifest:
        configs = json.loads(Path(args.manifest).read_text())["configs"]
        edges, _ = content_conflicts(Path(args.data), [c for c in configs if synthetic_time(c["dataset"].split("/")[0])])
        call_of = {key: {c for c, *_ in items} for key, items in groups.items()}
        shared = [sorted(map(list, e)) for e in edges if all(k in call_of for k in e) and set.intersection(*(call_of[k] for k in e))]
        content = {"m4_conflict_pairs": len(edges), "sharing_a_call": len(shared), "examples": shared[:5]}
    mixed = sum(1 for items in groups.values()
                for cohort in {c for c, _, _, _, _ in items}
                if len({t for c, _, t, _, _ in items if c == cohort}) > 1)
    out = {"what": "share of held-out label points readable from another instance's context, measured in absolute time",
           "input": str(args.input), "n_configs": len(index["configs"]), "n_cohorts": index.get("n_cohorts"),
           "n_series_groups": len(groups), "label_points_considered": int(total),
           "exposed_single_call": one, "exposed_cohort_wise": per_cohort,
           "exposed_single_call_any_item": one_dir, "exposed_cohort_wise_any_item": per_cohort_dir,
           "exposed_cohort_wise_any_item_real_timestamps": per_cohort_dir_real,
           "label_points_with_real_timestamps": int(real_total),
           "note": "same-item figures compare an instance with other windows of its own series; any-item figures with every "
                   "instance on its timeline (its dataset directory, or directories recording one system: "
                   "make_cases.RELATED_DIRECTORIES). Placeholder-time directories (M4) are excluded from the real-timestamps "
                   "figure, since their timestamps order nothing; they are checked by content instead (m4_content).",
           "m4_content": content,
           "cross_timeline_pairs_in_one_call": {"n_pairs": len(cross), "closest": closest},
           "exposed_cohort_wise_across_timelines_real_timestamps": across,
           "across_timelines_by_directory": across_by_dir,
           "per_directory": per_directory,
           "cohorts_mixing_cutoffs_for_one_series": mixed}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    if per_cohort > 0 or per_cohort_dir_real > 0 or mixed or (content and content["sharing_a_call"]):
        sys.exit("cohort grouping does not isolate labels")


if __name__ == "__main__":
    main()
