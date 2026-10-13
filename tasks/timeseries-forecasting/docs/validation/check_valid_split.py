#!/usr/bin/env python3
"""Assert that nothing in the agent-visible validation data lies inside any held-out test window.

    python3 check_valid_split.py --data /large-disk/timeseries --manifest ../environment/data/split.json \
        --valid-input /large-disk/timeseries/valid/input --out valid_split_leakage.json

A series appears in up to eight GIFT-Eval configs (three terms x aggregated frequencies of the same meter) whose
test regions begin at different times. This walks every staged validation instance, converts its context start,
its cutoff and its label window to absolute time, and intersects them with the test region of every config that
contains the same series (the test region of a config starts at its first cutoff and runs to the end of the
series). A series stored verbatim under two M4 directories is one series: each copy is checked against the earliest
test region of any copy, in index space. And no M4 series whose values hold another series' test label verbatim
may have a validation window at all (make_cases.content_conflicts). Any overlap is a leak into the agent's
read-only mount.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "environment" / "evaluation"))
import ts_io  # noqa: E402
from make_cases import content_conflicts, synthetic_time  # noqa: E402


def test_regions(data_root: Path, configs):
    """{(dataset directory, item): [(test region start, series end, config id)]} in absolute time."""
    regions, rows, first_idx = collections.defaultdict(list), [], {}
    for cfg in configs:
        directory = cfg["dataset"].split("/")[0]
        item_ids, starts, freq, lengths, fingerprints = ts_io.read_arrow_index(data_root / cfg["gifteval_subdir"], cfg["dataset"])
        L, W = cfg["prediction_length"], cfg["windows"]
        for i, T in enumerate(int(x) for x in lengths):
            rows.append(((directory, item_ids[i]), starts[i], freq, T, T - L * W, fingerprints[i], cfg["config_id"]))
            first_idx[fingerprints[i]] = min(first_idx.get(fingerprints[i], T - L * W), T - L * W)
    for key, start, freq, T, idx, fp, cid in rows:
        first = pd.Timestamp(ts_io.timestamp_after(start, freq, min(idx, first_idx[fp])))   # any copy's test region
        end = pd.Timestamp(ts_io.timestamp_after(start, freq, T))
        regions[key].append((first, end, cid))
    return regions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--valid-input", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    configs = json.loads(Path(args.manifest).read_text())["configs"]
    regions = test_regions(Path(args.data), configs)
    index = json.loads((Path(args.valid_input) / "index.json").read_text())
    by_id = {c["config_id"]: c for c in configs}
    checked = leaks = 0
    worst = []
    for staged in index["configs"]:
        cfg = by_id[staged["config_id"]]
        directory = cfg["dataset"].split("/")[0]
        case = ts_io.CaseFile(Path(args.valid_input) / "cases" / f"{staged['config_id']}.h5")
        freq = case.meta["freq"]
        for i in range(case.n):
            cutoff = pd.Timestamp(case.forecast_start[i])
            label_end = pd.Timestamp(ts_io.timestamp_after(cutoff.to_datetime64(), freq, case.prediction_length))
            ctx_start = pd.Timestamp(ts_io.timestamp_after(cutoff.to_datetime64(), freq, -int(case.context_length[i])))
            checked += 1
            for first, end, other in regions[(directory, case.item_id[i])]:
                if ctx_start < end and label_end > first:      # any overlap with a test region
                    leaks += 1
                    if len(worst) < 10:
                        worst.append({"config": staged["config_id"], "item": case.item_id[i], "overlaps_test_region_of": other,
                                      "valid_window": [str(ctx_start), str(label_end)], "test_region_starts": str(first)})
                    break
        case.close()
    # content: an M4 series whose visible values hold another series' test label verbatim must not be staged
    _, containers = content_conflicts(Path(args.data), [c for c in configs if synthetic_time(c["dataset"].split("/")[0])])
    staged_items = set()
    for staged in index["configs"]:
        directory = by_id[staged["config_id"]]["dataset"].split("/")[0]
        if synthetic_time(directory):
            case = ts_io.CaseFile(Path(args.valid_input) / "cases" / f"{staged['config_id']}.h5")
            staged_items.update((directory, it) for it in case.item_id)
            case.close()
    content_leaks = sorted(staged_items & containers)
    out = {"what": "overlap between agent-visible validation windows (context and labels) and any held-out test region, "
                   "in time (alias copies included) and by content (M4 series holding another series' test label)",
           "manifest": args.manifest, "valid_input": args.valid_input, "n_configs": len(index["configs"]),
           "instances_checked": checked, "instances_overlapping_a_test_region": leaks,
           "m4_series_holding_another_test_label": len(containers),
           "of_which_staged_for_validation": len(content_leaks), "examples": worst}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("n_configs", "instances_checked", "instances_overlapping_a_test_region",
                                          "m4_series_holding_another_test_label", "of_which_staged_for_validation")}))
    if leaks or content_leaks:
        sys.exit("validation data overlaps held-out test windows")


if __name__ == "__main__":
    main()
