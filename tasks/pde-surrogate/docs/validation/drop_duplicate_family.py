#!/usr/bin/env python3
"""Recompute a verifier report without the corpus family `ape2d/hyp`.

    python3 drop_duplicate_family.py [--write] validation/*.json

Upstream's full corpus release ships `hyp.hdf5` as a content copy of `diff.hdf5` (see
environment/data/AUDIT.md), so every replay made on the original split scored diffusion twice
under two names. The verifier's aggregation is exact at the dataset level: a group's VRMSE is
the unweighted mean of its datasets' mean VRMSE (environment/evaluation/score.py), the reward
is the mean over groups of a log-band map of the group VRMSE, and every other dataset's cases
are unchanged by the split change. Dropping the duplicate is therefore a recomputation from
numbers the report already holds, not a new run; the report says so in `derivation`.

Before touching a report the script re-derives its stored group VRMSE and reward from the
stored per-dataset means and refuses to proceed unless they agree to 1e-9, so the aggregation
it assumes is the one that produced the file.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "environment" / "evaluation"))
from score import reward_from_vrmse  # noqa: E402

DUPLICATE = "ape2d/hyp"
NOTE = ("recomputed on 2026-09-22 from this report's own per-dataset means after removing the corpus family "
        "'ape2d/hyp' (upstream ships hyp.hdf5 as a content copy of diff.hdf5); per-dataset numbers are the "
        "measured ones, group means and the reward follow score.py's aggregation exactly; case counts are the "
        "canonical split's (environment/data/split.json)")


def group_of(dataset):
    return "ape2d" if dataset.startswith("ape2d/") else dataset


def aggregate(datasets):
    """(groups, mean_vrmse, reward) from {dataset: mean vrmse}, exactly as score.py does it."""
    by = {}
    for ds, v in datasets.items():
        by.setdefault(group_of(ds), []).append(v)
    groups = {g: {"n_datasets": len(v), "vrmse": sum(v) / len(v), "reward": reward_from_vrmse(sum(v) / len(v))}
              for g, v in by.items()}
    mean_vrmse = sum(g["vrmse"] for g in groups.values()) / len(groups)
    reward = sum(g["reward"] for g in groups.values()) / len(groups)
    return groups, mean_vrmse, reward


def tolerance(stored):
    """Some records keep group values rounded to 4 decimals; accept half a unit of that precision."""
    text = repr(float(stored))
    decimals = len(text.split(".")[1]) if "." in text and "e" not in text else 12
    return max(1e-9, 0.51 * 10 ** -decimals)


def check(stored_groups, stored_reward, datasets, label):
    groups, _, reward = aggregate(datasets)
    for g, got in groups.items():
        want = stored_groups[g]["vrmse"] if isinstance(stored_groups[g], dict) else stored_groups[g]
        if abs(got["vrmse"] - want) > tolerance(want):
            raise SystemExit(f"{label}: stored group {g} vrmse {want} != recomputed {got['vrmse']}")
    if abs(reward - stored_reward) > tolerance(stored_reward):
        raise SystemExit(f"{label}: stored reward {stored_reward} != recomputed {reward}")


def transform_report(rep, label):
    """A verifier report (datasets: {name: {vrmse, n_cases, ...}}, groups, mean_vrmse, reward)."""
    per = {ds: d["vrmse"] for ds, d in rep["datasets"].items()}
    check(rep["groups"], rep["reward"], per, label)
    dropped = rep["datasets"].pop(DUPLICATE)
    per.pop(DUPLICATE)
    groups, mean_vrmse, reward = aggregate(per)
    before = {"reward": rep["reward"], "mean_vrmse": rep["mean_vrmse"], "n_cases": rep["n_cases"],
              "groups": {g: (v["vrmse"] if isinstance(v, dict) else v) for g, v in rep["groups"].items()},
              "dropped": {DUPLICATE: dropped}}
    rep["groups"] = {g: {**(rep["groups"][g] if isinstance(rep["groups"][g], dict) else {}), **groups[g]} for g in groups}
    rep["mean_vrmse"], rep["reward"] = mean_vrmse, reward
    rep["n_cases"] = rep["n_cases"] - dropped.get("n_cases", 100)     # the original split had 100 hyp cases
    rep["n_failed"] = rep.get("n_failed", 0) - dropped.get("n_failed", 0)
    rep["derivation"] = {"what": NOTE, "before": before}
    return before, reward


def transform_trivial(doc, label):
    """docs/validation/trivial_predictors_canonical.json: results[*] with flat datasets/groups maps."""
    changes = []
    for r in doc["results"]:
        check(r["groups"], r["reward"], r["datasets"], f"{label}:{r['predictor']}")
        dropped = r["datasets"].pop(DUPLICATE)
        groups, _, reward = aggregate(r["datasets"])
        before = {"reward": r["reward"], "n_cases": r["n_cases"], "groups": dict(r["groups"]), "dropped": {DUPLICATE: dropped}}
        r["groups"] = {g: v["vrmse"] for g, v in groups.items()}
        r["reward"] = reward
        r["n_cases"] = r["n_cases"] - 100
        r["derivation"] = {"what": NOTE, "before": before}
        changes.append((r["predictor"], before["reward"], reward))
    doc["held_out_set"] = "split.json test cases (1,085 canonical; the 100 hyp cases of the original 1,185 were the duplicate family)"
    return changes


def transform_result_record(doc, report, label):
    """A training record whose `result` block holds only group values (fine-tunes, CNextU-net runs).

    The run's full verifier report (report.json next to reward.json) carries the per-dataset means;
    it must agree with the record's stored result before anything is recomputed. The per-dataset
    means are then embedded in the record, so the next recomputation needs no external file."""
    res = doc["result"]
    per = {ds: d["vrmse"] for ds, d in report["datasets"].items()}
    check(report["groups"], report["reward"], per, f"{label}:report")
    for g, v in res["groups"].items():
        want = v["vrmse"] if isinstance(v, dict) else v
        if abs(report["groups"][g]["vrmse"] - want) > tolerance(want):
            raise SystemExit(f"{label}: result.groups[{g}] {want} != report {report['groups'][g]['vrmse']}")
    if abs(report["reward"] - res["reward"]) > tolerance(res["reward"]):
        raise SystemExit(f"{label}: result.reward {res['reward']} != report {report['reward']}")
    dropped = per.pop(DUPLICATE)
    groups, mean_vrmse, reward = aggregate(per)
    before = {"reward": res["reward"], "mean_vrmse": res["mean_vrmse"], "n_cases": res["n_cases"],
              "groups": {g: (v["vrmse"] if isinstance(v, dict) else v) for g, v in res["groups"].items()},
              "dropped": {DUPLICATE: report["datasets"][DUPLICATE]}}
    res["groups"] = {g: {"n_datasets": groups[g]["n_datasets"], "vrmse": round(groups[g]["vrmse"], 4), "reward": groups[g]["reward"]}
                     for g in groups}
    res["mean_vrmse"], res["reward"] = round(mean_vrmse, 4), reward
    res["n_cases"] = res["n_cases"] - report["datasets"][DUPLICATE]["n_cases"]
    res["datasets"] = {ds: {"n_cases": d["n_cases"], "n_failed": d.get("n_failed", 0), "vrmse": d["vrmse"]}
                       for ds, d in report["datasets"].items() if ds != DUPLICATE}
    doc["derivation"] = {"what": NOTE + "; the per-dataset means come from the run's verifier report.json and are now embedded",
                         "before": before}
    return before, reward


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--pair", action="append", default=[], metavar="RECORD.json=REPORT.json",
                    help="a training record whose result block is recomputed from the run's full verifier report")
    ap.add_argument("--write", action="store_true", help="rewrite the files in place (default: report only)")
    a = ap.parse_args()
    for pair in a.pair:
        rec_path, rep_path = (Path(x).expanduser() for x in pair.split("=", 1))
        doc = json.loads(rec_path.read_text())
        if "derivation" in doc:
            print(f"{rec_path.name:52s} already recomputed; skipped"); continue
        before, reward = transform_result_record(doc, json.loads(rep_path.read_text()), rec_path.name)
        print(f"{rec_path.name:52s} reward {before['reward']:.4f} -> {reward:.4f}   n_cases {before['n_cases']} -> {doc['result']['n_cases']}   "
              f"corpus vrmse {before['groups']['ape2d']:.4f} -> {doc['result']['groups']['ape2d']['vrmse']:.4f}")
        if a.write:
            rec_path.write_text(json.dumps(doc, indent=1) + "\n")
    for path in map(Path, a.paths):
        doc = json.loads(path.read_text())
        if "results" in doc and isinstance(doc["results"], list) and any(DUPLICATE in r.get("datasets", {}) for r in doc["results"]):
            for predictor, old, new in transform_trivial(doc, path.name):
                print(f"{path.name:52s} {predictor:28s} reward {old:.4f} -> {new:.4f}")
        elif DUPLICATE in doc.get("datasets", {}):
            if "derivation" in doc:
                print(f"{path.name:52s} already recomputed; skipped"); continue
            before, reward = transform_report(doc, path.name)
            print(f"{path.name:52s} reward {before['reward']:.4f} -> {reward:.4f}   n_cases {before['n_cases']} -> {doc['n_cases']}   "
                  f"corpus vrmse {before['groups']['ape2d']:.4f} -> {doc['groups']['ape2d']['vrmse']:.4f}")
        else:
            print(f"{path.name:52s} no '{DUPLICATE}' dataset; unchanged"); continue
        if a.write:
            path.write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
