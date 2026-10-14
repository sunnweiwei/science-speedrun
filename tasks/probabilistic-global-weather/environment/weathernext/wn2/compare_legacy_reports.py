#!/usr/bin/env python3
"""Paired CRPS skill panels; never average incompatible physical variables."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate import sha256

KEY = ("variable", "level_hpa", "lead_hours", "pool_cells")


def compare(candidate, baseline, resamples=2000, seed=0):
    if candidate["status"] != "passed" or baseline["status"] != "passed":
        raise ValueError("Both evaluations must have passed")
    for field in ("protocol_sha256", "weatherbenchx_revision", "evaluator_sha256", "estimator", "truth_values_sha256"):
        if candidate[field] != baseline[field]: raise ValueError(f"Mismatched {field}")
    def index(report):
        result = {}
        for row in report["per_case"]:
            key = tuple(row[k] for k in KEY) + (row["case"],)
            if key in result: raise ValueError("Duplicate case/metric panel")
            if not np.isfinite(row["crps"]) or row["crps"] < -1e-8: raise ValueError("Invalid CRPS")
            result[key] = row["crps"]
        return result
    a, b = index(candidate), index(baseline)
    if not a or a.keys() != b.keys(): raise ValueError("Paired case coverage differs")
    groups = {}
    for key in a: groups.setdefault(key[:-1], []).append(key[-1])
    result = []
    # Reuse the seed for panels: the same dates receive the same resampling.
    for group, dates in groups.items():
        dates = sorted(dates)
        av, bv = [np.array([d[group+(date,)] for date in dates]) for d in (a,b)]
        delta = av-bv
        ci = None
        if len(dates) >= 8:
            selection = np.random.default_rng(seed).integers(0, len(dates), size=(resamples,len(dates)))
            ci = np.quantile(delta[selection].mean(axis=1), [.025,.975]).tolist()
        result.append(dict(zip(KEY,group), cases=len(dates), candidate_crps=float(av.mean()),
                           baseline_crps=float(bv.mean()), skill=float(1-av.mean()/bv.mean()) if bv.mean()>0 else None,
                           paired_crps_difference=float(delta.mean()), paired_date_bootstrap_95pct=ci))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = dict(status="passed", candidate_report_sha256=sha256(args.candidate),
                  baseline_report_sha256=sha256(args.baseline),
                  interpretation="Positive skill / negative paired difference favors candidate; date-bootstrap intervals are descriptive, unadjusted for multiple comparisons and temporal dependence",
                  panels=compare(json.loads(args.candidate.read_text()), json.loads(args.baseline.read_text())))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix(args.output.suffix+".tmp")
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    temporary.replace(args.output)


if __name__ == "__main__": main()
