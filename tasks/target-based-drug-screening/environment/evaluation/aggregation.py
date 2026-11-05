"""Public, versioned aggregation for the complete screening benchmark."""

from collections import Counter
import math

SCORING_PROTOCOL = "drugclip-dataset-bedroc/v2"
REPORT_SCHEMA = "target-based-drug-screening-metrics/v2"
PRIMARY_METRIC = "equal-dataset mean BEDROC(alpha=80.5)"
TARGET_COUNTS = {"dude": 102, "lit_pcba": 15}
METRIC_KEYS = ("roc_auc", "bedroc_80.5", "ef_0.005", "ef_0.01", "ef_0.02", "ef_0.05")


def summarize_targets(per_target, *, require_full_suite=False):
    """Keep target means; emit a primary score only for complete 102+15 coverage."""
    rows = list(per_target)
    if not rows:
        raise ValueError("no target metrics")
    seen = set()
    for row in rows:
        key = (row["benchmark"], row["target_id"])
        if key[0] not in TARGET_COUNTS or not isinstance(key[1], str) or not key[1]:
            raise ValueError(f"unknown benchmark or invalid target: {key}")
        if key in seen:
            raise ValueError(f"duplicate target metrics: {key}")
        seen.add(key)
        for metric in METRIC_KEYS:
            value = row[metric]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"nonnumeric {metric} for {key}")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {metric} for {key}")
            if metric in ("roc_auc", "bedroc_80.5") and value > 1:
                raise ValueError(f"out-of-range {metric} for {key}")
    counts = Counter(row["benchmark"] for row in rows)
    complete = dict(counts) == TARGET_COUNTS
    if require_full_suite and not complete:
        raise ValueError(f"primary score requires 102 DUD-E and 15 LIT-PCBA targets; got {dict(counts)}")
    benchmarks = {}
    for benchmark in sorted(counts):
        selected = [row for row in rows if row["benchmark"] == benchmark]
        benchmarks[benchmark] = {
            "targets": len(selected),
            **{f"mean_{key}": math.fsum(row[key] for row in selected) / len(selected)
               for key in METRIC_KEYS},
        }
    result = {
        "schema": REPORT_SCHEMA,
        "scoring_protocol": SCORING_PROTOCOL,
        "primary_metric": PRIMARY_METRIC,
        "complete_benchmark": complete,
        "aggregation": {"dataset_weights": {"dude": 0.5, "lit_pcba": 0.5},
                        "within_dataset": "equal-target mean", "bedroc_alpha": 80.5},
        **{f"overall_mean_{key}": math.fsum(row[key] for row in rows) / len(rows)
           for key in METRIC_KEYS},
        "benchmarks": benchmarks,
    }
    if complete:
        result["dataset_balanced_bedroc"] = math.fsum(
            benchmarks[name]["mean_bedroc_80.5"] for name in TARGET_COUNTS) / 2
    return result
