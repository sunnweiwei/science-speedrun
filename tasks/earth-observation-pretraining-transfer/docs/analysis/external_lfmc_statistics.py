"""Pure scoring kernels for the prospective LFMC policy; no file or model access.

This module cannot authorize unsealing or establish provenance. A future scoring
entrypoint must verify the complete input/model/seed lock before calling it.
"""

from collections import Counter
from numbers import Real

import numpy as np


def aggregate_target(source_values):
    """Return lower-clipped mean and explicit exclusions, retaining duplicate values."""
    valid, excluded, counts = [], [], Counter()
    seen = set()
    for row_id, value in source_values:
        if row_id in seen:
            raise ValueError("Duplicate source row ID")
        seen.add(row_id)
        reason = None
        if value is None:
            reason = "missing"
        elif isinstance(value, (bool, np.bool_)):
            reason = "boolean"
        elif not isinstance(value, Real):
            reason = "nonnumeric"
        elif not np.isfinite(value):
            reason = "nonfinite"
        if reason:
            excluded.append({"source_row": row_id, "reason": reason})
            counts[reason] += 1
        else:
            valid.append(max(float(value), 0.0))
            counts["lower_clipped"] += int(value < 0)
    return {
        "value": float(np.mean(valid)) if valid else None,
        "scoreable": bool(valid),
        "source_rows": len(seen),
        "valid_source_rows": len(valid),
        "excluded": excluded,
        "counts": dict(counts),
    }


def paired_statistics(rows, labels, pairs):
    """Score all ordered rows; reject misalignment, missingness and duplicate seeds.

    rows: dictionaries with id, site_id, country. labels and each pair's random /
    trained prediction are dictionaries with ids and values. pairs include seed.
    Array positions never silently realign or drop records.
    """
    ids = [r["id"] for r in rows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Empty or duplicate evaluation IDs")
    seeds = [p["seed"] for p in pairs]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("At least three distinct paired seeds required")

    def values(record):
        if record["ids"] != ids:
            raise ValueError("Labels/predictions must match every fixed row in order")
        array = np.asarray(record["values"], dtype=np.float64)
        if array.shape != (len(ids),) or not np.isfinite(array).all():
            raise ValueError("Missing, wrongly shaped or nonfinite values")
        return array

    targets = values(labels)
    random = np.stack([values(p["random"]) for p in pairs])
    trained = np.stack([values(p["trained"]) for p in pairs])
    absolute_random = np.abs(random - targets)
    absolute_trained = np.abs(trained - targets)
    gains = absolute_random - absolute_trained
    sites, inverse = np.unique([r["site_id"] for r in rows], return_inverse=True)
    counts = np.bincount(inverse)
    seed_totals = np.stack([np.bincount(inverse, weights=g) for g in gains])
    rng = np.random.default_rng(42)
    site_draws = rng.integers(0, len(sites), size=(5000, len(sites)))
    denominator = counts[site_draws].sum(axis=1)
    site_boot = seed_totals.mean(axis=0)[site_draws].sum(axis=1) / denominator
    seed_draws = rng.integers(0, len(seeds), size=(5000, len(seeds)))
    joint_boot = seed_totals[seed_draws[:, :, None], site_draws[:, None, :]].sum(
        axis=(1, 2)
    ) / (len(seeds) * denominator)
    site_interval = np.quantile(site_boot, [0.025, 0.975], method="linear").tolist()
    joint_interval = np.quantile(joint_boot, [0.025, 0.975], method="linear").tolist()
    per_seed = []
    for i, seed in enumerate(seeds):
        metrics = {"seed": seed, "mae_gain": float(gains[i].mean())}
        for name, predictions, errors in (
            ("random", random, absolute_random),
            ("trained", trained, absolute_trained),
        ):
            metrics[name] = {
                "mae": float(errors[i].mean()),
                "rmse": float(np.sqrt(np.mean((predictions[i] - targets) ** 2))),
                "site_macro_mae": float(
                    (np.bincount(inverse, weights=errors[i]) / counts).mean()
                ),
            }
        metrics["site_macro_mae_gain"] = (
            metrics["random"]["site_macro_mae"] - metrics["trained"]["site_macro_mae"]
        )
        per_seed.append(metrics)
    countries = np.array([r["country"] for r in rows])
    country_metrics = []
    for country in np.unique(countries):
        keep = countries == country
        country_metrics.append(
            {
                "country": str(country),
                "rows": int(keep.sum()),
                "sites": len(set(np.array([r["site_id"] for r in rows])[keep])),
                "per_seed_random_mae": absolute_random[:, keep].mean(axis=1).tolist(),
                "per_seed_trained_mae": absolute_trained[:, keep].mean(axis=1).tolist(),
            }
        )
    return {
        "rows": len(ids),
        "sites": len(sites),
        "per_seed": per_seed,
        "mean_seed_mae_gain": float(gains.mean()),
        "site_bootstrap_95_percent_interval": site_interval,
        "seed_and_site_bootstrap_95_percent_interval": joint_interval,
        "bootstrap_replicates": 5000,
        "bootstrap_rng_seed": 42,
        "statistical_criterion_met": bool(
            (gains.mean(axis=1) > 0).all()
            and site_interval[0] > 0
            and joint_interval[0] > 0
        ),
        "countries_descriptive_only": country_metrics,
        "scope": "Statistical kernel only; confirmation additionally requires independently verified cohort/model/seed provenance and complete locked coverage.",
    }
