"""Fixed scores: historical three-suite v1 and expanded eight-suite v2."""

import math

VERSION = "crystal-potential-score-v1"
# Frozen before the scratch trial, from standard-500k-20260923-01 validation.
# HF release 1177b47d5c221602ec13216bf1dc3955ca9dc247,
# records/physical/3ep/score.json. Reuse these constants on validation AND test.
REFERENCE = {
    "F1": 0.8563218390804598,
    "avg_freq_MAE_K": 19.767337214745005,
    "bulk_MAE_GPa": 4.852471740287598,
    "shear_MAE_GPa": 9.989963350875858,
}


def aggregate_score(metrics, *, valid=True):
    """100 is ideal; the calibration profile scores 50; failed runs score zero.

    A partial diagnostic panel has no aggregate. All raw metrics remain available.
    This is a task-specific index, not an official score of the source benchmarks.
    """
    result = {"version": VERSION, "score": 0.0, "higher_is_better": True}
    if not valid:
        return {**result, "reason": "invalid evaluation"}
    if not {"discovery", "phonons", "elasticity"} <= metrics.keys():
        return {**result, "score": None, "reason": "requires all three suites"}
    f1 = metrics["discovery"]["F1"]
    phonon = metrics["phonons"]["avg_freq"]["MAE"]
    bulk = metrics["elasticity"]["bulk_modulus_vrh"]["MAE"]
    shear = metrics["elasticity"]["shear_modulus_vrh"]["MAE"]
    if (
        not all(
            isinstance(v, (int, float)) and math.isfinite(v)
            for v in [f1, phonon, bulk, shear]
        )
        or not 0 <= f1 <= 1
        or min(phonon, bulk, shear) < 0
    ):
        raise ValueError("Invalid primary metric for aggregate score")
    errors = {
        "discovery": (1 - f1) / (1 - REFERENCE["F1"]),
        "phonons": phonon / REFERENCE["avg_freq_MAE_K"],
        "elasticity": 0.5
        * (bulk / REFERENCE["bulk_MAE_GPa"] + shear / REFERENCE["shear_MAE_GPa"]),
    }
    if not all(math.isfinite(v) for v in errors.values()):
        raise ValueError("Nonfinite normalized aggregate error")
    result.update(
        score=100 * math.exp(-sum(math.log1p(v) for v in errors.values()) / 3),
        suite_scores={k: 100 / (1 + v) for k, v in errors.items()},
        normalized_errors=errors,
        reference=REFERENCE,
    )
    return result


VERSION_V2 = "crystal-potential-score-v2"
# Frozen from the rounded VALIDATION column of PR90's MP three-epoch reference
# (uma-mp-finetune/RESULTS.md at 6d61c37). These are scale constants, not targets
# fitted to scratch results or test labels. Reuse unchanged on validation/test.
REFERENCE_V2 = {
    "discovery_F1": 0.856,
    "phonon_avg_freq_MAE_K": 18.2,
    "bulk_modulus_MAE_GPa": 5.560,
    "shear_modulus_MAE_GPa": 7.277,
    "kappa_SRME": 0.455,
    "mof_bulk_modulus_MAPE_pct": 26.6,
    "amorphous_energy_RelRMSE_pct": 8.780,
    "amorphous_force_RelRMSE_pct": 14.8,
    "cleavage_relMAE_pct": 6.598,
    "neb_barrier_MAE_eV": 0.147,
}
SUITE_INPUTS_V2 = {
    "discovery": ("discovery_F1",),
    "phonons": ("phonon_avg_freq_MAE_K",),
    "elasticity": ("bulk_modulus_MAE_GPa", "shear_modulus_MAE_GPa"),
    "kappa": ("kappa_SRME",),
    "mof": ("mof_bulk_modulus_MAPE_pct",),
    "amorphous": ("amorphous_energy_RelRMSE_pct", "amorphous_force_RelRMSE_pct"),
    "cleavage": ("cleavage_relMAE_pct",),
    "neb": ("neb_barrier_MAE_eV",),
}


def expanded_score(metrics, *, valid=True):
    """Consume the grader's flat raw metrics; missing/invalid suites score zero.

    Average normalized errors within each suite, then geometrically average
    100/(1+error) across eight suites. Case counts cannot change suite weights.
    The historical three-suite function above is deliberately unchanged.
    """
    result = {"version": VERSION_V2, "score": 0.0, "higher_is_better": True}
    if not valid:
        return {**result, "reason": "invalid evaluation"}
    normalized = {}
    for key, reference in REFERENCE_V2.items():
        value = metrics.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (key == "discovery_F1" and value > 1)
        ):
            return {**result, "reason": f"missing or invalid primary metric: {key}"}
        normalized[key] = (
            (1 - value) / (1 - reference)
            if key == "discovery_F1"
            else value / reference
        )
        if not math.isfinite(normalized[key]):
            return {**result, "reason": f"nonfinite normalized error: {key}"}
    errors = {
        suite: sum(normalized[key] / len(keys) for key in keys)
        for suite, keys in SUITE_INPUTS_V2.items()
    }
    return {
        **result,
        "score": 100 * math.exp(-sum(math.log1p(v) for v in errors.values()) / 8),
        "suite_scores": {key: 100 / (1 + value) for key, value in errors.items()},
        "normalized_errors": errors,
        "reference": REFERENCE_V2,
    }


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Score a JSON of flat validation metrics"
    )
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            expanded_score(json.loads(args.metrics.read_text())),
            indent=2,
            allow_nan=False,
        )
    )
