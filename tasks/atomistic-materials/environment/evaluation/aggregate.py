"""Fixed scores: historical v1/v2 diagnostics and the primary SSI V4 eight-suite score."""

import json
import math
from pathlib import Path
from statistics import mean

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


VERSION_V4 = "ssi-atomistic-score-v4"
DISCOVERY_POSITIVE_PREVALENCE = 0.1648
DISCOVERY_F1_BASE = 2.0 * DISCOVERY_POSITIVE_PREVALENCE / (1.0 + DISCOVERY_POSITIVE_PREVALENCE)
ANCHORS_JSON_PATH = Path(__file__).with_name("atom_zero_force_anchors.json")
HARNESS_INPUTS_PATH = next((p / "harness" / "ssi_index_v3_inputs.json" for p in Path(__file__).resolve().parents
                            if (p / "harness" / "ssi_index_v3_inputs.json").is_file()),
                           Path(__file__).resolve().parents[3] / "harness" / "ssi_index_v3_inputs.json")
DEFAULT_ATOM_ZERO_FORCE_ANCHORS = {
    "phonon_avg_freq_MAE_K": 419.2277237301322,
    "bulk_modulus_MAE_GPa": 107.020216,
    "shear_modulus_MAE_GPa": 54.186936,
    "kappa_SRME": 2.0,
    "mof_bulk_modulus_MAPE_pct": 100.0,
    "amorphous_energy_RelRMSE_pct": 100.0,
    "amorphous_force_RelRMSE_pct": 100.0,
    "cleavage_relMAE_pct": 100.0,
    "neb_barrier_MAE_eV": 0.6803895877435844,
}


def load_atom_zero_force_anchors(anchors_path: Path | None = None, overrides: dict | None = None) -> dict[str, float]:
    """Load zero-force error denominators in exact parity with harness/ssi_index_v3.py."""
    anchors = dict(DEFAULT_ATOM_ZERO_FORCE_ANCHORS)
    target = anchors_path if anchors_path is not None else ANCHORS_JSON_PATH
    if target.is_file():
        payload = json.loads(target.read_text())
        raw = payload.get("atom_zero_force_anchors", payload)
        anchors.update({k: float(v) for k, v in raw.items() if k in anchors})
    if anchors_path is None and HARNESS_INPUTS_PATH.is_file():
        harness_payload = json.loads(HARNESS_INPUTS_PATH.read_text())
        anchors.update(
            {k: float(v) for k, v in harness_payload.get("atom_zero_force_anchors", {}).items() if k in anchors}
        )
    if overrides:
        anchors.update({k: float(v) for k, v in overrides.items() if k in anchors})
    return anchors


ATOM_ZERO_FORCE_ANCHORS = load_atom_zero_force_anchors()
ATOM_ERRORS = ATOM_ZERO_FORCE_ANCHORS


def _valid_metric(value, *, is_f1=False) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0.0
        and (not is_f1 or value <= 1.0)
    )


def ssi_v4_score(
    metrics: dict,
    *,
    valid: bool = True,
    anchors: dict | None = None,
    anchors_path: Path | None = None,
) -> dict:
    """Primary SSI V4 / V3 eight-suite equal arithmetic mean (1/8 per suite) in [0, 1]."""
    result = {
        "version": VERSION_V4,
        "score": 0.0,
        "signed_score": 0.0,
        "higher_is_better": True,
    }
    if not valid:
        return {**result, "reason": "invalid evaluation"}
    atom_errors = load_atom_zero_force_anchors(anchors_path=anchors_path, overrides=anchors)
    f1 = metrics.get("discovery_F1")
    if not _valid_metric(f1, is_f1=True):
        return {**result, "reason": "missing or invalid primary metric: discovery_F1"}
    f1 = float(f1)
    v = {}
    diagnostics = {"discovery_F1": f1}
    for key, denom in atom_errors.items():
        val = metrics.get(key)
        if not _valid_metric(val):
            return {**result, "reason": f"missing or invalid primary metric: {key}"}
        val = float(val)
        if not (math.isfinite(denom) and denom > 0.0):
            return {**result, "reason": f"invalid anchor denominator: {key}"}
        diagnostics[key] = val
        v[key] = 1.0 - val / denom
    terms = {
        "Discovery": (f1 - DISCOVERY_F1_BASE) / (1.0 - DISCOVERY_F1_BASE),
        "Phonons": v["phonon_avg_freq_MAE_K"],
        "Elasticity": mean((v["bulk_modulus_MAE_GPa"], v["shear_modulus_MAE_GPa"])),
        "Thermal conductivity": v["kappa_SRME"],
        "Metal\u2013organic frameworks": v["mof_bulk_modulus_MAPE_pct"],
        "Amorphous materials": mean((v["amorphous_energy_RelRMSE_pct"], v["amorphous_force_RelRMSE_pct"])),
        "Cleavage": v["cleavage_relMAE_pct"],
        "Migration barriers": v["neb_barrier_MAE_eV"],
    }
    raw = float(mean(terms.values()))
    clipped = max(0.0, min(1.0, raw))
    return {
        **result,
        "score": clipped,
        "signed_score": raw,
        "components": terms,
        "diagnostics": diagnostics,
        "anchors": {"discovery_f1_base": DISCOVERY_F1_BASE, **atom_errors},
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Score a JSON of flat validation metrics"
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--version", choices=("v4", "v2"), default="v4")
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    out = ssi_v4_score(data) if args.version == "v4" else expanded_score(data)
    print(json.dumps(out, indent=2, allow_nan=False))
