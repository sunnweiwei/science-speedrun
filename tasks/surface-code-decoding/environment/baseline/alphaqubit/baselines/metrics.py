"""Surface-code metrics without mixing round counts, bases, or chip patches.

The fitted model is log(1 - 2 P_fail(r)) = intercept + slope * r,
with epsilon = (1 - exp(slope)) / 2. Round one is excluded. Confidence
intervals condition on the observed circuits and assume independent shots;
they do not account for hardware drift, common-mode events, or seed variance.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import re
from statistics import NormalDist
from typing import Any

import numpy as np
from numpy.typing import NDArray


FIT_ROUNDS = tuple(range(3, 26, 2))
CONDITION_PATTERN = re.compile(
    r"surface_code_b(?P<basis>[XZ])_d(?P<distance>[35])_r(?P<rounds>\d+)_"
    r"center_(?P<row>\d+)_(?P<col>\d+)"
)


@dataclass(frozen=True)
class Condition:
    basis: str
    distance: int
    rounds: int
    row: int
    col: int

    @property
    def group(self) -> str:
        return f"d{self.distance}_b{self.basis}_center_{self.row}_{self.col}"


def parse_condition(name: str) -> Condition:
    match = CONDITION_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Invalid Sycamore surface-code condition: {name!r}")
    fields = match.groupdict()
    return Condition(
        basis=fields["basis"],
        distance=int(fields["distance"]),
        rounds=int(fields["rounds"]),
        row=int(fields["row"]),
        col=int(fields["col"]),
    )


def _validate_count(record: Mapping[str, Any], name: str) -> tuple[int, int]:
    errors, total = record["errors"], record["total"]
    if isinstance(errors, bool) or isinstance(total, bool):
        raise ValueError(f"{name}: counts must be integers, not booleans")
    if int(errors) != errors or int(total) != total:
        raise ValueError(f"{name}: counts must be integers")
    if total <= 0 or errors < 0 or errors > total:
        raise ValueError(f"{name}: require 0 <= errors <= total and total > 0")
    return int(errors), int(total)


def wilson_interval(errors: int, total: int, confidence: float = 0.95) -> list[float]:
    """Wilson score interval, including the zero-error case."""
    _validate_count({"errors": errors, "total": total}, "Wilson interval")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = errors / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(max(0, center - radius)), float(min(1, center + radius))]


def _fit_arrays(
    rounds: NDArray[np.float64], errors: NDArray[Any], total: NDArray[Any]
) -> dict[str, NDArray[np.float64]]:
    """Vectorized weighted least squares; final axis indexes round counts."""
    probability = np.asarray(errors, dtype=np.float64) / total
    fidelity = 1 - 2 * probability
    if not np.all(np.isfinite(fidelity)) or np.any(fidelity <= 0):
        raise ValueError("Log-fidelity fit requires finite P_fail < 0.5 at every round")
    log_fidelity = np.log(fidelity)
    # A Jeffreys adjustment is used only to avoid zero variance at p=0.
    variance_probability = (np.asarray(errors) + 0.5) / (total + 1)
    variance_log = 4 * variance_probability * (1 - variance_probability) / total / fidelity**2
    weights = 1 / variance_log
    weight_sum = np.sum(weights, axis=-1)
    mean_x = np.sum(weights * rounds, axis=-1) / weight_sum
    mean_y = np.sum(weights * log_fidelity, axis=-1) / weight_sum
    centered_x = rounds - np.expand_dims(mean_x, -1)
    centered_y = log_fidelity - np.expand_dims(mean_y, -1)
    denominator = np.sum(weights * centered_x**2, axis=-1)
    slope = np.sum(weights * centered_x * centered_y, axis=-1) / denominator
    intercept = mean_y - slope * mean_x
    prediction = np.expand_dims(intercept, -1) + np.expand_dims(slope, -1) * rounds
    residual = log_fidelity - prediction
    chi_square = np.sum(weights * residual**2, axis=-1)
    ler = -np.expm1(slope) / 2
    output = {
        "logical_error_per_round": ler,
        "slope": slope,
        "intercept": intercept,
        "reduced_chi_square": chi_square / (len(rounds) - 2),
        "weighted_log_fidelity_rmse": np.sqrt(chi_square / weight_sum),
    }
    if not all(np.all(np.isfinite(value)) for value in output.values()):
        raise ValueError("Nonfinite logical-error fit; no conditions were dropped")
    return output


def _group_counts(
    counts: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[tuple[str, Condition, int, int]]]:
    if not counts:
        raise ValueError("At least one complete condition group is required")
    groups: dict[str, list[tuple[str, Condition, int, int]]] = defaultdict(list)
    for name, record in sorted(counts.items()):
        condition = parse_condition(name)
        if condition.rounds not in (1, *FIT_ROUNDS):
            raise ValueError(f"{name}: unsupported round count")
        errors, total = _validate_count(record, name)
        groups[condition.group].append((name, condition, errors, total))
    for group, rows in groups.items():
        actual = sorted(row[1].rounds for row in rows if row[1].rounds != 1)
        if actual != list(FIT_ROUNDS):
            raise ValueError(f"{group}: expected each fit round {FIT_ROUNDS}, got {actual}")
        rows.sort(key=lambda row: row[1].rounds)
    return dict(groups)


def _fit_diagnostics(
    rounds: NDArray[np.float64], errors: NDArray[Any], total: NDArray[Any],
    fit: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose poor decay fits without silently filtering data or clipping rates."""
    fidelity = 1 - 2 * errors / total
    y = np.log(fidelity)
    variance_probability = (errors + 0.5) / (total + 1)
    weights = total * fidelity**2 / (4 * variance_probability * (1 - variance_probability))
    residual = y - (fit["intercept"] + fit["slope"] * rounds)
    constant = bool(np.ptp(y) == 0)
    weighted_r_squared = None if constant else float(
        1 - np.sum(weights * residual**2)
        / np.sum(weights * (y - np.average(y, weights=weights))**2)
    )
    unweighted_r_squared = None if constant else float(
        1 - np.sum(residual**2) / np.sum((y - np.mean(y))**2)
    )
    f0 = float(np.exp(fit["intercept"]))
    values = [f0, weighted_r_squared, unweighted_r_squared]
    if any(value is not None and not np.isfinite(value) for value in values):
        raise ValueError("Nonfinite fit diagnostics; no conditions were dropped")
    flags: list[str] = []
    if fit["logical_error_per_round"] < 0:
        flags.append("unphysical_negative_logical_error_per_round")
    if f0 > 1:
        flags.append("extrapolated_zero_round_fidelity_above_one")
    if f0 < 0.1:
        flags.append("extrapolated_zero_round_fidelity_below_0.1")
    if constant:
        flags.append("constant_log_fidelity_r_squared_undefined")
    return {
        "zero_round_fidelity": f0,
        "weighted_log_fidelity_r_squared": weighted_r_squared,
        "unweighted_log_fidelity_r_squared": unweighted_r_squared,
        "diagnostic_flags": flags,
    }


def _interval(samples: NDArray[Any]) -> list[float]:
    if not np.all(np.isfinite(samples)):
        raise ValueError("Nonfinite bootstrap results; no samples were dropped")
    return np.quantile(samples, [0.025, 0.975]).tolist()


def _aggregate(
    group_fits: Mapping[str, Mapping[str, Any]],
    group_samples: Mapping[str, NDArray[Any]],
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    macro: dict[str, Any] = {}
    samples: dict[str, NDArray[Any]] = {}
    for distance in (3, 5):
        names = [name for name, fit in group_fits.items() if fit["distance"] == distance]
        if not names:
            continue
        key = f"d{distance}"
        macro[key] = {
            "logical_error_per_round": float(np.mean([
                group_fits[name]["logical_error_per_round"] for name in names
            ])),
            "groups": names,
            "aggregation": "equal weight over basis/center fits, without pooling shot probabilities",
            "diagnostic_flags_by_group": {
                name: group_fits[name]["diagnostic_flags"]
                for name in names if group_fits[name].get("diagnostic_flags")
            },
        }
        if group_samples:
            samples[key] = np.mean([group_samples[name] for name in names], axis=0)
            macro[key]["ci95"] = _interval(samples[key])
    if "d3" in macro and "d5" in macro:
        positive_rates = all(macro[key]["logical_error_per_round"] > 0 for key in ("d3", "d5"))
        macro["error_suppression_d3_over_d5"] = {
            "value": (
                macro["d3"]["logical_error_per_round"] / macro["d5"]["logical_error_per_round"]
                if positive_rates else None
            ),
            "interpretation": "descriptive ratio across the measured d3 patches and d5 patch",
            "diagnostic_flags": [] if positive_rates else ["suppression_undefined_for_nonpositive_macro_logical_error"],
        }
        if samples:
            if not positive_rates or np.any(samples["d5"] <= 0) or np.any(samples["d3"] <= 0):
                macro["error_suppression_d3_over_d5"]["ci95"] = None
                macro["error_suppression_d3_over_d5"]["diagnostic_flags"].append(
                    "ratio_ci_undefined_due_to_nonpositive_logical_error_no_bootstrap_draws_dropped"
                )
            else:
                samples["error_suppression_d3_over_d5"] = samples["d3"] / samples["d5"]
                macro["error_suppression_d3_over_d5"]["ci95"] = _interval(
                    samples["error_suppression_d3_over_d5"]
                )
    return macro, samples


def summarize_conditions(
    counts: Mapping[str, Mapping[str, Any]], *, bootstrap_samples: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """Summarize ``{condition_name: {errors: int, total: int}}``.

    Every included basis/center group must contain r=3,5,...,25. Round one
    may be present for diagnostics, but never enters the slope fit. Missing
    groups are allowed for single-patch model training; missing fit rounds
    inside an included group are an error. Full-suite enforcement belongs
    to the caller. Set bootstrap_samples=0 for validation-only point scores.
    """
    if bootstrap_samples < 0 or int(bootstrap_samples) != bootstrap_samples:
        raise ValueError("bootstrap_samples must be a nonnegative integer")
    groups = _group_counts(counts)
    rng = np.random.default_rng(seed)
    condition_records: dict[str, Any] = {}
    fits: dict[str, Any] = {}
    samples: dict[str, NDArray[Any]] = {}
    for group, rows in groups.items():
        for name, condition, errors, total in rows:
            condition_records[name] = {
                "distance": condition.distance,
                "basis": condition.basis,
                "center": [condition.row, condition.col],
                "rounds": condition.rounds,
                "errors": errors,
                "total": total,
                "shot_error_rate": errors / total,
                "shot_error_rate_ci95": wilson_interval(errors, total),
                "included_in_fit": condition.rounds != 1,
            }
        fit_rows = [row for row in rows if row[1].rounds != 1]
        rounds = np.array([row[1].rounds for row in fit_rows], dtype=np.float64)
        errors = np.array([row[2] for row in fit_rows], dtype=np.int64)
        total = np.array([row[3] for row in fit_rows], dtype=np.int64)
        condition = fit_rows[0][1]
        fits[group] = {
            "distance": condition.distance,
            "basis": condition.basis,
            "center": [condition.row, condition.col],
            "fit_rounds": list(FIT_ROUNDS),
            "degrees_of_freedom": len(FIT_ROUNDS) - 2,
            **{key: float(value) for key, value in _fit_arrays(rounds, errors, total).items()},
        }
        fits[group].update(_fit_diagnostics(rounds, errors, total, fits[group]))
        if bootstrap_samples:
            draw = rng.binomial(total, errors / total, size=(bootstrap_samples, len(rounds)))
            samples[group] = _fit_arrays(rounds, draw, total)["logical_error_per_round"]
            fits[group]["logical_error_per_round_ci95"] = _interval(samples[group])
    macro, _ = _aggregate(fits, samples)
    return {
        "metric": "weighted log-fidelity slope with free intercept; epsilon=(1-exp(slope))/2",
        "diagnostic_policy": (
            "F0=exp(intercept) extrapolates fidelity to zero rounds. F0<0.1 and F0>1 are "
            "flagged without filtering. Negative fitted LER is retained and flagged; "
            "do not interpret it as a physically valid error rate. R-squared is null "
            "for constant log fidelity. Neither counts nor bootstrap draws are dropped."
        ),
        "uncertainty": {
            "method": "per-condition parametric binomial bootstrap with complete refitting",
            "confidence": 0.95,
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
            "assumptions": "independent shots and conditions; conditional on circuits; no drift or model-seed uncertainty",
        },
        "per_condition": condition_records,
        "per_group": fits,
        "macro": macro,
    }


def compare_paired_conditions(
    errors_a: Mapping[str, NDArray[Any]],
    errors_b: Mapping[str, NDArray[Any]],
    *,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """CI of A minus B using matched-shot error outcomes, not independent CIs.

    Arrays must align to the same shot indices. Multinomial resampling of
    (A correct/B correct, A correct/B wrong, A wrong/B correct, both wrong)
    is equivalent to independent-shot paired nonparametric bootstrapping.
    """
    if set(errors_a) != set(errors_b):
        raise ValueError("Paired methods must cover identical conditions")
    if bootstrap_samples <= 0:
        raise ValueError("Paired comparison requires positive bootstrap_samples")
    rng = np.random.default_rng(seed)
    counts_a: dict[str, dict[str, int]] = {}
    counts_b: dict[str, dict[str, int]] = {}
    draws_a: dict[str, NDArray[Any]] = {}
    draws_b: dict[str, NDArray[Any]] = {}
    for name in sorted(errors_a):
        a, b = np.asarray(errors_a[name]), np.asarray(errors_b[name])
        if a.ndim != 1 or a.shape != b.shape or not a.size:
            raise ValueError(f"{name}: paired arrays must have matching nonempty 1D shapes")
        if not np.all(np.isin(a, [0, 1])) or not np.all(np.isin(b, [0, 1])):
            raise ValueError(f"{name}: error arrays must be binary")
        counts_a[name] = {"errors": int(a.sum()), "total": a.size}
        counts_b[name] = {"errors": int(b.sum()), "total": b.size}
        outcomes = np.bincount(2 * a.astype(np.int8) + b.astype(np.int8), minlength=4)
        draws = rng.multinomial(a.size, outcomes / a.size, size=bootstrap_samples)
        draws_a[name] = draws[:, 2] + draws[:, 3]
        draws_b[name] = draws[:, 1] + draws[:, 3]
    summary_a = summarize_conditions(counts_a, bootstrap_samples=0)
    summary_b = summarize_conditions(counts_b, bootstrap_samples=0)
    samples_by_method: list[dict[str, NDArray[Any]]] = []
    for counts, draws, summary in (
        (counts_a, draws_a, summary_a), (counts_b, draws_b, summary_b)
    ):
        group_samples: dict[str, NDArray[Any]] = {}
        for group, rows in _group_counts(counts).items():
            rows = [row for row in rows if row[1].rounds != 1]
            rounds = np.array([row[1].rounds for row in rows], dtype=np.float64)
            total = np.array([row[3] for row in rows])
            draw = np.stack([draws[row[0]] for row in rows], axis=-1)
            group_samples[group] = _fit_arrays(rounds, draw, total)["logical_error_per_round"]
        _, macro_samples = _aggregate(summary["per_group"], group_samples)
        samples_by_method.append({**group_samples, **macro_samples})
    comparisons: dict[str, Any] = {}
    for group in samples_by_method[0]:
        if group.startswith("d") and "_b" in group:
            point_a = summary_a["per_group"][group]["logical_error_per_round"]
            point_b = summary_b["per_group"][group]["logical_error_per_round"]
        elif group in ("d3", "d5"):
            point_a = summary_a["macro"][group]["logical_error_per_round"]
            point_b = summary_b["macro"][group]["logical_error_per_round"]
        else:
            continue
        comparisons[group] = {
            "logical_error_per_round_a_minus_b": point_a - point_b,
            "ci95": _interval(samples_by_method[0][group] - samples_by_method[1][group]),
        }
    return {
        "method": "paired independent-shot multinomial bootstrap, stratified by condition",
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "negative_difference_favors": "A",
        "comparisons": comparisons,
    }
