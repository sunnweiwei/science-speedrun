"""Trusted scorer for first-recorded disease time and mortality risk."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from labels import death_label
from panel_scoring import PRIMARY as PANEL_PRIMARY, history_code, panel_score

HORIZONS = tuple(range(30, 361, 30)) + (365,)
MAX_NORMALIZATION_WEIGHT = 1000.0


def mean_or_none(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def percentile_interval(values) -> list[float] | None:
    available = [x for x in values if x is not None]
    return np.quantile(available, [.025, .975]).tolist() if available else None


def weighted_auc(y: np.ndarray, p: np.ndarray, weight: np.ndarray) -> float | None:
    """Exact tie-aware weighted AUROC via prediction-value groups."""
    positive = float(weight[y == 1].sum())
    negative = float(weight[y == 0].sum())
    if not positive or not negative:
        return None
    _, group = np.unique(p, return_inverse=True)
    positive_weight = np.bincount(group, weights=weight * (y == 1))
    negative_weight = np.bincount(group, weights=weight * (y == 0))
    lower_negative = np.cumsum(negative_weight) - negative_weight
    concordance = np.sum(positive_weight * (lower_negative + .5 * negative_weight))
    return float(concordance / (positive * negative))


def auc_and_average_precision(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Tie-aware AUROC and AP from one stable prediction sort."""
    order = np.argsort(p, kind="stable")
    y_sorted, p_sorted = y[order], p[order]
    starts = np.r_[0, np.flatnonzero(p_sorted[1:] != p_sorted[:-1]) + 1]
    positive_group = np.add.reduceat(y_sorted, starts).astype(np.float64)
    group_size = np.diff(np.r_[starts, len(y)]).astype(np.float64)
    negative_group = group_size - positive_group
    positive, negative = float(positive_group.sum()), float(negative_group.sum())
    lower_negative = np.cumsum(negative_group) - negative_group
    auc = float(np.sum(positive_group * (lower_negative + .5 * negative_group))
                / (positive * negative))
    descending_positive = positive_group[::-1]
    descending_size = group_size[::-1]
    cumulative_positive = np.cumsum(descending_positive)
    precision = cumulative_positive / np.cumsum(descending_size)
    ap = float(np.sum(precision * descending_positive) / positive)
    return auc, ap


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict | None:
    if not len(y):
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    ece, nonempty = 0.0, 0
    for k in range(bins):
        use = bucket == k
        if use.any():
            nonempty += 1
            ece += float(use.mean()) * abs(float(p[use].mean()) - float(y[use].mean()))
    observed, predicted = float(y.mean()), float(p.mean())
    return {"observed_rate": observed, "mean_prediction": predicted,
            "calibration_in_the_large": predicted - observed,
            "expected_calibration_error_10bin": ece, "nonempty_bins": nonempty}


def metrics(y: np.ndarray, p: np.ndarray, valid: np.ndarray) -> dict:
    y, p = y[valid].astype(np.uint8), p[valid].astype(np.float64)
    n, positives = len(y), int(y.sum())
    result = {"valid": n, "positives": positives, "negatives": n - positives,
              "auroc": None, "auprc": None, "brier": None,
              "calibration": calibration(y, p), "reason": None}
    if not n:
        result["reason"] = "no_valid_labels"
        return result
    result["brier"] = float(np.mean((p - y) ** 2))
    if not positives or positives == n:
        result["reason"] = "single_class"
        return result
    result["auroc"], result["auprc"] = auc_and_average_precision(y, p)
    return result


def read_predictions(path: Path, sample_ids: list[str], disease_count: int,
                     horizons: tuple[int, ...] = HORIZONS) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["sample_id", "disease_risk", "death_risk"])
    if table.num_rows != len(sample_ids):
        raise ValueError("Prediction row count differs from public sample count")
    by_id = {sid: i for i, sid in enumerate(sample_ids)}
    disease = np.empty((len(sample_ids), disease_count, len(horizons)), dtype=np.float32)
    death = np.empty((len(sample_ids), len(horizons)), dtype=np.float32)
    seen = set()
    for row in table.to_pylist():
        sid = row["sample_id"]
        if sid not in by_id or sid in seen:
            raise ValueError("Unknown or repeated sample_id")
        seen.add(sid)
        dr = np.asarray(row["disease_risk"], dtype=np.float32)
        mr = np.asarray(row["death_risk"], dtype=np.float32)
        if dr.size != disease_count * len(horizons) or mr.size != len(horizons):
            raise ValueError("Prediction dimensions do not match frozen diseases and direct time grid")
        disease[by_id[sid]] = dr.reshape(disease_count, len(horizons))
        death[by_id[sid]] = mr
    if seen != set(sample_ids):
        raise ValueError("Missing prediction sample_id")
    if not np.isfinite(disease).all() or not np.isfinite(death).all():
        raise ValueError("Non-finite probability")
    if (disease < 0).any() or (disease > 1).any() or (death < 0).any() or (death > 1).any():
        raise ValueError("Probability outside [0,1]")
    if (np.diff(disease, axis=2) < -1e-6).any() or (np.diff(death, axis=1) < -1e-6).any():
        raise ValueError("Cumulative risk decreases across horizons")
    return disease, death


def disease_time_brier(labels: np.ndarray, prediction: np.ndarray,
                       valid: np.ndarray) -> tuple[float | None, list[dict]]:
    """Equal-weight time average per disease, then macro average diseases."""
    per_disease = []
    for j in range(labels.shape[1]):
        by_time = []
        for k in range(labels.shape[2]):
            use = valid[:, j, k]
            by_time.append(float(np.mean((prediction[use, j, k] - labels[use, j, k]) ** 2))
                           if use.any() else None)
        available = [x for x in by_time if x is not None]
        per_disease.append({"time_brier": float(np.mean(available)) if available else None,
                            "timepoint_brier": by_time,
                            "reason": None if available else "no_valid_labels"})
    return mean_or_none([x["time_brier"] for x in per_disease]), per_disease


def disease_time_discrimination(labels: np.ndarray, prediction: np.ndarray,
                                valid: np.ndarray) -> tuple[dict, list[dict]]:
    """Disease-first, time-averaged discrimination on the frozen valid cells.

    A disease/time cell with no positive or no negative label is unavailable.  It
    is deliberately not assigned a neutral value: available time points are
    averaged within each disease, then available diseases are averaged equally.
    """
    per_disease = []
    for j in range(labels.shape[1]):
        by_time = []
        for k in range(labels.shape[2]):
            value = metrics(labels[:, j, k], prediction[:, j, k], valid[:, j, k])
            by_time.append(value)
        available_auc = [cell["auroc"] for cell in by_time if cell["auroc"] is not None]
        available_auprc = [cell["auprc"] for cell in by_time if cell["auprc"] is not None]
        per_disease.append({
            "time_averaged_auroc": mean_or_none(available_auc),
            "time_averaged_auprc": mean_or_none(available_auprc),
            "valid_timepoint_cells": len(available_auc),
            "timepoints": by_time,
            "reason": None if available_auc else "no_two_class_timepoint",
        })
    available = [item for item in per_disease if item["time_averaged_auroc"] is not None]
    scorable_cells = [cell for item in available for cell in item["timepoints"]
                      if cell["auroc"] is not None]
    summary = {
        "macro_time_averaged_auroc": mean_or_none(
            [item["time_averaged_auroc"] for item in available]),
        "macro_time_averaged_auprc": mean_or_none(
            [item["time_averaged_auprc"] for item in available]),
        "valid_diseases": len(available),
        "frozen_diseases": labels.shape[1],
        "valid_disease_time_cells": len(scorable_cells),
        "frozen_disease_time_cells": labels.shape[1] * labels.shape[2],
        "positive_support": int(sum(cell["positives"] for cell in scorable_cells)),
        "negative_support": int(sum(cell["negatives"] for cell in scorable_cells)),
    }
    return summary, per_disease


def weighted_macro_time_auroc(labels: np.ndarray, prediction: np.ndarray,
                              valid: np.ndarray, weight: np.ndarray) -> float | None:
    """Recompute the primary metric for a patient-bootstrap frequency vector."""
    disease_auc = []
    present = weight > 0
    for j in range(labels.shape[1]):
        time_auc = []
        for k in range(labels.shape[2]):
            use = valid[:, j, k] & present
            value = weighted_auc(labels[use, j, k], prediction[use, j, k], weight[use])
            if value is not None:
                time_auc.append(value)
        if time_auc:
            disease_auc.append(float(np.mean(time_auc)))
    return mean_or_none(disease_auc)


def normalized_disease_time_brier(labels: np.ndarray, prediction: np.ndarray,
                                  valid: np.ndarray,
                                  denominators: np.ndarray) -> tuple[float | None, list[dict]]:
    """Divide each disease/time Brier by its frozen train-only reference."""
    if denominators.shape != labels.shape[1:] or not np.isfinite(denominators).all():
        raise ValueError("Normalization denominator dimensions or values are invalid")
    if (denominators <= 0).any():
        raise ValueError("Normalization denominators must be positive")
    _, raw = disease_time_brier(labels, prediction, valid)
    per_disease = []
    for j, detail in enumerate(raw):
        by_time = [None if value is None else value / float(denominators[j, k])
                   for k, value in enumerate(detail["timepoint_brier"])]
        available = [value for value in by_time if value is not None]
        per_disease.append({
            "normalized_time_brier": float(np.mean(available)) if available else None,
            "normalized_timepoint_brier": by_time,
            "raw_time_brier": detail["time_brier"],
            "raw_timepoint_brier": detail["timepoint_brier"],
            "reason": None if available else "no_valid_labels",
        })
    return mean_or_none([x["normalized_time_brier"] for x in per_disease]), per_disease


def read_normalization(path: Path, disease_codes: list[str]) -> tuple[np.ndarray, dict]:
    table = pq.read_table(path)
    required = {"disease_code", "horizon_days", "age_sex_oof_brier",
                "denominator_brier", "normalization_weight", "floor_applied"}
    if not required.issubset(table.schema.names):
        raise ValueError("Train-only normalization artifact has missing fields")
    index = {code: j for j, code in enumerate(disease_codes)}
    denominators = np.full((len(disease_codes), len(HORIZONS)), np.nan, dtype=np.float64)
    raw_reference = np.full_like(denominators, np.nan)
    horizon_index = {h: k for k, h in enumerate(HORIZONS)}
    floor_applied = np.zeros_like(denominators, dtype=np.bool_)
    for row in table.to_pylist():
        if row["disease_code"] not in index or row["horizon_days"] not in horizon_index:
            raise ValueError("Normalization artifact contains an unknown disease or horizon")
        j, k = index[row["disease_code"]], horizon_index[row["horizon_days"]]
        if np.isfinite(denominators[j, k]):
            raise ValueError("Duplicate disease/time normalization cell")
        denominators[j, k] = row["denominator_brier"]
        raw_reference[j, k] = row["age_sex_oof_brier"]
        floor_applied[j, k] = row["floor_applied"]
        if not np.isclose(row["normalization_weight"], 1.0 / denominators[j, k]):
            raise ValueError("Normalization weight and denominator disagree")
    if not np.isfinite(denominators).all() or (denominators <= 0).any():
        raise ValueError("Normalization artifact does not cover the frozen score grid")
    if (1.0 / denominators).max() > MAX_NORMALIZATION_WEIGHT + 1e-9:
        raise ValueError("Normalization artifact exceeds the frozen weight cap")
    metadata = {"cells": int(denominators.size),
                "floor_applied_cells": int(floor_applied.sum()),
                "minimum_denominator": float(denominators.min()),
                "maximum_weight": float((1.0 / denominators).max()),
                "mean_train_reference_ratio": float(np.mean(raw_reference / denominators))}
    return denominators, metadata


def _date(value):
    return value.date() if hasattr(value, "time") else value


def _full_year(row: dict) -> bool:
    cutoff = _date(row["cutoff"])
    return (cutoff + timedelta(days=365) <= _date(row["diagnosis_observation_end"])
            and cutoff + timedelta(days=365) <= _date(row["death_observation_end"]))


def score(input_dir: Path, trusted_dir: Path, prediction_file: Path,
          bootstrap: int = 0, normalization_file: Path | None = None) -> dict:
    manifest = json.loads((input_dir / "manifest.json").read_text())
    if manifest.get("horizons_days") != list(HORIZONS):
        raise ValueError("Submission must predict every frozen time point directly")
    disease_codes = manifest["scored_diseases"]
    denominators, normalization_metadata = read_normalization(
        normalization_file or trusted_dir / "risk_normalization.parquet", disease_codes)
    public_samples = pq.read_table(input_dir / "samples.parquet").to_pylist()
    sample_ids = [r["sample_id"] for r in public_samples]
    sample_index = {sid: i for i, sid in enumerate(sample_ids)}
    disease_index = {code: i for i, code in enumerate(disease_codes)}
    if len(sample_index) != len(sample_ids):
        raise ValueError("Duplicate public sample ID")
    prediction, death_prediction = read_predictions(
        prediction_file, sample_ids, len(disease_codes))
    db = duckdb.connect()
    db.register("sample_order", pa.table({"sample_id": sample_ids,
        "i": np.arange(len(sample_ids), dtype=np.int32)}))
    db.register("disease_order", pa.table({"disease_code": disease_codes,
        "j": np.arange(len(disease_codes), dtype=np.int16)}))
    label_path = str(trusted_dir / "label_windows.parquet")
    total, unique = db.execute("""SELECT count(*),count(DISTINCT (sample_id,disease_code))
      FROM read_parquet(?)""", [label_path]).fetchone()
    expected = len(sample_ids) * len(disease_codes)
    if total != expected or unique != expected:
        db.close()
        raise ValueError("Trusted label keys are duplicated or do not cover the public grid")
    fields = [f"l.{field}_{horizon}" for horizon in HORIZONS
              for field in ("disease_label", "disease_valid")]
    aligned = db.execute(f"""SELECT {','.join(fields)} FROM sample_order s
      CROSS JOIN disease_order d JOIN read_parquet(?) l USING(sample_id,disease_code)
      ORDER BY s.i,d.j""", [label_path]).fetchnumpy()
    db.close()
    labels = np.stack([aligned[f"disease_label_{h}"] for h in HORIZONS], axis=1)
    valid = np.stack([aligned[f"disease_valid_{h}"] for h in HORIZONS], axis=1)
    labels = labels.reshape(prediction.shape).astype(np.uint8)
    valid = valid.reshape(prediction.shape).astype(bool)

    boundaries = {r["sample_id"]: r for r in pq.read_table(
        trusted_dir / "sample_boundaries.parquet").to_pylist()
        if r["sample_id"] in sample_index}
    if len(boundaries) != len(sample_ids):
        raise ValueError("Missing trusted observation boundary")
    death_y = np.zeros((len(sample_ids), len(HORIZONS)), dtype=np.uint8)
    death_valid = np.zeros_like(death_y, dtype=np.bool_)
    primary_patients = np.zeros(len(sample_ids), dtype=np.bool_)
    for sid, i in sample_index.items():
        row = boundaries[sid]
        primary_patients[i] = _full_year(row)
        for k, horizon in enumerate(HORIZONS):
            death_y[i, k], death_valid[i, k] = death_label(row, horizon)

    primary_valid = valid & primary_patients[:, None, None]
    history = [set() for _ in sample_ids]
    for sid, code in duckdb.connect().execute("""SELECT DISTINCT sample_id, code
        FROM read_parquet(?) WHERE event_type='diagnosis'""",
            [str(input_dir / "events.parquet")]).fetchall():
        if sid in sample_index:
            history[sample_index[sid]].add(history_code(code))
    k365 = HORIZONS.index(365)
    panel = panel_score(labels[:, :, k365], valid[:, :, k365], prediction[:, :, k365],
                        primary_patients, disease_codes, history)
    discrimination, discrimination_by_disease = disease_time_discrimination(
        labels, prediction, primary_valid)
    primary_score = discrimination["macro_time_averaged_auroc"]
    raw_primary_score, _ = disease_time_brier(labels, prediction, primary_valid)
    normalized_brier, brier_by_disease = normalized_disease_time_brier(
        labels, prediction, primary_valid, denominators)
    death_primary = [metrics(death_y[:, k], death_prediction[:, k],
                             death_valid[:, k] & primary_patients)
                     for k in range(len(HORIZONS))]
    death_time_brier = mean_or_none([x["brier"] for x in death_primary])

    by_horizon, per_disease = {}, {"time_brier": {}, "discrimination": {}}
    for k, horizon in enumerate(HORIZONS):
        one = {}
        for j, code in enumerate(disease_codes):
            horizon_valid = valid[:, j, k] & primary_patients
            value = discrimination_by_disease[j]["timepoints"][k]
            value["coverage_in_primary_cohort"] = float(
                horizon_valid.sum() / max(primary_patients.sum(), 1))
            one[code] = value
        auc = [x for x in one.values() if x["auroc"] is not None]
        calibrated = [x["calibration"] for x in one.values() if x["calibration"] is not None]
        by_horizon[str(horizon)] = {
            "macro_auroc": mean_or_none([x["auroc"] for x in auc]),
            "macro_auprc": mean_or_none([x["auprc"] for x in auc]),
            "macro_brier": mean_or_none([x["brier"] for x in one.values()]),
            "macro_normalized_brier": mean_or_none([
                value["brier"] / denominators[disease_index[code], k]
                for code, value in one.items() if value["brier"] is not None]),
            "macro_expected_calibration_error_10bin": mean_or_none(
                [x["expected_calibration_error_10bin"] for x in calibrated]),
            "valid_pairs": int(primary_valid[:, :, k].sum()),
            "positive_pairs": int((labels[:, :, k] & primary_valid[:, :, k]).sum()),
            "valid_patients": int(primary_valid[:, :, k].any(axis=1).sum()),
            "scorable_auroc_diseases": len(auc), "frozen_diseases": len(disease_codes),
            "disease_without_auc": {code: value["reason"] for code, value in one.items()
                                    if value["auroc"] is None},
            "death": metrics(death_y[:, k], death_prediction[:, k],
                             death_valid[:, k] & primary_patients),
        }
        per_disease[str(horizon)] = one
    for j, code in enumerate(disease_codes):
        brier_by_disease[j]["horizons_days"] = list(HORIZONS)
        brier_by_disease[j]["normalization_denominator"] = denominators[j].tolist()
        per_disease["time_brier"][code] = brier_by_disease[j]
        per_disease["discrimination"][code] = discrimination_by_disease[j]

    ages = np.asarray([r["approx_age"] for r in public_samples])
    sexes = np.asarray([r["gender"] for r in public_samples])
    any_event = np.asarray([bool(boundaries[sid]["any_future_disease_record_365"])
                            for sid in sample_ids])
    subsets = {"age_18_64": ages < 65, "age_65_plus": ages >= 65,
               "sex_F": sexes == "F", "sex_M": sexes == "M",
               "no_future_disease_record": primary_patients & ~any_event,
               "future_disease_record": primary_patients & any_event}
    strata = {}
    for name, subset in subsets.items():
        fixed = subset & primary_patients
        raw_score_value, _ = disease_time_brier(
            labels, prediction, valid & fixed[:, None, None])
        score_value, _ = normalized_disease_time_brier(
            labels, prediction, valid & fixed[:, None, None], denominators)
        parts = [metrics(labels[:, j, -1], prediction[:, j, -1],
                         valid[:, j, -1] & fixed) for j in range(len(disease_codes))]
        auc = [x for x in parts if x["auroc"] is not None]
        strata[name] = {"n": int(fixed.sum()),
            "age_sex_normalized_disease_time_brier": score_value,
            "raw_disease_time_brier_macro": raw_score_value,
            "macro_auroc_365d": mean_or_none([x["auroc"] for x in auc]),
            "macro_auprc_365d": mean_or_none([x["auprc"] for x in auc]),
            "macro_brier_365d": mean_or_none([x["brier"] for x in parts]),
            "scorable_auroc_diseases": len(auc),
            "death_time_brier": mean_or_none([metrics(
                death_y[:, k], death_prediction[:, k], death_valid[:, k] & fixed)["brier"]
                for k in range(len(HORIZONS))])}

    confidence = None
    if bootstrap > 0:
        rng = np.random.default_rng(20260924)
        primary_rows = np.flatnonzero(primary_patients)
        boot_labels = labels[primary_rows]
        boot_prediction = prediction[primary_rows]
        boot_valid = valid[primary_rows]
        squared_error = (boot_prediction - boot_labels) ** 2
        boot_death_y = death_y[primary_rows]
        boot_death_prediction = death_prediction[primary_rows]
        boot_death_valid = death_valid[primary_rows]
        death_squared_error = (boot_death_prediction - boot_death_y) ** 2
        disease_values, death_values, auc365_values, primary_auc_values = [], [], [], []
        for _ in range(bootstrap):
            sampled = rng.integers(0, len(primary_rows), len(primary_rows))
            weights = np.bincount(sampled, minlength=len(primary_rows)).astype(np.float64)
            disease_denominator = np.tensordot(weights, boot_valid, axes=(0, 0))
            disease_numerator = np.tensordot(
                weights, squared_error * boot_valid, axes=(0, 0))
            with np.errstate(invalid="ignore", divide="ignore"):
                time_brier = np.divide(disease_numerator, disease_denominator,
                    out=np.full_like(disease_numerator, np.nan),
                    where=disease_denominator > 0)
            disease_values.append(float(np.nanmean(
                np.nanmean(time_brier / denominators, axis=1))))
            death_denominator = np.tensordot(weights, boot_death_valid, axes=(0, 0))
            death_numerator = np.tensordot(
                weights, death_squared_error * boot_death_valid, axes=(0, 0))
            with np.errstate(invalid="ignore", divide="ignore"):
                death_brier = np.divide(death_numerator, death_denominator,
                    out=np.full_like(death_numerator, np.nan),
                    where=death_denominator > 0)
            death_values.append(float(np.nanmean(death_brier)))
            auc_parts = []
            for j in range(len(disease_codes)):
                use = boot_valid[:, j, -1] & (weights > 0)
                value = weighted_auc(boot_labels[use, j, -1],
                                     boot_prediction[use, j, -1], weights[use])
                if value is not None:
                    auc_parts.append(value)
            auc365_values.append(mean_or_none(auc_parts))
            primary_auc_values.append(weighted_macro_time_auroc(
                boot_labels, boot_prediction, boot_valid, weights))
        confidence = {"bootstrap_patients": bootstrap,
            "macro_time_averaged_auroc_95pct": percentile_interval(primary_auc_values),
            "age_sex_normalized_disease_time_brier_95pct": percentile_interval(disease_values),
            "death_time_brier_95pct": percentile_interval(death_values),
            "macro_auroc_365d_95pct": percentile_interval(auc365_values)}

    return {"schema": "natural-disease-progression-report/v3",
            # The reward. "primary" keeps the earlier disease-first AUROC as a reported metric.
            "reward_metric": PANEL_PRIMARY,
            "panel": panel,
            "primary": discrimination,
            "auxiliary": {
                "raw_disease_time_brier_macro": raw_primary_score,
                "age_sex_normalized_disease_time_brier": normalized_brier,
            },
            "normalization": normalization_metadata,
            "mortality": {"death_time_brier": death_time_brier,
                          "by_time": dict(zip(map(str, HORIZONS), death_primary))},
            "cohort": {"all_samples": len(sample_ids),
                       "primary_complete_year": int(primary_patients.sum()),
                       "partial_observation_outside_primary": int((~primary_patients).sum())},
            "horizons": by_horizon, "per_disease": per_disease,
            "strata": strata, "confidence_intervals": confidence}
