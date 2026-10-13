"""Risk-only v2 protocol, metric behavior, and leakage-boundary regressions."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environment" / "data"))
sys.path.insert(0, str(ROOT / "environment" / "evaluation"))
sys.path.insert(0, str(ROOT / "environment" / "references" / "scaffold"))
sys.path.insert(0, str(ROOT / "docs" / "maintenance" / "legacy-neural"))

from data import HORIZONS, write_predictions  # noqa: E402
from adapter import history_to_ids  # noqa: E402
from audit import run as audit_run  # noqa: E402
from baseline import Predictor, RiskNet  # noqa: E402
from label_rules import death_label, disease_label  # noqa: E402
from materialize import run as materialize_run  # noqa: E402
from protocol import split_subjects  # noqa: E402
from stage_runtime_data import stage as stage_runtime_data  # noqa: E402
from risk_scoring import (auc_and_average_precision, disease_time_brier,
                          disease_time_discrimination,
                          percentile_interval,
                          normalized_disease_time_brier, read_predictions,
                          score, weighted_auc)  # noqa: E402
from fixture_data import fixture_source  # noqa: E402


def test_split_is_reproducible_and_patient_disjoint():
    a = split_subjects(list(range(100)))
    b = split_subjects(list(reversed(range(100))))
    assert a == b
    assert [list(a.values()).count(x) for x in ("train", "validation", "test")] == [70, 15, 15]


def test_competing_death_and_censor_masks():
    row = {"risk_set": True, "ambiguous_prior": False, "event_days": None,
           "cutoff": date(2150, 1, 1), "diagnosis_observation_end": date(2150, 1, 20),
           "death_days": None, "death_observation_end": date(2150, 1, 20)}
    assert disease_label(row, 30) == (0, False)
    assert death_label(row, 30) == (0, False)
    row["death_days"] = 10
    assert disease_label(row, 30) == (0, True)
    assert death_label(row, 30) == (1, True)
    row["event_days"] = 11
    assert disease_label(row, 30) == (0, True)
    row["event_days"] = 8
    assert disease_label(row, 30) == (1, True)
    row["risk_set"] = False
    assert disease_label(row, 30) == (0, False)
    row["risk_set"] = True
    row["ambiguous_prior"] = True
    assert disease_label(row, 30) == (0, False)


def test_future_changes_and_diagnosis_row_order_do_not_change_inputs_or_labels(tmp_path):
    outputs = []
    for name, future_value, reverse in (("a", "8", False), ("b", "999", True)):
        source, output = tmp_path / f"source-{name}", tmp_path / f"output-{name}"
        fixture_source(source, future_value=future_value, reverse_diagnoses=reverse)
        audit_run(source, output, ROOT / "environment" / "data" / "cms-2018-gem.zip")
        materialize_run(source, output, 1)
        outputs.append(output)
    for split in ("validation", "test"):
        for filename in ("samples.parquet", "events.parquet"):
            left = pq.read_table(outputs[0] / "public" / split / filename)
            right = pq.read_table(outputs[1] / "public" / split / filename)
            order = [("sample_id", "ascending")]
            if filename == "events.parquet":
                order.extend([("available_time", "ascending"),
                              ("event_time", "ascending"),
                              ("event_type", "ascending"), ("code", "ascending")])
            assert left.sort_by(order).equals(right.sort_by(order))
    for split in ("validation", "test"):
        assert pq.read_table(outputs[0] / "trusted" / split / "label_windows.parquet").equals(
            pq.read_table(outputs[1] / "trusted" / split / "label_windows.parquet"))


def test_materialization_and_runtime_stage_keep_test_truth_private(tmp_path):
    source, materialized = tmp_path / "source", tmp_path / "materialized"
    fixture_source(source)
    audit_run(source, materialized, ROOT / "environment" / "data" / "cms-2018-gem.zip")
    materialize_run(source, materialized, 1)
    for split in ("validation", "test"):
        truth = materialized / "trusted" / split
        assert {path.name for path in truth.iterdir()} == {
            "labels.parquet", "label_windows.parquet", "sample_boundaries.parquet"}
        boundaries = pq.read_table(truth / "sample_boundaries.parquet").to_pylist()
        assert boundaries and {row["split"] for row in boundaries} == {split}
    for legacy in ("labels.parquet", "label_windows.parquet", "sample_boundaries.parquet"):
        assert not (materialized / "trusted" / legacy).exists()
    development, private = stage_runtime_data(materialized, tmp_path / "runtime")
    assert not (development / "public" / "test").exists()
    assert not (development / "trusted" / "test").exists()
    assert not (development / "working.duckdb").exists()
    assert (development / "trusted" / "validation" / "label_windows.parquet").is_file()
    assert (private / "trusted" / "test" / "label_windows.parquet").is_file()
    assert (private / "public" / "test" / "samples.parquet").is_file()


def test_adapter_ignores_ids_and_future_metadata():
    adapter = {"vocab": {"gender:F": 1, "age_decade:50": 2, "diagnosis:A00": 3},
               "lab_bins": {}, "max_events": 10}
    base = {"gender": "F", "approx_age": 55,
            "events": [{"event_type": "diagnosis", "code": "A00"}]}
    altered = {**base, "subject_id": 99, "sample_id": "future", "last_followup": 999,
               "future_admission_count": 11,
               "events": [{"event_type": "diagnosis", "code": "A00",
                            "hadm_id": 123, "future_status": "died later"}]}
    assert history_to_ids(base, adapter) == history_to_ids(altered, adapter)


def test_prediction_contract_is_direct_13_point_and_monotone(tmp_path):
    path = tmp_path / "predictions.parquet"
    good = np.linspace(0.01, 0.2, len(HORIZONS)).tolist()
    pq.write_table(pa.Table.from_pylist([{
        "sample_id": "s", "disease_risk": good, "death_risk": good}]), path)
    disease, death = read_predictions(path, ["s"], 1)
    assert disease.shape == (1, 1, 13) and death.shape == (1, 13)
    bad = good.copy()
    bad[5] = bad[4] - .01
    pq.write_table(pa.Table.from_pylist([{
        "sample_id": "s", "disease_risk": bad, "death_risk": good}]), path)
    with pytest.raises(ValueError, match="decreases"):
        read_predictions(path, ["s"], 1)
    pq.write_table(pa.Table.from_pylist([{
        "sample_id": "s", "disease_risk": [.1] * 4, "death_risk": [.1] * 4}]), path)
    with pytest.raises(ValueError, match="direct time grid"):
        read_predictions(path, ["s"], 1)


def test_primary_brier_detects_timing_identity_and_miscalibration():
    # Four patients, two diseases. Disease A first appears by day 60 in two;
    # disease B by day 180 in the other two.
    labels = np.zeros((4, 2, len(HORIZONS)), dtype=np.uint8)
    labels[:2, 0, np.asarray(HORIZONS) >= 60] = 1
    labels[2:, 1, np.asarray(HORIZONS) >= 180] = 1
    valid = np.ones_like(labels, dtype=bool)
    exact = labels.astype(np.float32)
    exact_score, _ = disease_time_brier(labels, exact, valid)
    early = np.roll(exact, -1, axis=2)
    early[:, :, -1] = exact[:, :, -1]
    late = np.roll(exact, 1, axis=2)
    late[:, :, 0] = 0
    swapped = exact[:, ::-1, :]
    assert exact_score == 0
    assert disease_time_brier(labels, early, valid)[0] > exact_score
    assert disease_time_brier(labels, late, valid)[0] > exact_score
    assert disease_time_brier(labels, swapped, valid)[0] > exact_score

    calibrated = np.broadcast_to(labels.mean(axis=0), labels.shape).copy()
    overconfident = np.where(calibrated > 0, .99, .01).astype(np.float32)
    assert disease_time_brier(labels, overconfident, valid)[0] > disease_time_brier(
        labels, calibrated, valid)[0]


def test_normalized_brier_uses_frozen_cell_denominators():
    labels = np.zeros((2, 2, 1), dtype=np.uint8)
    valid = np.ones_like(labels, dtype=bool)
    prediction = np.asarray([[[.01], [.1]], [[.01], [.1]]], dtype=np.float32)
    denominators = np.asarray([[.001], [.01]])
    raw, _ = disease_time_brier(labels, prediction, valid)
    normalized, details = normalized_disease_time_brier(
        labels, prediction, valid, denominators)
    assert raw == pytest.approx(.00505)
    assert normalized == pytest.approx(.55)
    assert details[0]["normalized_time_brier"] == pytest.approx(.1)
    assert details[1]["normalized_time_brier"] == pytest.approx(1.0)


def test_primary_auc_is_disease_first_time_average_and_skips_single_class_cells():
    labels = np.asarray([
        [[0, 0], [0, 0]],
        [[1, 0], [1, 0]],
        [[0, 1], [1, 0]],
        [[1, 1], [1, 0]],
    ], dtype=np.uint8)
    prediction = np.asarray([
        [[.1, .1], [.1, .1]],
        [[.9, .2], [.6, .2]],
        [[.8, .8], [.7, .3]],
        [[.2, .9], [.8, .4]],
    ], dtype=np.float32)
    valid = np.ones_like(labels, dtype=bool)
    summary, details = disease_time_discrimination(labels, prediction, valid)
    # Disease 0 has AUCs .75 and 1; disease 1 contributes only its first cell.
    expected = ((.75 + 1.0) / 2 + 1.0) / 2
    assert summary["macro_time_averaged_auroc"] == pytest.approx(expected)
    assert summary["valid_diseases"] == 2
    assert summary["valid_disease_time_cells"] == 3
    assert details[1]["timepoints"][1]["auroc"] is None
    assert details[1]["timepoints"][1]["reason"] == "single_class"
    pooled = roc_auc_score(labels.reshape(-1), prediction.reshape(-1))
    assert summary["macro_time_averaged_auroc"] != pytest.approx(pooled)


def test_bootstrap_interval_reports_unavailable_metric():
    assert percentile_interval([None, None]) is None


def test_weighted_auc_matches_expanded_patient_bootstrap():
    y = np.asarray([0, 1, 0, 1], dtype=np.uint8)
    p = np.asarray([.2, .8, .2, .4])
    weights = np.asarray([2, 1, 3, 2], dtype=np.float64)
    expanded = np.repeat(np.arange(len(y)), weights.astype(int))
    assert weighted_auc(y, p, weights) == pytest.approx(
        roc_auc_score(y[expanded], p[expanded]))


def test_combined_discrimination_matches_reference_with_ties():
    y = np.asarray([0, 1, 0, 1, 1], dtype=np.uint8)
    p = np.asarray([.2, .8, .2, .4, .8])
    auc, ap = auc_and_average_precision(y, p)
    assert auc == pytest.approx(roc_auc_score(y, p))
    assert ap == pytest.approx(average_precision_score(y, p))


def _write_scoring_fixture(root: Path):
    public, trusted = root / "public", root / "trusted"
    public.mkdir(parents=True)
    trusted.mkdir()
    codes = ["A00", "B00"]
    sample_rows = [
        {"sample_id": "s0", "gender": "F", "approx_age": 50, "cutoff": date(2150, 1, 1)},
        {"sample_id": "s1", "gender": "M", "approx_age": 70, "cutoff": date(2150, 1, 1)},
        {"sample_id": "s2", "gender": "F", "approx_age": 80, "cutoff": date(2150, 1, 1)},
        {"sample_id": "s3", "gender": "M", "approx_age": 40, "cutoff": date(2150, 1, 1)},
    ]
    pq.write_table(pa.Table.from_pylist(sample_rows), public / "samples.parquet")
    # The panel reward reads diagnosis history; one unrelated diagnosis is enough here.
    pq.write_table(pa.Table.from_pylist([{"sample_id": "s0", "event_type": "diagnosis",
        "code": "Z00"}]), public / "events.parquet")
    (public / "manifest.json").write_text(json.dumps({
        "schema": "natural-disease-progression-input/v2",
        "task": "first-recorded-disease-time-and-death-risk",
        "horizons_days": list(HORIZONS), "scored_diseases": codes, "split": "validation"}))
    label_rows = []
    event_day = {("s0", "A00"): 60, ("s0", "B00"): 60, ("s1", "A00"): 180}
    for sample in sample_rows:
        for code in codes:
            row = {"sample_id": sample["sample_id"], "disease_code": code,
                   "risk_set": True, "ambiguous_prior": False,
                   "event_days": event_day.get((sample["sample_id"], code)),
                   "cutoff": date(2150, 1, 1),
                   "diagnosis_observation_end": date(2152, 1, 1),
                   "death_days": None, "death_observation_end": date(2152, 1, 1)}
            for horizon in HORIZONS:
                row[f"disease_label_{horizon}"], row[f"disease_valid_{horizon}"] = disease_label(row, horizon)
                row[f"death_label_{horizon}"], row[f"death_valid_{horizon}"] = death_label(row, horizon)
            label_rows.append(row)
    pq.write_table(pa.Table.from_pylist(label_rows), trusted / "label_windows.parquet")
    normalization_rows = [{"disease_code": code, "horizon_days": horizon,
        "train_valid": 100, "train_positive": 10,
        "train_negative": 90, "age_sex_oof_brier": 1.0,
        "denominator_brier": 1.0, "normalization_weight": 1.0,
        "floor_applied": False}
        for code in codes for horizon in HORIZONS]
    pq.write_table(pa.Table.from_pylist(normalization_rows),
                   trusted / "risk_normalization.parquet")
    boundaries = [{"sample_id": row["sample_id"], "split": "validation",
        "cutoff": date(2150, 1, 1), "death_date": None, "death_days": None,
        "diagnosis_observation_end": date(2152, 1, 1),
        "death_observation_end": date(2152, 1, 1),
        "any_future_disease_record_365": row["sample_id"] in {"s0", "s1"}}
        for row in sample_rows]
    boundaries[-1]["diagnosis_observation_end"] = date(2150, 6, 1)
    pq.write_table(pa.Table.from_pylist(boundaries), trusted / "sample_boundaries.parquet")
    return public, trusted, sample_rows, codes, event_day


def test_end_to_end_scorer_has_only_risk_outputs(tmp_path):
    public, trusted, samples, codes, event_day = _write_scoring_fixture(tmp_path)
    prediction_rows = []
    for sample in samples:
        disease = []
        for code in codes:
            day = event_day.get((sample["sample_id"], code))
            disease.extend([float(day is not None and day <= h) for h in HORIZONS])
        prediction_rows.append({"sample_id": sample["sample_id"],
                                "disease_risk": disease,
                                "death_risk": [0.0] * len(HORIZONS)})
    path = tmp_path / "predictions.parquet"
    pq.write_table(pa.Table.from_pylist(prediction_rows), path)
    report = score(public, trusted, path, bootstrap=2)
    assert report["schema"] == "natural-disease-progression-report/v3"
    assert report["primary"]["macro_time_averaged_auroc"] == 1.0
    assert report["primary"]["valid_diseases"] == 2
    assert report["auxiliary"] == {
        "raw_disease_time_brier_macro": 0.0,
        "age_sex_normalized_disease_time_brier": 0.0,
    }
    assert report["cohort"] == {"all_samples": 4, "primary_complete_year": 3,
                                 "partial_observation_outside_primary": 1}
    assert set(report["horizons"]) == set(map(str, HORIZONS))
    assert report["horizons"]["365"]["frozen_diseases"] == 2
    assert "trajectories" not in pq.read_schema(path).names
    assert report["strata"]["no_future_disease_record"]["n"] == 1
    assert report["strata"]["future_disease_record"]["n"] == 2


def test_starter_writes_risk_only_predictions(tmp_path):
    output_dir = tmp_path / "output"
    disease = np.linspace(.01, .2, len(HORIZONS), dtype=np.float32).reshape(1, 1, -1)
    death = np.linspace(.02, .1, len(HORIZONS), dtype=np.float32).reshape(1, -1)
    write_predictions(output_dir, ["s"], disease, death, disease_count=1)
    schema = pq.read_schema(output_dir / "predictions.parquet")
    assert schema.names == ["sample_id", "disease_risk", "death_risk"]
    with pytest.raises(ValueError, match="nondecreasing"):
        write_predictions(tmp_path / "bad", ["s"], disease[:, :, ::-1], death, 1)


def test_obsolete_trajectory_protocol_modules_are_absent():
    for relative in ("environment/data/derive_trajectory.py",
                     "environment/evaluation/joint_scoring.py",
                     "environment/evaluation/trajectory.py"):
        assert not (ROOT / relative).exists()
