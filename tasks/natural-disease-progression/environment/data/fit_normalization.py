"""Fit train-only age/sex Brier denominators for normalized risk scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from protocol import HORIZONS


FOLDS = 5
FLOOR_QUANTILE = 0.05
MIN_RELIABLE_POSITIVES = 20
MIN_RELIABLE_NEGATIVES = 20
MAX_NORMALIZATION_WEIGHT = 1000.0


def update_manifests(materialized: Path, report: dict) -> None:
    manifest_path = materialized / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        manifest["primary_metric"] = "age_sex_normalized_disease_time_brier"
        manifest["normalization"] = {
            "artifact": "trusted/risk_normalization.parquet",
            "reference": report["reference"], "source": report["source"],
            "floor_rule": report["floor_rule"],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for split in ("validation", "test"):
        path = materialized / "public" / split / "manifest.json"
        if path.is_file():
            manifest = json.loads(path.read_text())
            manifest["primary_metric"] = "age_sex_normalized_disease_time_brier"
            path.write_text(json.dumps(manifest, indent=2) + "\n")


def fit(materialized: Path) -> dict:
    """Cross-fit a demographic reference entirely within the frozen train split."""
    db = duckdb.connect()
    db.execute("SET threads=4")
    db.execute("SET memory_limit='4GB'")
    labels = (materialized / "train" / "label_windows.parquet").as_posix()
    samples = (materialized / "train" / "samples.parquet").as_posix()
    rows = []
    for horizon in HORIZONS:
        values = db.execute(f"""WITH base AS (
          SELECT l.sample_id,l.disease_code,
            CAST(l.disease_label_{horizon} AS INTEGER) outcome,
            s.gender,
            CASE WHEN s.approx_age<45 THEN 0 WHEN s.approx_age<65 THEN 1
                 WHEN s.approx_age<80 THEN 2 ELSE 3 END age_bin,
            hash(l.sample_id)%{FOLDS} fold
          FROM read_parquet(?) l JOIN read_parquet(?) s USING(sample_id)
          WHERE l.disease_valid_{horizon}
            AND l.diagnosis_observation_end>=l.cutoff+INTERVAL 365 DAY
            AND l.death_observation_end>=l.cutoff+INTERVAL 365 DAY),
        by_fold AS (
          SELECT disease_code,gender,age_bin,fold,count(*) n,sum(outcome) positive
          FROM base GROUP BY ALL),
        totals AS (
          SELECT disease_code,gender,age_bin,sum(n) n,sum(positive) positive
          FROM by_fold GROUP BY ALL),
        predicted AS (
          SELECT b.*,(t.positive-f.positive+1.0)/(t.n-f.n+2.0) prediction
          FROM base b JOIN by_fold f USING(disease_code,gender,age_bin,fold)
          JOIN totals t USING(disease_code,gender,age_bin))
        SELECT disease_code,count(*) train_valid,sum(outcome) train_positive,
          avg(power(prediction-outcome,2)) age_sex_oof_brier
        FROM predicted GROUP BY disease_code ORDER BY disease_code""",
          [labels, samples]).fetchall()
        rows.extend((code, horizon, valid, positive, brier)
                    for code, valid, positive, brier in values)
    db.close()
    frame = pd.DataFrame(rows, columns=["disease_code", "horizon_days", "train_valid",
                                        "train_positive", "age_sex_oof_brier"])
    frame["train_negative"] = frame.train_valid - frame.train_positive
    reliable = ((frame.train_positive >= MIN_RELIABLE_POSITIVES)
                & (frame.train_negative >= MIN_RELIABLE_NEGATIVES)
                & np.isfinite(frame.age_sex_oof_brier)
                & (frame.age_sex_oof_brier > 0))
    reliability_fallback = False
    if not reliable.any():
        # Synthetic smoke fixtures can be smaller than the production
        # reliability threshold. Production MIMIC materialization never uses
        # this fallback; record it explicitly if a tiny fixture does.
        reliable = (np.isfinite(frame.age_sex_oof_brier)
                    & (frame.age_sex_oof_brier > 0))
        reliability_fallback = True
    if not reliable.any():
        raise ValueError("No finite train cells available to freeze normalization floor")
    empirical_floor = float(frame.loc[reliable, "age_sex_oof_brier"].quantile(
        FLOOR_QUANTILE, interpolation="linear"))
    if not np.isfinite(empirical_floor) or empirical_floor <= 0:
        raise ValueError("Invalid train-derived Brier floor")
    floor = max(empirical_floor, 1.0 / MAX_NORMALIZATION_WEIGHT)
    frame["denominator_brier"] = np.maximum(frame.age_sex_oof_brier, floor)
    frame["normalization_weight"] = 1.0 / frame.denominator_brier
    frame["floor_applied"] = frame.age_sex_oof_brier < floor
    frame = frame.sort_values(["disease_code", "horizon_days"])
    target = materialized / "trusted" / "risk_normalization.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    report = {
        "schema": "natural-disease-progression-risk-normalization/v1",
        "source": "train_complete_year_only",
        "reference": "5_fold_out_of_fold_age_sex_laplace_rate",
        "folds": FOLDS,
        "fold_assignment": f"duckdb_hash(sample_id)_mod_{FOLDS}",
        "age_bins": ["18-44", "45-64", "65-79", "80+"],
        "floor_rule": {
            "quantile": FLOOR_QUANTILE,
            "minimum_train_positives": MIN_RELIABLE_POSITIVES,
            "minimum_train_negatives": MIN_RELIABLE_NEGATIVES,
            "empirical_brier_floor": empirical_floor,
            "predeclared_maximum_weight": MAX_NORMALIZATION_WEIGHT,
            "brier_floor": floor,
            "maximum_weight": 1.0 / floor,
        },
        "cells": int(len(frame)),
        "reliable_cells_for_floor": int(reliable.sum()),
        "reliability_fallback": reliability_fallback,
        "floor_applied_cells": int(frame.floor_applied.sum()),
        "raw_brier_quantiles": {str(q): float(frame.age_sex_oof_brier.quantile(q))
                                for q in (0, .01, .05, .1, .25, .5, .75, .9, .99, 1)},
    }
    (materialized / "risk_normalization.json").write_text(
        json.dumps(report, indent=2) + "\n")
    update_manifests(materialized, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized", type=Path, required=True)
    print(json.dumps(fit(parser.parse_args().materialized), indent=2))
