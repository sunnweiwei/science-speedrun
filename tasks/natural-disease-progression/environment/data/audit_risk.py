"""Audit the frozen first-recorded-disease risk target and time grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from protocol import HORIZONS


def run(materialized: Path) -> dict:
    db = duckdb.connect(str(materialized / "working.duckdb"), read_only=True)
    vocabulary = dict(zip(("train_cutoff_input_diagnosis", "train_full_timeline_diagnosis",
                           "scored_disease"), db.execute("""SELECT
      (SELECT count(DISTINCT code) FROM read_parquet(?) WHERE event_type='diagnosis'),
      (SELECT count(DISTINCT code) FROM diagnosis_events e JOIN split s USING(subject_id)
        WHERE s.split='train'),
      (SELECT count(*) FROM scored_diseases)""",
      [str(materialized / "train" / "events.parquet")]).fetchone()))
    coverage = {}
    for split in ("train", "validation", "test"):
        coverage[split] = dict(zip(("samples", "diagnosis_365d_complete",
                                    "death_365d_complete", "both_365d_complete"),
            db.execute("""SELECT count(*),
              count(*) FILTER(WHERE diagnosis_observation_end>=CAST(dischtime AS DATE)+INTERVAL 365 DAY),
              count(*) FILTER(WHERE death_observation_end>=CAST(dischtime AS DATE)+INTERVAL 365 DAY),
              count(*) FILTER(WHERE diagnosis_observation_end>=CAST(dischtime AS DATE)+INTERVAL 365 DAY
                AND death_observation_end>=CAST(dischtime AS DATE)+INTERVAL 365 DAY)
              FROM samples WHERE split=?""", [split]).fetchone()))
    paths = {"train": str(materialized / "train" / "label_windows.parquet"),
             "validation": str(materialized / "trusted" / "validation" / "label_windows.parquet"),
             "test": str(materialized / "trusted" / "test" / "label_windows.parquet")}
    codes = [r[0] for r in db.execute(
        "SELECT disease_code FROM scored_diseases ORDER BY 1").fetchall()]
    per_disease = {code: {"disease_code": code} for code in codes}
    columns = ",".join(
        f"count(*) FILTER(WHERE l.disease_valid_{h}),"
        f"count(*) FILTER(WHERE l.disease_valid_{h} AND l.disease_label_{h})"
        for h in HORIZONS)
    for split, path in paths.items():
        rows = db.execute(f"""SELECT l.disease_code,{columns}
          FROM read_parquet(?) l JOIN samples s USING(sample_id)
          WHERE s.split=? GROUP BY l.disease_code ORDER BY l.disease_code""",
          [path, split]).fetchall()
        for row in rows:
            values = row[1:]
            per_disease[row[0]][split] = {
                str(h): {"valid": values[2*k], "positive": values[2*k+1]}
                for k, h in enumerate(HORIZONS)}
    result = {"schema": "natural-disease-progression-risk-audit/v1",
              "horizons_days": list(HORIZONS), "vocabulary": vocabulary,
              "coverage": coverage, "per_disease": list(per_disease.values())}
    normalization_path = materialized / "risk_normalization.json"
    if normalization_path.is_file():
        normalization = json.loads(normalization_path.read_text())
        result["normalization"] = {
            "reference": normalization["reference"],
            "source": normalization["source"],
            "floor_rule": normalization["floor_rule"],
            "cells": normalization["cells"],
            "floor_applied_cells": normalization["floor_applied_cells"],
        }
    (materialized / "risk_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    db.close()
    return {k: v for k, v in result.items() if k != "per_disease"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().materialized), indent=2))
