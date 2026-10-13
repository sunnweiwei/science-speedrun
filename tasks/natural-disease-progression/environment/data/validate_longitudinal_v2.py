"""Fail-closed checks for a longitudinal-v2 materialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from protocol import HORIZONS


PUBLIC_EVENT_COLUMNS = {
    "sample_id", "stay_key", "event_type", "code", "value", "unit",
    "event_time", "available_time", "source_table", "event_group_key",
    "mapping_status", "seconds_before_cutoff",
}


def validate_manifest_contract(root: Path) -> tuple[dict, list[str]]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema"] != "natural-disease-progression/v2":
        raise ValueError("Not a v2 materialization")
    if manifest.get("protocol") != "longitudinal-v2":
        raise ValueError("Unsupported materialization protocol")
    if manifest.get("horizons_days") != list(HORIZONS):
        raise ValueError("Materialization does not use the frozen 13 horizons")
    if manifest.get("primary_metric") != "macro_time_averaged_auroc":
        raise ValueError("Materialization does not use the frozen AUROC reward")
    diseases = manifest["scored_diseases"]
    if manifest.get("num_scored_diseases") != len(diseases) or len(diseases) != len(set(diseases)):
        raise ValueError("Invalid or duplicate scored disease vocabulary")
    for split in ("validation", "test"):
        split_path = root / "public" / split / "manifest.json"
        split_manifest = json.loads(split_path.read_text())
        expected = {
            "protocol": manifest["protocol"],
            "horizons_days": list(HORIZONS),
            "scored_diseases": diseases,
            "split": split,
            "primary_metric": manifest["primary_metric"],
        }
        for key, value in expected.items():
            if split_manifest.get(key) != value:
                raise ValueError(f"{split_path}: {key} disagrees with root manifest")
    return manifest, diseases


def validate(root: Path) -> dict:
    manifest, diseases = validate_manifest_contract(root)
    db = duckdb.connect()
    db.execute("SET threads=4")
    report: dict[str, object] = {"schema": "natural-disease-progression-validation/v2"}
    patient_map = root / "trusted" / "controller_patient_splits.parquet"
    patient_counts = db.execute("""SELECT count(*),count(DISTINCT patient_key),
      count(DISTINCT sample_id),count(DISTINCT (patient_key,split))
      FROM read_parquet(?)""", [str(patient_map)]).fetchone()
    if len(set(patient_counts)) != 1:
        raise ValueError(f"Stable patient split keys are duplicated: {patient_counts}")
    cross_split = db.execute("""SELECT count(*) FROM (
      SELECT patient_key FROM read_parquet(?) GROUP BY patient_key
      HAVING count(DISTINCT split)>1)""", [str(patient_map)]).fetchone()[0]
    if cross_split:
        raise ValueError(f"{cross_split} stable patient identifiers cross data splits")
    ids: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        base = root / "train" if split == "train" else root / "public" / split
        samples = base / "samples.parquet"
        events = base / "events.parquet"
        sample_rows = db.execute("SELECT count(*),count(DISTINCT sample_id),min(index_rank),max(index_rank) FROM read_parquet(?)", [str(samples)]).fetchone()
        if sample_rows[0] != sample_rows[1] or sample_rows[2:] != (2, 2):
            raise ValueError(f"Invalid sample cardinality/rank in {split}: {sample_rows}")
        ids[split] = {x[0] for x in db.execute("SELECT sample_id FROM read_parquet(?)", [str(samples)]).fetchall()}
        columns = {x[0] for x in db.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(events)]).fetchall()}
        if columns != PUBLIC_EVENT_COLUMNS:
            raise ValueError(f"Unexpected public event columns in {split}: {columns}")
        bad = db.execute("""SELECT count(*) FROM read_parquet(?) e
          LEFT JOIN read_parquet(?) s USING(sample_id)
          WHERE s.sample_id IS NULL OR e.event_time>s.cutoff
             OR e.available_time>s.cutoff
             OR abs(e.seconds_before_cutoff-date_diff('second',e.event_time,s.cutoff))>1""",
          [str(events), str(samples)]).fetchone()[0]
        if bad:
            raise ValueError(f"{split}: {bad} unavailable/future/orphan event rows")
        history = db.execute("""WITH h AS (SELECT sample_id,count(*) events,
          count(DISTINCT stay_key) FILTER (WHERE stay_key IS NOT NULL) stays,
          count(*) FILTER (WHERE event_type='lab' AND try_cast(value AS DOUBLE) IS NOT NULL) numeric_labs
          FROM read_parquet(?) GROUP BY sample_id)
          SELECT count(*),median(events),quantile_cont(events,.9),
            avg(CASE WHEN stays>=2 THEN 1.0 ELSE 0.0 END),sum(numeric_labs) FROM h""",
          [str(events)]).fetchone()
        report[split] = {
            "samples": sample_rows[0], "median_events": float(history[1]),
            "p90_events": float(history[2]),
            "fraction_with_two_or_more_stays": float(history[3]),
            "numeric_lab_rows": int(history[4]),
        }
    if ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"]:
        raise ValueError("Public sample identifier overlap between splits")
    mapped = db.execute("SELECT split,count(*) FROM read_parquet(?) GROUP BY split",
                        [str(patient_map)]).fetchall()
    if dict(mapped) != {split: len(values) for split, values in ids.items()}:
        raise ValueError("Stable patient map does not match public split cardinalities")

    expected_cells = len(diseases) * len(HORIZONS)
    normalization = root / "trusted" / "risk_normalization.parquet"
    n_cells, n_codes, min_weight, max_weight = db.execute(
        "SELECT count(*),count(DISTINCT disease_code),min(normalization_weight),max(normalization_weight) FROM read_parquet(?)",
        [str(normalization)]).fetchone()
    if n_cells != expected_cells or n_codes != len(diseases):
        raise ValueError("Normalization does not cover the frozen output grid")
    for split in ("train", "validation", "test"):
        base = root / "train" if split == "train" else root / "trusted" / split
        labels = base / "labels.parquet"
        windows = base / "label_windows.parquet"
        expected = len(ids[split]) * len(diseases)
        counts = db.execute("SELECT count(*),count(DISTINCT sample_id),count(DISTINCT disease_code) FROM read_parquet(?)", [str(labels)]).fetchone()
        if counts != (expected, len(ids[split]), len(diseases)):
            raise ValueError(f"{split}: incomplete label grid {counts}, expected {expected}")
        bad_time = db.execute("""SELECT count(*) FROM read_parquet(?)
          WHERE first_future_record_time IS NOT NULL
            AND (first_future_record_time<=cutoff OR event_days<0)""", [str(labels)]).fetchone()[0]
        if bad_time:
            raise ValueError(f"{split}: {bad_time} future targets at/before cutoff")
        monotonic_terms = []
        for left, right in zip(HORIZONS, HORIZONS[1:]):
            monotonic_terms.append(f"(disease_label_{left} AND NOT disease_label_{right})")
            monotonic_terms.append(f"(death_label_{left} AND NOT death_label_{right})")
        bad_monotonic = db.execute(
            f"SELECT count(*) FROM read_parquet(?) WHERE {' OR '.join(monotonic_terms)}",
            [str(windows)]).fetchone()[0]
        if bad_monotonic:
            raise ValueError(f"{split}: {bad_monotonic} non-monotonic cumulative targets")
    db.close()
    report["scored_diseases"] = len(diseases)
    report["stable_patient_split_keys"] = patient_counts[0]
    report["normalization_cells"] = n_cells
    report["normalization_weight_range"] = [float(min_weight), float(max_weight)]
    report["status"] = "passed"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized", required=True, type=Path)
    result = validate(parser.parse_args().materialized)
    print(json.dumps(result, indent=2))
