"""Read-only provenance/time audit for the paired admission-time lab coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def run(materialized: Path) -> dict:
    db = duckdb.connect(str(materialized / "working.duckdb"), read_only=True)
    paths = [str(materialized / section / "events.parquet") for section in
             ("train", "public/validation", "public/test")]
    db.execute("""CREATE TEMP TABLE candidate AS SELECT e.sample_id,e.hadm_id,e.source_table,
      e.event_time,e.available_time,s.hadm_id index_hadm,s.admittime
      FROM read_parquet(?) e JOIN samples s USING(sample_id)
      WHERE e.event_type='lab' AND e.event_time<=s.admittime""", [paths])
    results = db.execute("""SELECT
      count(DISTINCT sample_id) FILTER(WHERE available_time<=admittime) patients_usable,
      count(*) FILTER(WHERE available_time<=admittime) rows_usable,
      count(DISTINCT sample_id) FILTER(WHERE available_time>admittime) patients_chart_only_not_available,
      count(*) FILTER(WHERE available_time>admittime) rows_chart_only_not_available,
      count(*) FILTER(WHERE available_time<=admittime AND hadm_id=index_hadm) index_stay_rows,
      count(*) FILTER(WHERE available_time<=admittime AND hadm_id<>index_hadm) prior_stay_rows,
      count(*) FILTER(WHERE available_time<=admittime AND hadm_id IS NULL) no_hadm_rows,
      count(DISTINCT sample_id) FILTER(WHERE available_time<=admittime AND hadm_id=index_hadm) index_stay_patients,
      count(*) FILTER(WHERE available_time<=admittime AND available_time>=event_time) valid_order_rows,
      count(*) FILTER(WHERE available_time<=admittime AND source_table<>'labevents') wrong_source_rows,
      count(*) FILTER(WHERE available_time<=admittime AND
        date_diff('minute',available_time,admittime)<=60) available_within_1h,
      count(*) FILTER(WHERE available_time<=admittime AND
        date_diff('minute',available_time,admittime)<=360) available_within_6h,
      count(*) FILTER(WHERE available_time<=admittime AND
        date_diff('minute',available_time,admittime)<=1440) available_within_24h,
      quantile_cont(date_diff('hour',available_time,admittime),[.1,.5,.9])
        FILTER(WHERE available_time<=admittime) available_hours_before_admission_p10_p50_p90
      FROM candidate""").fetchone()
    names = ("patients_usable", "rows_usable", "patients_chart_only_not_available",
             "rows_chart_only_not_available", "index_stay_rows", "prior_stay_rows",
             "no_hadm_rows", "index_stay_patients", "valid_order_rows",
             "wrong_source_rows", "available_within_1h", "available_within_6h",
             "available_within_24h", "available_hours_before_admission_p10_p50_p90")
    result = dict(zip(names, results))
    near = db.execute("""SELECT
      count(DISTINCT sample_id) FILTER(WHERE available_time<=admittime
        AND event_time>=admittime-INTERVAL 24 HOUR) patients_24h,
      count(DISTINCT sample_id) FILTER(WHERE available_time<=admittime
        AND event_time>=admittime-INTERVAL 7 DAY) patients_7d,
      count(DISTINCT sample_id) FILTER(WHERE available_time<=admittime
        AND hadm_id IS NULL) patients_no_hadm,
      count(*) FILTER(WHERE available_time<=admittime
        AND event_time>=admittime-INTERVAL 24 HOUR) rows_24h,
      count(*) FILTER(WHERE available_time<=admittime
        AND event_time>=admittime-INTERVAL 24 HOUR AND hadm_id IS NULL) rows_24h_no_hadm,
      count(*) FILTER(WHERE available_time<=admittime
        AND event_time>=admittime-INTERVAL 24 HOUR AND hadm_id=index_hadm) rows_24h_index_hadm
      FROM candidate""").fetchone()
    result.update(zip(("patients_24h", "patients_7d", "patients_no_hadm",
                       "rows_24h", "rows_24h_no_hadm", "rows_24h_index_hadm"), near))
    if result["rows_usable"] != result["valid_order_rows"] or result["wrong_source_rows"]:
        raise AssertionError("Admission lab availability or source invariant failed")
    db.close()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().materialized), indent=2))
