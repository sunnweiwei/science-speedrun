"""Build the versioned longitudinal-v2 cohort from an audited v1 controller DB.

This is intentionally a separate output rather than an in-place migration.  It
uses the frozen patient split and source-derived timeline in working.duckdb,
selects the second eligible live adult discharge, and never exposes source IDs.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import duckdb

from protocol import HORIZONS, SPLIT_SEED


def sql_copy(db: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    db.execute(
        f"COPY ({query}) TO '{path.as_posix()}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )


def build(source: Path, output: Path, min_train_events: int = 50) -> dict:
    source_db = source / "working.duckdb"
    if not source_db.is_file():
        raise FileNotFoundError(f"Missing audited controller DB: {source_db}")
    if output.resolve() == source.resolve():
        raise ValueError("v2 must use a separate output directory; v1 is immutable")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    db = duckdb.connect(str(source_db), read_only=True)
    db.execute("SET threads=4")
    db.execute("SET memory_limit='6GB'")

    # Eligibility uses only facts observed by each discharge. Selecting rank 2
    # guarantees a prior eligible stay without selecting on a future return.
    db.execute("""CREATE TEMP TABLE v2_samples AS WITH eligible AS (
      SELECT a.*,s.patient_key,s.split,p.anchor_age,p.anchor_year,
        p.anchor_year_group,p.gender,p.dod,
        ROW_NUMBER() OVER (
          PARTITION BY a.subject_id ORDER BY a.dischtime,a.hadm_id) eligible_rank,
        LAG(a.dischtime) OVER (
          PARTITION BY a.subject_id ORDER BY a.dischtime,a.hadm_id) previous_eligible_dischtime
      FROM admissions a JOIN split s USING(subject_id) JOIN patients p USING(subject_id)
      WHERE a.dischtime IS NOT NULL AND a.admittime IS NOT NULL
        AND a.dischtime>=a.admittime
        AND p.anchor_age+year(a.admittime)-p.anchor_year>=18
        AND a.hospital_expire_flag='0'
        AND (a.deathtime IS NULL OR a.deathtime>a.dischtime)
        AND (p.dod IS NULL OR p.dod>CAST(a.dischtime AS DATE))
    ), selected AS (
      SELECT *,concat(split,'_v2_',left(patient_key,16)) sample_id,
        anchor_age+year(dischtime)-anchor_year age_at_cutoff,
        make_date(anchor_year+2022-CAST(right(anchor_year_group,4) AS INTEGER),
                  12,31) diagnosis_observation_end
      FROM eligible WHERE eligible_rank=2
    ), followup AS (
      SELECT subject_id,CAST(max(dischtime) AS DATE)+INTERVAL 365 DAY
        AS death_observation_end
      FROM admissions GROUP BY subject_id
    )
    SELECT x.*,f.death_observation_end,
      date_diff('day',CAST(x.dischtime AS DATE),x.dod) death_days
    FROM selected x JOIN followup f USING(subject_id)""")

    # Freeze supported diseases using v2 TRAIN outcomes only.
    db.execute("""CREATE TEMP TABLE v2_train_events AS
      SELECT f.disease_code,count(DISTINCT c.subject_id) positives
      FROM v2_samples c JOIN first_disease f USING(subject_id)
      WHERE c.split='train' AND f.first_record_time>c.dischtime
        AND CAST(f.first_record_time AS DATE)<=CAST(c.dischtime AS DATE)+INTERVAL 365 DAY
        AND CAST(f.first_record_time AS DATE)<=c.diagnosis_observation_end
        AND (left(f.disease_code,1) BETWEEN 'A' AND 'Q'
             OR left(f.disease_code,1)='U')
        AND NOT EXISTS (
          SELECT 1 FROM ambiguous_history a
          WHERE a.subject_id=c.subject_id AND a.disease_code=f.disease_code
            AND a.event_time<=c.dischtime)
      GROUP BY f.disease_code""")
    diseases = [r[0] for r in db.execute(
        "SELECT disease_code FROM v2_train_events WHERE positives>=? ORDER BY 1",
        [min_train_events],
    ).fetchall()]
    if not diseases:
        raise ValueError("No disease satisfies the train-only support threshold")
    db.execute("CREATE TEMP TABLE v2_diseases(disease_code VARCHAR)")
    db.executemany("INSERT INTO v2_diseases VALUES (?)", [(x,) for x in diseases])

    sql_copy(db, """SELECT e.disease_code,e.positives train_365d_positives
      FROM v2_train_events e JOIN v2_diseases d USING(disease_code) ORDER BY 1""",
      output / "scored_diseases.parquet")
    # Controller-only stable keys prove patient disjointness independently of
    # split-prefixed public sample IDs. Runtime staging intentionally omits it.
    sql_copy(db, """SELECT patient_key,split,sample_id FROM v2_samples
      ORDER BY patient_key""", output / "trusted" / "controller_patient_splits.parquet")

    # Public inputs use opaque encounter/group keys. Raw subject_id, hadm_id,
    # labevent_id and emar_id are never written to the v2 artifact.
    for split in ("train", "validation", "test"):
        base = output / "train" if split == "train" else output / "public" / split
        sql_copy(db, f"""SELECT sample_id,gender,age_at_cutoff approx_age,
          (anchor_age>=91) older_than_89,dischtime cutoff,eligible_rank index_rank
          FROM v2_samples WHERE split='{split}' ORDER BY sample_id""",
          base / "samples.parquet")
        sql_copy(db, f"""SELECT c.sample_id,
          CASE WHEN t.hadm_id IS NOT NULL THEN left(sha256(concat(
            'natural-disease-progression-v2:stay:',c.patient_key,':',
            CAST(t.hadm_id AS VARCHAR))),24) END stay_key,
          t.event_type,t.code,t.\"value\",t.unit,t.event_time,t.available_time,
          t.source_table,left(sha256(concat(
            'natural-disease-progression-v2:group:',c.patient_key,':',
            coalesce(t.same_time_group,''))),24) event_group_key,
          t.mapping_status,
          date_diff('second',t.event_time,c.dischtime) seconds_before_cutoff
          FROM v2_samples c JOIN timeline t USING(subject_id)
          WHERE c.split='{split}' AND t.event_time<=c.dischtime
            AND t.available_time<=c.dischtime
          ORDER BY c.sample_id,t.event_time,t.available_time,t.event_type,t.code,
                   event_group_key""", base / "events.parquet")

    db.execute("""CREATE TEMP TABLE v2_label_state AS
      SELECT c.sample_id,c.subject_id,c.split,d.disease_code,
        (f.first_record_time IS NULL OR f.first_record_time>c.dischtime) risk_set,
        EXISTS(SELECT 1 FROM ambiguous_history a
          WHERE a.subject_id=c.subject_id AND a.disease_code=d.disease_code
            AND a.event_time<=c.dischtime) ambiguous_prior,
        CASE WHEN f.first_record_time>c.dischtime
             THEN f.first_record_time END first_future_record_time,
        CASE WHEN f.first_record_time>c.dischtime
             THEN left(sha256(concat('natural-disease-progression-v2:future-group:',
                  c.patient_key,':',f.first_same_time_group)),24) END future_same_time_group,
        CASE WHEN f.first_record_time>c.dischtime THEN date_diff(
          'day',CAST(c.dischtime AS DATE),CAST(f.first_record_time AS DATE)) END event_days,
        c.dischtime cutoff,c.diagnosis_observation_end,c.dod,c.death_days,
        c.death_observation_end
      FROM v2_samples c CROSS JOIN v2_diseases d LEFT JOIN first_disease f
        ON c.subject_id=f.subject_id AND d.disease_code=f.disease_code""")

    label_cols = """sample_id,disease_code,risk_set,ambiguous_prior,
      first_future_record_time,future_same_time_group,event_days,
      CAST(cutoff AS DATE) cutoff,diagnosis_observation_end,death_days,
      CAST(death_observation_end AS DATE) death_observation_end"""
    sql_copy(db, f"SELECT {label_cols} FROM v2_label_state WHERE split='train' ORDER BY sample_id,disease_code",
             output / "train" / "labels.parquet")
    for split in ("validation", "test"):
        sql_copy(db, f"SELECT {label_cols} FROM v2_label_state WHERE split='{split}' ORDER BY sample_id,disease_code",
                 output / "trusted" / split / "labels.parquet")
        sql_copy(db, f"""SELECT c.sample_id,c.split,CAST(c.dischtime AS DATE) cutoff,
          c.dod death_date,c.death_days,c.diagnosis_observation_end,
          CAST(c.death_observation_end AS DATE) death_observation_end,
          EXISTS(SELECT 1 FROM first_disease f
            WHERE f.subject_id=c.subject_id AND f.first_record_time>c.dischtime
              AND CAST(f.first_record_time AS DATE)<=CAST(c.dischtime AS DATE)+INTERVAL 365 DAY
              AND CAST(f.first_record_time AS DATE)<=c.diagnosis_observation_end
              AND (c.death_days IS NULL OR CAST(f.first_record_time AS DATE)
                   <=CAST(c.dischtime AS DATE)+c.death_days*INTERVAL 1 DAY))
            any_future_disease_record_365
          FROM v2_samples c WHERE c.split='{split}' ORDER BY c.sample_id""",
          output / "trusted" / split / "sample_boundaries.parquet")

    counts = dict(db.execute("SELECT split,count(*) FROM v2_samples GROUP BY split").fetchall())
    history = db.execute("""WITH h AS (
      SELECT c.split,c.sample_id,count(t.code) event_count,
        count(DISTINCT CASE WHEN t.hadm_id IS NOT NULL THEN t.hadm_id END) stay_count,
        date_diff('day',min(t.event_time),c.dischtime) history_days,
        date_diff('day',c.previous_eligible_dischtime,c.dischtime) eligible_discharge_gap_days
      FROM v2_samples c LEFT JOIN timeline t ON c.subject_id=t.subject_id
        AND t.event_time<=c.dischtime AND t.available_time<=c.dischtime
      GROUP BY c.split,c.sample_id,c.dischtime,c.previous_eligible_dischtime)
      SELECT split,median(event_count),quantile_cont(event_count,.9),
        avg(CASE WHEN stay_count>=2 THEN 1.0 ELSE 0.0 END),
        quantile_cont(history_days,[.1,.5,.9,.99]),
        quantile_cont(eligible_discharge_gap_days,[.1,.5,.9,.99])
      FROM h GROUP BY split ORDER BY split""").fetchall()
    audit = {
        "schema": "natural-disease-progression-longitudinal-audit/v2",
        "source_materialization": str(source),
        "selection": "second eligible live adult discharge",
        "sample_counts": {k: int(v) for k, v in counts.items()},
        "num_scored_diseases": len(diseases),
        "train_min_365d_events": min_train_events,
        "history_by_split": {
            row[0]: {
                "median_events": float(row[1]),
                "p90_events": float(row[2]),
                "fraction_with_two_or_more_stays": float(row[3]),
                "history_days_p10_p50_p90_p99": [float(x) for x in row[4]],
                "eligible_discharge_gap_days_p10_p50_p90_p99": [float(x) for x in row[5]],
            } for row in history
        },
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    manifest = {
        "schema": "natural-disease-progression/v2",
        "protocol": "longitudinal-v2",
        "source": "MIMIC-IV 3.1",
        "split_seed": SPLIT_SEED,
        "task": "first-recorded-disease-time-and-death-risk",
        "index_time": "second_eligible_live_adult_discharge",
        "horizons_days": HORIZONS,
        "train_min_365d_events": min_train_events,
        "num_scored_diseases": len(diseases),
        "scored_diseases": diseases,
        "primary_metric": "macro_time_averaged_auroc",
        "patient_weighting": "one_index_sample_per_patient",
        "notes": "All inputs obey event and availability cutoff. Endpoint is first inpatient record, not biological onset.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for split in ("validation", "test"):
        (output / "public" / split / "manifest.json").write_text(json.dumps({
            "schema": "natural-disease-progression-input/v2",
            "protocol": "longitudinal-v2",
            "task": manifest["task"],
            "horizons_days": HORIZONS,
            "scored_diseases": diseases,
            "split": split,
            "primary_metric": manifest["primary_metric"],
        }, indent=2) + "\n")
    db.close()

    # Reuse the separately tested expansion/normalization implementation.  It
    # reads only the newly generated v2 train labels for fitted artifacts.
    from expand_labels import expand
    from fit_normalization import fit as fit_normalization
    expand(output)
    normalization = fit_normalization(output)
    manifest["normalization"] = {
        "artifact": "trusted/risk_normalization.parquet",
        "reference": normalization["reference"],
        "source": normalization["source"],
        "floor_rule": normalization["floor_rule"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    protocol = Path(__file__).resolve().parent / "protocols" / "longitudinal-v2.json"
    # copy2/copyfile metadata updates fail on common WSL/NTFS bind mounts even
    # though content writes are supported; protocol provenance needs bytes,
    # not host-specific timestamps.
    shutil.copyfile(protocol, output / "protocol.json")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-materialized", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-train-events", type=int, default=50)
    args = parser.parse_args()
    print(json.dumps(build(args.source_materialized, args.output,
                           args.min_train_events), indent=2))
