"""Reproducible MIMIC-IV 3.1 conversion. Run audit.py first and review audit.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from audit import csv_view
from protocol import HORIZONS, SPLIT_SEED, patient_key, split_subjects


def sql_copy(db, query: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    db.execute(f"COPY ({query}) TO '{path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")


def run(root: Path, output: Path, min_train_events: int) -> None:
    if not (output / "audit.json").is_file():
        raise FileNotFoundError("Run and review audit.py before materialize.py")
    audit = json.loads((output / "audit.json").read_text())
    if audit["source_version"] != "MIMIC-IV 3.1":
        raise ValueError("Audit source version mismatch")
    db = duckdb.connect(str(output / "working.duckdb"))
    db.execute("SET threads=4")
    db.execute("SET memory_limit='4GB'")
    for name in ("labevents", "prescriptions", "pharmacy", "emar"):
        csv_view(db, name, root)
    subjects = [x[0] for x in db.execute("SELECT DISTINCT subject_id FROM adult_admissions").fetchall()]
    frozen_split = output / "trusted" / "subject_split.parquet"
    if frozen_split.exists():
        split_frame = pd.read_parquet(frozen_split)
        if set(split_frame.subject_id) != set(subjects) or split_frame.subject_id.duplicated().any():
            raise ValueError("Existing frozen patient split does not match audited cohort")
        if any(split_frame.patient_key != split_frame.subject_id.map(patient_key)):
            raise ValueError("Existing frozen patient keys changed")
    else:
        split = split_subjects(subjects)
        split_frame = pd.DataFrame([(s, patient_key(s), group) for s, group in split.items()], columns=["subject_id", "patient_key", "split"])
    db.register("split_frame", split_frame)
    db.execute("CREATE OR REPLACE TABLE split AS SELECT * FROM split_frame")
    sql_copy(db, "SELECT * FROM split ORDER BY subject_id", output / "trusted" / "subject_split.parquet")

    # The first qualifying live discharge is chosen using only facts known at that discharge.
    db.execute("""CREATE OR REPLACE TABLE cohort AS WITH eligible AS (
      SELECT a.*,s.patient_key,s.split,p.anchor_age,p.anchor_year,p.anchor_year_group,p.gender,p.dod,
        ROW_NUMBER() OVER(PARTITION BY a.subject_id ORDER BY a.dischtime,a.hadm_id) AS rn
      FROM admissions a JOIN split s USING(subject_id) JOIN patients p USING(subject_id)
      WHERE a.dischtime IS NOT NULL AND a.admittime IS NOT NULL AND a.dischtime>=a.admittime
        AND p.anchor_age+year(a.admittime)-p.anchor_year>=18
        AND a.hospital_expire_flag='0' AND (a.deathtime IS NULL OR a.deathtime>a.dischtime)
        AND (p.dod IS NULL OR p.dod>CAST(a.dischtime AS DATE)) )
      SELECT *, concat(split,'_',left(patient_key,16)) AS sample_id,
        p.anchor_age+year(dischtime)-p.anchor_year AS age_at_cutoff,
        make_date(anchor_year+2022-CAST(right(anchor_year_group,4) AS INTEGER),12,31) AS diagnosis_observation_end
      FROM eligible p WHERE rn=1""")
    db.execute("""CREATE OR REPLACE TABLE death_followup AS SELECT c.subject_id,
      CAST(max(a.dischtime) AS DATE)+INTERVAL 365 DAY AS death_observation_end
      FROM cohort c JOIN admissions a USING(subject_id) GROUP BY c.subject_id""")
    db.execute("""CREATE OR REPLACE TABLE samples AS SELECT c.*, f.death_observation_end,
      date_diff('day',CAST(c.dischtime AS DATE),c.dod) AS death_days
      FROM cohort c JOIN death_followup f USING(subject_id)""")

    # Same-admission diagnoses have a common, unordered discharge-time group.
    db.execute("""CREATE OR REPLACE TABLE diagnosis_events AS SELECT m.subject_id,m.hadm_id,
      'diagnosis' event_type, CASE WHEN m.mapping_status IN ('direct','clear') THEN m.category ELSE concat('ICD9:',m.icd_code) END code,
      CAST(NULL AS VARCHAR) AS "value",CAST(NULL AS VARCHAR) unit,a.dischtime event_time,a.dischtime available_time,
      'diagnoses_icd' source_table,concat('hadm:',m.hadm_id) same_time_group,m.mapping_status,
      m.category AS disease_code,m.candidates AS ambiguous_candidates
      FROM mapped_diagnoses m JOIN admissions a USING(subject_id,hadm_id)
      JOIN patients p USING(subject_id)
      WHERE a.dischtime IS NOT NULL AND a.admittime IS NOT NULL AND a.dischtime>=a.admittime
        AND (p.dod IS NULL OR CAST(a.dischtime AS DATE)<=p.dod)""")
    db.execute("""CREATE OR REPLACE TABLE first_disease AS SELECT subject_id,disease_code,
      min(event_time) first_record_time,arg_min(same_time_group,event_time) first_same_time_group FROM diagnosis_events
      WHERE disease_code IS NOT NULL AND mapping_status IN ('direct','clear')
      GROUP BY subject_id,disease_code""")
    db.execute("""CREATE OR REPLACE TABLE ambiguous_history AS SELECT subject_id,
      unnest(string_split(ambiguous_candidates,',')) disease_code,event_time
      FROM diagnosis_events WHERE mapping_status='ambiguous' AND ambiguous_candidates IS NOT NULL""")
    sql_copy(db, "SELECT s.patient_key,s.split,f.disease_code,f.first_record_time,f.first_same_time_group FROM first_disease f JOIN split s USING(subject_id)", output / "trusted" / "first_disease.parquet")

    # Use documented availability times. Missing availability is excluded; no
    # retrospective interpolation or observed future stop/status is exposed.
    db.execute("""CREATE OR REPLACE VIEW lab_events AS SELECT CAST(l.subject_id AS BIGINT) subject_id,
      TRY_CAST(l.hadm_id AS BIGINT) hadm_id,'lab' event_type,concat('LAB:',l.itemid) code,
      COALESCE(l.valuenum,l.value) AS "value",l.valueuom unit,
      TRY_CAST(l.charttime AS TIMESTAMP) event_time,TRY_CAST(l.storetime AS TIMESTAMP) available_time,
      'labevents' source_table,concat('lab:',l.labevent_id) same_time_group,'direct' mapping_status,
      CAST(NULL AS VARCHAR) disease_code,CAST(NULL AS VARCHAR) ambiguous_candidates
      FROM raw_labevents l JOIN split s ON CAST(l.subject_id AS BIGINT)=s.subject_id
      WHERE TRY_CAST(l.charttime AS TIMESTAMP) IS NOT NULL AND TRY_CAST(l.storetime AS TIMESTAMP) IS NOT NULL
        AND TRY_CAST(l.storetime AS TIMESTAMP)>=TRY_CAST(l.charttime AS TIMESTAMP)""")
    db.execute("""CREATE OR REPLACE TABLE rx_orders AS SELECT CAST(subject_id AS BIGINT) subject_id,
      CAST(hadm_id AS BIGINT) hadm_id,pharmacy_id,MIN(TRY_CAST(entertime AS TIMESTAMP)) order_time
      FROM raw_pharmacy WHERE pharmacy_id IS NOT NULL AND pharmacy_id<>'' GROUP BY 1,2,3""")
    db.execute("""CREATE OR REPLACE VIEW prescription_events AS SELECT CAST(p.subject_id AS BIGINT) subject_id,
      CAST(p.hadm_id AS BIGINT) hadm_id,'prescription' event_type,concat('NDC:',p.ndc) code,
      p.dose_val_rx AS "value",p.dose_unit_rx unit,TRY_CAST(p.starttime AS TIMESTAMP) event_time,
      o.order_time available_time,'prescriptions' source_table,concat('rx:',p.pharmacy_id) same_time_group,
      'ndc' mapping_status,CAST(NULL AS VARCHAR) disease_code,CAST(NULL AS VARCHAR) ambiguous_candidates
      FROM raw_prescriptions p JOIN rx_orders o ON p.subject_id=CAST(o.subject_id AS VARCHAR)
        AND p.hadm_id=CAST(o.hadm_id AS VARCHAR) AND p.pharmacy_id=o.pharmacy_id
      JOIN split s ON CAST(p.subject_id AS BIGINT)=s.subject_id
      WHERE p.ndc IS NOT NULL AND regexp_full_match(p.ndc,'[0-9]{11}')
        AND o.order_time IS NOT NULL AND TRY_CAST(p.starttime AS TIMESTAMP) IS NOT NULL""")
    # Resolve the NDC using only prescription records whose linked pharmacy
    # order was already entered by the administration time. A later row with
    # the same pharmacy_id must neither supply nor invalidate a past code.
    db.execute("""CREATE OR REPLACE TABLE emar_code_sources AS SELECT
      CAST(p.subject_id AS BIGINT) subject_id,CAST(p.hadm_id AS BIGINT) hadm_id,
      p.pharmacy_id,p.ndc,MIN(greatest(o.order_time,
        TRY_CAST(p.starttime AS TIMESTAMP))) code_available_time
      FROM raw_prescriptions p JOIN rx_orders o
        ON p.subject_id=CAST(o.subject_id AS VARCHAR)
        AND p.hadm_id=CAST(o.hadm_id AS VARCHAR) AND p.pharmacy_id=o.pharmacy_id
      WHERE p.pharmacy_id IS NOT NULL AND regexp_full_match(p.ndc,'[0-9]{11}')
        AND o.order_time IS NOT NULL AND TRY_CAST(p.starttime AS TIMESTAMP) IS NOT NULL
      GROUP BY 1,2,3,4""")
    db.execute("""CREATE OR REPLACE VIEW administration_events AS WITH linked AS (
      SELECT e.subject_id,e.hadm_id,e.emar_id,e.pharmacy_id,e.charttime,e.storetime,
        MIN(c.ndc) ndc
      FROM raw_emar e JOIN emar_code_sources c
        ON CAST(e.subject_id AS BIGINT)=c.subject_id
        AND CAST(e.hadm_id AS BIGINT)=c.hadm_id AND e.pharmacy_id=c.pharmacy_id
        AND c.code_available_time<=TRY_CAST(e.charttime AS TIMESTAMP)
      JOIN split s ON CAST(e.subject_id AS BIGINT)=s.subject_id
      WHERE lower(e.event_txt) IN ('administered','given','started')
        AND TRY_CAST(e.charttime AS TIMESTAMP) IS NOT NULL
        AND TRY_CAST(e.storetime AS TIMESTAMP)>=TRY_CAST(e.charttime AS TIMESTAMP)
      GROUP BY 1,2,3,4,5,6 HAVING count(DISTINCT c.ndc)=1)
      SELECT CAST(e.subject_id AS BIGINT) subject_id,
      CAST(e.hadm_id AS BIGINT) hadm_id,'administration' event_type,concat('NDC:',e.ndc) code,
      CAST(NULL AS VARCHAR) AS "value",CAST(NULL AS VARCHAR) unit,
      TRY_CAST(e.charttime AS TIMESTAMP) event_time,TRY_CAST(e.storetime AS TIMESTAMP) available_time,
      'emar' source_table,concat('emar:',e.emar_id) same_time_group,'ndc' mapping_status,
      CAST(NULL AS VARCHAR) disease_code,CAST(NULL AS VARCHAR) ambiguous_candidates
      FROM linked e""")
    db.execute("""CREATE OR REPLACE TABLE timeline AS SELECT * FROM diagnosis_events
      UNION ALL SELECT * FROM lab_events UNION ALL SELECT * FROM prescription_events
      UNION ALL SELECT * FROM administration_events""")
    db.execute("CREATE OR REPLACE TABLE patient_origin AS SELECT subject_id,min(admittime) origin FROM admissions GROUP BY 1")
    sql_copy(db, """SELECT s.patient_key,s.split,t.hadm_id,
      CASE WHEN t.hadm_id IS NOT NULL THEN left(sha256(concat('natural-disease-progression-v1:stay:',
        CAST(t.subject_id AS VARCHAR),':',CAST(t.hadm_id AS VARCHAR))),24) END AS stay_key,
      t.event_type,t.code,t."value",t.unit,t.event_time,
      t.available_time,t.source_table,t.same_time_group,t.mapping_status,
      date_diff('second',o.origin,t.event_time) AS relative_seconds
      FROM timeline t JOIN split s USING(subject_id) JOIN patient_origin o USING(subject_id)
      WHERE s.split='train'""", output / "train" / "timeline.parquet")
    # Training is not tied to the benchmark's first cutoff. These tables give
    # every train patient and every admission (including diagnosis-empty stays)
    # a stable anonymous join key, plus the full available outcome boundary.
    sql_copy(db, """SELECT s.patient_key,p.gender,(p.anchor_age>=91) older_than_89,
      p.dod AS death_date,
      make_date(p.anchor_year+2022-CAST(right(p.anchor_year_group,4) AS INTEGER),12,31)
        AS diagnosis_observation_end,
      CAST(max(a.dischtime) AS DATE)+INTERVAL 365 DAY AS death_observation_end
      FROM split s JOIN patients p USING(subject_id)
      LEFT JOIN admissions a USING(subject_id)
      WHERE s.split='train'
      GROUP BY s.patient_key,p.gender,p.anchor_age,p.dod,p.anchor_year,p.anchor_year_group""",
      output / "train" / "patients.parquet")
    sql_copy(db, """SELECT s.patient_key,
      left(sha256(concat('natural-disease-progression-v1:stay:',
        CAST(a.subject_id AS VARCHAR),':',CAST(a.hadm_id AS VARCHAR))),24) AS stay_key,
      a.admittime,a.dischtime,
      (a.admittime IS NOT NULL AND a.dischtime IS NOT NULL AND a.dischtime>=a.admittime)
        AS chronology_valid,
      p.anchor_age+year(a.admittime)-p.anchor_year AS approx_age_at_admit,
      p.anchor_age+year(a.dischtime)-p.anchor_year AS approx_age_at_discharge,
      coalesce(list_sort(list_distinct(list(struct_pack(
        code:=CASE WHEN m.mapping_status IN ('direct','clear') THEN m.category
                   ELSE concat('ICD9:',m.icd_code) END,
        mapping_status:=m.mapping_status)) FILTER (WHERE m.icd_code IS NOT NULL))),
        []::STRUCT(code VARCHAR,mapping_status VARCHAR)[]) AS diagnoses
      FROM admissions a JOIN split s USING(subject_id) JOIN patients p USING(subject_id)
      LEFT JOIN mapped_diagnoses m ON m.subject_id=a.subject_id AND m.hadm_id=a.hadm_id
      WHERE s.split='train'
      GROUP BY s.patient_key,a.subject_id,a.hadm_id,a.admittime,a.dischtime,
        p.anchor_age,p.anchor_year
      ORDER BY s.patient_key,a.admittime,a.hadm_id""", output / "train" / "admissions.parquet")

    # Public sample input files contain neither raw IDs nor post-cutoff rows.
    for group in ("train", "validation", "test"):
        input_dir = output / "train" if group == "train" else output / "public" / group
        sql_copy(db, f"""SELECT sample_id,gender,age_at_cutoff AS approx_age,(anchor_age>=91) AS older_than_89,
          dischtime AS cutoff FROM samples WHERE split='{group}' ORDER BY sample_id""", input_dir / "samples.parquet")
        sql_copy(db, f"""SELECT c.sample_id,t.hadm_id,t.event_type,t.code,t."value",t.unit,t.event_time,
          t.available_time,t.source_table,t.same_time_group,t.mapping_status,
          date_diff('second',t.event_time,c.dischtime) AS seconds_before_cutoff
          FROM samples c JOIN timeline t USING(subject_id) WHERE c.split='{group}'
          AND t.event_time<=c.dischtime AND t.available_time<=c.dischtime
          ORDER BY c.sample_id,t.available_time,t.event_time,t.event_type,t.code,t.same_time_group""", input_dir / "events.parquet")

    # Freeze the disease list from TRAIN labels only, before test scoring.
    db.execute("""CREATE OR REPLACE TABLE train_events AS SELECT f.disease_code,count(DISTINCT c.subject_id) positives
      FROM samples c JOIN first_disease f USING(subject_id)
      WHERE c.split='train' AND f.first_record_time>c.dischtime
        AND CAST(f.first_record_time AS DATE)<=CAST(c.dischtime AS DATE)+INTERVAL 365 DAY
        AND CAST(f.first_record_time AS DATE)<=c.diagnosis_observation_end
        AND (left(f.disease_code,1) BETWEEN 'A' AND 'Q' OR left(f.disease_code,1)='U')
        AND NOT EXISTS(SELECT 1 FROM ambiguous_history a WHERE a.subject_id=c.subject_id
          AND a.disease_code=f.disease_code AND a.event_time<=c.dischtime)
      GROUP BY f.disease_code""")
    diseases = [r[0] for r in db.execute("SELECT disease_code FROM train_events WHERE positives>=? ORDER BY disease_code", [min_train_events]).fetchall()]
    if not diseases:
        raise ValueError("No disease meets the frozen train event threshold")
    db.register("disease_frame", pd.DataFrame({"disease_code": diseases}))
    db.execute("CREATE OR REPLACE TABLE scored_diseases AS SELECT * FROM disease_frame")
    sql_copy(db, "SELECT e.disease_code,e.positives AS train_365d_positives FROM train_events e JOIN scored_diseases d USING(disease_code) ORDER BY 1", output / "scored_diseases.parquet")
    db.execute("""CREATE OR REPLACE TABLE label_state AS SELECT c.sample_id,c.subject_id,c.split,d.disease_code,
      (f.first_record_time IS NULL OR f.first_record_time>c.dischtime) AS risk_set,
      EXISTS(SELECT 1 FROM ambiguous_history a WHERE a.subject_id=c.subject_id AND a.disease_code=d.disease_code AND a.event_time<=c.dischtime) AS ambiguous_prior,
      CASE WHEN f.first_record_time>c.dischtime THEN f.first_record_time END AS first_future_record_time,
      CASE WHEN f.first_record_time>c.dischtime THEN f.first_same_time_group END AS future_same_time_group,
      CASE WHEN f.first_record_time>c.dischtime THEN date_diff('day',CAST(c.dischtime AS DATE),CAST(f.first_record_time AS DATE)) END AS event_days,
      c.dischtime AS cutoff,c.diagnosis_observation_end,c.dod,c.death_days,c.death_observation_end
      FROM samples c CROSS JOIN scored_diseases d LEFT JOIN first_disease f
      ON c.subject_id=f.subject_id AND d.disease_code=f.disease_code""")
    for group in ("validation", "test"):
        sql_copy(db, f"""SELECT sample_id,disease_code,risk_set,ambiguous_prior,first_future_record_time,future_same_time_group,event_days,
          CAST(cutoff AS DATE) cutoff,diagnosis_observation_end,death_days,CAST(death_observation_end AS DATE) death_observation_end
          FROM label_state WHERE split='{group}' ORDER BY sample_id,disease_code""",
          output / "trusted" / group / "labels.parquet")
    sql_copy(db, """SELECT sample_id,disease_code,risk_set,ambiguous_prior,first_future_record_time,future_same_time_group,event_days,
      CAST(cutoff AS DATE) cutoff,diagnosis_observation_end,death_days,CAST(death_observation_end AS DATE) death_observation_end
      FROM label_state WHERE split='train'""", output / "train" / "labels.parquet")
    for group in ("validation", "test"):
        sql_copy(db, f"""SELECT c.sample_id,c.split,CAST(c.dischtime AS DATE) cutoff,
          c.dod AS death_date,c.death_days,c.diagnosis_observation_end,
          CAST(c.death_observation_end AS DATE) death_observation_end,
          EXISTS(SELECT 1 FROM first_disease f WHERE f.subject_id=c.subject_id
            AND f.first_record_time>c.dischtime
            AND CAST(f.first_record_time AS DATE)<=CAST(c.dischtime AS DATE)+INTERVAL 365 DAY
            AND CAST(f.first_record_time AS DATE)<=c.diagnosis_observation_end
            AND (c.death_days IS NULL OR
                 CAST(f.first_record_time AS DATE)<=CAST(c.dischtime AS DATE)+c.death_days*INTERVAL 1 DAY))
            AS any_future_disease_record_365
          FROM samples c WHERE c.split='{group}' ORDER BY c.sample_id""",
          output / "trusted" / group / "sample_boundaries.parquet")
    manifest={"schema":"natural-disease-progression/v1","source":"MIMIC-IV 3.1",
              "task":"first-recorded-disease-time-and-death-risk",
              "split_seed":SPLIT_SEED,"horizons_days":HORIZONS,"train_min_365d_events":min_train_events,
              "num_scored_diseases":len(diseases),"scored_diseases":diseases,
              "primary_metric":"age_sex_normalized_disease_time_brier",
              "notes":"Predictions are direct cumulative risks at every listed day. Raw MIMIC and trusted labels stay local; only public/{validation,test} is model-visible."}
    (output / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    for group in ("validation","test"):
        (output / "public" / group / "manifest.json").write_text(json.dumps({"schema":"natural-disease-progression-input/v1",
          "task":"first-recorded-disease-time-and-death-risk",
          "horizons_days":HORIZONS,"scored_diseases":diseases,"split":group,
          "primary_metric":"age_sex_normalized_disease_time_brier"},indent=2)+"\n")
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
    (output / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    db.close()
    # Remove superseded v1 trajectory targets if this is an in-place rerun.
    # Full train timelines and all-admission exports above remain available.
    for obsolete in (
        output / "output_vocabulary.parquet",
        output / "trajectory_audit.json",
        output / "train" / "trajectories.parquet",
        output / "train" / "trajectory_exclusions.parquet",
        output / "train" / "admission_trajectories.parquet",
        output / "trusted" / "trajectories.parquet",
        output / "trusted" / "trajectory_exclusions.parquet",
        output / "trusted" / "admission_trajectories.parquet",
        # Pre-boundary layouts combined validation and test truth. Never leave
        # these files behind after an in-place regeneration.
        output / "trusted" / "labels.parquet",
        output / "trusted" / "label_windows.parquet",
        output / "trusted" / "sample_boundaries.parquet",
    ):
        obsolete.unlink(missing_ok=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--mimic-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--min-train-events",type=int,default=50)
    args=parser.parse_args()
    run(args.mimic_root,args.output,args.min_train_events)
