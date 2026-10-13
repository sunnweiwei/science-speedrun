"""First-pass MIMIC-IV v3.1 audit; no cohort or scoring thresholds are inferred here."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from protocol import HORIZONS, GEM_SHA256, read_gem


def csv_view(db, name: str, root: Path) -> None:
    path = root / "hosp" / f"{name}.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    db.execute(f"CREATE OR REPLACE VIEW raw_{name} AS SELECT * FROM read_csv('{path.as_posix()}', header=true, all_varchar=true)")


def q(db, statement):
    return db.execute(statement).fetchall()


def run(root: Path, output: Path, gem: Path) -> dict:
    if not (root / "CHANGELOG.txt").is_file() or "MIMIC-IV v3.1" not in (root / "CHANGELOG.txt").read_text()[:200]:
        raise ValueError("This protocol requires the isolated MIMIC-IV 3.1 directory")
    output.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect(str(output / "working.duckdb"))
    db.execute("SET threads=4")
    db.execute("SET memory_limit='4GB'")
    for name in ("patients", "admissions", "diagnoses_icd", "d_icd_diagnoses", "labevents", "prescriptions", "emar"):
        csv_view(db, name, root)
    db.execute("CREATE OR REPLACE TABLE patients AS SELECT CAST(subject_id AS BIGINT) subject_id, gender, CAST(anchor_age AS INTEGER) anchor_age, CAST(anchor_year AS INTEGER) anchor_year, anchor_year_group, TRY_CAST(dod AS DATE) dod FROM raw_patients")
    db.execute("CREATE OR REPLACE TABLE admissions AS SELECT CAST(subject_id AS BIGINT) subject_id, CAST(hadm_id AS BIGINT) hadm_id, TRY_CAST(admittime AS TIMESTAMP) admittime, TRY_CAST(dischtime AS TIMESTAMP) dischtime, TRY_CAST(deathtime AS TIMESTAMP) deathtime, hospital_expire_flag FROM raw_admissions")
    db.execute("CREATE OR REPLACE TABLE diagnoses AS SELECT CAST(subject_id AS BIGINT) subject_id, CAST(hadm_id AS BIGINT) hadm_id, icd_code, CAST(icd_version AS INTEGER) icd_version FROM raw_diagnoses_icd")
    db.execute("CREATE OR REPLACE TABLE icd10_dictionary AS SELECT DISTINCT icd_code FROM raw_d_icd_diagnoses WHERE icd_version='10'")
    db.execute("CREATE OR REPLACE TABLE adult_admissions AS SELECT a.*, p.anchor_age + year(a.admittime)-p.anchor_year AS approx_age, p.gender, p.anchor_year, p.anchor_year_group, p.dod FROM admissions a JOIN patients p USING(subject_id) WHERE a.admittime IS NOT NULL AND p.anchor_age + year(a.admittime)-p.anchor_year >= 18")
    counts = q(db, "SELECT COUNT(DISTINCT subject_id), COUNT(*) FROM adult_admissions")[0]
    stays = q(db, "SELECT stays, COUNT(*) FROM (SELECT subject_id, COUNT(*) stays FROM adult_admissions GROUP BY 1) GROUP BY 1 ORDER BY 1")
    gem_map = read_gem(gem)
    mapping_rows = [(code, status, target, ",".join(candidates)) for code, (status, target, candidates) in gem_map.items()]
    db.register("gem_frame", pd.DataFrame(mapping_rows, columns=["icd_code","mapping_status","category","candidates"]))
    db.execute("CREATE OR REPLACE TABLE gem_map AS SELECT * FROM gem_frame")
    mapping = q(db, """SELECT d.icd_version, COALESCE(CASE WHEN d.icd_version=10 THEN 'direct' ELSE g.mapping_status END,'unmapped') status, COUNT(*)
        FROM diagnoses d LEFT JOIN gem_map g ON d.icd_version=9 AND d.icd_code=g.icd_code GROUP BY 1,2 ORDER BY 1,2""")
    db.execute("""CREATE OR REPLACE TABLE mapped_diagnoses AS SELECT d.*, CASE WHEN d.icd_version=10 AND dict.icd_code IS NOT NULL
          AND regexp_full_match(upper(left(replace(d.icd_code,'.',''),3)),'[A-Z][0-9][0-9A-Z]')
          THEN upper(left(replace(d.icd_code,'.',''),3)) WHEN d.icd_version=9 THEN g.category END category,
        CASE WHEN d.icd_version=10 AND dict.icd_code IS NOT NULL THEN 'direct'
          WHEN d.icd_version=10 THEN 'invalid' ELSE COALESCE(g.mapping_status,'unmapped') END mapping_status,
        g.candidates FROM diagnoses d LEFT JOIN gem_map g ON d.icd_version=9 AND d.icd_code=g.icd_code
        LEFT JOIN icd10_dictionary dict ON d.icd_version=10 AND d.icd_code=dict.icd_code""")
    new_events = q(db, """WITH first_stay AS (SELECT subject_id, MIN(dischtime) first_discharge FROM adult_admissions GROUP BY 1),
        first_code AS (SELECT m.subject_id,m.category,MIN(a.dischtime) first_time FROM mapped_diagnoses m JOIN adult_admissions a USING(subject_id,hadm_id)
        WHERE m.category IS NOT NULL AND a.dischtime IS NOT NULL GROUP BY 1,2)
        SELECT CASE WHEN f.first_time > s.first_discharge THEN 'after_first' ELSE 'initial' END phase, COUNT(*)
        FROM first_code f JOIN first_stay s USING(subject_id) GROUP BY 1""")
    per_patient_new = q(db, """WITH first_stay AS (SELECT subject_id,MIN(dischtime) first_discharge FROM adult_admissions GROUP BY 1),
      events AS (SELECT m.subject_id,m.category,MIN(a.dischtime) first_time FROM mapped_diagnoses m JOIN adult_admissions a USING(subject_id,hadm_id) WHERE m.category IS NOT NULL GROUP BY 1,2),
      c AS (SELECT s.subject_id, COUNT(e.category) n FROM first_stay s LEFT JOIN events e ON s.subject_id=e.subject_id AND e.first_time>s.first_discharge GROUP BY 1)
      SELECT quantile_cont(n,[0,.25,.5,.75,.9,.99,1]) FROM c""")[0][0]
    raw_coverage = {}
    for name in ("labevents", "prescriptions", "emar"):
        raw_coverage[name] = dict(zip(("rows", "patients", "admissions"), q(db, f"SELECT count(*),count(DISTINCT subject_id),count(DISTINCT hadm_id) FROM raw_{name}")[0]))
    timestamps = {
        "admissions": q(db, "SELECT SUM(admittime IS NULL),SUM(dischtime IS NULL),SUM(dischtime<admittime),SUM(deathtime IS NOT NULL AND deathtime>dischtime) FROM admissions")[0],
        "labevents": q(db, "SELECT SUM(charttime IS NULL OR charttime=''),SUM(storetime IS NULL OR storetime=''),SUM(TRY_CAST(storetime AS TIMESTAMP)<TRY_CAST(charttime AS TIMESTAMP)),quantile_cont(date_diff('hour',TRY_CAST(charttime AS TIMESTAMP),TRY_CAST(storetime AS TIMESTAMP)),[.5,.9,.99]) FROM raw_labevents")[0],
        "prescriptions": q(db, "SELECT SUM(starttime IS NULL OR starttime=''),SUM(stoptime IS NULL OR stoptime=''),SUM(TRY_CAST(stoptime AS TIMESTAMP)<TRY_CAST(starttime AS TIMESTAMP)) FROM raw_prescriptions")[0],
        "emar": q(db, "SELECT SUM(charttime IS NULL OR charttime=''),SUM(storetime IS NULL OR storetime=''),SUM(TRY_CAST(storetime AS TIMESTAMP)<TRY_CAST(charttime AS TIMESTAMP)),quantile_cont(date_diff('hour',TRY_CAST(charttime AS TIMESTAMP),TRY_CAST(storetime AS TIMESTAMP)),[.5,.9,.99]) FROM raw_emar")[0],
        "death": q(db, "SELECT COUNT(*) FILTER(WHERE dod IS NOT NULL),COUNT(*) FILTER(WHERE dod<CAST(admittime AS DATE)),COUNT(*) FILTER(WHERE dod<CAST(dischtime AS DATE)),COUNT(*) FILTER(WHERE deathtime IS NOT NULL AND dod IS NOT NULL AND CAST(deathtime AS DATE)<>dod) FROM adult_admissions")[0],
    }
    windows = {}
    for h in HORIZONS:
        windows[str(h)] = q(db, f"""WITH index_stay AS (SELECT *,ROW_NUMBER() OVER(PARTITION BY subject_id ORDER BY dischtime,hadm_id) rn FROM adult_admissions
          WHERE dischtime IS NOT NULL AND hospital_expire_flag='0' AND (dod IS NULL OR dod>CAST(dischtime AS DATE)) AND (deathtime IS NULL OR deathtime>dischtime)),
          eligible AS (SELECT * FROM index_stay WHERE rn=1)
          SELECT count(*),count(*) FILTER(WHERE CAST(dischtime AS DATE) + INTERVAL {h} DAY <=
          make_date(anchor_year+2022-CAST(right(anchor_year_group,4) AS INTEGER),12,31)) FROM eligible""")[0]
    target_counts = q(db, """WITH eligible AS (SELECT *,ROW_NUMBER() OVER(PARTITION BY subject_id ORDER BY dischtime,hadm_id) rn
      FROM adult_admissions WHERE dischtime IS NOT NULL AND hospital_expire_flag='0'
      AND (dod IS NULL OR dod>CAST(dischtime AS DATE)) AND (deathtime IS NULL OR deathtime>dischtime)),
      index_stay AS (SELECT * FROM eligible WHERE rn=1),
      first_code AS (SELECT m.subject_id,m.category,MIN(a.dischtime) first_time FROM mapped_diagnoses m
        JOIN adult_admissions a USING(subject_id,hadm_id) WHERE m.category IS NOT NULL AND a.dischtime IS NOT NULL GROUP BY 1,2)
      SELECT f.category,COUNT(*) FILTER(WHERE f.first_time>i.dischtime
        AND CAST(f.first_time AS DATE)<=CAST(i.dischtime AS DATE)+INTERVAL 365 DAY) positives_365d
      FROM first_code f JOIN index_stay i USING(subject_id) GROUP BY 1 ORDER BY positives_365d DESC,1""")
    result = {"source_version":"MIMIC-IV 3.1", "gem_sha256":GEM_SHA256,
              "adult_inpatient_patients":counts[0],"adult_admissions":counts[1],
              "admissions_per_patient":dict(stays),"new_disease_events":dict(new_events),
              "new_disease_events_per_patient_quantiles":per_patient_new,
              "mapping_rows":[{"version":v,"status":s,"rows":n} for v,s,n in mapping],
              "modality_coverage":raw_coverage,"timestamp_audit":timestamps,
              "candidate_windows":{k:{"surviving_index":v[0],"conservative_diagnosis_coverage":v[1]} for k,v in windows.items()},
              "candidate_disease_positive_counts_365d":dict(target_counts)}
    (output / "audit.json").write_text(json.dumps(result,indent=2,default=str)+"\n")
    db.close()
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--mimic-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gem",type=Path,default=Path(__file__).with_name("cms-2018-gem.zip"))
    args=parser.parse_args()
    print(json.dumps(run(args.mimic_root,args.output,args.gem),indent=2,default=str))
