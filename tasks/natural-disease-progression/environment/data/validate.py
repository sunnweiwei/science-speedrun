"""Full-materialization invariants; writes aggregate-only validation.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


def validate(root: Path) -> dict:
    db = duckdb.connect(str(root / "working.duckdb"), read_only=True)
    split = db.execute("SELECT count(*),count(DISTINCT subject_id),count(*) FILTER(WHERE split='train'),count(*) FILTER(WHERE split='validation'),count(*) FILTER(WHERE split='test') FROM split").fetchone()
    if split[0] != split[1]:
        raise AssertionError("Patients overlap across splits")
    cohort = db.execute("""WITH stay_counts AS (SELECT subject_id,count(*) n FROM adult_admissions GROUP BY 1)
      SELECT count(*),count(*) FILTER(WHERE stay_counts.n=1) FROM samples JOIN stay_counts USING(subject_id)""").fetchone()
    result = {"split_patients":{"train":split[2],"validation":split[3],"test":split[4]},
              "index_samples":cohort[0],"single_admission_index_samples":cohort[1],
              "public_input_violations":{},"label_violations":{}}
    train_admissions=(root/"train"/"admissions.parquet").as_posix()
    train_patients=(root/"train"/"patients.parquet").as_posix()
    expected_stays=db.execute("""SELECT count(*) FROM admissions a JOIN split s USING(subject_id)
      WHERE s.split='train'""").fetchone()[0]
    exported_stays=db.execute(f"SELECT count(*),count(*) FILTER(WHERE diagnoses IS NULL) FROM read_parquet('{train_admissions}')").fetchone()
    exported_patients=db.execute(f"SELECT count(*),count(DISTINCT patient_key) FROM read_parquet('{train_patients}')").fetchone()
    result["train_full_history"]={"admissions":exported_stays[0],
      "expected_admissions":expected_stays,"patients":exported_patients[0]}
    if exported_stays!=(expected_stays,0) or exported_patients!=(split[2],split[2]):
        raise AssertionError("Train full-admission or patient export incomplete")
    medication_bad=db.execute("""SELECT count(*) FROM timeline t
      WHERE t.event_type='administration' AND NOT EXISTS (
        SELECT 1 FROM emar_code_sources c WHERE c.subject_id=t.subject_id
          AND c.hadm_id=t.hadm_id AND concat('NDC:',c.ndc)=t.code
          AND c.code_available_time<=t.event_time)""").fetchone()[0]
    result["administration_code_source_violations"]=medication_bad
    if medication_bad:
        raise AssertionError("Administration NDC came from a later prescription")
    for group in ("validation","test"):
        folder=root/"public"/group
        e,s=(folder/"events.parquet").as_posix(),(folder/"samples.parquet").as_posix()
        schema=pq.read_schema(e).names + pq.read_schema(s).names
        if "subject_id" in schema or "patient_key" in schema or "dod" in schema or "death_days" in schema:
            raise AssertionError("Raw identifier or future outcome in public schema")
        bad=db.execute(f"""SELECT count(*) FROM read_parquet('{e}') e JOIN read_parquet('{s}') s USING(sample_id)
          WHERE e.event_time>s.cutoff OR e.available_time>s.cutoff""").fetchone()[0]
        result["public_input_violations"][group]=bad
        if bad:
            raise AssertionError("Post-cutoff event in public input")
    for section in ("train", "trusted/validation", "trusted/test"):
        path=(root/section/"label_windows.parquet").as_posix()
        bad=db.execute(f"""SELECT count(*) FILTER(WHERE disease_label_365 AND NOT disease_valid_365),
          count(*) FILTER(WHERE (NOT risk_set OR ambiguous_prior) AND disease_valid_365),
          count(*) FILTER(WHERE event_days IS NULL AND death_days IS NULL
            AND cutoff+INTERVAL 365 DAY>diagnosis_observation_end AND disease_valid_365),
          count(*) FILTER(WHERE event_days>death_days AND disease_label_365)
          FROM read_parquet('{path}')""").fetchone()
        result["label_violations"][section]={"positive_invalid":bad[0],"outside_risk_set":bad[1],
          "partial_followup_negative":bad[2],"after_death_positive":bad[3]}
        if any(bad):
            raise AssertionError(f"Label invariant failed in {section}: {bad}")
        expected_columns={f"{kind}_{h}" for h in range(30,361,30)
                          for kind in ("disease_label","disease_valid","death_label","death_valid")}
        expected_columns.update({f"{kind}_365" for kind in
                                 ("disease_label","disease_valid","death_label","death_valid")})
        missing=expected_columns-set(pq.read_schema(path).names)
        if missing:
            raise AssertionError(f"Missing direct time-grid labels in {section}: {sorted(missing)}")
    normalization=(root/"trusted"/"risk_normalization.parquet").as_posix()
    norm=db.execute(f"""SELECT count(*),count(DISTINCT disease_code),
      count(DISTINCT concat(disease_code,':',CAST(horizon_days AS VARCHAR))),
      min(denominator_brier),max(normalization_weight),
      count(*) FILTER(WHERE abs(normalization_weight-1.0/denominator_brier)>1e-8),
      count(*) FILTER(WHERE horizon_days NOT IN ({','.join(map(str, list(range(30,361,30))+[365]))})),
      count(*) FILTER(WHERE d.disease_code IS NULL)
      FROM read_parquet('{normalization}') n LEFT JOIN scored_diseases d USING(disease_code)""").fetchone()
    expected_cells=db.execute("SELECT count(*)*13 FROM scored_diseases").fetchone()[0]
    result["risk_normalization"]={"cells":norm[0],"diseases":norm[1],
      "minimum_denominator":norm[3],"maximum_weight":norm[4]}
    if (norm[0]!=expected_cells or norm[2]!=expected_cells or norm[3]<=0
            or norm[4]>1000.0+1e-9 or norm[5] or norm[6] or norm[7]):
        raise AssertionError(f"Train-only risk normalization invariant failed: {norm}")
    frozen=(root/"trusted"/"subject_split.parquet").as_posix()
    mismatches=db.execute(f"""SELECT count(*) FROM split s FULL OUTER JOIN
      read_parquet('{frozen}') f USING(subject_id) WHERE s.subject_id IS NULL
      OR f.subject_id IS NULL OR s.split<>f.split OR s.patient_key<>f.patient_key""").fetchone()[0]
    if mismatches:
        raise AssertionError("Frozen patient split was modified")
    for group in ("validation", "test"):
        public_ids = (root / "public" / group / "samples.parquet").as_posix()
        labels = (root / "trusted" / group / "label_windows.parquet").as_posix()
        boundaries = (root / "trusted" / group / "sample_boundaries.parquet").as_posix()
        coverage = db.execute(f"""SELECT
          (SELECT count(DISTINCT sample_id) FROM read_parquet('{public_ids}')),
          (SELECT count(DISTINCT sample_id) FROM read_parquet('{labels}')),
          (SELECT count(DISTINCT sample_id) FROM read_parquet('{boundaries}')),
          (SELECT count(*) FROM read_parquet('{labels}') WHERE NOT starts_with(sample_id, '{group}_')),
          (SELECT count(*) FROM read_parquet('{boundaries}') WHERE split<>'{group}')""").fetchone()
        if coverage[0] != coverage[1] or coverage[0] != coverage[2] or coverage[3] or coverage[4]:
            raise AssertionError(f"{group} trusted truth is mixed with another split: {coverage}")
    db.close()
    (root/"validation.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--materialized",required=True,type=Path)
    print(json.dumps(validate(parser.parse_args().materialized),indent=2))
