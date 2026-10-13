"""Read-only, paired cutoff/horizon audit; does not change v1 artifacts.

Both origins refer to the admission selected for the frozen first eligible
live-discharge cohort. This is a retrospective comparison, not an admission-
time deployment cohort: eligibility itself uses the eventual live discharge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


HORIZONS = (365, 730, 1095)
MODES = ("first_new", "all_recorded")


def run(materialized: Path) -> list[dict]:
    db = duckdb.connect(str(materialized / "working.duckdb"), read_only=True)
    db.execute("SET threads=4")
    db.execute("SET memory_limit='4GB'")
    split_path = materialized / "trusted" / "subject_split.parquet"
    mismatch = db.execute("""SELECT count(*) FROM split s FULL OUTER JOIN
      read_parquet(?) p USING(subject_id) WHERE s.subject_id IS NULL OR p.subject_id IS NULL
      OR s.split<>p.split OR s.patient_key<>p.patient_key""", [str(split_path)]).fetchone()[0]
    if mismatch:
        raise ValueError("Frozen patient split differs from materialized split")

    # Existing public/train input is already filtered by the established
    # event-time and availability-time rules at discharge. For admission, the
    # earliest time when both conditions hold must be no later than admittime.
    event_paths = [str(materialized / section / "events.parquet") for section in
                   ("train", "public/validation", "public/test")]
    db.execute("""CREATE TEMP TABLE modality AS SELECT sample_id,event_type,
      min(greatest(event_time,available_time)) first_usable_time
      FROM read_parquet(?) WHERE event_type IN ('lab','prescription','administration')
      GROUP BY 1,2""", [event_paths])
    db.execute("""CREATE TEMP TABLE input_coverage AS SELECT s.sample_id,
      bool_or(m.event_type='lab' AND m.first_usable_time<=s.admittime) lab_admission,
      bool_or(m.event_type IN ('prescription','administration')
        AND m.first_usable_time<=s.admittime) medication_admission,
      bool_or(m.event_type='lab') lab_discharge,
      bool_or(m.event_type IN ('prescription','administration')) medication_discharge
      FROM samples s LEFT JOIN modality m USING(sample_id) GROUP BY s.sample_id""")

    # A first-new disease is a clear, mapped first category record after the
    # selected origin. Historical ambiguous candidate categories are masked.
    # The all-recorded alternative retains every diagnosis code (including
    # explicit ICD-9 ambiguous/unmapped tokens) and permits repeats across stays.
    db.execute("""CREATE TEMP VIEW future_groups AS
      SELECT s.sample_id,'first_new' AS mode,
        'admission' AS origin,f.first_record_time event_time
      FROM samples s JOIN first_disease f USING(subject_id)
      WHERE f.first_record_time>s.admittime
        AND CAST(f.first_record_time AS DATE)<=CAST(s.admittime AS DATE)+INTERVAL 1095 DAY
        AND CAST(f.first_record_time AS DATE)<=s.diagnosis_observation_end
        AND (s.dod IS NULL OR CAST(f.first_record_time AS DATE)<=s.dod)
        AND NOT EXISTS(SELECT 1 FROM ambiguous_history a WHERE a.subject_id=s.subject_id
          AND a.disease_code=f.disease_code AND a.event_time<=s.admittime)
      UNION ALL
      SELECT s.sample_id,'first_new','discharge',f.first_record_time
      FROM samples s JOIN first_disease f USING(subject_id)
      WHERE f.first_record_time>s.dischtime
        AND CAST(f.first_record_time AS DATE)<=CAST(s.dischtime AS DATE)+INTERVAL 1095 DAY
        AND CAST(f.first_record_time AS DATE)<=s.diagnosis_observation_end
        AND (s.dod IS NULL OR CAST(f.first_record_time AS DATE)<=s.dod)
        AND NOT EXISTS(SELECT 1 FROM ambiguous_history a WHERE a.subject_id=s.subject_id
          AND a.disease_code=f.disease_code AND a.event_time<=s.dischtime)
      UNION ALL
      SELECT s.sample_id,'all_recorded','admission',d.event_time
      FROM samples s JOIN diagnosis_events d USING(subject_id)
      WHERE d.event_time>s.admittime
        AND CAST(d.event_time AS DATE)<=CAST(s.admittime AS DATE)+INTERVAL 1095 DAY
        AND CAST(d.event_time AS DATE)<=s.diagnosis_observation_end
        AND (s.dod IS NULL OR CAST(d.event_time AS DATE)<=s.dod)
      UNION ALL
      SELECT s.sample_id,'all_recorded','discharge',d.event_time
      FROM samples s JOIN diagnosis_events d USING(subject_id)
      WHERE d.event_time>s.dischtime
        AND CAST(d.event_time AS DATE)<=CAST(s.dischtime AS DATE)+INTERVAL 1095 DAY
        AND CAST(d.event_time AS DATE)<=s.diagnosis_observation_end
        AND (s.dod IS NULL OR CAST(d.event_time AS DATE)<=s.dod)""")
    db.execute("""CREATE TEMP TABLE group_counts AS SELECT f.sample_id,f.mode,f.origin,h.days,
      count(DISTINCT f.event_time) n_groups FROM future_groups f JOIN samples s USING(sample_id)
      CROSS JOIN (VALUES (365),(730),(1095)) h(days)
      WHERE CAST(f.event_time AS DATE)<=CAST(CASE WHEN f.origin='admission'
        THEN s.admittime ELSE s.dischtime END AS DATE)
        + h.days * INTERVAL 1 DAY GROUP BY 1,2,3,4""")
    missing_full_group = db.execute("""SELECT count(*) FROM group_counts n
      LEFT JOIN group_counts a ON a.sample_id=n.sample_id AND a.origin=n.origin
        AND a.days=n.days AND a.mode='all_recorded'
      WHERE n.mode='first_new' AND n.n_groups>coalesce(a.n_groups,0)""").fetchone()[0]
    if missing_full_group:
        raise AssertionError("First-new group count exceeds all-recorded group count")

    eligible_patients = db.execute("SELECT count(*) FROM samples").fetchone()[0]
    rows = []
    for origin in ("admission", "discharge"):
        cutoff = "s.admittime" if origin == "admission" else "s.dischtime"
        for days in HORIZONS:
            for mode in MODES:
                values = db.execute(f"""SELECT count(*) diagnosis_complete,
                  count(*) FILTER(WHERE s.diagnosis_observation_end>=CAST({cutoff} AS DATE)
                    + INTERVAL {days} DAY AND s.death_observation_end>=CAST({cutoff} AS DATE)
                    + INTERVAL {days} DAY) diagnosis_and_death_complete,
                  count(*) FILTER(WHERE i.lab_{origin}) input_lab,
                  count(*) FILTER(WHERE i.medication_{origin}) input_medication,
                  count(*) FILTER(WHERE coalesce(g.n_groups,0)=0) groups_0,
                  count(*) FILTER(WHERE g.n_groups=1) groups_1,
                  count(*) FILTER(WHERE g.n_groups=2) groups_2,
                  count(*) FILTER(WHERE g.n_groups>=3) groups_3_plus
                  FROM samples s JOIN input_coverage i USING(sample_id)
                  LEFT JOIN group_counts g ON g.sample_id=s.sample_id AND g.origin=?
                    AND g.mode=? AND g.days=?
                  WHERE s.diagnosis_observation_end>=CAST({cutoff} AS DATE)
                    + INTERVAL {days} DAY""", [origin, mode, days]).fetchone()
                rows.append(dict(zip(("diagnosis_complete", "diagnosis_and_death_complete", "input_lab",
                                      "input_medication", "groups_0", "groups_1",
                                      "groups_2", "groups_3_plus"), values),
                                 eligible_patients=eligible_patients, origin=origin,
                                 horizon_days=days, target=mode))
    db.close()
    for row in rows:
        if sum(row[f"groups_{part}"] for part in ("0", "1", "2", "3_plus")) != row["diagnosis_complete"]:
            raise AssertionError(f"Group counts do not sum to covered cohort: {row}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.materialized), indent=2))
