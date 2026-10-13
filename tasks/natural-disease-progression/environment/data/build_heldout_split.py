#!/usr/bin/env python3
"""Re-split the longitudinal-v2 development release into the evaluated runtime trees.

    python build_heldout_split.py --source DEV_RELEASE --output RUNTIME_ROOT

SOURCE is the controlled development release (`runtime/development` of longitudinal-v2, HF
revision f7da2d0): train and validation inputs with their labels. The output holds
`development/` (mounted into the agent container) and `private/` (verifier only), each with a
SHA-256 inventory.

Why a re-split: MIMIC-IV coded diagnoses in ICD-9 until October 2015 and in ICD-10 afterwards.
The first-record labels count ICD-9-era outcomes only where the mapping is unambiguous, so one-year
incidence after an ICD-9-era index admission is a fraction of the true rate (sepsis 0.15 % vs
3.3 %, heart failure 0.13 % vs 2.5 %). With both eras in every split, recognising the coding era is
worth more legacy AUROC than age and sex. Validation and test therefore hold only ICD-10-era index
admissions, whose outcomes are completely coded. Training keeps every other patient; the
incomplete ICD-9-era labels are part of the problem.

The era of a sample is the coding system of the diagnoses in its index (second) admission. Event
dates are shifted per patient, but `mapping_status` survives: `direct` is native ICD-10.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import duckdb

SALT = "ndp-heldout-v3"
TEST_FRACTION = 0.30
VALIDATION_FRACTION = 0.10
SPLIT_SCHEME = "icd10-era-heldout/v3"
PRIMARY_METRIC = "panel_recall_at_5pct"


def bucket(sample_id: str) -> float:
    return int(hashlib.sha256(f"{SALT}:{sample_id}".encode()).hexdigest()[:12], 16) / 16**12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path, kind: str) -> None:
    files = {str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(root.rglob("*")) if p.is_file() and p.name != "runtime-manifest.json"}
    (root / "runtime-manifest.json").write_text(json.dumps(
        {"schema": "natural-disease-progression-runtime-data/v1", "kind": kind,
         "split_scheme": SPLIT_SCHEME, "files": files}, indent=1) + "\n")


def build(source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"{output} exists; use a new output directory")
    db = duckdb.connect()
    origin = {"train": (source / "train", source / "train"),
              "validation": (source / "public/validation", source / "trusted/validation")}
    union = lambda name, part: " UNION ALL ".join(  # noqa: E731
        f"SELECT * FROM read_parquet('{origin[k][part]}/{name}.parquet')" for k in origin)
    db.execute(f"CREATE VIEW samples AS {union('samples', 0)}")
    db.execute(f"CREATE VIEW events AS {union('events', 0)}")
    db.execute(f"CREATE VIEW windows AS {union('label_windows', 1)}")
    db.execute(f"CREATE VIEW labels AS {union('labels', 1)}")
    db.execute(f"""CREATE TABLE era AS
      WITH dx AS (SELECT * FROM events WHERE event_type='diagnosis'),
      idx AS (SELECT sample_id, arg_min(stay_key, seconds_before_cutoff) stay_key FROM dx GROUP BY 1)
      SELECT s.sample_id, avg((dx.mapping_status='direct')::INT) icd10_share
      FROM samples s LEFT JOIN idx USING(sample_id) LEFT JOIN dx USING(sample_id, stay_key)
      GROUP BY 1""")
    rows = db.execute("SELECT sample_id, icd10_share FROM era ORDER BY sample_id").fetchall()
    split = {}
    for sample_id, share in rows:
        if share == 1:
            b = bucket(sample_id)
            split[sample_id] = ("test" if b < TEST_FRACTION else
                                "validation" if b < TEST_FRACTION + VALIDATION_FRACTION else "train")
        else:  # ICD-9-era, mixed or no index diagnoses: training only
            split[sample_id] = "train"
    db.execute("CREATE TABLE assignment(sample_id VARCHAR, split VARCHAR)")
    db.executemany("INSERT INTO assignment VALUES (?, ?)", list(split.items()))
    # Observation boundaries for scored splits: the release's own rows where it has them (the
    # former validation split); otherwise derived from the label rows, which carry the same
    # cutoff, death and observation fields. `any_future_disease_record_365` is then limited to the
    # 262 scored diseases; it only feeds a diagnostic stratum.
    db.execute(f"""CREATE VIEW boundaries AS
      WITH derived AS (
        SELECT sample_id, any_value(cutoff) cutoff,
               any_value(CASE WHEN death_days IS NULL THEN NULL
                              ELSE (cutoff + to_days(death_days::INT))::DATE END) death_date,
               any_value(death_days) death_days,
               any_value(diagnosis_observation_end) diagnosis_observation_end,
               any_value(death_observation_end) death_observation_end,
               bool_or(coalesce(risk_set AND event_days <= 365, FALSE)) any_future_disease_record_365
        FROM windows GROUP BY 1),
      released AS (SELECT * EXCLUDE (split)
                   FROM read_parquet('{source}/trusted/validation/sample_boundaries.parquet'))
      SELECT d.sample_id, coalesce(r.cutoff, d.cutoff) cutoff, coalesce(r.death_date, d.death_date) death_date,
             coalesce(r.death_days, d.death_days) death_days,
             coalesce(r.diagnosis_observation_end, d.diagnosis_observation_end) diagnosis_observation_end,
             coalesce(r.death_observation_end, d.death_observation_end) death_observation_end,
             coalesce(r.any_future_disease_record_365, d.any_future_disease_record_365) any_future_disease_record_365
      FROM derived d LEFT JOIN released r USING(sample_id)""")

    root_manifest = json.loads((source / "manifest.json").read_text())
    public_manifest = json.loads((source / "public/validation/manifest.json").read_text())
    evaluated = dict(split_scheme=SPLIT_SCHEME, primary_metric=PRIMARY_METRIC,
                     evaluation_cohort="ICD-10-era index admissions (validation and test)")
    dev, private = output / "development", output / "private"

    def copy(query: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        db.execute(f"COPY ({query}) TO '{path}' (FORMAT parquet, COMPRESSION zstd)")

    for name, base in (("train", dev / "train"), ("validation", dev), ("test", private)):
        pick = f"(SELECT sample_id FROM assignment WHERE split='{name}')"
        public = base if name == "train" else base / "public" / name
        truth = base if name == "train" else base / "trusted" / name
        copy(f"SELECT * FROM samples WHERE sample_id IN {pick} ORDER BY sample_id", public / "samples.parquet")
        # Time order, then every column: several events share a timestamp, and the inventory must
        # be reproducible.
        copy(f"SELECT * FROM events WHERE sample_id IN {pick} ORDER BY sample_id, event_time, "
             "available_time, stay_key, event_type, code, value, unit, source_table, "
             "event_group_key, mapping_status, seconds_before_cutoff",
             public / "events.parquet")
        copy(f"SELECT * FROM labels WHERE sample_id IN {pick} ORDER BY sample_id, disease_code", truth / "labels.parquet")
        copy(f"SELECT * FROM windows WHERE sample_id IN {pick} ORDER BY sample_id, disease_code",
             truth / "label_windows.parquet")
        if name != "train":
            copy(f"SELECT sample_id, '{name}' AS split, * EXCLUDE (sample_id) FROM boundaries "
                 f"WHERE sample_id IN {pick} ORDER BY sample_id", truth / "sample_boundaries.parquet")
            (public / "manifest.json").write_text(json.dumps(
                dict(public_manifest, split=name, **evaluated), indent=1) + "\n")
    for tree in (dev, private):
        (tree / "trusted").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "trusted/risk_normalization.parquet", tree / "trusted/risk_normalization.parquet")
    (dev / "manifest.json").write_text(json.dumps(dict(root_manifest, **evaluated), indent=1) + "\n")
    shutil.copyfile(source / "scored_diseases.parquet", dev / "scored_diseases.parquet")

    counts = dict(db.execute("SELECT split, count(*) FROM assignment GROUP BY 1 ORDER BY 1").fetchall())
    era = dict(db.execute("""SELECT CASE WHEN icd10_share=1 THEN 'icd10' WHEN icd10_share=0 THEN 'icd9'
        ELSE 'mixed_or_none' END, count(*) FROM era GROUP BY 1 ORDER BY 1""").fetchall())
    scored_era = db.execute("""SELECT count(*) FROM assignment a JOIN era e USING(sample_id)
        WHERE a.split<>'train' AND e.icd10_share IS DISTINCT FROM 1""").fetchone()[0]
    if scored_era:
        raise ValueError("A validation/test sample is not an ICD-10-era index admission")
    for name, base in (("validation", dev), ("test", private)):
        n = counts[name]
        diseases = len(public_manifest["scored_diseases"])
        grid = db.execute(f"SELECT count(*) FROM read_parquet('{base}/trusted/{name}/label_windows.parquet')").fetchone()[0]
        bounds = db.execute(f"SELECT count(*) FROM read_parquet('{base}/trusted/{name}/sample_boundaries.parquet')").fetchone()[0]
        if grid != n * diseases or bounds != n:
            raise ValueError(f"{name}: labels/boundaries do not cover the sample grid")
    summary = {"split_scheme": SPLIT_SCHEME, "salt": SALT, "source_release": source.name,
               "index_era": era, "samples": counts,
               "fractions_of_icd10_era": {"test": TEST_FRACTION, "validation": VALIDATION_FRACTION}}
    (dev / "split.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    inventory(dev, "development")
    inventory(private, "private")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), indent=1))


if __name__ == "__main__":
    main()
