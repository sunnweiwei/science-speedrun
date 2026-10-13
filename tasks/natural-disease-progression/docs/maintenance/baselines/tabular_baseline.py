"""Strong same-information LightGBM baseline for longitudinal-v2.

Features are derived only from cutoff-bounded public events.  The feature
dictionary (including selected lab code/unit pairs) is fitted on train and is
reused unchanged for validation/test.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HORIZONS = tuple(range(30, 361, 30)) + (365,)
CHAPTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIAGNOSIS_BINS = 128
MEDICATION_BINS = 64


def fit_feature_spec(train: Path, top_labs: int = 48) -> dict:
    db = duckdb.connect()
    rows = db.execute("""SELECT code,coalesce(unit,'') unit,
      count(try_cast(value AS DOUBLE)) numeric_n
      FROM read_parquet(?) WHERE event_type='lab'
      GROUP BY code,unit HAVING numeric_n>=100
      ORDER BY numeric_n DESC,code,unit LIMIT ?""",
      [str(train / "events.parquet"), top_labs]).fetchall()
    db.close()
    return {"schema": "ndp-tabular-features/v1",
            "fit_source": "train/events.parquet only",
            "top_labs": [{"code": code, "unit": unit, "train_numeric_n": int(n)}
                         for code, unit, n in rows],
            "diagnosis_hash_bins": DIAGNOSIS_BINS,
            "medication_hash_bins": MEDICATION_BINS}


def build_features(input_dir: Path, output: Path, spec: dict) -> dict:
    started = time.monotonic()
    samples = sorted(pq.read_table(input_dir / "samples.parquet").to_pylist(),
                     key=lambda row: row["sample_id"])
    ids = [row["sample_id"] for row in samples]
    index = {sid: i for i, sid in enumerate(ids)}
    basic_names = ["age", "sex_f", "log_events", "log_stays", "history_days"]
    for modality in ("diagnosis", "lab", "prescription", "administration"):
        basic_names.extend((f"log_{modality}_events", f"days_since_last_{modality}"))
    feature_names = (basic_names + [f"diagnosis_chapter_{x}" for x in CHAPTERS]
        + [f"diagnosis_hash_{i}" for i in range(DIAGNOSIS_BINS)]
        + [f"medication_hash_{i}" for i in range(MEDICATION_BINS)])
    lab_metrics = ("last", "mean", "std", "min", "max", "log_count", "missing")
    feature_names += [f"lab_{i}_{metric}" for i in range(len(spec["top_labs"]))
                      for metric in lab_metrics]
    x = np.zeros((len(ids), len(feature_names)), dtype=np.float32)
    x[:, 0] = [row["approx_age"] for row in samples]
    x[:, 1] = [row["gender"] == "F" for row in samples]
    db = duckdb.connect()
    event_file = str(input_dir / "events.parquet")
    aggregate = db.execute("""SELECT sample_id,count(*) n,count(DISTINCT stay_key) stays,
      max(seconds_before_cutoff)/86400.0 history_days,
      count(*) FILTER(WHERE event_type='diagnosis') diagnosis_n,
      min(seconds_before_cutoff) FILTER(WHERE event_type='diagnosis')/86400.0 diagnosis_recent,
      count(*) FILTER(WHERE event_type='lab') lab_n,
      min(seconds_before_cutoff) FILTER(WHERE event_type='lab')/86400.0 lab_recent,
      count(*) FILTER(WHERE event_type='prescription') prescription_n,
      min(seconds_before_cutoff) FILTER(WHERE event_type='prescription')/86400.0 prescription_recent,
      count(*) FILTER(WHERE event_type='administration') administration_n,
      min(seconds_before_cutoff) FILTER(WHERE event_type='administration')/86400.0 administration_recent
      FROM read_parquet(?) GROUP BY sample_id""", [event_file]).fetchall()
    for row in aggregate:
        i = index[row[0]]
        x[i, 2:5] = (np.log1p(row[1]), np.log1p(row[2]), row[3] or 0)
        for m in range(4):
            x[i, 5 + 2*m] = np.log1p(row[4 + 2*m] or 0)
            x[i, 6 + 2*m] = row[5 + 2*m] if row[5 + 2*m] is not None else -1
    chapter_offset = len(basic_names)
    chapters = db.execute("""SELECT sample_id,left(code,1) chapter,count(*) n
      FROM read_parquet(?) WHERE event_type='diagnosis'
      GROUP BY sample_id,chapter""", [event_file]).fetchall()
    chapter_index = {chapter: j for j, chapter in enumerate(CHAPTERS)}
    for sid, chapter, n in chapters:
        if chapter in chapter_index:
            x[index[sid], chapter_offset + chapter_index[chapter]] = np.log1p(n)
    diagnosis_offset = chapter_offset + len(CHAPTERS)
    medication_offset = diagnosis_offset + DIAGNOSIS_BINS
    hashed = db.execute(f"""SELECT sample_id,
      CASE WHEN event_type='diagnosis' THEN 0 ELSE 1 END AS feature_family,
      hash(code)%CASE WHEN event_type='diagnosis' THEN {DIAGNOSIS_BINS} ELSE {MEDICATION_BINS} END AS bucket,
      count(*) n FROM read_parquet(?)
      WHERE event_type IN ('diagnosis','prescription','administration')
      GROUP BY sample_id,feature_family,bucket""", [event_file]).fetchall()
    for sid, family, bucket, n in hashed:
        offset = diagnosis_offset if family == 0 else medication_offset
        x[index[sid], offset + int(bucket)] = np.log1p(n)
    lab_offset = medication_offset + MEDICATION_BINS
    lab_spec = pa.table({"code": [row["code"] for row in spec["top_labs"]],
                         "unit": [row["unit"] for row in spec["top_labs"]],
                         "lab_index": np.arange(len(spec["top_labs"]), dtype=np.int16)})
    db.register("lab_spec", lab_spec)
    lab_rows = db.execute("""SELECT e.sample_id,s.lab_index,
      arg_min(try_cast(e.value AS DOUBLE),e.seconds_before_cutoff) last_value,
      avg(try_cast(e.value AS DOUBLE)) mean_value,
      stddev_pop(try_cast(e.value AS DOUBLE)) std_value,
      min(try_cast(e.value AS DOUBLE)) min_value,max(try_cast(e.value AS DOUBLE)) max_value,
      count(try_cast(e.value AS DOUBLE)) numeric_n
      FROM read_parquet(?) e JOIN lab_spec s
        ON e.code=s.code AND coalesce(e.unit,'')=s.unit
      WHERE e.event_type='lab' GROUP BY e.sample_id,s.lab_index""", [event_file]).fetchall()
    x[:, lab_offset + 6::len(lab_metrics)] = 1.0
    for sid, lab_index, last, mean, std, minimum, maximum, n in lab_rows:
        base = lab_offset + int(lab_index) * len(lab_metrics)
        x[index[sid], base:base+7] = (last, mean, std or 0, minimum, maximum,
                                      np.log1p(n), 0)
    db.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=np.asarray(ids), features=x,
                        feature_names=np.asarray(feature_names))
    return {"samples": len(ids), "features": x.shape[1], "output": str(output),
            "duration_seconds": round(time.monotonic() - started, 2)}


def _label_grid(materialized: Path, ids: np.ndarray, diseases: list[str]):
    db = duckdb.connect()
    db.register("sample_order", pa.table({"sample_id": ids.tolist(),
        "i": np.arange(len(ids), dtype=np.int32)}))
    db.register("disease_order", pa.table({"disease_code": diseases,
        "j": np.arange(len(diseases), dtype=np.int16)}))
    labels = str(materialized / "train" / "label_windows.parquet")
    total, unique = db.execute("""SELECT count(*),count(DISTINCT (sample_id,disease_code))
      FROM read_parquet(?)""", [labels]).fetchone()
    expected = len(ids) * len(diseases)
    if total != expected or unique != expected:
        db.close()
        raise ValueError("Train label keys are duplicated or do not match the feature grid")
    fields = []
    for horizon in HORIZONS:
        fields.extend((f"l.disease_label_{horizon}", f"l.disease_valid_{horizon}"))
    query = f"""SELECT {','.join(fields)},
      l.cutoff+INTERVAL 365 DAY<=l.diagnosis_observation_end
        AND l.cutoff+INTERVAL 365 DAY<=l.death_observation_end AS full_year
      FROM sample_order s CROSS JOIN disease_order d
      JOIN read_parquet(?) l USING(sample_id,disease_code) ORDER BY s.i,d.j"""
    data = db.execute(query, [labels]).fetchnumpy()
    db.close()
    if not data or len(next(iter(data.values()))) != expected:
        raise ValueError("Keyed train label grid does not match feature rows")
    y = np.stack([data[f"disease_label_{h}"] for h in HORIZONS], axis=1)
    valid = np.stack([data[f"disease_valid_{h}"] for h in HORIZONS], axis=1)
    shape = (len(ids), len(diseases), len(HORIZONS))
    return y.reshape(shape).astype(np.uint8), valid.reshape(shape).astype(bool), \
        data["full_year"].reshape(len(ids), len(diseases))[:, 0].astype(bool)


def fit(materialized: Path, features: Path, output: Path, seed: int = 20260929) -> dict:
    from lightgbm import LGBMClassifier
    started = time.monotonic()
    manifest = json.loads((materialized / "manifest.json").read_text())
    diseases = manifest["scored_diseases"]
    cache = np.load(features, allow_pickle=False)
    x, ids = cache["features"], cache["ids"]
    y, valid, primary = _label_grid(materialized, ids, diseases)
    models, time_scale = [], np.zeros((len(diseases), len(HORIZONS)), np.float32)
    for j, disease in enumerate(diseases):
        use = valid[:, j, -1] & primary
        model = LGBMClassifier(n_estimators=48, learning_rate=.08, num_leaves=31,
            min_child_samples=60, subsample=.8, colsample_bytree=.8,
            max_bin=63, force_col_wise=True, reg_lambda=2.0,
            random_state=seed, n_jobs=-1, verbosity=-1)
        model.fit(x[use], y[use, j, -1])
        models.append(model)
        rates = []
        for k in range(len(HORIZONS)):
            cell = valid[:, j, k] & primary
            rates.append((float(y[cell, j, k].sum()) + 1) / (int(cell.sum()) + 2))
        rates = np.maximum.accumulate(rates)
        time_scale[j] = np.clip(rates / max(rates[-1], 1e-8), 0, 1)
    first = diseases[0]
    db = duckdb.connect()
    labels = str(materialized / "train" / "label_windows.parquet")
    death = []
    for horizon in HORIZONS:
        n, p = db.execute(f"""SELECT count(*) FILTER(WHERE death_valid_{horizon}),
          count(*) FILTER(WHERE death_label_{horizon}) FROM read_parquet(?)
          WHERE disease_code=? AND cutoff+INTERVAL 365 DAY<=diagnosis_observation_end
            AND cutoff+INTERVAL 365 DAY<=death_observation_end""", [labels, first]).fetchone()
        death.append((p + 1) / (n + 2))
    db.close()
    state = {"schema": "ndp-lightgbm/v1", "diseases": diseases,
             "horizons": HORIZONS, "feature_names": cache["feature_names"].tolist(),
             "models": models, "time_scale": time_scale,
             "death": np.maximum.accumulate(death).astype(np.float32), "seed": seed}
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, output, compress=3)
    report = {"model": "LightGBM", "seed": seed, "models": len(models),
              "features": x.shape[1], "parameters": "tree ensemble; see checkpoint",
              "duration_seconds": round(time.monotonic() - started, 2),
              "selection": "fixed recipe; train labels only; no test tuning"}
    output.with_suffix(".training.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def predict(features: Path, checkpoint: Path, output: Path) -> dict:
    state = joblib.load(checkpoint)
    cache = np.load(features, allow_pickle=False)
    x, ids = cache["features"], cache["ids"]
    end_risk = np.stack([model.booster_.predict(x) for model in state["models"]], axis=1)
    risk = end_risk[:, :, None] * state["time_scale"][None, :, :]
    risk = np.maximum.accumulate(np.clip(risk, 0, 1), axis=2).astype(np.float32)
    death = np.broadcast_to(state["death"], (len(ids), len(HORIZONS))).astype(np.float32)
    rows = [{"sample_id": str(sid), "disease_risk": risk[i].reshape(-1).tolist(),
             "death_risk": death[i].tolist()} for i, sid in enumerate(ids)]
    schema = pa.schema([pa.field("sample_id", pa.string()),
        pa.field("disease_risk", pa.list_(pa.float32(), len(state["diseases"])*len(HORIZONS))),
        pa.field("death_risk", pa.list_(pa.float32(), len(HORIZONS)))])
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), output, compression="zstd")
    return {"samples": len(rows), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("cache")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--adapter", type=Path)
    command = sub.add_parser("fit")
    command.add_argument("--materialized", type=Path, required=True)
    command.add_argument("--features", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--seed", type=int, default=20260929)
    command = sub.add_parser("predict")
    command.add_argument("--features", type=Path, required=True)
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "cache":
        spec = json.loads(args.adapter.read_text()) if args.adapter else fit_feature_spec(args.input)
        args.output.with_suffix(".adapter.json").write_text(json.dumps(spec, indent=2) + "\n")
        print(json.dumps(build_features(args.input, args.output, spec), indent=2))
    elif args.command == "fit":
        print(json.dumps(fit(args.materialized, args.features, args.output, args.seed), indent=2))
    else:
        print(json.dumps(predict(args.features, args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
