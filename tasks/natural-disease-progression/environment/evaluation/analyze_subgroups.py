"""Compare fixed predictions by disease frequency, horizon, and history span."""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

from risk_scoring import HORIZONS, disease_time_discrimination, read_predictions


def primary(labels: np.ndarray, prediction: np.ndarray, valid: np.ndarray) -> dict:
    summary, _ = disease_time_discrimination(labels, prediction, valid)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--trusted", type=Path, required=True)
    parser.add_argument("--normalization", type=Path,
                        help="accepted for backwards-compatible commands; unused by AUROC")
    parser.add_argument("--prediction", action="append", required=True, help="name=parquet")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.input / "manifest.json").read_text())
    diseases = manifest["scored_diseases"]
    samples = pq.read_table(args.input / "samples.parquet").to_pylist()
    sample_ids = [row["sample_id"] for row in samples]
    sample_index = {sample_id: i for i, sample_id in enumerate(sample_ids)}
    disease_index = {code: i for i, code in enumerate(diseases)}
    shape = (len(sample_ids), len(diseases), len(HORIZONS))
    labels = np.zeros(shape, np.uint8)
    valid = np.zeros(shape, bool)
    columns = ["sample_id", "disease_code"] + [
        f"{prefix}_{horizon}" for horizon in HORIZONS
        for prefix in ("disease_label", "disease_valid")
    ]
    seen = set()
    for batch in pq.ParquetFile(args.trusted / "label_windows.parquet").iter_batches(
            batch_size=65536, columns=columns):
        values = batch.to_pydict()
        for row in range(batch.num_rows):
            key = (values["sample_id"][row], values["disease_code"][row])
            if key in seen:
                raise ValueError(f"duplicate label key: {key}")
            seen.add(key)
            i, j = sample_index[key[0]], disease_index[key[1]]
            for k, horizon in enumerate(HORIZONS):
                labels[i, j, k] = values[f"disease_label_{horizon}"][row]
                valid[i, j, k] = values[f"disease_valid_{horizon}"][row]
    if len(seen) != len(sample_ids) * len(diseases):
        raise ValueError("label key coverage is incomplete")

    boundaries = {
        row["sample_id"]: row
        for row in pq.read_table(args.trusted / "sample_boundaries.parquet").to_pylist()
    }
    year = timedelta(days=365)
    def as_date(value):
        return value.date() if hasattr(value, "date") else value
    primary_cohort = np.asarray([
        as_date(row["cutoff"]) + year <= boundaries[row["sample_id"]]["diagnosis_observation_end"]
        and as_date(row["cutoff"]) + year <= boundaries[row["sample_id"]]["death_observation_end"]
        for row in samples
    ])
    valid &= primary_cohort[:, None, None]

    db = duckdb.connect()
    span_rows = db.execute("""
        SELECT s.sample_id,
               date_diff('day', min(e.event_time), s.cutoff) AS history_days
        FROM read_parquet(?) s
        JOIN read_parquet(?) e USING (sample_id)
        GROUP BY s.sample_id, s.cutoff
        """, [str(args.input / "samples.parquet"), str(args.input / "events.parquet")]).fetchall()
    db.close()
    spans = dict(span_rows)
    history_days = np.asarray([spans[sample_id] for sample_id in sample_ids], dtype=np.float64)
    history_quartiles = np.quantile(history_days, [.25, .5, .75])

    normalization = pq.read_table(args.trusted.parent / "risk_normalization.parquet").to_pandas()
    frequency = (normalization[normalization.horizon_days == 365]
                 .set_index("disease_code").loc[diseases, "train_positive"].to_numpy())
    frequency_quartiles = np.quantile(frequency, [.25, .5, .75])
    history_groups = [
        history_days <= history_quartiles[0],
        (history_days > history_quartiles[0]) & (history_days <= history_quartiles[1]),
        (history_days > history_quartiles[1]) & (history_days <= history_quartiles[2]),
        history_days > history_quartiles[2],
    ]
    frequency_groups = [
        frequency <= frequency_quartiles[0],
        (frequency > frequency_quartiles[0]) & (frequency <= frequency_quartiles[1]),
        (frequency > frequency_quartiles[1]) & (frequency <= frequency_quartiles[2]),
        frequency > frequency_quartiles[2],
    ]

    result = {
        "schema": "ndp-auroc-subgroup-comparison/v2",
        "history_days_quartiles": history_quartiles.tolist(),
        "train_365d_positive_quartiles": frequency_quartiles.tolist(),
        "models": {},
    }
    for item in args.prediction:
        name, path = item.split("=", 1)
        prediction, _ = read_predictions(Path(path), sample_ids, len(diseases))
        model = {"overall": primary(labels, prediction, valid),
                 "by_horizon": {}, "by_history": {}, "by_frequency": {}}
        for k, horizon in enumerate(HORIZONS):
            model["by_horizon"][str(horizon)] = primary(
                labels[:, :, k:k + 1], prediction[:, :, k:k + 1], valid[:, :, k:k + 1])
        for i, group in enumerate(history_groups):
            model["by_history"][f"q{i + 1}"] = primary(
                labels[group], prediction[group], valid[group])
        for i, group in enumerate(frequency_groups):
            model["by_frequency"][f"q{i + 1}"] = primary(
                labels[:, group], prediction[:, group], valid[:, group])
        result["models"][name] = model

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
