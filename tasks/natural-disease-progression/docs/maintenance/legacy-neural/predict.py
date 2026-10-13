#!/usr/bin/env python3
"""Historical raw-input inference wrapper for the deprecated baseline."""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from adapter import iter_histories
from baseline import Predictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    manifest = json.loads((args.input / "manifest.json").read_text())
    predictor = Predictor(args.checkpoint, args.device)
    if predictor.diseases != manifest["scored_diseases"]:
        raise ValueError("Checkpoint disease list does not match input manifest")
    if predictor.horizons != tuple(manifest["horizons_days"]):
        raise ValueError("Checkpoint time grid does not match input manifest")
    args.output.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        pa.field("sample_id", pa.string()),
        pa.field("disease_risk", pa.list_(pa.float32(),
                                          len(predictor.diseases) * len(predictor.horizons))),
        pa.field("death_risk", pa.list_(pa.float32(), len(predictor.horizons))),
    ])
    writer = pq.ParquetWriter(args.output / "predictions.parquet", schema,
                              compression="zstd")
    ids, histories = [], []

    def flush() -> None:
        result = predictor.predict_risk(histories, manifest["horizons_days"])
        rows = [{"sample_id": sample_id,
                 "disease_risk": result["disease_risk"][i].reshape(-1).tolist(),
                 "death_risk": result["death_risk"][i].tolist()}
                for i, sample_id in enumerate(ids)]
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))

    for sample_id, history in iter_histories(args.input, predictor.adapter["max_events"]):
        ids.append(sample_id)
        histories.append(history)
        if len(ids) == args.batch_size:
            flush()
            ids, histories = [], []
    if ids:
        flush()
    writer.close()


if __name__ == "__main__":
    main()
