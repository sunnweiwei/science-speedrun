#!/usr/bin/env python3
"""Average compatible prediction files by key, never by implicit row order."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input) < 2:
        parser.error("at least two --input files are required")

    frames = []
    expected_ids = None
    for path in args.input:
        table = pq.read_table(path, columns=["sample_id", "disease_risk", "death_risk"])
        ids = table["sample_id"].to_pylist()
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate sample_id in {path}")
        by_id = {sample_id: i for i, sample_id in enumerate(ids)}
        if expected_ids is None:
            expected_ids = sorted(ids)
        elif set(ids) != set(expected_ids):
            raise ValueError(f"sample_id coverage differs in {path}")
        order = np.asarray([by_id[sample_id] for sample_id in expected_ids])
        disease = np.asarray(table["disease_risk"].to_pylist(), dtype=np.float32)[order]
        death = np.asarray(table["death_risk"].to_pylist(), dtype=np.float32)[order]
        frames.append((disease, death))

    disease = np.mean([frame[0] for frame in frames], axis=0, dtype=np.float64).astype(np.float32)
    death = np.mean([frame[1] for frame in frames], axis=0, dtype=np.float64).astype(np.float32)
    schema = pa.schema([
        pa.field("sample_id", pa.string()),
        pa.field("disease_risk", pa.list_(pa.float32(), disease.shape[1])),
        pa.field("death_risk", pa.list_(pa.float32(), death.shape[1])),
    ])
    rows = [
        {"sample_id": sample_id, "disease_risk": disease[i].tolist(), "death_risk": death[i].tolist()}
        for i, sample_id in enumerate(expected_ids)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), args.output, compression="zstd")
    print(f"wrote {len(rows)} keyed predictions averaged across {len(frames)} inputs")


if __name__ == "__main__":
    main()
