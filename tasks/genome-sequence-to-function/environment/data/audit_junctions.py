from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from agbench.zarr_store import close_store, open_group


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("splits", nargs="+")
    args = parser.parse_args()
    for split in args.splits:
        rows = pd.read_parquet(args.root / "splits" / f"{split}_materialized.parquet")
        count = np.zeros(734, np.int64)
        nonzero = np.zeros(734, np.int64)
        total_rows = 0
        max_rows = 0
        for item in rows.itertuples(index=False):
            root, store = open_group(args.root / item.target_shard, mode="r")
            try:
                group = root[item.target_group]["splice_junctions"]
                track = np.asarray(group["track"], np.int64)
                value = np.asarray(group["value"], np.float64)
            finally:
                close_store(store)
            total_rows += len(track)
            max_rows = max(max_rows, len(track))
            count += np.bincount(track, minlength=734)
            nonzero += np.bincount(track[value != 0], minlength=734)
        print({
            "split": split,
            "examples": len(rows),
            "sparse_rows": total_rows,
            "max_rows_per_example": max_rows,
            "tracks_with_rows": int((count > 0).sum()),
            "tracks_with_two_rows": int((count > 1).sum()),
            "tracks_with_nonzero": int((nonzero > 0).sum()),
            "min_rows": int(count.min()),
            "median_rows": float(np.median(count)),
            "max_track_rows": int(count.max()),
        }, flush=True)


if __name__ == "__main__":
    main()
