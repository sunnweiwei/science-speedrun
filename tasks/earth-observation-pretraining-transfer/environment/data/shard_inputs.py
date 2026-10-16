"""Copy unlabeled HDF5 rows into contiguous shards, preserving inference batches."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--start", type=int, default=0, help="Diagnostic interval only")
    p.add_argument("--stop", type=int, help="Diagnostic interval only")
    a = p.parse_args()
    if a.shards < 1 or a.batch_size < 1:
        p.error("shards and batch-size must be positive")
    with h5py.File(a.input, "r") as source:
        allowed = {
            "ids",
            "sentinel1",
            "sentinel2_l2a",
            "sentinel1_valid",
            "sentinel2_l2a_valid",
        }
        if set(source) - allowed or source.attrs.get("task") not in {"awf", "lfmc"}:
            raise ValueError("Expected the declared unlabeled evaluation input schema")
        total = len(source["ids"])
        stop = total if a.stop is None else a.stop
        if not 0 <= a.start < stop <= total:
            p.error("invalid source interval")
        if a.start % a.batch_size or (stop != total and stop % a.batch_size):
            p.error("interval must preserve original batch boundaries")
        batches = math.ceil((stop - a.start) / a.batch_size)
        if a.shards > batches:
            p.error("more shards than inference batches")
        a.output.mkdir(parents=True, exist_ok=False)
        records = []
        for shard in range(a.shards):
            begin = a.start + (batches * shard // a.shards) * a.batch_size
            end = min(
                stop, a.start + (batches * (shard + 1) // a.shards) * a.batch_size
            )
            path = a.output / f"part{shard:03d}.h5"
            hashes = {}
            with h5py.File(path.with_suffix(".tmp"), "w") as target:
                for key, value in source.attrs.items():
                    target.attrs[key] = value
                for name, dataset in source.items():
                    if not isinstance(dataset, h5py.Dataset):
                        raise ValueError("Expected flat input HDF5 datasets")
                    if not dataset.shape or dataset.shape[0] != total:
                        raise ValueError(f"Expected row-aligned dataset: {name}")
                    destination = target.create_dataset(
                        name,
                        shape=(end - begin, *dataset.shape[1:]),
                        dtype=dataset.dtype,
                    )
                    for key, value in dataset.attrs.items():
                        destination.attrs[key] = value
                    digest = hashlib.sha256()
                    for offset in range(begin, end, 128):
                        limit = min(end, offset + 128)
                        values = dataset[offset:limit]
                        destination[offset - begin : limit - begin] = values
                        if name == "ids":
                            for value in values:
                                encoded = (
                                    value
                                    if isinstance(value, bytes)
                                    else str(value).encode()
                                )
                                digest.update(len(encoded).to_bytes(8, "little"))
                                digest.update(encoded)
                        else:
                            digest.update(np.ascontiguousarray(values).tobytes())
                    hashes[name] = digest.hexdigest()
            path.with_suffix(".tmp").replace(path)
            records.append(
                {
                    "file": path.name,
                    "start": begin,
                    "stop": end,
                    "rows": end - begin,
                    "data_sha256": hashes,
                }
            )
            print(json.dumps(records[-1]), flush=True)
        report = {
            "source": str(a.input.resolve()),
            "source_rows": total,
            "start": a.start,
            "stop": stop,
            "rows": stop - a.start,
            "batch_size": a.batch_size,
            "shards": records,
            "full_release": a.start == 0 and stop == total,
            "labels_copied": False,
        }
        (a.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
