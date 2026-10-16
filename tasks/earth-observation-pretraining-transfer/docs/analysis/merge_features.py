"""Merge contiguous inference shards and enforce exact evaluation ID coverage."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-partial-diagnostic", action="store_true")
    a = p.parse_args()
    if a.output.suffix != ".npz":
        p.error("output must end in .npz")
    manifest = json.loads(a.manifest.read_text())
    complete = manifest["start"] == 0 and manifest["stop"] == manifest["source_rows"]
    if complete != manifest["full_release"]:
        raise ValueError("Inconsistent full-release declaration")
    if not complete and not a.allow_partial_diagnostic:
        raise ValueError("Evaluation requires the complete release")
    with h5py.File(manifest["source"], "r") as source:
        if len(source["ids"]) != manifest["source_rows"]:
            raise ValueError("Source population changed")
        expected = source["ids"].asstr()[manifest["start"] : manifest["stop"]]
    arrays, offset, width = [], manifest["start"], None
    for index, shard in enumerate(manifest["shards"]):
        if shard["start"] != offset or shard["stop"] <= offset:
            raise ValueError("Shard intervals have gaps or overlap")
        if offset % manifest["batch_size"]:
            raise ValueError("Shard split an inference batch")
        with np.load(
            a.features / f"part{index:03d}" / "features.npz", allow_pickle=False
        ) as result:
            ids, features = result["ids"], result["features"]
            first, last = offset - manifest["start"], shard["stop"] - manifest["start"]
            if not np.array_equal(ids, expected[first:last]):
                raise ValueError("Shard IDs do not match the exact source interval")
            if (
                features.ndim != 2
                or features.shape[0] != len(ids)
                or not np.isfinite(features).all()
            ):
                raise ValueError("Malformed shard features")
            if width is None:
                width = features.shape[1]
                if width < 1:
                    raise ValueError("Empty embedding dimension")
            if features.shape[1] != width or features.dtype != np.float32:
                raise ValueError("Inconsistent feature dimension or dtype")
            arrays.append(features)
        offset = shard["stop"]
    if offset != manifest["stop"] or len(np.unique(expected)) != len(expected):
        raise ValueError("Incomplete or duplicate source IDs")
    features = np.concatenate(arrays)
    if len(features) != len(expected):
        raise ValueError("Incomplete feature population")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise FileExistsError(a.output)
    np.savez_compressed(a.output, ids=expected.astype(str), features=features)
    print(
        json.dumps(
            {
                "rows": len(expected),
                "dimension": width,
                "full_release": manifest["full_release"],
            }
        )
    )


if __name__ == "__main__":
    main()
