#!/usr/bin/env python3
"""Measure one incompressible FP16 example using production Zarr chunks."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from numcodecs import Blosc
import numpy as np
import zarr

from common import DENSE_TRACKS


COMPRESSOR = Blosc(cname="zstd", clevel=7, shuffle=Blosc.BITSHUFFLE)


def _fill(array, rng) -> None:
    chunks = array.chunks
    for block in np.ndindex(*(int(np.ceil(n / c)) for n, c in zip(array.shape, chunks))):
        selection = tuple(slice(i * c, min((i + 1) * c, n)) for i, c, n in zip(block, chunks, array.shape))
        shape = tuple(item.stop - item.start for item in selection)
        array[selection] = rng.standard_normal(shape).astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        shutil.rmtree(args.output)
    root = zarr.open_group(str(args.output), mode="w")
    example = root.create_group("examples/measurement")
    rng = np.random.default_rng(17)
    specs = {
        **{name: ((1536, tracks), (256, min(64, tracks))) for name, tracks in DENSE_TRACKS.items()},
        "splice_sites": ((196608, 4), (4096, 4)),
        "splice_usage": ((196608, 734), (4096, 64)),
        "junctions": ((512, 512, 734), (128, 128, 64)),
        "contact_maps": ((131328, 28), (8192, 28)),
    }
    for name, (shape, chunks) in specs.items():
        array = example.create_dataset(name, shape=shape, chunks=chunks, dtype="f2", compressor=COMPRESSOR)
        _fill(array, rng)
    example.create_dataset("junction_site_positions", data=np.arange(2048, dtype=np.int32).reshape(4, 512))
    stored = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
    raw = sum(np.prod(shape) * 2 for shape, _ in specs.values()) + 2048 * 4
    print({"stored_bytes": stored, "raw_bytes": int(raw), "stored_gib": stored / 2**30,
           "projected_1000_gib": stored * 1000 / 2**30})


if __name__ == "__main__":
    main()
