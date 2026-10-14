"""Reject executable/object codecs before opening an untrusted local Zarr v2."""
import json
from pathlib import Path

import numpy as np

MAX_METADATA_BYTES = 16 * 1024 * 1024
# One 8-member, 40-lead native WN2 case is about 103 GiB decompressed.
MAX_LOGICAL_BYTES = 160 * 1024**3
MAX_CHUNK_BYTES = 256 * 1024**2
SAFE_COMPRESSORS = {"blosc", "zlib", "gzip", "bz2", "lz4", "zstd"}


def audit_zarr(path):
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("prediction store root must be an ordinary directory")
    for index, child in enumerate(path.rglob("*")):
        if child.is_symlink() or index >= 250000:
            raise ValueError("prediction store contains symlinks or too many entries")
    metadata_path = path / ".zmetadata"
    if metadata_path.is_symlink() or metadata_path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("unsafe or oversized consolidated Zarr metadata")
    document = json.loads(metadata_path.read_text())
    if document.get("zarr_consolidated_format") != 1:
        raise ValueError("consolidated Zarr v2 metadata is required")
    total = 0
    for key, entry in document["metadata"].items():
        if key.startswith("/") or ".." in key.split("/") or "\\" in key:
            raise ValueError("invalid Zarr metadata path")
        if key.endswith(".zgroup") and entry.get("zarr_format") != 2:
            raise ValueError("only Zarr v2 groups are permitted")
        if not key.endswith(".zarray"):
            continue
        if entry.get("zarr_format") != 2:
            raise ValueError("only Zarr v2 arrays are permitted")
        dtype = np.dtype(entry["dtype"])
        if dtype.hasobject or dtype.fields is not None or dtype.kind not in "biufmM":
            raise ValueError("prediction arrays must have primitive numeric dtypes")
        if entry.get("filters") not in (None, []):
            raise ValueError("prediction array filters are forbidden, including pickle codecs")
        compressor = entry.get("compressor")
        if compressor is not None and compressor.get("id") not in SAFE_COMPRESSORS:
            raise ValueError("unsafe prediction compressor")
        shape, chunks = entry["shape"], entry["chunks"]
        if len(shape) != len(chunks) or not 1 <= len(shape) <= 8 or any(type(n) is not int or n <= 0 for n in shape + chunks):
            raise ValueError("invalid prediction array dimensions")
        elements, chunk_elements = 1, 1
        for size, chunk in zip(shape, chunks):
            elements *= size
            chunk_elements *= chunk
        total += elements * dtype.itemsize
        if total > MAX_LOGICAL_BYTES or chunk_elements * dtype.itemsize > MAX_CHUNK_BYTES:
            raise ValueError("prediction decompressed size/chunk exceeds safety limit")
