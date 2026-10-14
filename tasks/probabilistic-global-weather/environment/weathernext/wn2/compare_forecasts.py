"""Compare decoded forecast bits and metadata independently of compression."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import zarr

from zarr_safety import audit_zarr


def compare(left, right):
    for path in (left, right):
        audit_zarr(path)
    groups = [zarr.open_consolidated(str(path), mode='r') for path in (left, right)]
    a, b = groups
    names = sorted(a.array_keys())
    if names != sorted(b.array_keys()) or a.attrs.asdict() != b.attrs.asdict():
        raise ValueError('Forecast variables or group attributes differ')
    hashes = {}
    for name in names:
        x, y = a[name], b[name]
        if x.shape != y.shape or x.dtype != y.dtype or x.attrs.asdict() != y.attrs.asdict():
            raise ValueError(f'Forecast schema differs: {name}')
        h = hashlib.sha256()
        for starts in itertools.product(*(range(0, n, c) for n, c in zip(x.shape, x.chunks))):
            index = tuple(slice(s, min(s+c, n)) for s, c, n in zip(starts, x.chunks, x.shape))
            values = [np.asarray(array[index]) for array in (x, y)]
            if any(v.dtype.kind == 'f' and not np.isfinite(v).all() for v in values):
                raise ValueError(f'Missing or nonfinite forecast: {name}, {starts}')
            raw = [v.tobytes(order='C') for v in values]
            if raw[0] != raw[1]:
                raise ValueError(f'Decoded forecast bits differ: {name}, {starts}')
            h.update(raw[0])
        hashes[name] = h.hexdigest()
    return dict(status='passed', comparison='Exact decoded array bits and schema; compression bytes may differ',
                arrays=len(names), array_sha256=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('left', 'right', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.left, args.right)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
