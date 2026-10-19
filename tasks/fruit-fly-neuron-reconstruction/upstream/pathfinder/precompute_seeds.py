"""Fill the exact PolicyPeaks seed cache for every FFN window in parallel.

Seeds depend only on the image context (seed_cache.py), not on the checkpoint, so
computing them up front in a process pool replaces the serial per-window seeding inside
each FFN worker. Workers then read the same seeds from the cache (cache hits are
validated against the context bytes), so window outputs are unchanged.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import external, sha256, write_json
from seed_cache import get_seeds
from stitch_ffn import windows

_raw = _root = _cache = _raw_hash = None


def _init(root, cache, raw_path, raw_hash):
    global _raw, _root, _cache, _raw_hash
    _root, _cache, _raw_hash = root, cache, raw_hash
    _raw = np.load(raw_path, mmap_mode='r', allow_pickle=False)


def _one(window):
    start, stop = np.asarray(window['start']), np.asarray(window['stop'])
    context = np.asarray(_raw[tuple(slice(int(a) - 16, int(b) + 16) for a, b in zip(start, stop))])
    seeds, evidence = get_seeds(context, _root, _cache,
                                dict(raw_sha256=_raw_hash, start=start.tolist(), stop=stop.tolist()))
    return dict(index=window['index'], seeds=len(seeds), hit=evidence['hit'], key=evidence['key'])


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    for key in ('root', 'image-manifest', 'cache', 'receipt'):
        p.add_argument('--' + key, required=True, type=Path)
    p.add_argument('--roi', required=True, type=int, nargs=6)
    p.add_argument('--core-size', type=int, default=436); p.add_argument('--margin', type=int, default=32)
    p.add_argument('--clip-boundary-to-image', action='store_true')
    p.add_argument('--workers', type=int, default=os.cpu_count())
    a = p.parse_args()
    root, cache, image_path = external(a.root), external(a.cache), external(a.image_manifest)
    if root not in cache.parents or root not in image_path.parents:
        raise ValueError('Cache and image must be inside the root')
    image = json.loads(image_path.read_text())
    raw_path = image_path.parent / 'raw.npy'; raw_hash = sha256(raw_path)
    if raw_hash != image['files']['raw.npy']:
        raise ValueError('Raw data hash mismatch')
    boundary = image['shape'] if a.clip_boundary_to_image else None
    all_windows = windows(a.roi, a.core_size, a.margin, boundary)
    # Largest windows first so the pool finishes evenly.
    order = sorted(all_windows, key=lambda w: -int(np.prod(np.asarray(w['stop']) - np.asarray(w['start']))))
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=max(1, a.workers), initializer=_init,
                             initargs=(root, cache, raw_path, raw_hash)) as pool:
        rows = list(pool.map(_one, order, chunksize=1))
    write_json(a.receipt, dict(schema='policy-peaks-precompute/v1', windows=len(rows), workers=a.workers,
                               seconds=time.monotonic() - started, raw_sha256=raw_hash,
                               seeds=sum(r['seeds'] for r in rows), entries=sorted(rows, key=lambda r: r['index'])))


if __name__ == '__main__':
    main()
