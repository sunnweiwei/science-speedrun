"""Exact PolicyPeaks cache; checkpoint-independent and prediction-only.

Hash the actual context bytes as well as caller provenance. Cache hits validate
original NPY identity; no reduced precision or seed subsampling is permitted.
"""
import os
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import numpy as np
from ffn_seeds import policy_peaks
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,sha256,write_json
from fetch_axonem import reserve


def get_seeds(image,root,cache,identity):
    root=external(root);cache=external(cache)
    from execution_resources import valid_root
    if not valid_root(root) or root not in cache.parents:
        raise ValueError('Cache must be inside Xtemp experiment root')
    if image.ndim!=3 or image.dtype!=np.uint8:raise ValueError('Original uint8 context required')
    if not isinstance(identity,dict) or not identity:raise ValueError('Explicit input provenance required')
    digest=hashlib.sha256()
    for plane in image:digest.update(np.ascontiguousarray(plane).tobytes())
    contract=dict(schema='exact-policy-peaks-cache/v1',input_identity=identity,context_sha256=digest.hexdigest(),
        shape=list(image.shape),dtype=str(image.dtype),margin=16,voxel_size=[16.,16.,16.],exclusion=None,
        policy_sha256=sha256(Path(__file__).with_name('ffn_seeds.py')),
        dependencies={name:importlib.metadata.version(name) for name in ('numpy','scipy','scikit-image','edt')})
    key=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    cache.mkdir(parents=True,exist_ok=True);entry=cache/key;entry.mkdir(exist_ok=True)
    with (entry/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=entry/'receipt.json';data=entry/'seeds.npy'
        if receipt.exists():
            record=json.loads(receipt.read_text())
            if record['contract']!=contract or sha256(data)!=record['sha256']:raise ValueError('Seed cache integrity mismatch')
            seeds=np.load(data,allow_pickle=False)
            if seeds.dtype!=np.int64 or seeds.shape!=(record['count'],3):raise ValueError('Seed cache array contract mismatch')
            hit=True
        else:
            if data.exists() or (entry/'seeds.npy.partial').exists():raise ValueError('Incomplete cache entry requires explicit recovery')
            start=time.monotonic();seeds=policy_peaks(image)
            reserve(root,seeds.nbytes+65536)
            partial=entry/'seeds.npy.partial'
            with partial.open('xb') as stream:np.save(stream,seeds,allow_pickle=False)
            original=sha256(partial);restored=np.load(partial,allow_pickle=False)
            if not np.array_equal(restored,seeds):raise ValueError('Seed cache round-trip differs')
            partial.replace(data)
            record=dict(contract=contract,sha256=original,count=len(seeds),policy_seconds=time.monotonic()-start,ground_truth_used=False)
            write_json(receipt,record);hit=False
    seeds.flags.writeable=False
    return seeds,dict(hit=hit,key=key,receipt=str(receipt),receipt_sha256=sha256(receipt),seed_sha256=record['sha256'])
