"""Reuse only byte-verified complete tiles from a finished FFN shard receipt."""
import json
from pathlib import Path
import numpy as np
from dataset_contract import sha256


def completed_tiles(receipt_path, expected, tiles):
    receipt_path=Path(receipt_path)
    previous=json.loads(receipt_path.read_text())
    if previous.get('ground_truth_used') is not False:
        raise ValueError('Prediction-only source receipt required')
    if previous.get('seed_order','ascending')!=expected.get('seed_order','ascending'):
        raise ValueError('Resume seed order differs')
    for key,value in expected.items():
        if previous.get(key,'ascending' if key=='seed_order' else None)!=value:
            raise ValueError('Resume provenance differs: '+key)
    planned={t['index']:t for t in tiles};seen=set();retained=[]
    stride=expected['id_stride']
    for row in previous['tile_results']:
        index=row['index']
        if type(index) is not int or index in seen or index not in planned:
            raise ValueError('Duplicate or foreign resume tile')
        seen.add(index)
        if any(row[k]!=planned[index][k] for k in ('start','stop')):
            raise ValueError('Resume tile geometry differs')
        if type(row.get('budget_exhausted')) is not bool:
            raise ValueError('Explicit tile completion status required')
        if row['budget_exhausted']:
            continue
        path=Path(row['path'])
        if not path.is_absolute() or sha256(path)!=row['sha256']:
            raise ValueError('Resume tile bytes differ')
        arr=np.load(path,mmap_mode='r',allow_pickle=False)
        if arr.dtype.kind!='u' or list(arr.shape)!=(np.array(row['stop'])-row['start']).tolist():
            raise ValueError('Invalid resume tile array')
        ids=arr[arr>0]
        if len(ids) and (ids.min()<=index*stride or ids.max()>=(index+1)*stride):
            raise ValueError('Resume tile IDs outside reserved interval')
        retained.append(row)
    return retained,dict(receipt=str(receipt_path.resolve()),sha256=sha256(receipt_path),
        reused_complete_tiles=len(retained),partial_tiles_recomputed=True,
        policy='Reference original verified tile files without copying; retain them until final archive')
