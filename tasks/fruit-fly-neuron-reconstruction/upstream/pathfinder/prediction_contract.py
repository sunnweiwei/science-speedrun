"""Seal and validate a prediction volume against independently supplied geometry.

The contract checks data integrity and coordinates, not biological correctness,
training compliance, or a submitter's claims about use of ground truth.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,sha256,write_json

FIELDS=('axes','coordinate_convention','shape','origin_nm','voxel_size_nm')


def geometry(meta,shape):
    if meta.get('axes')!='zyx' or meta.get('coordinate_convention')!='voxel_outer_corner':
        raise ValueError('Explicit ZYX outer-corner convention required')
    if len(shape)!=3 or meta.get('shape')!=list(shape) or any(n<1 for n in shape):raise ValueError('Volume extent mismatch')
    for key in ('origin_nm','voxel_size_nm'):
        value=np.asarray(meta.get(key),dtype=float)
        if value.shape!=(3,) or not np.isfinite(value).all() or (key=='voxel_size_nm' and (value<=0).any()):
            raise ValueError('Invalid geometry: '+key)
    return {k:meta[k] for k in FIELDS}


def validate(folder,expected):
    folder=Path(folder).resolve();manifest=json.loads((folder/'prediction.json').read_text())
    if manifest.get('schema')!='pathfinder-prediction/v1' or manifest.get('zero_label')!='unknown_or_omitted':
        raise ValueError('Unsupported prediction contract')
    if set(manifest['files'])!={'segmentation.npy','metadata.json'}:raise ValueError('Explicit prediction files required')
    for name,digest in manifest['files'].items():
        path=folder/name
        if path.is_symlink() or not path.is_file() or sha256(path)!=digest:raise ValueError('Prediction integrity mismatch: '+name)
    seg=np.load(folder/'segmentation.npy',mmap_mode='r',allow_pickle=False)
    if seg.dtype not in (np.dtype('uint32'),np.dtype('uint64')):raise ValueError('Unsigned32/64 instance IDs required')
    recorded=geometry(json.loads((folder/'metadata.json').read_text()),seg.shape)
    trusted=geometry(expected,seg.shape)
    if recorded!=trusted:raise ValueError('Prediction geometry disagrees with independent expected ROI')
    if manifest.get('geometry')!=recorded or manifest.get('dtype')!=str(seg.dtype):raise ValueError('Manifest/header mismatch')
    return dict(schema='pathfinder-prediction-check/v1',files=manifest['files'],geometry=recorded,
        manifest_sha256=sha256(folder/'prediction.json'),paper_protocol_verified=False,
        scope='Artifact and independent geometry checks only; quality requires trusted evaluation')


def seal(segmentation,metadata):
    segmentation=external(segmentation)
    if segmentation.name!='segmentation.npy':raise ValueError('Prediction volume must be named segmentation.npy')
    out=segmentation.parent
    if (out/'prediction.json').exists() or (out/'metadata.json').exists():raise FileExistsError('Prediction contract sidecar already exists')
    seg=np.load(segmentation,mmap_mode='r',allow_pickle=False)
    if seg.dtype not in (np.dtype('uint32'),np.dtype('uint64')):raise ValueError('Unsigned32/64 instance IDs required')
    canonical=geometry(json.loads(Path(metadata).read_text()),seg.shape)
    write_json(out/'metadata.json',canonical)
    manifest=dict(schema='pathfinder-prediction/v1',geometry=canonical,dtype=str(seg.dtype),
        zero_label='unknown_or_omitted',files={name:sha256(out/name) for name in ('segmentation.npy','metadata.json')},
        source_metadata_sha256=sha256(metadata),paper_comparable=False,
        completeness='not inferred from label values; consult inference coverage receipts separately')
    write_json(out/'prediction.json',manifest)
    return validate(out,canonical)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['seal','validate'],required=True)
    p.add_argument('--prediction',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    result=seal(a.prediction/'segmentation.npy',a.metadata) if a.mode=='seal' else validate(a.prediction,json.loads(a.metadata.read_text()))
    write_json(external(a.report),result)

if __name__=='__main__':main()
