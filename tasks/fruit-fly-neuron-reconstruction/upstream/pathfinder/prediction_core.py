"""Persist the scoring core of an already fully decoded prediction, without relabeling."""
import json
from pathlib import Path
import numpy as np

from evaluate_physical import grid, reference_bounds, EVALUATOR_SHA
from fetch_axonem import reserve
from common import external, sha256, write_json


def core_slices(prediction_grid, reference_metadata):
    origin, spacing, shape, order = grid(prediction_grid)
    low, high = reference_bounds(reference_metadata)
    if (origin > low).any() or (origin+spacing*shape < high).any():
        raise ValueError('Decoded prediction does not cover scoring core')
    start=np.floor((low-origin)/spacing).astype(int)
    stop=np.ceil((high-origin)/spacing).astype(int)
    inverse=np.argsort(order)
    native_start=start[inverse];native_stop=stop[inverse]
    metadata=dict(axes=prediction_grid['axes'],coordinate_convention='voxel_outer_corner',
        origin_nm=(origin+start*spacing)[inverse].tolist(),
        voxel_size_nm=spacing[inverse].tolist(),shape=(native_stop-native_start).tolist())
    return tuple(slice(int(a),int(b)) for a,b in zip(native_start,native_stop)),metadata


def export_core(segmentation, prediction_grid, suite, output, root):
    """Call after global decoding; cropping affinities before decoding is not equivalent."""
    suite=Path(suite);output=external(output);root=external(root)
    if root not in output.parents:raise ValueError('Output outside budget root')
    grid(prediction_grid,segmentation.shape)
    if segmentation.ndim!=3 or segmentation.dtype.kind!='u':
        raise ValueError('Unsigned 3D decoded labels required')
    manifest=json.loads((suite/'suite.json').read_text())
    if (manifest.get('schema')!='pathfinder-evaluation-suite/v1' or
            manifest.get('boundary_margin_voxels')!=[0,0,0] or
            set(manifest['files'])!={'metadata.json','skeletons.json','run_length.py'}):
        raise ValueError('Frozen zero-margin suite required')
    for name,digest in manifest['files'].items():
        if sha256(suite/name)!=digest:raise ValueError('Reference integrity mismatch')
    if sha256(suite/'run_length.py')!=EVALUATOR_SHA:raise ValueError('Pinned evaluator required')
    reference=json.loads((suite/'metadata.json').read_text())
    slices,metadata=core_slices(prediction_grid,reference)
    crop=segmentation[slices]
    reserve(root,int(crop.size)*crop.dtype.itemsize+1_000_000)
    output.mkdir(parents=True,exist_ok=False)
    destination=np.lib.format.open_memmap(output/'segmentation.npy',mode='w+',dtype=crop.dtype,shape=crop.shape)
    for start in range(0,crop.shape[0],16):
        destination[start:start+16]=crop[start:start+16]
    destination.flush();del destination
    write_json(output/'metadata.json',metadata)
    receipt=dict(schema='pathfinder-decoded-core-export/v1',
        source_prediction_grid=prediction_grid,suite_sha256=sha256(suite/'suite.json'),
        segmentation_sha256=sha256(output/'segmentation.npy'),metadata_sha256=sha256(output/'metadata.json'),
        adapter_sha256=sha256(Path(__file__)),
        policy='Crop globally decoded labels only; no interpolation, renumbering or core-local decoding')
    write_json(output/'export-receipt.json',receipt)
    return receipt
