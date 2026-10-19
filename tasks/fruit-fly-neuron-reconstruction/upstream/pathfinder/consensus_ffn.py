"""Run pinned Google split consensus on complete, explicitly ordered FFN predictions.

This CPU adapter uses the author function directly. Optional tiled execution
requires verified first-input label confinement and preserves global partitions
up to positive-ID renaming. Both modes enforce a declared working-set estimate.
It is not a cross-tile stitching operation.
"""
import os
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from assemble_tiles import validate_shards
from prepare_ffn_decisions import FFN_COMMIT, CONNECTOMICS_COMMIT
from fetch_axonem import reserve
from evaluate_physical import grid
from common import external, sha256, write_json


def author_consensus(ffn_source, connectomics_source):
    for path, commit in ((ffn_source, FFN_COMMIT), (connectomics_source, CONNECTOMICS_COMMIT)):
        if (subprocess.check_output(['git','rev-parse','HEAD'], cwd=path, text=True).strip() != commit or
                subprocess.check_output(['git','diff','HEAD','--name-only'], cwd=path, text=True).strip()):
            raise ValueError('Pinned clean author source required')
        sys.path.insert(0, str(Path(path).resolve()))
    from ffn.inference import segmentation
    expected = Path(ffn_source).resolve()/'ffn/inference/segmentation.py'
    if Path(segmentation.__file__).resolve() != expected:
        raise ValueError('Imported FFN module is not the pinned source')
    return segmentation.split_segmentation_by_intersection, sha256(expected)


def verify_input(path, memory_limit_bytes=None):
    path = Path(path)
    receipt = json.loads((path/'receipt.json').read_text())
    assembled = receipt.get('all_tiles_complete') is True and 'inputs' in receipt
    if assembled:
        shards=[]
        for row in receipt['inputs']:
            source=Path(row['path'])
            if sha256(source)!=row['sha256']:raise ValueError('Assembly source receipt changed')
            shards.append(json.loads(source.read_text()))
        first, expected, tiles=validate_shards(shards)
        if receipt.get('ground_truth_used') is not False or receipt['metadata']!=first['metadata']:
            raise ValueError('Assembly metadata or provenance differs')
        if sha256(path/'segmentation.npy')!=receipt['segmentation_sha256']:
            raise ValueError('Assembly hash mismatch')
    else:
        if receipt.get('shards') != 1 or receipt.get('all_shard_tiles_complete') is not True:
            raise ValueError('Complete assembled prediction required')
        first, expected, tiles = validate_shards([receipt])
    if set(tiles) != set(expected) or any(t.get('budget_exhausted') is not False for t in tiles.values()):
        raise ValueError('Incomplete tile coverage')
    array = np.load(path/'segmentation.npy', mmap_mode='r', allow_pickle=False)
    if array.dtype.kind != 'u' or list(array.shape) != first['metadata']['shape']:
        raise ValueError('Prediction shape or dtype mismatch')
    if first['metadata'].get('axes') != 'zyx':
        raise ValueError('FFN predictions require an explicit ZYX grid')
    grid(first['metadata'],array.shape)
    origin = np.asarray(first['roi'][::2])
    if list(np.asarray(first['roi'][1::2])-origin) != list(array.shape):
        raise ValueError('Prediction ROI differs from array shape')
    for index, tile in tiles.items():
        lo = np.asarray(tile['start'])-origin
        hi = np.asarray(tile['stop'])-origin
        if memory_limit_bytes is not None and int(np.prod(hi-lo))*128>memory_limit_bytes:
            raise ValueError('Input tile verification exceeds memory limit')
        section=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))
        if assembled:
            # The assembled file and its source receipts were verified above;
            # archived original tile arrays need not be restored.
            data=array[section]
        else:
            src = Path(tile['path'])
            if sha256(src) != tile['sha256']:raise ValueError('Tile hash mismatch')
            data = np.load(src, mmap_mode='r', allow_pickle=False)
        if data.dtype.kind != 'u' or list(data.shape) != (hi-lo).tolist():
            raise ValueError('Tile array geometry differs')
        positive = data[data > 0]
        if len(positive) and (positive.min() <= index*first['id_stride'] or
                              positive.max() >= (index+1)*first['id_stride']):
            raise ValueError('Tile label namespace differs')
        if not assembled and not np.array_equal(array[section], data):
            raise ValueError('Assembled prediction differs from verified tiles')
    return array, dict(first,tile_results=list(tiles.values()))


def consensus_blocks(arrays, tiles, roi_start, function, min_size):
    """Exact partition equivalence when first-input IDs are confined to tiles.

    verify_input establishes confinement via disjoint per-tile ID namespaces.
    Consequently all counts for every (A,B) intersection live in one tile.
    Split consensus preserves confinement inductively across further inputs.
    Compact output labels differ from a global call only by positive-ID renaming.
    """
    offset=0
    for tile in sorted(tiles,key=lambda t:t['index']):
        lo=np.asarray(tile['start'])-roi_start;hi=np.asarray(tile['stop'])-roi_start
        section=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))
        result=np.array(arrays[0][section],dtype=np.uint64,order='C')
        for array in arrays[1:]:
            if int(result.max(initial=0)) > np.iinfo(np.uint64).max-result.size:
                raise ValueError('Insufficient uint64 label headroom')
            function(result,np.array(array[section],dtype=np.uint64,order='C'),min_size)
        ids,inverse=np.unique(result,return_inverse=True)
        positive=ids>0
        mapping=np.zeros(len(ids),dtype=np.uint64)
        count=int(positive.sum())
        if offset+count>np.iinfo(np.uint64).max:raise ValueError('Consensus label overflow')
        mapping[positive]=np.arange(offset+1,offset+count+1,dtype=np.uint64)
        offset+=count
        yield section,mapping[inverse].reshape(result.shape),offset


def validate_six_runs(receipts):
    """Check variant coverage only, not checkpoint-selection or paper parity."""
    pairs=[(r['checkpoint_sha256'],r.get('seed_order','ascending')) for r in receipts]
    checkpoints={checkpoint for checkpoint,_ in pairs}
    if (len(pairs)!=6 or len(checkpoints)!=3 or len(set(pairs))!=6 or
            set(pairs)!={(checkpoint,order) for checkpoint in checkpoints
                         for order in ('ascending','descending')}):
        raise ValueError('Require three checkpoint hashes, each with ascending and descending seeds')
    return dict(checkpoint_sha256s=sorted(checkpoints),variants=6,
                scope='File-hash and seed-order coverage only; weight distinctness and checkpoint selection require separate evidence')


def validate_checkpoint_runs(receipts, checkpoint_count=3):
    """Explicit single-checkpoint protocol; retain the original six-run guard."""
    if type(checkpoint_count) is not int or checkpoint_count not in (1,3):
        raise ValueError('One or three checkpoints required')
    if checkpoint_count==3:return validate_six_runs(receipts)
    pairs=[(r['checkpoint_sha256'],r.get('seed_order')) for r in receipts]
    if (len(pairs)!=2 or len({p[0] for p in pairs})!=1 or
            [p[1] for p in pairs]!=['ascending','descending']):
        raise ValueError('Single checkpoint requires ordered ascending and descending predictions')
    return dict(checkpoint_sha256s=[pairs[0][0]],variants=2,
                scope='Explicit single-checkpoint seed-order consensus; not the paper three-checkpoint protocol')


def run(inputs, root, output, ffn_source, connectomics_source, min_size, memory_gib,
        tiled=False, output_dtype='uint64', require_six_runs=False):
    adapter_sha=sha256(Path(__file__))
    root, output = external(root), external(output)
    if root not in output.parents or output.exists():
        raise ValueError('New output inside the Xtemp budget root required')
    if not str(root).startswith(os.environ.get('PATHFINDER_SCRATCH_ROOT', '/scratch').rstrip('/') + '/'):
        raise ValueError('Xtemp root required')
    if any(root not in Path(p).resolve().parents for p in inputs):
        raise ValueError('Consensus inputs must remain inside the same budget root')
    if len(inputs) < 2 or type(min_size) is not int or min_size < 0 or not np.isfinite(memory_gib) or memory_gib <= 0:
        raise ValueError('At least two inputs, nonnegative minimum size and positive memory limit required')
    headers = [np.load(Path(p)/'segmentation.npy', mmap_mode='r', allow_pickle=False) for p in inputs]
    if any(a.ndim != 3 or a.dtype.kind != 'u' or a.shape != headers[0].shape for a in headers):
        raise ValueError('Matching unsigned 3D prediction arrays required')
    estimated_bytes = headers[0].size * 128
    if not tiled and estimated_bytes > memory_gib * 2**30:
        raise ValueError('Conservative in-memory consensus estimate exceeds declared memory limit')
    arrays, receipts = zip(*(verify_input(p,memory_gib*2**30) for p in inputs))
    six_runs=validate_six_runs(receipts) if require_six_runs else None
    if output_dtype not in ('uint32','uint64'):raise ValueError('Unsigned output type required')
    if tiled:
        estimated_bytes=max(int(np.prod(np.asarray(t['stop'])-t['start']))
                            for r in receipts for t in r['tile_results'])*128
        if estimated_bytes>memory_gib*2**30:raise ValueError('Tile working set exceeds memory limit')
    if any(r['metadata'] != receipts[0]['metadata'] or r['raw_sha256'] != receipts[0]['raw_sha256']
           or r['roi'] != receipts[0]['roi'] for r in receipts[1:]):
        raise ValueError('Consensus inputs must share image, ROI and physical grid')
    function, function_sha = author_consensus(ffn_source, connectomics_source)
    reserve(root, arrays[0].size*np.dtype(output_dtype).itemsize + 1048576)
    provenance = [dict(path=str(Path(p).resolve()), receipt_sha256=sha256(Path(p)/'receipt.json'),
                       segmentation_sha256=sha256(Path(p)/'segmentation.npy'),
                       checkpoint_sha256=r['checkpoint_sha256'], seed_order=r.get('seed_order','ascending'))
                  for p,r in zip(inputs,receipts)]
    output.mkdir(parents=True)
    if tiled:
        result=np.lib.format.open_memmap(output/'segmentation.npy',mode='w+',dtype=output_dtype,shape=arrays[0].shape)
        for section,block,last_id in consensus_blocks(arrays,receipts[0]['tile_results'],
                np.asarray(receipts[0]['roi'][::2]),function,min_size):
            if last_id>np.iinfo(result.dtype).max:raise ValueError('Output label type overflow; no complete receipt emitted')
            result[section]=block
        result.flush();del result
    else:
        result = np.array(arrays[0], dtype=np.uint64, order='C')
        for array in arrays[1:]:
            if int(result.max(initial=0)) > np.iinfo(np.uint64).max-result.size:
                raise ValueError('Insufficient uint64 label headroom for author relabeling')
            function(result, np.asarray(array, dtype=np.uint64), min_size)
        if int(result.max(initial=0))>np.iinfo(np.dtype(output_dtype)).max:raise ValueError('Output label type overflow')
        np.save(output/'segmentation.npy', result.astype(output_dtype,copy=False), allow_pickle=False)
    write_json(output/'metadata.json', receipts[0]['metadata'])
    if sha256(Path(__file__))!=adapter_sha:
        raise ValueError('Consensus adapter changed during execution; no complete receipt emitted')
    for row in provenance:
        path=Path(row['path'])
        if (sha256(path/'receipt.json')!=row['receipt_sha256'] or
                sha256(path/'segmentation.npy')!=row['segmentation_sha256']):
            raise ValueError('Consensus input changed during execution; no complete receipt emitted')
    report = dict(schema='author-ffn-consensus/v1', complete=True, inputs=provenance,
                  min_size=min_size, input_order_matters=True, estimated_working_bytes=estimated_bytes,
                  required_six_runs=require_six_runs, six_run_coverage=six_runs,
                  declared_memory_gib=memory_gib, author_commit=FFN_COMMIT,
                  tiled=tiled, output_dtype=output_dtype,
                  id_policy='Compact disjoint positive IDs per first-input tile' if tiled else 'Author IDs retained',
                  connectomics_commit=CONNECTOMICS_COMMIT, author_function_sha256=function_sha,
                  segmentation_sha256=sha256(output/'segmentation.npy'),
                  metadata_sha256=sha256(output/'metadata.json'), adapter_sha256=adapter_sha,
                  ground_truth_used=False, paper_comparable=False, cross_tile_stitching=False,
                  scope='Ordered pairwise calls to the pinned official split-consensus function; '
                        'checkpoint selection, input variants and minimum size are experiment settings.')
    write_json(output/'receipt.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', nargs='+', type=Path, required=True)
    for name in ('root','output','ffn-source','connectomics-source'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--min-size', type=int, required=True)
    p.add_argument('--memory-gib', type=float, required=True)
    p.add_argument('--tiled',action='store_true',help='Bound memory by verified first-input tile confinement')
    p.add_argument('--output-dtype',choices=['uint32','uint64'],default='uint64')
    p.add_argument('--require-six-runs',action='store_true',help='Require three checkpoint hashes and both seed orders for each')
    a = p.parse_args()
    print(json.dumps(run(a.inputs,a.root,a.output,a.ffn_source,a.connectomics_source,a.min_size,a.memory_gib,a.tiled,a.output_dtype,a.require_six_runs),indent=2))
