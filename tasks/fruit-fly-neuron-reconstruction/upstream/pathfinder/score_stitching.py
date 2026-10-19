"""Score the same window predictions before and after inter-window stitching.

Exports frozen scoring cores without renumbering or rerunning connected
components. Reference labels never enter inference or stitching.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from stitch_ffn import validate
from prediction_core import core_slices,export_core
from evaluate_physical import score
from common import external,sha256,write_json
from fetch_axonem import reserve


def export_pair(manifest_path,stitched,suite,root,output):
    root=external(root);output=external(output);stitched=Path(stitched).resolve();suite=Path(suite).resolve();manifest_path=Path(manifest_path).resolve()
    if root not in output.parents or output.exists():raise ValueError('New output under budget root required')
    if any(root not in p.parents for p in (stitched,suite,manifest_path)):raise ValueError('Inputs outside budget root')
    manifest=json.loads(manifest_path.read_text());records=validate(manifest)
    receipt_path=stitched/'receipt.json';receipt=json.loads(receipt_path.read_text())
    inputs={str(manifest_path):sha256(manifest_path),str(receipt_path):sha256(receipt_path)}
    if receipt.get('complete') is not True or receipt.get('cross_tile_stitching') is not True or receipt.get('ground_truth_used') is not False:
        raise ValueError('Complete prediction-only stitched result required')
    if receipt['manifest_sha256']!=inputs[str(manifest_path)]:raise ValueError('Stitch manifest changed')
    for name,key in [('segmentation.npy','segmentation_sha256'),('metadata.json','metadata_sha256')]:
        path=stitched/name;inputs[str(path)]=sha256(path)
        if inputs[str(path)]!=receipt[key]:raise ValueError('Stitched output changed')
    metadata=json.loads((stitched/'metadata.json').read_text())
    if metadata!=manifest['metadata']:raise ValueError('Stitched geometry differs')
    cores=receipt['cores']
    if len(cores)!=len(records) or sorted(c['index'] for c in cores)!=list(range(len(records))):raise ValueError('Missing or duplicate core components')
    cores=sorted(cores,key=lambda c:c['index'])
    for c in cores:
        path=Path(c['path']).resolve()
        if root not in path.parents or sha256(path)!=c['sha256']:raise ValueError('Core component integrity mismatch')
        inputs[str(path)]=c['sha256']
    reference=json.loads((suite/'metadata.json').read_text())
    slices,core_meta=core_slices(metadata,reference)
    lo=np.asarray([s.start for s in slices])+manifest['roi'][::2]
    hi=np.asarray([s.stop for s in slices])+manifest['roi'][::2]
    reserve(root,2*int(np.prod(core_meta['shape']))*4+10_000_000)
    output.mkdir()
    full=np.load(stitched/'segmentation.npy',mmap_mode='r',allow_pickle=False)
    try:
        if full.dtype!=np.uint32:raise ValueError('Stitched labels must be uint32')
        export_core(full,metadata,suite,output/'stitched',root)
    finally:full._mmap.close()
    before=output/'unstitched';before.mkdir()
    dest=np.lib.format.open_memmap(before/'segmentation.npy',mode='w+',dtype=np.uint32,shape=tuple(core_meta['shape']))
    covered=0
    try:
        for r,c in zip(records,cores):
            start=np.maximum(lo,r['core_start']);stop=np.minimum(hi,r['core_stop'])
            if (stop<=start).any():continue
            array=np.load(c['path'],mmap_mode='r',allow_pickle=False)
            try:
                if array.dtype!=np.uint32 or list(array.shape)!=(np.asarray(r['core_stop'])-r['core_start']).tolist():raise ValueError('Core component grid mismatch')
                source=tuple(slice(int(a),int(b)) for a,b in zip(start-r['core_start'],stop-r['core_start']))
                target=tuple(slice(int(a),int(b)) for a,b in zip(start-lo,stop-lo))
                dest[target]=array[source];covered+=int(np.prod(stop-start))
            finally:array._mmap.close()
        if covered!=dest.size:raise ValueError('Incomplete scoring core coverage')
        after=np.load(output/'stitched/segmentation.npy',mmap_mode='r',allow_pickle=False)
        try:
            for z in range(0,dest.shape[0],16):
                if not np.array_equal(dest[z:z+16]>0,after[z:z+16]>0):raise ValueError('Stitching altered foreground support')
        finally:after._mmap.close()
        dest.flush()
    finally:dest._mmap.close()
    write_json(before/'metadata.json',core_meta)
    for path,digest in inputs.items():
        if sha256(path)!=digest:raise ValueError('Input changed during paired core extraction')
    result=dict(complete=True,input_sha256=inputs,suite_sha256=sha256(suite/'suite.json'),
        exporter_sha256=sha256(Path(__file__)),core_exporter_sha256=sha256(Path(__file__).with_name('prediction_core.py')),
        predictions={name:{key:sha256(output/name/key) for key in ('segmentation.npy','metadata.json')} for name in ('unstitched','stitched')},
        scope='Same window predictions, same trimmed connected components and same frozen scoring graph; only inter-window label unions differ. No core-local decoding or renumbering.')
    write_json(output/'export-receipt.json',result);return result


def run(manifest,stitched,suite,root,output):
    output=Path(output);receipt=export_pair(manifest,stitched,suite,root,output)
    results={name:score(suite,output/name,output/('score-'+name)) for name in ('unstitched','stitched')}
    if any(r['suite_sha256']!=receipt['suite_sha256'] for r in results.values()):raise ValueError('Reference changed across paired scores')
    summary=dict(complete=True,scope=receipt['scope'],export_receipt_sha256=sha256(output/'export-receipt.json'),
        results={name:{k:r[k] for k in ('nerl','erl_nm','split_errors','merge_errors','omitted_node_fraction')} for name,r in results.items()},
        metric_sha256={name:sha256(output/('score-'+name)/'metrics.json') for name in results})
    write_json(output/'comparison.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','stitched','suite','root','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.manifest,a.stitched,a.suite,a.root,a.output),indent=2))
