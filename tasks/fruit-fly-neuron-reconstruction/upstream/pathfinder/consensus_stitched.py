"""Global ordered consensus of six verified, losslessly packed stitched volumes.

Only one compressed input is decoded at a time. The mutable output is a memmap;
input decode buffers, output pages and the global pair table count toward RAM.
"""
import os
import argparse
import json
from pathlib import Path
import numpy as np
from stitch_ffn import validate
from window_io import window_array
from consensus_ffn import validate_six_runs,validate_checkpoint_runs
from global_consensus import intersect_inplace
from common import external,sha256,write_json
from fetch_axonem import reserve


def inside(root,path):
    path=Path(path).resolve()
    if root not in path.parents:raise ValueError('Package artifacts must remain in budget root')
    return path


def verify_package(path,root):
    path=inside(root,path);receipt_path=path/'receipt.json';r=json.loads(receipt_path.read_text())
    if r.get('schema')!='stitched-ffn-package/v1' or r.get('complete') is not True or r.get('ground_truth_used') is not False:
        raise ValueError('Complete prediction-only stitched package required')
    manifest_path=inside(root,r['manifest']);prediction=inside(root,r['prediction'])
    required={str(manifest_path),str(prediction/'receipt.json'),str(prediction/'metadata.json')}
    if set(r['input_sha256'])!=required:raise ValueError('Incomplete package provenance')
    pins={str(receipt_path):sha256(receipt_path)}
    for p,digest in r['input_sha256'].items():
        if sha256(inside(root,p))!=digest:raise ValueError('Package parent changed')
        pins[p]=digest
    manifest=json.loads(manifest_path.read_text());validate(manifest)
    parent=json.loads((prediction/'receipt.json').read_text())
    if parent.get('complete') is not True or parent.get('cross_tile_stitching') is not True or parent.get('ground_truth_used') is not False:
        raise ValueError('Incomplete parent stitching')
    if parent['manifest_sha256']!=pins[str(manifest_path)] or parent['metadata_sha256']!=pins[str(prediction/'metadata.json')]:raise ValueError('Parent provenance mismatch')
    for key in ('metadata','roi','raw_sha256','checkpoint_sha256','seed_order'):
        if r[key]!=manifest[key]:raise ValueError('Package differs from inference manifest')
    if json.loads((prediction/'metadata.json').read_text())!=r['metadata']:raise ValueError('Package grid differs')
    storage=r['storage'];archive=inside(root,storage['path'])
    if (storage.get('storage')!='npz' or storage['shape']!=r['metadata']['shape'] or
            storage['sha256']!=parent['segmentation_sha256'] or sha256(archive)!=storage['archive_sha256']):
        raise ValueError('Packaged array integrity mismatch')
    pins[str(archive)]=storage['archive_sha256']
    return r,pins


def bind_model_package(model_package,packages,root,checkpoint_count=3):
    """Bind completed predictions to the exact verified ensemble assets.

    Six different file identities alone do not prove three distinct models.
    This also rejects predictions from an unrelated checkpoint ensemble.
    """
    from model_package import verify
    model_package=inside(root,model_package)
    check=verify(model_package,require_ensemble=checkpoint_count==3)
    manifest_path=model_package/'package.json';manifest=json.loads(manifest_path.read_text())
    if checkpoint_count==3:
        binding=check['ffn_ensemble'];expected=binding['ordered_runs']
    else:
        if 'ffn_ensemble' in manifest:raise ValueError('Explicit single-checkpoint model package required')
        expected=[dict(checkpoint_index=0,checkpoint_path=str(model_package/'weights/ffn.pt'),
                       checkpoint_sha256=manifest['files']['weights/ffn.pt'],seed_order=order)
                  for order in ('ascending','descending')]
        binding=dict(checkpoints=1,variants=2,ordered_runs=expected,paper_comparable=False)
    if len(packages)!=len(expected) or any(
        p['checkpoint_sha256']!=r['checkpoint_sha256'] or p['seed_order']!=r['seed_order']
        for p,r in zip(packages,expected)):
        raise ValueError('Predictions differ from ordered model-package checkpoint assets')
    if sha256(manifest_path)!=check['package_manifest_sha256']:raise ValueError('Model package changed during binding')
    pins={str(manifest_path):check['package_manifest_sha256']}
    pins.update({str(model_package/name):digest for name,digest in manifest['files'].items()})
    return dict(model_package=str(model_package),package_manifest_sha256=check['package_manifest_sha256'],
                **binding),pins


def run(spec,root,output,min_size,memory_gib,slab_depth=8,max_pairs=1000000,model_package=None,checkpoint_count=3):
    root=external(root);output=external(output);spec=inside(root,spec)
    if not str(root).startswith(os.environ.get('PATHFINDER_SCRATCH_ROOT', '/scratch').rstrip('/') + '/') or root not in output.parents or output.exists():raise ValueError('New Xtemp output required')
    if any(type(v) is not int for v in (min_size,slab_depth,max_pairs)) or min_size<0 or slab_depth<1 or max_pairs<1 or not np.isfinite(memory_gib) or memory_gib<=0:
        raise ValueError('Finite positive resource bounds and nonnegative minimum size required')
    paths=json.loads(spec.read_text())
    if type(checkpoint_count) is not int or checkpoint_count not in (1,3):raise ValueError('One or three checkpoints required')
    if not isinstance(paths,list) or len(paths)!=2*checkpoint_count:raise ValueError('Exactly two ordered package paths per checkpoint required')
    pins={str(spec):sha256(spec)};packages=[]
    for path in paths:
        r,p=verify_package(path,root);packages.append(r);pins.update(p)
    coverage=validate_checkpoint_runs(packages,checkpoint_count);first=packages[0]
    if any(any(r[key]!=first[key] for key in ('metadata','roi','raw_sha256')) for r in packages[1:]):raise ValueError('All packages must share image and grid')
    binding=None
    if model_package is not None:
        binding,model_pins=bind_model_package(model_package,packages,root,checkpoint_count);pins.update(model_pins)
    shape=first['metadata']['shape'];array_bytes=int(np.prod(shape))*4
    kernel_bytes=128*min(slab_depth,shape[0])*shape[1]*shape[2]+512*max_pairs
    estimated=3*array_bytes+kernel_bytes+2**30
    if estimated>memory_gib*2**30:raise ValueError('Decoded input/output/table exceed declared memory')
    reserve(root,array_bytes+10000000)
    sources={str(Path(__file__).with_name(name)):sha256(Path(__file__).with_name(name)) for name in
             ('consensus_stitched.py','global_consensus.py','window_io.py','pack_stitched.py','consensus_ffn.py','stitch_ffn.py','fetch_axonem.py')}
    if binding is not None:
        for name in ('model_package.py','ffn_model.py','sense.py','shape_model.py'):
            path=Path(__file__).with_name(name);sources[str(path)]=sha256(path)
    common=Path(__file__).resolve().parents[1]/'common.py';sources[str(common)]=sha256(common)
    output.mkdir();dest=np.lib.format.open_memmap(output/'segmentation.npy',mode='w+',dtype=np.uint32,shape=tuple(shape));stages=[]
    try:
        for index,r in enumerate(packages):
            with window_array(r['storage']) as incoming:
                if index==0:
                    for z in range(0,shape[0],slab_depth):dest[z:z+slab_depth]=incoming[z:z+slab_depth]
                else:stages.append(intersect_inplace(dest,incoming,min_size,slab_depth,max_pairs,kernel_bytes/2**30))
            del incoming
            dest.flush()
            write_json(output/'progress.json',dict(completed_inputs=index+1,total_inputs=len(packages),complete=False))
    finally:dest._mmap.close()
    for p,digest in {**pins,**sources}.items():
        if sha256(p)!=digest:raise ValueError('Input or source changed during consensus')
    write_json(output/'metadata.json',first['metadata'])
    result=dict(schema='stitched-global-consensus/v1',complete=True,ground_truth_used=False,paper_comparable=False,
        metadata=first['metadata'],roi=first['roi'],raw_sha256=first['raw_sha256'],required_six_runs=coverage,
        ordered_packages=paths,input_sha256=pins,source_sha256=sources,stages=stages,min_size=min_size,
        conservative_memory_bytes=estimated,segmentation_sha256=sha256(output/'segmentation.npy'),metadata_sha256=sha256(output/'metadata.json'),
        scope='Global split consensus, partition-equivalent with compact IDs. Three file hashes x two seed orders; tensor distinctness/checkpoint-selection and final reconstruction quality require separate evidence.')
    if binding is not None:
        result['model_ensemble']=binding
        result['scope']='Global split consensus bound to three distinct full FFN model-package states and six ordered predictions; paper checkpoint selection and final reconstruction quality remain unverified.'
    if checkpoint_count==1:
        result['schema']='stitched-single-checkpoint-consensus/v1'
        result['required_checkpoint_runs']=result.pop('required_six_runs')
        result['checkpoint_count']=1
        if binding is not None:result['model_checkpoint']=result.pop('model_ensemble')
        result['scope']='User-selected single-checkpoint ascending/descending split consensus; full original ROI, prediction-only inputs; not the paper three-checkpoint protocol.'
    write_json(output/'receipt.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('spec','root','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--min-size',type=int,required=True);p.add_argument('--memory-gib',type=float,required=True)
    p.add_argument('--slab-depth',type=int,default=8);p.add_argument('--max-pairs',type=int,default=1000000)
    p.add_argument('--model-package',type=Path,help='Require exact ordered predictions from a verified three-checkpoint full-model package')
    p.add_argument('--checkpoint-count',type=int,choices=(1,3),default=3,help='Explicit simplified single-checkpoint protocol; default retains six required runs')
    a=p.parse_args();print(json.dumps(run(a.spec,a.root,a.output,a.min_size,a.memory_gib,a.slab_depth,a.max_pairs,a.model_package,a.checkpoint_count),indent=2))
