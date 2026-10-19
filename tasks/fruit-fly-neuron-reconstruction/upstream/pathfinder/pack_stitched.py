"""Package a completed stitched prediction losslessly; never remove the source.

This package can be read one volume at a time for global consensus. It retains
both original NPY identity and the stitching/inference metadata provenance.
"""
import os
import argparse
import json
from pathlib import Path
import numpy as np
from stitch_ffn import validate
from window_io import save_compressed
from common import external,sha256,write_json
from fetch_axonem import reserve


def run(prediction,manifest_path,root,output,max_bytes,memory_gib,*,compression_level=1):
    root=external(root);output=external(output);prediction=Path(prediction).resolve();manifest_path=Path(manifest_path).resolve()
    if not str(root).startswith(os.environ.get('PATHFINDER_SCRATCH_ROOT', '/scratch').rstrip('/') + '/') or output.exists() or root not in output.parents or any(root not in p.parents for p in (prediction,manifest_path)):
        raise ValueError('New Xtemp output and inputs inside budget root required')
    if type(max_bytes) is not int or max_bytes<1 or not np.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Finite positive storage and memory bounds required')
    if type(compression_level) is not int or not 0<=compression_level<=9:raise ValueError('Invalid compression level')
    manifest=json.loads(manifest_path.read_text());validate(manifest)
    receipt_path=prediction/'receipt.json';receipt=json.loads(receipt_path.read_text())
    if receipt.get('complete') is not True or receipt.get('cross_tile_stitching') is not True or receipt.get('ground_truth_used') is not False:raise ValueError('Complete prediction-only stitching required')
    pins={str(manifest_path):sha256(manifest_path),str(receipt_path):sha256(receipt_path),str(prediction/'metadata.json'):sha256(prediction/'metadata.json')}
    if receipt['manifest_sha256']!=pins[str(manifest_path)] or receipt['metadata_sha256']!=pins[str(prediction/'metadata.json')]:raise ValueError('Stitched provenance changed')
    if json.loads((prediction/'metadata.json').read_text())!=manifest['metadata']:raise ValueError('Stitched grid differs from manifest')
    shape=manifest['metadata']['shape'];array_bytes=int(np.prod(shape))*4
    # Conservative source-page residency plus bounded compression/verification buffers.
    # Verification streams 8 MiB chunks instead of constructing a full decoded array.
    estimated_working_bytes=array_bytes+2**30
    if estimated_working_bytes>memory_gib*2**30:raise ValueError('Packing working set exceeds declared memory')
    source=prediction/'segmentation.npy'
    if sha256(source)!=receipt['segmentation_sha256']:raise ValueError('Original stitched prediction changed')
    reserve(root,max_bytes+1000000)
    sources={str(Path(__file__).with_name(name)):sha256(Path(__file__).with_name(name)) for name in ('pack_stitched.py','window_io.py','stitch_ffn.py','fetch_axonem.py')}
    common=Path(__file__).resolve().parents[1]/'common.py';sources[str(common)]=sha256(common)
    output.mkdir();array=np.load(source,mmap_mode='r',allow_pickle=False)
    try:
        if list(array.shape)!=shape or array.dtype!=np.uint32:raise ValueError('Stitched uint32 geometry required')
        storage=save_compressed(array,output/'segmentation.npz',max_bytes,compression_level=compression_level)
        if storage['sha256']!=receipt['segmentation_sha256']:raise ValueError('Packaged NPY bytes differ')
    finally:array._mmap.close()
    for path,digest in {**pins,**sources}.items():
        if sha256(path)!=digest:raise ValueError('Provenance/source changed during packaging')
    result=dict(schema='stitched-ffn-package/v1',complete=True,ground_truth_used=False,original_retained=True,
        prediction=str(prediction),manifest=str(manifest_path),input_sha256=pins,source_sha256=sources,storage=storage,
        metadata=manifest['metadata'],raw_sha256=manifest['raw_sha256'],checkpoint_sha256=manifest['checkpoint_sha256'],
        seed_order=manifest['seed_order'],roi=manifest['roi'],memory_gib=memory_gib,
        estimated_working_bytes=estimated_working_bytes,verification='streaming-exact-npy-sha256',
        scope='Lossless storage of a completed stitched prediction; no rerun, selection, quality or paper-parity claim')
    write_json(output/'receipt.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('prediction','manifest','root','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--max-bytes',type=int,required=True);p.add_argument('--memory-gib',type=float,required=True)
    p.add_argument('--compression-level',type=int,choices=range(10),default=1)
    a=p.parse_args();print(json.dumps(run(a.prediction,a.manifest,a.root,a.output,a.max_bytes,a.memory_gib,compression_level=a.compression_level),indent=2))
