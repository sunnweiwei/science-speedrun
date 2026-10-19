"""Stream the shared EM image onto the frozen BANIS+ XYZ 9/9/20 nm grid."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from evaluate_physical import grid
from fetch_axonem import reserve
from common import external, sha256, write_json

SPACING_XYZ = np.array([9.,9.,20.])


def target_grid(metadata):
    origin, spacing, shape, _ = grid(metadata)
    if metadata['axes'] != 'zyx' or not np.array_equal(spacing,[16,16,16]):
        raise ValueError('Shared ZYX 16nm EM required')
    target_shape = np.floor((shape*spacing)[::-1]/SPACING_XYZ).astype(int)
    if (target_shape <= 0).any():
        raise ValueError('Source too small for native grid')
    return dict(axes='xyz', coordinate_convention='voxel_outer_corner',
        origin_nm=origin[::-1].tolist(), voxel_size_nm=SPACING_XYZ.tolist(),
        shape=target_shape.tolist())


def image_chunk(raw, source, target, start, stop):
    """XYZ output, physical center sampling, bounded output scratch memory."""
    so, ss, _, _ = grid(source,raw.shape)
    to, ts, shape, _ = grid(target)
    start=np.asarray(start);stop=np.asarray(stop)
    if (start.shape!=(3,) or stop.shape!=(3,) or start.dtype.kind not in 'iu' or
            stop.dtype.kind not in 'iu' or (start<0).any() or (stop<=start).any() or
            (stop>shape[::-1]).any() or source['axes']!='zyx' or target['axes']!='xyz'):
        raise ValueError('Valid XYZ chunk and ZYX source required')
    vectors=[(to[::-1][a]+(np.arange(start[a],stop[a],dtype=float)+.5)*ts[::-1][a]-so[::-1][a])/ss[::-1][a]-.5 for a in range(3)]
    xyz=np.meshgrid(*vectors,indexing='ij')
    values=map_coordinates(raw,np.stack(xyz[::-1]),order=1,mode='nearest',prefilter=False,output=np.float32)
    return np.rint(values).clip(0,255).astype(np.uint8)


def convert(source, output, root):
    import zarr
    from numcodecs import Blosc
    source=Path(source);output=external(output);root=external(root)
    if root not in output.parents:raise ValueError('Output must be inside budget root')
    meta=json.loads((source/'manifest.json').read_text())
    if meta.get('schema')!='pathfinder-continuous-image/v1':raise ValueError('Image manifest required')
    for name in ('raw.npy','image_valid_z.npy'):
        if sha256(source/name)!=meta['files'][name]:raise ValueError('Source hash mismatch: '+name)
    raw=np.load(source/'raw.npy',mmap_mode='r',allow_pickle=False)
    valid=np.load(source/'image_valid_z.npy',allow_pickle=False)
    grid(meta,raw.shape)
    if raw.dtype!=np.uint8 or valid.dtype!=np.bool_ or valid.shape!=(raw.shape[0],) or not valid.all():
        raise ValueError('Complete uint8 image and valid Z planes required')
    target=target_grid(meta);shape=target['shape'];n=int(np.prod(shape,dtype=np.int64))
    reserve(root,n+max(n//50,16_000_000))
    output.mkdir(parents=True,exist_ok=False)
    arr=zarr.open(str(output/'data.zarr/img'),mode='w',shape=tuple(shape)+(1,),
        chunks=(128,128,128,1),dtype='u1',compressor=Blosc(cname='lz4',clevel=5,shuffle=1))
    count=0
    for start in itertools.product(*(range(0,s,128) for s in shape)):
        stop=tuple(min(a+128,s) for a,s in zip(start,shape))
        arr[tuple(slice(a,b) for a,b in zip(start,stop))+(0,)]=image_chunk(raw,meta,target,start,stop)
        count+=1
        if count%100==0:print(json.dumps(dict(completed_chunks=count)),flush=True)
    target.update(source_manifest_sha256=sha256(source/'manifest.json'),
        source_raw_sha256=meta['files']['raw.npy'],adapter_sha256=sha256(Path(__file__)),
        conversion='Trilinear sampling at physical voxel centers; nearest edge extension; round-to-even uint8; no antialias filter',
        extent_policy='Same lower outer corner; trim less than one target voxel at each high face',
        native_resolution_restored=False,ground_truth_used=False,chunks=count,
        files={str(p.relative_to(output)):sha256(p) for p in sorted((output/'data.zarr').rglob('*')) if p.is_file()})
    write_json(output/'metadata.json',target)
    return target


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','output','root'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();convert(a.source,a.output,a.root)
