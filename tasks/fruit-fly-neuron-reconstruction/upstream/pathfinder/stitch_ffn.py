"""Assemble complete overlapping FFN windows with explicit conservative stitching.

Uses Google's connected-component implementation; interface projection and tie
handling are local, documented choices. No ground truth enters this stage.
"""
import argparse
import hashlib
import json
from execution_resources import valid_root
from pathlib import Path
import numpy as np
from batched_ffn import MIN_INSTANCE_VOXELS,grid_tiles,window_finished
from common import external,sha256,write_json
from consensus_ffn import author_consensus
from evaluate_physical import grid
from fetch_axonem import reserve
from merge_ffn_agglomeration import union_mapping,remap
from overlap_stitching import component_interface_plane,mutual_maximum_pairs
from window_io import window_array,reserved_window_bytes


def windows(roi,core_size=436,margin=32,boundary_image_shape=None):
    if (len(roi)!=6 or any(type(v) is not int for v in roi) or
            type(core_size) is not int or core_size<1 or type(margin) is not int or margin<1):
        raise ValueError('Integer ROI, positive core size and overlap margin required')
    lo=np.asarray(roi[::2]);hi=np.asarray(roi[1::2])
    if boundary_image_shape is None:
        if (hi<=lo).any() or (lo<16+margin).any():raise ValueError('ROI needs real overlap and network context')
    else:
        if len(boundary_image_shape)!=3 or any(type(n) is not int or n<=32 for n in boundary_image_shape):
            raise ValueError('Explicit integer raw image shape required')
        if (hi<=lo).any() or (lo<16).any() or (hi>np.asarray(boundary_image_shape)-16).any():
            raise ValueError('Entire ROI needs real network context')
    result=[dict(index=t['index'],core_start=t['start'],core_stop=t['stop'],
                 start=(np.asarray(t['start'])-margin).tolist(),stop=(np.asarray(t['stop'])+margin).tolist())
            for t in grid_tiles(lo,hi,core_size)]
    if boundary_image_shape is not None:
        for row in result:
            row['start']=np.maximum(row['start'],16).tolist()
            row['stop']=np.minimum(row['stop'],np.asarray(boundary_image_shape)-16).tolist()
    return result


def drop_small_components(core,min_voxels=MIN_INSTANCE_VOXELS):
    """Clear disconnected pieces below FFN's own instance minimum and relabel 1..n in label order.

    Splitting a grown FFN object into face-connected components can leave hundreds of millions of
    one-voxel specks for a noisy checkpoint; they would exceed the context label bound and each
    would become a SENSE center. Growth already rejects objects below min_voxels; the same rule
    applies to their pieces. Cleared voxels are unlabelled (omissions when scored).
    """
    ids=np.ascontiguousarray(core).view(np.int64) if core.dtype==np.uint64 else core.astype(np.intp)
    sizes=np.bincount(ids.ravel())
    keep=sizes>=min_voxels;keep[0]=False
    remap=np.zeros(len(sizes),core.dtype);remap[keep]=np.arange(1,int(keep.sum())+1,dtype=core.dtype)
    small=(sizes>0)&~keep;small[0]=False
    return remap[ids],int(keep.sum()),(int(small.sum()),int(sizes[small].sum()))


def validate(manifest):
    if manifest.get('schema')!='overlapping-ffn-windows/v1' or manifest.get('ground_truth_used') is not False:
        raise ValueError('Prediction-only overlap manifest required')
    expected=windows(manifest['roi'],manifest['core_size'],manifest['margin'],manifest.get('boundary_image_shape'))
    records=manifest['windows']
    if len(records)!=len(expected) or sorted(r['index'] for r in records)!=list(range(len(expected))):
        raise ValueError('Missing or duplicate prediction windows')
    records=sorted(records,key=lambda r:r['index'])
    storage_limits=manifest.get('window_storage_limits')
    if storage_limits is not None:
        if (not isinstance(storage_limits,dict) or set(storage_limits)!={str(w['index']) for w in expected}
            or any(type(v) is not int or v<1 for v in storage_limits.values())
            or manifest.get('compressed_window_bytes')!=max(storage_limits.values())):
            raise ValueError('Exact positive per-window storage reservations required')
    for r,w in zip(records,expected):
        if any(r[k]!=v for k,v in w.items()):raise ValueError('Window geometry differs from declared grid')
        cap=manifest.get('compressed_window_bytes',0) if storage_limits is None else storage_limits[str(w['index'])]
        expected_bytes=reserved_window_bytes(np.asarray(w['stop'])-w['start'],cap)
        if cap and (r.get('storage')!='npz' or r.get('reserved_bytes')!=expected_bytes):
            raise ValueError('Window storage differs from declared reservation')
        if not window_finished(r):raise ValueError('Incomplete prediction window')
        for key in ('raw_sha256','checkpoint_sha256','seed_order'):
            if r[key]!=manifest[key]:raise ValueError('Mixed inference inputs')
    if manifest['seed_order'] not in ('ascending','descending'):raise ValueError('Unknown seed order')
    shape=(np.asarray(manifest['roi'][1::2])-manifest['roi'][::2]).tolist()
    grid(manifest['metadata'],shape)
    if manifest['metadata'].get('axes')!='zyx' or manifest['metadata']['shape']!=shape:
        raise ValueError('ZYX output geometry required')
    return records


def run(manifest_path,root,output,ffn_source,connectomics_source,memory_gib,*,compressed_core_bytes=0,compression_level=1,mapping_bytes=0):
    manifest_path=Path(manifest_path);manifest_hash=sha256(manifest_path)
    manifest=json.loads(manifest_path.read_text());records=validate(manifest)
    root=external(root);output=external(output)
    if not valid_root(root) or root not in output.parents or output.exists():
        raise ValueError('New Xtemp output under budget root required')
    if not np.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Positive memory bound required')
    image_path=Path(manifest['image_manifest']).resolve()
    if root not in image_path.parents or sha256(image_path)!=manifest['image_manifest_sha256']:raise ValueError('Image manifest changed')
    image=json.loads(image_path.read_text());raw_origin,spacing,raw_shape,_=grid(image)
    if manifest.get('boundary_image_shape') is not None and manifest['boundary_image_shape']!=raw_shape.tolist():
        raise ValueError('Boundary clipping image shape differs from raw data')
    if image['axes']!='zyx' or image['files']['raw.npy']!=manifest['raw_sha256']:raise ValueError('Raw image provenance differs')
    if manifest['metadata'].get('raw_origin_nm') is not None and manifest['metadata']['raw_origin_nm']!=raw_origin.tolist():
        raise ValueError('Raw origin provenance differs from image')
    if (not np.array_equal(manifest['metadata']['voxel_size_nm'],spacing) or
            not np.array_equal(manifest['metadata']['origin_nm'],raw_origin+np.asarray(manifest['roi'][::2])*spacing)):
        raise ValueError('Physical output origin differs from raw-grid ROI')
    if any((np.asarray(r['start'])<16).any() or (np.asarray(r['stop'])>raw_shape-16).any() for r in records):
        raise ValueError('Window lacks real network context')
    source_files={str(Path(__file__).with_name(name)):sha256(Path(__file__).with_name(name)) for name in
                  ('stitch_ffn.py','overlap_stitching.py','merge_ffn_agglomeration.py','consensus_ffn.py','batched_ffn.py',
                   'evaluate_physical.py','fetch_axonem.py','window_io.py','bounded_json.py')}
    common_path=Path(__file__).resolve().parents[1]/'common.py'
    source_files[str(common_path)]=sha256(common_path)
    largest=max(int(np.prod(np.asarray(r['stop'])-r['start'])) for r in records)
    shape=manifest['metadata']['shape'];output_bytes=int(np.prod(shape))*4
    starts={tuple(r['core_start']) for r in records};boundary_voxels=0
    for row in records:
        for axis in range(3):
            neighbor=list(row['core_start']);neighbor[axis]=row['core_stop'][axis]
            if tuple(neighbor) in starts:
                boundary_voxels+=int(np.prod([row['core_stop'][i]-row['core_start'][i] for i in range(3) if i!=axis]))
    # Two exact uint32 planes per interface, generated during the existing core pass.
    plane_cache_bytes=8*boundary_voxels
    by_start={tuple(r['core_start']):r for r in records}
    interface_specs=[];requests={r['index']:[] for r in records}
    for left in records:
        for axis in range(3):
            neighbor=list(left['core_start']);neighbor[axis]=left['core_stop'][axis]
            right=by_start.get(tuple(neighbor))
            if right is None:continue
            coordinate=right['core_start'][axis]-1
            bounds=[[max(left['core_start'][i],right['core_start'][i]),min(left['core_stop'][i],right['core_stop'][i])] for i in range(3)]
            spec=dict(left=left['index'],right=right['index'],axis=axis,coordinate_voxels=coordinate)
            number=len(interface_specs);interface_specs.append(spec)
            for side,row in enumerate((left,right)):
                requests[row['index']].append((number,side,axis,coordinate,bounds))
    # Allow serialized interface records/mappings, not only the output arrays.
    auxiliary_bytes=512*boundary_voxels+10_000_000
    if 96*largest+output_bytes+2*auxiliary_bytes+plane_cache_bytes>memory_gib*2**30:raise ValueError('Conservative working set exceeds declared memory')
    if type(mapping_bytes) is not int or mapping_bytes<0:raise ValueError('Nonnegative mapping byte cap required')
    if mapping_bytes:auxiliary_bytes=mapping_bytes+10_000_000
    from window_io import reserved_window_bytes,save_compressed
    if type(compression_level) is not int or not 0<=compression_level<=9:raise ValueError('Invalid compression level')
    core_bytes=sum(reserved_window_bytes(np.asarray(r['core_stop'])-r['core_start'],compressed_core_bytes) for r in records)
    reserve(root,output_bytes+128+core_bytes+auxiliary_bytes)
    for r in records:
        path=Path(r['path']).resolve()
        if root not in path.parents:raise ValueError('Prediction outside budget root')
        with window_array(r) as array:
            if array.dtype.kind!='u' or list(array.shape)!=(np.asarray(r['stop'])-r['start']).tolist():raise ValueError('Invalid window array')
    author_consensus(ffn_source,connectomics_source)  # Checks pinned clean repositories.
    from connectomics.segmentation import labels
    component_source=Path(labels.__file__).resolve()
    if component_source!=Path(connectomics_source).resolve()/'connectomics/segmentation/labels.py':raise ValueError('Unexpected component implementation')
    component_hash=sha256(component_source)
    output.mkdir();(output/'cores').mkdir();offset=0;core_receipts=[]
    plane_cache={};plane_receipts=[];cached_bytes=0
    for r in records:
        lo=np.asarray(r['core_start'])-r['start'];hi=np.asarray(r['core_stop'])-r['start']
        with window_array(r) as array:
            crop=np.array(array[tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))],dtype=np.uint64)
            core,count,dropped=drop_small_components(labels.split_disconnected_components(crop,connectivity=1))
            if offset+count>np.iinfo(np.uint32).max:raise ValueError('Global component ID overflow')
            positive=core>0;core[positive]+=offset;offset+=count
            core=core.astype(np.uint32)
            for number,side,axis,coordinate,bounds in requests[r['index']]:
                plane=component_interface_plane(array,core,r['start'],r['core_start'],axis,coordinate,bounds)
                cached_bytes+=plane.nbytes
                if cached_bytes>plane_cache_bytes:raise ValueError('Interface plane cache exceeds bound')
                plane_cache[(number,side)]=plane
                plane_receipts.append(dict(interface=number,side=side,window=r['index'],shape=list(plane.shape),
                    sha256=hashlib.sha256(plane.tobytes(order='C')).hexdigest()))
        if compressed_core_bytes:
            path=output/'cores'/f"{r['index']:07d}.npz"
            storage=save_compressed(core,path,reserved_window_bytes(core.shape,compressed_core_bytes),compression_level=compression_level)
        else:
            path=output/'cores'/f"{r['index']:07d}.npy";np.save(path,core)
            storage=dict(path=str(path),sha256=sha256(path))
        core_receipts.append(dict(index=r['index'],components=count,dropped_small_components=dropped[0],dropped_small_voxels=dropped[1],**storage))
    pairs=[];interfaces=[]
    for number,spec in enumerate(interface_specs):
        matching=mutual_maximum_pairs(plane_cache.pop((number,0)),plane_cache.pop((number,1)))
        pairs.extend([p['a'],p['b']] for p in matching['pairs'])
        interfaces.append(dict(spec,matching=matching))
    if plane_cache:raise ValueError('Unconsumed interface planes')
    write_json(output/'interface-planes.json',dict(schema='exact-component-interface-planes/v1',
        cache_bytes=cached_bytes,cache_bound_bytes=plane_cache_bytes,planes=plane_receipts,ground_truth_used=False))
    mapping=union_mapping(pairs)
    from bounded_json import write_bounded
    mapping_value=dict(mapping={str(k):v for k,v in mapping.items()},interfaces=interfaces,ground_truth_used=False)
    if mapping_bytes:write_bounded(output/'mapping.json',mapping_value,mapping_bytes)
    else:write_json(output/'mapping.json',mapping_value)
    dest=np.lib.format.open_memmap(output/'segmentation.npy',mode='w+',dtype=np.uint32,shape=tuple(shape));origin=np.asarray(manifest['roi'][::2])
    try:
        for row in records:
            lo=np.asarray(row['core_start'])-origin;hi=np.asarray(row['core_stop'])-origin
            with window_array(core_receipts[row['index']]) as core:
                dest[tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))]=remap(core,mapping)
        dest.flush()
    finally:dest._mmap.close()
    for row in records:
        if sha256(row['path'])!=row.get('archive_sha256',row['sha256']):raise ValueError('Prediction changed during stitching')
    for p,h in source_files.items():
        if sha256(p)!=h:raise ValueError('Stitching source changed')
    if sha256(image_path)!=manifest['image_manifest_sha256'] or sha256(manifest_path)!=manifest_hash or sha256(component_source)!=component_hash:raise ValueError('Manifest/author implementation changed')
    if mapping_bytes:write_bounded(output/'metadata.json',manifest['metadata'],5_000_000)
    else:write_json(output/'metadata.json',manifest['metadata'])
    result=dict(schema='overlap-stitched-ffn/v1',complete=True,manifest_sha256=manifest_hash,source_sha256=source_files,
                roi=manifest['roi'],raw_sha256=manifest['raw_sha256'],metadata=manifest['metadata'],
                connected_components_source_sha256=component_hash,cores=core_receipts,windows=len(records),interfaces=len(interfaces),
                call_limited_windows=sorted(r['index'] for r in records if r.get('call_limit_reached') is True),
                min_component_voxels=MIN_INSTANCE_VOXELS,
                reserved_output_bytes=output_bytes+128+core_bytes+auxiliary_bytes,declared_memory_gib=memory_gib,
                compressed_core_bytes=compressed_core_bytes,compression_level=compression_level,
                mapping_bytes=mapping_bytes,interface_plane_cache_bytes=cached_bytes,
                interface_planes_sha256=sha256(output/'interface-planes.json'),
                matched_pairs=len(pairs),remapped_ids=len(mapping),mapping_sha256=sha256(output/'mapping.json'),
                segmentation_sha256=sha256(output/'segmentation.npy'),metadata_sha256=sha256(output/'metadata.json'),
                ground_truth_used=False,paper_comparable=False,cross_tile_stitching=True,
                scope='Official core connected components plus independently implemented mutual overlap matching; explicit local tie/background/interface conventions. Not paper parity.')
    if mapping_bytes:write_bounded(output/'receipt.json',result,5_000_000)
    else:write_json(output/'receipt.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','root','output','ffn-source','connectomics-source'):p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--memory-gib',type=float,required=True)
    p.add_argument('--compressed-core-bytes',type=int,default=0)
    p.add_argument('--compression-level',type=int,default=1)
    p.add_argument('--mapping-bytes',type=int,default=0,help='Hard mapping JSON cap; exceeding it fails without publishing a partial result')
    a=p.parse_args();print(json.dumps(run(a.manifest,a.root,a.output,a.ffn_source,a.connectomics_source,a.memory_gib,
        compressed_core_bytes=a.compressed_core_bytes,compression_level=a.compression_level,mapping_bytes=a.mapping_bytes),indent=2))
