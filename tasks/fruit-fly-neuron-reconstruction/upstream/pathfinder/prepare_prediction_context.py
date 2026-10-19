"""Attach verified image coverage and an instance index to a completed prediction.

The parent receipt is retained unchanged. Reference mode pins the original NPY
without creating another array path; legacy mode hardlinks and charges its full
size. The referenced parent must remain available. A whole-ROI coverage
record denotes completed computation, not legacy disjoint FFN tile IDs, complete
SHAPE candidate coverage, observed image everywhere, or biological accuracy.
"""
import argparse
import json
import os
from execution_resources import valid_root
from pathlib import Path
import numpy as np
from stitch_ffn import validate
from consensus_stitched import verify_package
from consensus_ffn import validate_six_runs,validate_checkpoint_runs
from instance_index import build_index
from compact_index import build_table,DTYPE
from index_storage import load_index,TableRecords
from common import external,sha256,write_json
from fetch_axonem import reserve
from context_storage import segmentation_path


SCHEMA='prepared-prediction-context/v1'


def inside(root,path):
    path=Path(path).resolve()
    if root not in path.parents:raise ValueError('All artifacts must remain inside budget root')
    return path


def parent_evidence(prediction,root,manifest=None):
    prediction=inside(root,prediction);rp=prediction/'receipt.json';mp=prediction/'metadata.json'
    receipt=json.loads(rp.read_text());metadata=json.loads(mp.read_text())
    if receipt.get('complete') is not True or receipt.get('ground_truth_used') is not False:
        raise ValueError('Complete prediction-only parent required')
    pins={str(rp):sha256(rp),str(mp):sha256(mp)}
    if receipt.get('metadata_sha256')!=pins[str(mp)]:raise ValueError('Parent metadata changed')
    schema=receipt.get('schema')
    if schema=='overlap-stitched-ffn/v1':
        if manifest is None or receipt.get('cross_tile_stitching') is not True:raise ValueError('Stitched manifest required')
        manifest=inside(root,manifest);m=json.loads(manifest.read_text());validate(m)
        pins[str(manifest)]=sha256(manifest)
        if receipt['manifest_sha256']!=pins[str(manifest)] or metadata!=m['metadata']:raise ValueError('Parent stitching provenance differs')
        roi=m['roi'];raw_sha=m['raw_sha256'];image_sha=m['image_manifest_sha256']
        image_path=inside(root,m['image_manifest']);pins[str(image_path)]=sha256(image_path)
        if pins[str(image_path)]!=m['image_manifest_sha256']:raise ValueError('Parent image manifest changed')
    elif schema in ('stitched-global-consensus/v1','stitched-single-checkpoint-consensus/v1'):
        if manifest is not None:raise ValueError('Consensus provenance is recorded in its parent receipt')
        packages=[]
        for path in receipt['ordered_packages']:
            package,package_pins=verify_package(path,root);packages.append(package);pins.update(package_pins)
        if schema=='stitched-global-consensus/v1':
            if validate_six_runs(packages)!=receipt['required_six_runs'] or len(receipt['stages'])!=5:raise ValueError('Incomplete six-run consensus')
        elif (receipt.get('checkpoint_count')!=1 or
              validate_checkpoint_runs(packages,1)!=receipt['required_checkpoint_runs'] or len(receipt['stages'])!=1):
            raise ValueError('Incomplete single-checkpoint seed-order consensus')
        roi=receipt['roi'];raw_sha=receipt['raw_sha256']
        if any(p['metadata']!=metadata or p['roi']!=roi or p['raw_sha256']!=raw_sha for p in packages):raise ValueError('Consensus input grids differ')
        image_hashes={json.loads(Path(p['manifest']).read_text())['image_manifest_sha256'] for p in packages}
        if len(image_hashes)!=1:raise ValueError('Consensus image manifests differ')
        image_sha=image_hashes.pop()
        for path,digest in receipt['input_sha256'].items():
            path=inside(root,path)
            if sha256(path)!=digest:raise ValueError('Consensus input changed')
            pins[str(path)]=digest
    else:raise ValueError('Unsupported parent prediction schema')
    if 'metadata' in receipt and receipt['metadata']!=metadata:raise ValueError('Parent embedded metadata differs')
    return receipt,metadata,roi,raw_sha,image_sha,pins


def image_evidence(image_path,root,metadata,roi,raw_sha):
    image_path=inside(root,image_path);image=json.loads(image_path.read_text());meta=dict(metadata)
    rawpath=image_path.parent/'raw.npy';validpath=image_path.parent/'image_valid_z.npy'
    pins={str(image_path):sha256(image_path)}
    for path in (rawpath,validpath):
        inside(root,path);digest=sha256(path)
        if digest!=image['files'][path.name]:raise ValueError('Image artifact changed')
        pins[str(path)]=digest
    if raw_sha!=pins[str(rawpath)]:raise ValueError('Parent raw image differs')
    raw=np.load(rawpath,mmap_mode='r',allow_pickle=False);valid=np.load(validpath,mmap_mode='r',allow_pickle=False)
    try:
        bounds=np.asarray(roi)
        if (bounds.shape!=(6,) or bounds.dtype.kind not in 'iu' or (bounds[::2]<0).any()
            or (bounds[1::2]>raw.shape).any() or (bounds[1::2]<=bounds[::2]).any()
            or (bounds[1::2]-bounds[::2]).tolist()!=meta.get('shape')):raise ValueError('Invalid prediction ROI')
        if raw.ndim!=3 or raw.dtype!=np.uint8 or image.get('shape')!=list(raw.shape):raise ValueError('Invalid raw image geometry')
        if valid.shape!=(raw.shape[0],) or valid.dtype!=np.bool_:raise ValueError('Boolean image validity required')
        for obj in (image,meta):
            origin=np.asarray(obj.get('origin_nm'),dtype=float)
            if (obj.get('axes')!='zyx' or obj.get('coordinate_convention')!='voxel_outer_corner'
                or obj.get('voxel_size_nm')!=[16,16,16] or origin.shape!=(3,) or not np.isfinite(origin).all()):raise ValueError('16nm ZYX physical grid required')
        if (meta.get('raw_origin_nm',image['origin_nm'])!=image['origin_nm'] or
            not np.array_equal(np.asarray(meta['origin_nm']),np.asarray(image['origin_nm'])+bounds[::2]*16)):
            raise ValueError('Physical ROI differs from raw grid')
        # Older immutable stitched receipts omitted this redundant origin. It is
        # derived here from the verified raw manifest, never written to parent.
        meta['raw_origin_nm']=image['origin_nm']
        z0,z1=roi[:2]
        observation=dict(manifest_sha256=pins[str(image_path)],validity_sha256=pins[str(validpath)],
            observed_z_fraction=float(valid[z0:z1].mean()),unknown_output_z=np.flatnonzero(~valid[z0:z1]).tolist(),
            scope=image['image_valid']['scope'],all_defects_detected=False)
        return meta,observation,pins,int(raw.nbytes)
    finally:raw._mmap.close();valid._mmap.close()


def coverage_record(roi,observation):
    return dict(schema='prediction-context-coverage/v1',tiles=[dict(index=0,start=roi[::2],stop=roi[1::2],status='complete')],
        total_tiles=1,complete_tiles=1,all_tiles_complete=True,image_observation=observation,
        coverage_units='One whole-ROI completion region; not disjoint FFN tiles',
        scope='Completed computation only; image gaps and limited downstream candidate coverage remain explicit')


def run(prediction,image_manifest,root,output,memory_gib,manifest=None,slab_depth=8,max_instances=1000000,*,reference_parent_array=False,numeric_max_label=0):
    root=external(root);output=external(output)
    if not valid_root(root) or root not in output.parents or output.exists():raise ValueError('New Xtemp output required')
    if (not np.isfinite(memory_gib) or memory_gib<=0 or type(slab_depth) is not int or slab_depth<1
        or type(max_instances) is not int or max_instances<1):raise ValueError('Finite positive resource bounds required')
    if type(numeric_max_label) is not int or numeric_max_label<0:raise ValueError('Nonnegative numeric label bound required')
    prediction=inside(root,prediction);image_manifest=inside(root,image_manifest)
    parent,metadata,roi,raw_sha,image_sha,pins=parent_evidence(prediction,root,manifest)
    if sha256(image_manifest)!=image_sha:raise ValueError('Image manifest differs from parent inference')
    metadata,observation,image_pins,raw_bytes=image_evidence(image_manifest,root,metadata,roi,raw_sha);pins.update(image_pins)
    source=prediction/'segmentation.npy';digest=sha256(source)
    if digest!=parent['segmentation_sha256']:raise ValueError('Parent prediction changed')
    shape=metadata['shape'];nbytes=int(np.prod(shape))*4
    memory=nbytes+raw_bytes+(256 if numeric_max_label else 128)*min(slab_depth,shape[0])*shape[1]*shape[2]+(128*(numeric_max_label+1) if numeric_max_label else 4096*max_instances)+2**30
    if memory>memory_gib*2**30:raise ValueError('Index working set exceeds declared memory')
    reserved=(0 if reference_parent_array else source.stat().st_size)+(DTYPE.itemsize*(numeric_max_label+1)+256 if numeric_max_label else 1024*max_instances)+10000000
    reserve(root,reserved)  # Hardlinks are deliberately charged in full.
    sources={str(Path(__file__).with_name(name)):sha256(Path(__file__).with_name(name)) for name in
        ('prepare_prediction_context.py','context_storage.py','compact_index.py','index_storage.py','instance_index.py','stitch_ffn.py','consensus_stitched.py','consensus_ffn.py','fetch_axonem.py')}
    common=Path(__file__).resolve().parents[1]/'common.py';sources[str(common)]=sha256(common)
    seg=np.load(source,mmap_mode='r',allow_pickle=False)
    try:
        if seg.dtype!=np.uint32 or list(seg.shape)!=shape:raise ValueError('Parent uint32 prediction geometry differs')
        records=build_table(seg,slab_depth,max_label=numeric_max_label,max_instances=max_instances) if numeric_max_label else build_index(seg,slab_depth,max_instances)
    finally:seg._mmap.close()
    output.mkdir()
    if not reference_parent_array:os.link(source,output/'segmentation.npy')
    stored=source if reference_parent_array else output/'segmentation.npy'
    index=dict(schema='predicted-instance-index/v1',shape=shape,axes='zyx',records=records,segmentation_sha256=digest,ground_truth_used=False)
    if numeric_max_label:
        table_path=output/'instance-table.npy';np.save(table_path,records,allow_pickle=False)
        if table_path.stat().st_size>DTYPE.itemsize*(numeric_max_label+1)+256:raise ValueError('Numeric index exceeds reserved bytes')
        index.pop('records');index.update(storage='numeric-table-v1',table=dict(file=table_path.name,sha256=sha256(table_path),bytes=table_path.stat().st_size,labels=len(records),instances=int(np.count_nonzero(records['size']))))
    write_json(output/'metadata.json',metadata);write_json(output/'coverage.json',coverage_record(roi,observation));write_json(output/'instance-index.json',index)
    for path,expected in {**pins,**sources,str(stored):digest}.items():
        if sha256(path)!=expected:raise ValueError('Input/source changed during context preparation')
    if sum(p.stat().st_size for p in output.iterdir())>reserved-1000000:raise ValueError('Context metadata exceeds reservation')
    result=dict(schema=SCHEMA,complete=True,ground_truth_used=False,paper_comparable=False,
        prediction=str(prediction),manifest=str(Path(manifest).resolve()) if manifest else None,
        image_manifest=str(image_manifest),root=str(root),input_sha256=pins,source_sha256=sources,
        roi=roi,raw_sha256=raw_sha,metadata=metadata,image_observation=observation,all_tiles_complete=True,
        segmentation_sha256=digest,metadata_sha256=sha256(output/'metadata.json'),coverage_sha256=sha256(output/'coverage.json'),
        index_sha256=sha256(output/'instance-index.json'),reserved_output_bytes=reserved,conservative_memory_bytes=memory,
        slab_depth=slab_depth,max_instances=max_instances,numeric_max_label=numeric_max_label,storage='hardlink; charged as full array in budget',
        scope='Parent prediction unchanged; derived image coverage and full global-instance index. Does not establish complete SHAPE context at every center or model/paper parity.')
    if reference_parent_array:result.update(storage='parent-array-reference',segmentation_path=str(source))
    write_json(output/'receipt.json',result);return result


def audit(output):
    output=Path(output);r=json.loads((output/'receipt.json').read_text());root=Path(r['root']).resolve()
    inside(root,output)
    if r.get('schema')!=SCHEMA or r.get('complete') is not True or r.get('ground_truth_used') is not False:raise ValueError('Complete prepared prediction required')
    parent,metadata,roi,raw_sha,image_sha,pins=parent_evidence(r['prediction'],root,r['manifest'])
    if sha256(r['image_manifest'])!=image_sha:raise ValueError('Image manifest differs from parent inference')
    metadata,observation,image_pins,_=image_evidence(r['image_manifest'],root,metadata,roi,raw_sha);pins.update(image_pins)
    if pins!=r['input_sha256'] or metadata!=r['metadata'] or roi!=r['roi'] or raw_sha!=r['raw_sha256'] or observation!=r['image_observation']:
        raise ValueError('Prepared context provenance changed')
    for name,key in [('segmentation.npy','segmentation_sha256'),('metadata.json','metadata_sha256'),('coverage.json','coverage_sha256'),('instance-index.json','index_sha256')]:
        path=segmentation_path(output) if name=='segmentation.npy' else output/name
        if sha256(path)!=r[key]:raise ValueError('Prepared prediction artifact changed')
    if r['segmentation_sha256']!=parent['segmentation_sha256']:raise ValueError('Prediction differs from parent')
    if json.loads((output/'metadata.json').read_text())!=metadata or json.loads((output/'coverage.json').read_text())!=coverage_record(roi,observation) or r['all_tiles_complete'] is not True:
        raise ValueError('Prepared coverage disagrees with parent')
    index=load_index(output/'instance-index.json')
    if index['segmentation_sha256']!=r['segmentation_sha256'] or index['shape']!=metadata['shape']:raise ValueError('Prepared index differs')
    if isinstance(index.get('records'),TableRecords):index['records'].table._mmap.close()
    return dict(full_context_reconstruction_complete=True,complete_tiles=1,total_tiles=1,
        coverage_units='One whole-ROI completion region; not disjoint FFN tiles',
        assembly_receipt_sha256=sha256(output/'receipt.json'),coverage_sha256=r['coverage_sha256'],segmentation_sha256=r['segmentation_sha256'],
        scope='Complete compute coverage only; no biological accuracy or full SHAPE candidate coverage claim')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('prediction','image-manifest','root','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--manifest',type=Path);p.add_argument('--memory-gib',type=float,required=True)
    p.add_argument('--slab-depth',type=int,default=8);p.add_argument('--max-instances',type=int,default=1000000)
    p.add_argument('--reference-parent-array',action='store_true',help='Pin and use original parent NPY without another path charged to disk')
    p.add_argument('--numeric-max-label',type=int,default=0,help='Positive explicit dense-label cap enables compact numeric index; zero keeps legacy records')
    a=p.parse_args();print(json.dumps(run(a.prediction,a.image_manifest,a.root,a.output,a.memory_gib,a.manifest,a.slab_depth,a.max_instances,reference_parent_array=a.reference_parent_array,numeric_max_label=a.numeric_max_label),indent=2))
