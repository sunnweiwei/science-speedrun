"""Sharded batched SENSE on indexed predicted instances, without ground truth.

Select only representative instance centers with a full geometric 10um SHAPE
radius in the assembled prediction. This deliberately does not claim full-volume
candidate coverage. Acquisition gaps remain explicit; zero labels are unknown.
"""
import argparse
import json
import re
from pathlib import Path
import time
import sys
import numpy as np
from index_storage import load_index,TableRecords,TableMapping,validate_table,select_table_centers
import torch
from sense import Sense, prompt_mask, candidates
from infer_sense import checkpoint_config
from inference_progress import Progress
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external, device_guard, sha256, write_json
from execution_resources import device_guard


def validate_inputs(raw, seg, valid_z, image, assembled, index):
    meta=assembled.get('metadata',{})
    if assembled.get('ground_truth_used') is not False or index.get('ground_truth_used') is not False:
        raise ValueError('Prediction-only assembled receipt and index required')
    if (index.get('schema')!='predicted-instance-index/v1' or index.get('axes')!='zyx'
        or index.get('segmentation_sha256')!=assembled.get('segmentation_sha256')
        or not isinstance(assembled.get('segmentation_sha256'),str)
        or not re.fullmatch('[0-9a-f]{64}',assembled['segmentation_sha256'])):
        raise ValueError('Instance index segmentation hash/provenance mismatch')
    if raw.ndim!=3 or raw.dtype!=np.uint8 or seg.ndim!=3 or seg.dtype.kind!='u':
        raise ValueError('Unsigned predicted volume and uint8 image required')
    if valid_z.shape!=(raw.shape[0],) or valid_z.dtype!=np.bool_:
        raise ValueError('Image availability must be a Z boolean vector')
    for obj,shape in [(image,raw.shape),(meta,seg.shape),(index,seg.shape)]:
        if obj.get('axes')!='zyx' or obj.get('shape')!=list(shape):raise ValueError('Axis/shape mismatch')
    for obj in (image,meta):
        if obj.get('coordinate_convention')!='voxel_outer_corner':raise ValueError('Outer-corner convention required')
        for name in ('origin_nm','voxel_size_nm'):
            v=np.asarray(obj.get(name),dtype=float)
            if v.shape!=(3,) or not np.isfinite(v).all():raise ValueError('Invalid physical metadata')
        if list(obj['voxel_size_nm'])!=[16,16,16]:raise ValueError('Full128 SENSE recipe requires16nm grid')
    offset=(np.asarray(meta['origin_nm'])-image['origin_nm'])/16
    if not np.array_equal(offset,np.round(offset)) or (offset<0).any() or (offset+seg.shape>raw.shape).any():
        raise ValueError('Prediction ROI does not align inside raw image')
    if meta.get('raw_origin_nm')!=image['origin_nm']:raise ValueError('Raw origin provenance mismatch')
    if index.get('shape')!=list(seg.shape):raise ValueError('Index geometry mismatch')
    rows=index.get('records')
    if isinstance(rows,TableRecords):
        validate_table(rows,seg);return offset.astype(int)
    if not isinstance(rows,list):raise ValueError('Index records required')
    seen=set()
    for row in rows:
        sid=row.get('id');size=row.get('size');center=np.asarray(row.get('center_zyx'));box=np.asarray(row.get('bbox_zyx'))
        if type(sid)!=int or sid<=0 or sid in seen or type(size)!=int or size<1:
            raise ValueError('Unique positive instance IDs/sizes required')
        seen.add(sid)
        if (center.shape!=(3,) or center.dtype.kind not in 'iu' or box.shape!=(3,2) or box.dtype.kind not in 'iu'
            or (box[:,0]<0).any() or (box[:,1]>seg.shape).any() or (box[:,1]<=box[:,0]).any()
            or (center<box[:,0]).any() or (center>=box[:,1]).any()
            or size>int(np.prod(box[:,1]-box[:,0]))):raise ValueError('Invalid instance bounding box/center/size')
        if int(seg[tuple(center)])!=sid:raise ValueError('Index center is not its positive predicted instance')
    return offset.astype(int)


def select_centers(records, shape, shard_index, shards):
    if type(shards)!=int or shards<1 or type(shard_index)!=int or not 0<=shard_index<shards:
        raise ValueError('Invalid shard assignment')
    if isinstance(records,TableRecords):return select_table_centers(records,shape,shard_index,shards)
    chosen=[];skipped=[];other=[];eligible=[]
    for row in sorted(records,key=lambda r:r['id']):
        physical=(np.asarray(row['center_zyx'])+.5)*16
        if not ((physical>=10000)&(physical<=np.asarray(shape)*16-10000)).all():
            skipped.append(row['id']);continue
        ordinal=len(eligible);eligible.append(row['id'])
        (chosen if ordinal%shards==shard_index else other).append(row if ordinal%shards==shard_index else row['id'])
    return chosen,skipped,other,eligible


def make_crop(raw,seg,valid_z,row,offset):
    center=np.asarray(row['center_zyx']);raw_center=center+offset
    if (raw_center<64).any() or (raw_center>np.asarray(raw.shape)-64).any():
        raise ValueError('Insufficient actual raw128 FOV')
    sl=tuple(slice(int(c-64),int(c+64)) for c in raw_center)
    raw_crop=np.asarray(raw[sl],dtype=np.float32)/255
    lo=np.maximum(center-64,0);hi=np.minimum(center+64,seg.shape)
    labels=np.zeros((128,)*3,seg.dtype)
    labels[tuple(slice(int(l-c+64),int(h-c+64)) for l,h,c in zip(lo,hi,center))]=seg[tuple(slice(int(l),int(h)) for l,h in zip(lo,hi))]
    prompt=prompt_mask(labels,row['id'],16);prompt[labels==0]=.5
    # The 10um sphere spans1250 planes on this grid; the fraction below is
    # explicitly a Z-plane availability fraction, not spherical voxel volume.
    z0=int(np.floor(raw_center[0]+.5-625));z1=int(np.ceil(raw_center[0]+.5+625))
    return raw_crop,prompt,labels,dict(instance_id=row['id'],center_zyx=row['center_zyx'],
        sense_input_missing_z_fraction=float((~valid_z[raw_center[0]-64:raw_center[0]+64]).mean()),
        shape_context_missing_z_fraction=float((~valid_z[z0:z1]).mean()),
        shape_context_raw_z_interval=[z0,z1],geometric_shape_context_complete=True)


def infer(raw,seg,valid_z,image,assembled,index,predict,*,shard_index=0,shards=1,batch_size=8,
          max_calls=256,seconds=1800,min_size=500,threshold=.5,progress=None):
    if not 1<=batch_size<=8 or max_calls<0 or seconds<=0 or min_size<1 or not 0<=threshold<=1:
        raise ValueError('Invalid inference batch/budget/threshold')
    started=time.monotonic();offset=validate_inputs(raw,seg,valid_z,image,assembled,index)
    chosen,skipped,other,eligible=select_centers(index['records'],seg.shape,shard_index,shards)
    sizes=TableMapping(index['records'],sizes=True) if isinstance(index['records'],TableRecords) else {r['id']:r['size'] for r in index['records']};edges={};done=[];calls=0;model_batches=0;expired=False
    if progress:progress.emit(force=True,status="running",processed=0,total=len(chosen),candidates=0)
    while calls<len(chosen):
        if calls>=max_calls or time.monotonic()-started>=seconds:expired=True;break
        rows=chosen[calls:calls+min(batch_size,max_calls-calls)]
        crops=[make_crop(raw,seg,valid_z,row,offset) for row in rows]
        if time.monotonic()-started>=seconds:expired=True;break
        probabilities=predict(np.stack([c[0] for c in crops]),np.stack([c[1] for c in crops]))
        if np.shape(probabilities)!=(len(rows),128,128,128):raise ValueError('Model must preserve full128 field')
        model_batches+=1
        for row,crop,prob in zip(rows,crops,probabilities):
            done.append(crop[3]);calls+=1
            for edge in candidates(prob,crop[2],sizes,row['id'],min_size=min_size,threshold=threshold):
                pair=tuple(sorted((edge['source'],edge['target'])))
                value=dict(source=pair[0],target=pair[1],sense=edge['overlap'],overlap=edge['overlap'],
                    evidence_seed_id=row['id'],center_zyx=row['center_zyx'],
                    sense_input_missing_z_fraction=crop[3]['sense_input_missing_z_fraction'],
                    shape_context_missing_z_fraction=crop[3]['shape_context_missing_z_fraction'])
                if pair not in edges or value['sense']>edges[pair]['sense']:edges[pair]=value
        if progress:progress.emit(status="running",processed=calls,total=len(chosen),candidates=len(edges))
    if progress:progress.emit(force=True,status="limit_reached" if expired else "complete",processed=calls,total=len(chosen),candidates=len(edges))
    elapsed=time.monotonic()-started
    result=dict(schema='pathfinder-candidate-edges/v1',edges=[edges[k] for k in sorted(edges)],metadata=assembled['metadata'],
        ground_truth_used=False,paper_comparable=False,shard_index=shard_index,shards=shards)
    audit=dict(schema='continuous-sense-shard/v1',shard_index=shard_index,shards=shards,batch_size=batch_size,
        calls=calls,model_batches=model_batches,seconds=elapsed,max_calls=max_calls,seconds_limit=seconds,
        budget_exhausted=expired,time_limit_overrun_seconds=max(0,elapsed-seconds),
        time_limit_policy='between whole model batches and preprocessing; in-flight work not interrupted',
        total_indexed_instances=len(sizes),geometrically_eligible_representative_centers=len(eligible),
        eligible_instance_ids=eligible,assigned_instance_ids=[r['id'] for r in chosen],other_shard_instance_ids=other,
        skipped_geometric_context_instance_ids=skipped,unprocessed_instance_ids=[r['id'] for r in chosen[calls:]],
        processed=done,empty_segmentation=not bool(sizes),
        assigned_instance_coverage=calls/len(chosen) if chosen else 0,
        total_indexed_instance_coverage=calls/len(sizes) if sizes else 0,
        selection='one indexed actual-positive representative center per instance; no search for alternate centers',
        candidate_coverage_scope='only geometric10um interior representatives; not full-volume candidates',
        denominator='full predicted instance size from index over entire assembled ROI',
        input_shape=[128]*3,prompt_radius=16,shape_radius_nm=10000,min_size=min_size,threshold=threshold,
        image_availability_scope=image.get('image_valid',{}).get('scope'),
        all_image_defects_detected=False,all_tiles_complete=assembled.get('all_tiles_complete'),
        segmentation_sha256=assembled['segmentation_sha256'],ground_truth_used=False,paper_comparable=False)
    return result,audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('image-manifest','assembled','index','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--max-calls',type=int,default=256)
    p.add_argument('--seconds',type=float,default=1800);p.add_argument('--min-size',type=int,default=500)
    p.add_argument('--threshold',type=float,default=.5);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args();device_guard(a.device);torch.set_num_threads(2)
    out=external(a.output);out.mkdir(parents=True,exist_ok=False)
    progress=Progress(out/'progress.jsonl','sense',a.shard_index)
    progress.emit(force=True,status='validating_inputs')
    image=json.loads(a.image_manifest.read_text());receipt_path=a.assembled/'receipt.json'
    assembled=json.loads(receipt_path.read_text());index=load_index(a.index)
    from context_storage import segmentation_path
    rawpath=a.image_manifest.parent/'raw.npy';validpath=a.image_manifest.parent/'image_valid_z.npy';segpath=segmentation_path(a.assembled)
    # Stream each large artifact once per process, never once per instance.
    for path,expected in [(rawpath,image['files']['raw.npy']),(validpath,image['files']['image_valid_z.npy']),
                          (segpath,assembled['segmentation_sha256'])]:
        if sha256(path)!=expected:raise ValueError(f'Artifact integrity mismatch: {path.name}')
    if (assembled.get('image_observation') or {}).get('manifest_sha256')!=sha256(a.image_manifest):
        raise ValueError('Assembled image-manifest provenance mismatch')
    if json.loads((a.assembled/'metadata.json').read_text())!=assembled['metadata']:
        raise ValueError('Assembled metadata sidecar mismatch')
    coverage_path=a.assembled/'coverage.json';coverage=json.loads(coverage_path.read_text())
    if (coverage.get('all_tiles_complete')!=assembled.get('all_tiles_complete')
        or coverage.get('image_observation')!=assembled.get('image_observation')):
        raise ValueError('Assembled coverage provenance mismatch')
    raw=np.load(rawpath,mmap_mode='r');seg=np.load(segpath,mmap_mode='r');valid=np.load(validpath,mmap_mode='r')
    validate_inputs(raw,seg,valid,image,assembled,index)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    config=checkpoint_config(checkpoint,image['voxel_size_nm'])
    if config.get('input_shape')!=[128]*3 or config.get('prompt_radius')!=16:
        raise ValueError('Full128 radius16 SENSE checkpoint required')
    model=Sense(config['width'],tuple(config['factors'])).to(a.device);model.load_state_dict(checkpoint['model'],strict=True);model.eval()
    def predict(images,prompts):
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.device=='cuda'):
            return model(torch.from_numpy(images[:,None]).to(a.device),torch.from_numpy(prompts[:,None]).to(a.device)).sigmoid()[:,0].float().cpu().numpy()
    result,audit=infer(raw,seg,valid,image,assembled,index,predict,shard_index=a.shard_index,shards=a.shards,
        batch_size=a.batch,max_calls=a.max_calls,seconds=a.seconds,min_size=a.min_size,threshold=a.threshold,progress=progress)
    audit.update(image_manifest_sha256=sha256(a.image_manifest),instance_index_sha256=sha256(a.index),
        assembled_receipt_sha256=sha256(receipt_path),coverage_sha256=sha256(coverage_path),
        checkpoint_sha256=sha256(a.checkpoint))
    write_json(out/'edges.json',result)
    audit['edges_sha256']=sha256(out/'edges.json')
    write_json(out/'receipt.json',audit)


if __name__=='__main__':main()
