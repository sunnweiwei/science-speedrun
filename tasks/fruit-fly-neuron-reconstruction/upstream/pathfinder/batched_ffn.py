"""Batch independent seed-and-grow tile requests without changing FFN semantics.

Same per-tile seed order,33-cube field,move/assignment thresholds as infer_ffn.
Tiles do not communicate: cross-tile fragmentation is explicit, not stitched.
No ground truth is read. Partial work and untouched tiles remain distinguishable.
"""
import argparse,json,time,sys
from ffn_movement import MOVE_THRESHOLD,FaceMaxMovement,disconnected_bias
from pathlib import Path
import numpy as np
from ffn_seeds import policy_peaks,seed_is_clear
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,device_guard,sha256,write_json,source_provenance



MIN_INSTANCE_VOXELS=10


def tile_id_stride(tile_size):
    if type(tile_size)!=int or tile_size<1:raise ValueError('Positive integer tile size required')
    # Instances are disjoint and each owns at least MIN_INSTANCE_VOXELS voxels.
    return tile_size**3//MIN_INSTANCE_VOXELS+1


def global_tile_labels(segmentation,tile_index,total_tiles,tile_size):
    stride=tile_id_stride(tile_size)
    if (segmentation.ndim!=3 or segmentation.dtype.kind!='u' or
        any(n>tile_size for n in segmentation.shape) or not 0<=tile_index<total_tiles or
        int(segmentation.max(initial=0))>=stride):
        raise ValueError('Tile labels violate the disjoint instance-ID bound')
    dtype=np.uint32 if total_tiles*stride<2**32 else np.uint64
    converted=segmentation.astype(dtype);mask=converted>0
    converted[mask]+=tile_index*stride
    return converted

def grow_tile(image,start,stop,max_calls=100000,seed_order='ascending',seed_provider=None):
    """Yield (raw,canvas) windows; send logits or None to stop and retain partial work."""
    start=np.asarray(start);stop=np.asarray(stop);shape=stop-start
    if max_calls<1 or (shape<=0).any() or (start<16).any() or (stop>np.array(image.shape)-16).any():
        raise ValueError('Positive bounded tile and16-voxel real context required')
    context=np.asarray(image[tuple(slice(int(a)-16,int(b)+16) for a,b in zip(start,stop))])
    seeds=(policy_peaks(context) if seed_provider is None else seed_provider(context,start.tolist(),stop.tolist()))-16
    if seed_order not in ('ascending','descending'):raise ValueError('Explicit ascending/descending seed order required')
    if seed_order=='descending':seeds=seeds[::-1]
    segmentation=np.zeros(tuple(shape),np.uint32);calls=objects=0;exhausted=call_limit=False;moves=dict(seeds_lost=0,stale_moves_skipped=0)
    canvas=None
    for seed in seeds:
        if not seed_is_clear(segmentation,seed):continue
        if calls>=max_calls:exhausted=call_limit=True;break
        if canvas is None:
            canvas=np.full(tuple(shape+32),-2.944439,np.float32)
            initialized=np.zeros(canvas.shape,bool)
        center=seed+16;canvas[tuple(center)]=2.944439;initialized[tuple(center)]=True
        touched_lo=center.copy();touched_hi=center+1
        movement=FaceMaxMovement(center,canvas.shape)
        while movement.queue:
            c=movement.next(canvas,moves,blocked=lambda point: segmentation[tuple(x-16 for x in point)]>0)
            if c is None:break
            if calls>=max_calls:exhausted=call_limit=True;break
            local=tuple(slice(int(x-16),int(x+17)) for x in c)
            global_c=start+c-16;global_sl=tuple(slice(int(x-16),int(x+17)) for x in global_c)
            predicted=yield (np.array(image[global_sl],np.float32)/255,canvas[local].copy())
            if predicted is None:exhausted=True;break
            predicted=np.asarray(predicted)
            if predicted.shape!=(33,33,33) or not np.isfinite(predicted).all():raise ValueError('Finite33-cube predictions required')
            predicted=disconnected_bias(canvas[local],predicted,initialized[local],audit=moves)
            canvas[local]=predicted;initialized[local]=True;calls+=1
            touched_lo=np.minimum(touched_lo,np.asarray(c)-16);touched_hi=np.maximum(touched_hi,np.asarray(c)+17)
            movement.update(predicted,c)
        # Outside the modified box logits remain at the background prior. Restrict
        # assignment and reset to this box, preserving the full movement canvas.
        lo=np.maximum(touched_lo,16);hi=np.minimum(touched_hi,shape+16)
        interior=canvas[tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))]
        target=segmentation[tuple(slice(int(a)-16,int(b)-16) for a,b in zip(lo,hi))]
        selected=(interior>=2.944439)&(target==0)&(canvas[tuple(center)]>=MOVE_THRESHOLD)
        if selected.sum()>=MIN_INSTANCE_VOXELS:objects+=1;target[selected]=objects
        if exhausted:break
        touched=tuple(slice(int(a),int(b)) for a,b in zip(touched_lo,touched_hi))
        canvas[touched]=-2.944439;initialized[touched]=False
    return segmentation,dict(calls=calls,objects=objects,seed_count=len(seeds),budget_exhausted=exhausted,call_limit_reached=call_limit,
        omitted_voxel_fraction=float((segmentation==0).mean()),**moves)


def window_finished(record):
    """A window is finished when growth ran out of seeds or reached the per-window call cap.

    The call cap is part of the inference protocol: segments grown before it are kept and the
    remaining seeds are left unlabelled (scored as omissions). Time-limit stops are unfinished.
    """
    return record.get('budget_exhausted') is False or record.get('call_limit_reached') is True


def schedule(image,tiles,predict,batch=8,seconds=1800,max_calls=100000,on_result=None,clock=time.monotonic,seed_order='ascending',on_progress=None,seed_provider=None):
    """Bounded round-robin scheduler. Every active tile contributes at most one window."""
    if batch<1 or seconds<=0:raise ValueError('Positive batch and time limit required')
    iterator=iter(tiles);active=[];completed=[];start=clock();pending_done=False;batches=windows=0;time_exhausted=False
    def finish(tile,result):
        if on_result:on_result(tile,*result)
        completed.append(dict(tile_index=tile['index'],**result[1]))
    while True:
        while len(active)<batch and not pending_done and clock()-start<seconds:
            try:tile=next(iterator)
            except StopIteration:pending_done=True;break
            gen=grow_tile(image,tile['start'],tile['stop'],max_calls,seed_order,seed_provider=seed_provider)
            try:req=next(gen);active.append((tile,gen,req))
            except StopIteration as done:finish(tile,done.value)
        if not active:break
        if clock()-start>=seconds:
            time_exhausted=True
            for tile,gen,_ in active:
                try:gen.send(None)
                except StopIteration as done:finish(tile,done.value)
            active=[];break
        raw=np.stack([r[0] for _,_,r in active]);canvas=np.stack([r[1] for _,_,r in active])
        predictions=np.asarray(predict(raw,canvas))
        if predictions.shape!=raw.shape:raise ValueError('Predictor must return B,33,33,33')
        batches+=1;windows+=len(active);next_active=[]
        for (tile,gen,_),prediction in zip(active,predictions):
            try:req=gen.send(prediction);next_active.append((tile,gen,req))
            except StopIteration as done:finish(tile,done.value)
        active=next_active
        if on_progress:on_progress(dict(model_batches=batches,model_windows=windows,
            completed_windows=len(completed),active_window_ids=[t['index'] for t,_,_ in active]))
    return dict(tiles=completed,model_batches=batches,model_windows=windows,seconds=clock()-start,
        time_exhausted=time_exhausted or (not pending_done and clock()-start>=seconds))


def grid_tiles(start,stop,tile_size):
    if tile_size<1:raise ValueError('Positive tile size required')
    index=0
    for z in range(start[0],stop[0],tile_size):
        for y in range(start[1],stop[1],tile_size):
            for x in range(start[2],stop[2],tile_size):
                lo=np.array([z,y,x]);hi=np.minimum(lo+tile_size,stop)
                yield dict(index=index,start=lo.tolist(),stop=hi.tolist());index+=1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['raw','checkpoint','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--roi',type=int,nargs=6);p.add_argument('--tiles',type=Path,help='JSON list of [z0,z1,y0,y1,x0,x1] tile bounds')
    p.add_argument('--tile-size',type=int,default=64);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--max-tiles',type=int);p.add_argument('--seconds',type=float,default=1800)
    p.add_argument('--max-calls-per-tile',type=int,default=100000)
    p.add_argument('--seed-order',choices=['ascending','descending'],default='ascending')
    p.add_argument('--resume-from',type=Path,help='Finished shard receipt; only verified complete tiles are reused')
    p.add_argument('--voxel-size-nm',type=float,nargs=3,default=[16]*3)
    p.add_argument('--origin-nm',type=float,nargs=3,default=[0]*3)
    p.add_argument('--shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--assemble',action='store_true');p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    a=p.parse_args();device_guard(a.device)
    if a.batch<1 or a.seconds<=0 or a.shards<1 or not 0<=a.shard_index<a.shards or (a.max_tiles is not None and a.max_tiles<1):
        raise ValueError('Invalid resource limits or shard')
    import torch
    from ffn_model import FFN15
    spacing=np.asarray(a.voxel_size_nm);origin=np.asarray(a.origin_nm)
    if not np.isfinite(spacing).all() or (spacing<=0).any() or not np.isfinite(origin).all():raise ValueError('Invalid physical metadata')
    image=np.load(a.raw,mmap_mode='r',allow_pickle=False)
    if image.ndim!=3 or image.dtype!=np.uint8:raise ValueError('Require uint8 ZYX raw EM')
    roi=a.roi or [16,image.shape[0]-16,16,image.shape[1]-16,16,image.shape[2]-16]
    start=np.asarray(roi[::2]);stop=np.asarray(roi[1::2])
    if (start<16).any() or (stop>np.array(image.shape)-16).any() or (stop<=start).any():raise ValueError('ROI requires16voxel real EM halo')
    if a.tiles:
        requested=json.loads(a.tiles.read_text());all_tiles=[]
        for i,bounds in enumerate(requested):
            if len(bounds)!=6:raise ValueError('Six tile bounds required')
            lo=np.asarray(bounds[::2]);hi=np.asarray(bounds[1::2])
            if (lo<start).any() or (hi>stop).any() or (hi<=lo).any() or (hi-lo>a.tile_size).any():raise ValueError('Invalid explicit tile')
            all_tiles.append(dict(index=i,start=lo.tolist(),stop=hi.tolist()))
        for i,t in enumerate(all_tiles):
            for other in all_tiles[:i]:
                if np.all(np.minimum(t['stop'],other['stop'])>np.maximum(t['start'],other['start'])):raise ValueError('Explicit tiles must not overlap')
    else:all_tiles=list(grid_tiles(start,stop,a.tile_size))
    shard_tiles=[t for t in all_tiles if t['index']%a.shards==a.shard_index]
    planned=shard_tiles[:a.max_tiles] if a.max_tiles else shard_tiles
    if a.resume_from and (a.tiles or a.max_tiles):raise ValueError('Resume requires the full regular grid')
    raw_hash=sha256(a.raw);checkpoint_hash=sha256(a.checkpoint)
    metadata=dict(axes='zyx',voxel_size_nm=spacing.tolist(),raw_origin_nm=origin.tolist(),
        origin_nm=(origin+start*spacing).tolist(),shape=(stop-start).tolist(),coordinate_convention='voxel_outer_corner')
    receipts=[];resume=None
    if a.resume_from:
        from resume_ffn import completed_tiles
        receipts,resume=completed_tiles(a.resume_from,dict(roi=roi,tile_size=a.tile_size,seed_order=a.seed_order,
            id_stride=tile_id_stride(a.tile_size),shard_index=a.shard_index,shards=a.shards,
            checkpoint_sha256=checkpoint_hash,raw_sha256=raw_hash,metadata=metadata),shard_tiles)
        reused={t['index'] for t in receipts};planned=[t for t in planned if t['index'] not in reused]
    out=external(a.output);out.mkdir(parents=True,exist_ok=False);(out/'tiles').mkdir()
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True);architecture=checkpoint['architecture']
    if architecture['stage']!='ffn' or not np.array_equal(spacing,architecture.get('voxel_size_nm',[16]*3)):
        raise ValueError('FFN checkpoint on matching voxel grid required')
    model=FFN15(activation_checkpointing=False).to(a.device);model.load_state_dict(checkpoint['model'],strict=True);model.eval();torch.set_num_threads(2)
    bf16=a.device=='cuda' and torch.cuda.is_bf16_supported()
    def predict(raw,canvas):
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
            return model(torch.from_numpy(raw[:,None]).to(a.device),torch.from_numpy(canvas[:,None]).to(a.device))[:,0].float().cpu().numpy()
    stride=tile_id_stride(a.tile_size);dtype=np.uint32 if len(all_tiles)*stride<2**32 else np.uint64
    assembled=np.lib.format.open_memmap(out/'segmentation.npy',mode='w+',dtype=dtype,shape=tuple(stop-start)) if a.assemble else None
    if assembled is not None:assembled[:]=0
    for tile in receipts:
        if assembled is not None:
            lo=np.array(tile['start'])-start;hi=np.array(tile['stop'])-start
            assembled[tuple(slice(int(x),int(y)) for x,y in zip(lo,hi))]=np.load(tile['path'],mmap_mode='r')
        with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(tile)+'\n')
    def save(tile,seg,stats):
        converted=global_tile_labels(seg,tile['index'],len(all_tiles),a.tile_size)
        path=out/'tiles'/f"{tile['index']:07d}.npy";np.save(path,converted,allow_pickle=False)
        if assembled is not None:
            lo=np.array(tile['start'])-start;hi=np.array(tile['stop'])-start
            assembled[tuple(slice(int(x),int(y)) for x,y in zip(lo,hi))]=converted;assembled.flush()
        record=dict(tile,path=str(path),sha256=sha256(path),**stats)
        receipts.append(record)
        with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    run=schedule(image,planned,predict,a.batch,a.seconds,a.max_calls_per_tile,save,seed_order=a.seed_order)
    completed_voxels=sum(int(np.prod(np.array(t['stop'])-t['start'])) for t in receipts if not t['budget_exhausted'])
    write_json(out/'receipt.json',dict(**run,tile_results=receipts,total_tiles=len(all_tiles),shard_tiles=len(shard_tiles),
        planned_tiles=len(planned),unstarted_shard_tiles=len(shard_tiles)-len(receipts),batch_size=a.batch,
        completed_voxels=completed_voxels,roi_voxels=int(np.prod(stop-start)),all_shard_tiles_complete=len(receipts)==len(shard_tiles) and not any(t['budget_exhausted'] for t in receipts),
        checkpoint_sha256=checkpoint_hash,raw_sha256=raw_hash,source=source_provenance(),resume=resume,
        roi=roi,tile_size=a.tile_size,seed_order=a.seed_order,shard_index=a.shard_index,shards=a.shards,id_stride=stride,bf16=bf16,
        ground_truth_used=False,cross_tile_fragmentation=True,cross_tile_stitching=False,consensus=False,paper_comparable=False,
        metadata=metadata))

if __name__=='__main__':main()
