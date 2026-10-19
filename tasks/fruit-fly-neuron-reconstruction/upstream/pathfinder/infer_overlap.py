"""Run unchanged FFN growth over overlapping windows; merge complete shards.

Window growth uses batched_ffn.schedule directly, with the same model and seed
semantics. Output assembly is a separate prediction-only stitch_ffn stage.
"""
import argparse
import json
import os
from execution_resources import valid_root
from pathlib import Path
import numpy as np
from batched_ffn import schedule,window_finished
from common import external,device_guard,sha256,write_json
from execution_resources import device_guard
from evaluate_physical import grid
from fetch_axonem import reserve
from stitch_ffn import windows,validate
from window_io import save_compressed,reserved_window_bytes
from inference_progress import Progress
from window_assignment import assign,POLICIES


def select_windows(rows,indices):
    if not isinstance(indices,list) or not indices or any(type(i) is not int for i in indices):
        raise ValueError('Explicit nonempty integer window indices required')
    if len(set(indices))!=len(indices) or not set(indices)<=set(w['index'] for w in rows):
        raise ValueError('Duplicate or unknown recovery window index')
    return [w for w in rows if w['index'] in set(indices)]


def merge(shards):
    if not shards:raise ValueError('No overlap shards')
    if any(s.get('window_selection') is not None for s in shards):
        raise ValueError('Recovery selections require an audited full-coverage recovery join')
    first=shards[0];count=first['shards']
    if type(count) is not int or count<1 or len(shards)!=count or sorted(s['shard_index'] for s in shards)!=list(range(count)):
        raise ValueError('All unique overlap shards required')
    keys=('roi','core_size','margin','raw_sha256','checkpoint_sha256','seed_order','metadata','image_manifest','image_manifest_sha256','source_sha256')
    boundary_shape=first.get('boundary_image_shape')
    expected=windows(first['roi'],first['core_size'],first['margin'],boundary_shape)
    policy=first.get('assignment_policy','modulo')
    backend=first.get('inference_backend','eager')
    if backend not in ('eager','norm32','norm32-graph'):raise ValueError('Unknown inference backend')
    owners=assign(expected,count,policy)
    records=[]
    for shard in shards:
        if shard.get('complete') is not True or shard.get('ground_truth_used') is not False or shard['shards']!=count:
            raise ValueError('Complete prediction-only shards required')
        if any(shard[k]!=first[k] for k in keys):raise ValueError('Inconsistent overlap shard provenance')
        if shard.get('boundary_image_shape')!=boundary_shape:raise ValueError('Inconsistent boundary clipping policy')
        if shard.get('compressed_window_bytes',0)!=first.get('compressed_window_bytes',0):raise ValueError('Inconsistent window storage budget')
        if shard.get('compression_level',1)!=first.get('compression_level',1):raise ValueError('Inconsistent compression level')
        if shard.get('assignment_policy','modulo')!=policy:raise ValueError('Inconsistent window assignment policy')
        if shard.get('inference_backend','eager')!=backend:raise ValueError('Inconsistent inference backend')
        if sorted(r['index'] for r in shard['windows'])!=sorted(w['index'] for w in expected if owners[w['index']]==shard['shard_index']):
            raise ValueError('Shard window assignment mismatch')
        records.extend(shard['windows'])
    result=dict(schema='overlapping-ffn-windows/v1',ground_truth_used=False,windows=records,**{k:first[k] for k in keys})
    result['assignment_policy']=policy
    result['inference_backend']=backend
    result['compression_level']=first.get('compression_level',1)
    if boundary_shape is not None:result['boundary_image_shape']=boundary_shape
    if first.get('compressed_window_bytes',0):result['compressed_window_bytes']=first['compressed_window_bytes']
    validate(result);return result


def worker(a):
    device_guard('cuda');root=external(a.root);out=external(a.output)
    if root not in out.parents or out.exists() or not valid_root(root):
        raise ValueError('New output inside Xtemp root required')
    if a.shards<1 or not 0<=a.shard_index<a.shards or a.batch<1 or not np.isfinite(a.seconds) or a.seconds<=0 or a.max_calls<1:
        raise ValueError('Invalid finite worker budget')
    from execution_resources import allocation
    job_id=allocation(root,a.seconds)
    image_path=Path(a.image_manifest).resolve();checkpoint_path=Path(a.checkpoint).resolve()
    if root not in image_path.parents or root not in checkpoint_path.parents:raise ValueError('Inputs must remain in budget root')
    image=json.loads(image_path.read_text());origin,spacing,shape,_=grid(image)
    if image['axes']!='zyx':raise ValueError('ZYX image required')
    raw_path=image_path.parent/'raw.npy';raw_hash=sha256(raw_path)
    if raw_hash!=image['files']['raw.npy']:raise ValueError('Raw data hash mismatch')
    raw=np.load(raw_path,mmap_mode='r',allow_pickle=False)
    if raw.dtype!=np.uint8 or list(raw.shape)!=image['shape']:raise ValueError('Raw array contract mismatch')
    boundary_shape=shape.tolist() if getattr(a,'clip_boundary_to_image',False) else None
    all_windows=windows(a.roi,a.core_size,a.margin,boundary_shape)
    selection_path=getattr(a,'window_indices',None)
    selection=None
    if selection_path is not None:
        selection_path=external(selection_path)
        if root not in selection_path.parents:raise ValueError('Recovery selection must remain inside budget root')
        indices=json.loads(selection_path.read_text())
        selection=dict(path=str(selection_path),sha256=sha256(selection_path),indices=indices,total_windows=len(all_windows))
        all_windows=select_windows(all_windows,indices)
    if any((np.asarray(w['stop'])>shape-16).any() for w in all_windows):raise ValueError('Missing real halo')
    policy=getattr(a,'assignment_policy','modulo');owners=assign(all_windows,a.shards,policy)
    assigned=[w for w in all_windows if owners[w['index']]==a.shard_index]
    # Controller must jointly reserve every worker; this worker also checks its own output.
    compressed_limit=getattr(a,'compressed_window_bytes',0)
    compression_level=getattr(a,'compression_level',1)
    if type(compression_level) is not int or not 0<=compression_level<=9:raise ValueError('Invalid compression level')
    reserve(root,sum(reserved_window_bytes(np.asarray(w['stop'])-w['start'],compressed_limit) for w in assigned)+1_000_000)
    import torch
    from ffn_model import FFN15
    checkpoint_hash=sha256(checkpoint_path);checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    architecture=checkpoint['architecture']
    if architecture['stage']!='ffn' or not np.array_equal(spacing,architecture.get('voxel_size_nm',[16]*3)):
        raise ValueError('Checkpoint voxel grid differs')
    model=FFN15(activation_checkpointing=False).to('cuda');model.load_state_dict(checkpoint['model'],strict=True);model.eval();torch.set_num_threads(2)
    bf16=torch.cuda.is_bf16_supported()
    source_files={str(p):sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    common_path=Path(__file__).resolve().parents[1]/'common.py';source_files[str(common_path)]=sha256(common_path)
    metadata=dict(axes='zyx',coordinate_convention='voxel_outer_corner',raw_origin_nm=origin.tolist(),shape=(np.asarray(a.roi[1::2])-a.roi[::2]).tolist(),origin_nm=(origin+np.asarray(a.roi[::2])*spacing).tolist(),voxel_size_nm=spacing.tolist())
    report=dict(schema='overlapping-ffn-shard/v1',roi=a.roi,core_size=a.core_size,margin=a.margin,shards=a.shards,shard_index=a.shard_index,seed_order=a.seed_order,raw_sha256=raw_hash,checkpoint_sha256=checkpoint_hash,metadata=metadata,image_manifest=str(image_path),image_manifest_sha256=sha256(image_path),source_sha256=source_files,ground_truth_used=False,windows=[],bf16=bf16)
    report['assignment_policy']=policy
    if selection is not None:report['window_selection']=selection
    report['compression_level']=compression_level
    if boundary_shape is not None:report['boundary_image_shape']=boundary_shape
    if compressed_limit:report['compressed_window_bytes']=compressed_limit
    out.mkdir()
    heartbeat=Progress(out/'heartbeat.jsonl','overlap-ffn',a.shard_index)
    heartbeat.emit(force=True,status='starting',total_windows=len(assigned),completed_windows=0,model_batches=0,model_windows=0)
    def predict(image_batch,canvas_batch):
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
            return model(torch.from_numpy(image_batch[:,None]).to('cuda'),torch.from_numpy(canvas_batch[:,None]).to('cuda'))[:,0].float().cpu().numpy()
    from ffn_backend import select_backend
    requested=getattr(a,'inference_backend','norm32')
    backend=select_backend(model,requested)
    report['requested_inference_backend']=requested
    if backend=='norm32-graph':
        from ffn_inference import CudaGraphFFN
        predict=CudaGraphFFN(model,max_batch=a.batch,bf16=bf16)
    elif backend not in ('eager','norm32'):raise ValueError('Unknown inference backend')
    report['inference_backend']=backend
    def save(w,seg,stats):
        if compressed_limit:
            path=out/f"window-{w['index']:07d}.npz"
            storage=save_compressed(seg,path,reserved_window_bytes(seg.shape,compressed_limit),compression_level=compression_level)
        else:
            path=out/f"window-{w['index']:07d}.npy";np.save(path,seg,allow_pickle=False)
            storage=dict(path=str(path),sha256=sha256(path))
        record=dict(w,**stats,**storage,raw_sha256=raw_hash,checkpoint_sha256=checkpoint_hash,seed_order=a.seed_order)
        report['windows'].append(record)
        with (out/'progress.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
    seed_provider=None
    cache_path=getattr(a,'seed_cache',None)
    if cache_path is not None:
        from seed_cache import get_seeds
        report['seed_cache']=dict(path=str(cache_path),entries=[])
        def seed_provider(context,start,stop):
            seeds,evidence=get_seeds(context,root,cache_path,dict(raw_sha256=raw_hash,start=start,stop=stop))
            report['seed_cache']['entries'].append(dict(start=start,stop=stop,**evidence))
            return seeds
    execution=schedule(raw,assigned,predict,a.batch,a.seconds,a.max_calls,save,seed_order=a.seed_order,seed_provider=seed_provider,
        on_progress=lambda values:heartbeat.emit(status='running',total_windows=len(assigned),**values))
    complete=len(report['windows'])==len(assigned) and all(window_finished(w) for w in report['windows'])
    for path,digest in source_files.items():
        if sha256(path)!=digest:raise ValueError('Scientific source changed')
    if sha256(image_path)!=report['image_manifest_sha256'] or sha256(checkpoint_path)!=checkpoint_hash:raise ValueError('Inputs changed')
    if selection is not None and sha256(selection_path)!=selection['sha256']:raise ValueError('Recovery selection changed')
    report.update(complete=complete,execution=execution,job_id=job_id)
    write_json(out/'receipt.json',report);raw._mmap.close()
    heartbeat.emit(force=True,status='complete' if complete else 'incomplete',total_windows=len(assigned),
        completed_windows=len(report['windows']),model_batches=execution['model_batches'],model_windows=execution['model_windows'])
    if not complete:raise ValueError('Incomplete overlap inference; receipt is diagnostic only')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    w=sub.add_parser('worker')
    for key in ('root','output','image-manifest','checkpoint'):w.add_argument('--'+key,required=True,type=Path)
    w.add_argument('--roi',required=True,type=int,nargs=6);w.add_argument('--core-size',type=int,default=436);w.add_argument('--margin',type=int,default=32)
    w.add_argument('--shards',type=int,default=4);w.add_argument('--shard-index',type=int,required=True)
    w.add_argument('--assignment-policy',choices=POLICIES,default='modulo')
    w.add_argument('--window-indices',type=Path,help='Recovery-only explicit indices; never accepted by ordinary full-volume merge')
    w.add_argument('--seed-cache',type=Path,help='Optional exact PolicyPeaks cache inside the experiment root')
    w.add_argument('--batch',type=int,default=4);w.add_argument('--seconds',type=float,required=True);w.add_argument('--max-calls',type=int,default=1000000)
    w.add_argument('--inference-backend',choices=['eager','norm32','norm32-graph'],default='norm32',help='Default norm32 on all CUDA GPUs; eager and graph are explicit overrides')
    w.add_argument('--seed-order',choices=['ascending','descending'],default='ascending')
    w.add_argument('--clip-boundary-to-image',action='store_true',help='Clip only outer overlap envelopes; preserve full ROI and real network halo')
    w.add_argument('--compressed-window-bytes',type=int,default=0,help='Positive per-window lossless archive cap; zero keeps raw NPY')
    w.add_argument('--compression-level',type=int,choices=range(10),default=1)
    m=sub.add_parser('merge');m.add_argument('--receipts',type=Path,nargs='+',required=True);m.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='worker':worker(a)
    else:
        if a.output.exists():raise ValueError('Output exists')
        result=merge([json.loads(path.read_text()) for path in a.receipts]);result['shard_receipts']={str(path):sha256(path) for path in a.receipts}
        write_json(external(a.output),result)
