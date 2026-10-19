"""Launch isolated GPU workers for a volume or one global shard group.

With --workers-per-gpu N, every visible GPU runs N workers (one shard each) and the
launcher merges all shards of the volume; run several workers per GPU under CUDA MPS so
their small FFN batches overlap. Without it, four workers on four GPUs form one group.

Run inside a registered gpu allocation. Stitching and evaluation follow
as separate CPU stages; no ground truth is available to the inference workers.
With total_shards > 4, only group-complete.json is published; the complete
volume requires join_overlap_groups.py after all registered allocations finish
their workers. A separate campaign reservation must bound concurrent storage.
"""
import argparse
from execution_resources import valid_root
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import socket
import sys
import time
import numpy as np
from infer_overlap import merge
from stitch_ffn import windows
from common import device_guard,external,sha256,write_json
from execution_resources import device_guard
from fetch_axonem import reserve
from window_io import reserved_window_bytes
from window_assignment import assign,POLICIES


def run_workers(commands,masks,output,timeout):
    if not commands or len(commands)!=len(masks) or any(not m.strip() for m in masks):
        raise ValueError('One visible GPU device per worker required')
    def worker(rank):
        with (output/f'worker-{rank}.log').open('x') as log:
            with subprocess.Popen(commands[rank],env=dict(os.environ,CUDA_VISIBLE_DEVICES=masks[rank]),
                                  stdout=log,stderr=subprocess.STDOUT) as process:
                try:code=process.wait(timeout=timeout)
                except BaseException:
                    process.kill();process.wait();raise
                if code:raise subprocess.CalledProcessError(code,commands[rank])
                return dict(rank=rank,pid=process.pid,returncode=code,command=commands[rank])
    with ThreadPoolExecutor(max_workers=len(commands)) as pool:return list(pool.map(worker,range(len(commands))))


def run(a):
    from ffn_backend import receipt_backend_matches
    device_guard('cuda');root=external(a.root);output=external(a.output)
    if not valid_root(root) or root not in output.parents or output.exists():
        raise ValueError('New output inside Xtemp budget root required')
    masks=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    per_gpu=getattr(a,'workers_per_gpu',None)
    if per_gpu is None:
        if len(masks)!=4 or len(set(masks))!=4 or any(not m.strip() for m in masks):raise ValueError('Exactly four visible GPUs required')
        ranks=4
    else:
        if type(per_gpu) is not int or per_gpu<1 or not masks or len(set(masks))!=len(masks) or any(not m.strip() for m in masks):
            raise ValueError('Distinct visible GPUs and a positive worker count per GPU required')
        ranks=len(masks)*per_gpu
    from execution_resources import allocation
    if not np.isfinite(a.seconds) or a.seconds<=0 or a.batch<1 or a.max_calls<1:
        raise ValueError('Finite positive inference limits required')
    job_id=allocation(root,a.seconds+600,gpus=len(set(masks)))
    checkpoint=external(a.checkpoint);image=external(a.image_manifest)
    if root not in checkpoint.parents or root not in image.parents:raise ValueError('Inputs must be within budget root')
    metadata=json.loads(image.read_text());shape=np.asarray(metadata['shape'])
    boundary_shape=shape.tolist() if getattr(a,'clip_boundary_to_image',False) else None
    total=getattr(a,'total_shards',4);offset=getattr(a,'shard_offset',0);policy=getattr(a,'assignment_policy','modulo')
    if per_gpu is not None:
        total,offset=ranks,0
    elif type(total) is not int or total not in (4,8,12,16,20) or type(offset) is not int or offset%4 or not 0<=offset<=total-4:
        raise ValueError('One four-rank group within at most twenty global shards required')
    all_windows=windows(a.roi,a.core_size,a.margin,boundary_shape)
    owners=assign(all_windows,total,policy)
    if any(rank not in owners.values() for rank in range(total)):raise ValueError('Every global GPU rank needs at least one window')
    assigned=[w for w in all_windows if offset<=owners[w['index']]<offset+ranks]
    if any((np.asarray(w['stop'])>shape-16).any() for w in assigned):raise ValueError('Real network halo missing')
    compressed_limit=getattr(a,'compressed_window_bytes',0)
    compression_level=getattr(a,'compression_level',1)
    if type(compression_level) is not int or not 0<=compression_level<=9:raise ValueError('Invalid compression level')
    reserved=sum(reserved_window_bytes(np.asarray(w['stop'])-w['start'],compressed_limit) for w in assigned)+10_000_000
    reserve(root,reserved)
    code=Path(__file__).resolve().parent
    source={str(p):sha256(p) for p in sorted(code.glob('*.py'))}
    common=code.parent/'common.py';source[str(common)]=sha256(common)
    inputs={str(p):sha256(p) for p in (checkpoint,image)}
    output.mkdir();started=time.monotonic()
    write_json(output/'start.json',dict(job_id=job_id,source_sha256=source,input_sha256=inputs,
              reserved_output_bytes=reserved,windows=len(assigned),ground_truth_used=False,arguments=vars(a)))
    commands=[]
    seed_cache=getattr(a,'seed_cache',None)
    for rank in range(ranks):
        commands.append([sys.executable,str(code/'infer_overlap.py'),'worker','--root',str(root),'--output',str(output/f'shard-{offset+rank}'),
                         '--image-manifest',str(image),'--checkpoint',str(checkpoint),'--roi',*map(str,a.roi),
                         '--core-size',str(a.core_size),'--margin',str(a.margin),'--shards',str(total),'--shard-index',str(offset+rank),
                         '--assignment-policy',policy,
                         '--inference-backend',getattr(a,'inference_backend','norm32'),
                         '--compression-level',str(compression_level),
                         '--batch',str(a.batch),'--seconds',str(a.seconds),'--max-calls',str(a.max_calls),'--seed-order',a.seed_order])
        if boundary_shape is not None:commands[-1].append('--clip-boundary-to-image')
        if compressed_limit:commands[-1].extend(['--compressed-window-bytes',str(compressed_limit)])
        if seed_cache:commands[-1].extend(['--seed-cache',str(external(seed_cache))])
    rank_masks=[masks[rank%len(masks)] for rank in range(ranks)]
    workers=run_workers(commands,rank_masks,output,a.seconds+540)
    receipts=[output/f'shard-{offset+rank}'/'receipt.json' for rank in range(ranks)]
    rows=[json.loads(p.read_text()) for p in receipts]
    for rank,row in enumerate(rows):
        if (row.get('complete') is not True or row.get('ground_truth_used') is not False
            or row['shards']!=total or row['shard_index']!=offset+rank
            or row.get('assignment_policy','modulo')!=policy
            or sorted(w['index'] for w in row['windows'])!=sorted(w['index'] for w in assigned if owners[w['index']]==offset+rank)):
            raise ValueError('Complete exact global-rank window assignment required')
        if (row['source_sha256']!=source or row['checkpoint_sha256']!=inputs[str(checkpoint)]
            or not receipt_backend_matches(row,getattr(a,'inference_backend','norm32'))
            or row['image_manifest_sha256']!=inputs[str(image)]
            or row.get('boundary_image_shape')!=boundary_shape
            or row.get('compressed_window_bytes',0)!=compressed_limit
            or row.get('compression_level',1)!=compression_level
            or any(row[k]!=getattr(a,k) for k in ('roi','core_size','margin','seed_order'))):
            raise ValueError('Worker provenance differs from controller')
    manifest=merge(rows) if total==ranks else rows[0]
    if manifest.get('boundary_image_shape')!=boundary_shape or any(manifest[k]!=getattr(a,k) for k in ('roi','core_size','margin','seed_order')):
        raise ValueError('Worker geometry differs from requested inference')
    if manifest.get('compressed_window_bytes',0)!=compressed_limit:raise ValueError('Worker storage budget differs from request')
    for path,digest in {**source,**inputs}.items():
        if sha256(path)!=digest:raise ValueError('Source or inputs changed during inference')
    if manifest['source_sha256']!=source or manifest['checkpoint_sha256']!=inputs[str(checkpoint)] or manifest['image_manifest_sha256']!=inputs[str(image)]:
        raise ValueError('Worker provenance differs from controller')
    manifest['shard_receipts']={str(p):sha256(p) for p in receipts}
    if (not isinstance(workers,list) or len(workers)!=ranks or [r['rank'] for r in workers]!=list(range(ranks))
        or any(r['returncode']!=0 or r['command']!=commands[r['rank']] for r in workers)):
        raise ValueError('All worker processes must join successfully')
    write_json(output/'workers-exited.json',dict(schema='joined-ffn-workers/v1',job_id=job_id,
               hostname=socket.gethostname(),controller_pid=os.getpid(),workers=workers))
    common_receipt=dict(complete=True,job_id=job_id,seconds=time.monotonic()-started,
               start_sha256=sha256(output/'start.json'),workers_exited_sha256=sha256(output/'workers-exited.json'),ground_truth_used=False)
    if total==ranks:
        write_json(output/'manifest.json',manifest)
        write_json(output/'complete.json',dict(common_receipt,manifest_sha256=sha256(output/'manifest.json')))
    else:
        # Deliberately no full-volume manifest/complete marker until every group joins.
        write_json(output/'group-complete.json',dict(common_receipt,schema='overlap-worker-group/v1',
            total_shards=total,shard_offset=offset,assignment_policy=policy,
            shard_receipts={str(p):sha256(p) for p in receipts}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','output','image-manifest','checkpoint'):p.add_argument('--'+key,required=True)
    p.add_argument('--roi',type=int,nargs=6,required=True)
    p.add_argument('--core-size',type=int,default=436);p.add_argument('--margin',type=int,default=32)
    p.add_argument('--batch',type=int,default=4);p.add_argument('--seconds',type=float,required=True)
    p.add_argument('--inference-backend',choices=['eager','norm32','norm32-graph'],default='norm32')
    p.add_argument('--max-calls',type=int,default=1000000)
    p.add_argument('--seed-order',choices=['ascending','descending'],default='ascending')
    p.add_argument('--clip-boundary-to-image',action='store_true',help='Preserve full ROI at raw image edges with truncated overlap envelopes')
    p.add_argument('--compressed-window-bytes',type=int,default=0,help='Per-window lossless archive cap; zero keeps raw NPY')
    p.add_argument('--compression-level',type=int,choices=range(10),default=1)
    p.add_argument('--total-shards',type=int,default=4)
    p.add_argument('--shard-offset',type=int,default=0)
    p.add_argument('--assignment-policy',choices=POLICIES,default='modulo')
    p.add_argument('--workers-per-gpu',type=int,help='Workers on every visible GPU; all shards of the volume in one launch')
    p.add_argument('--seed-cache',help='Exact checkpoint-independent PolicyPeaks cache inside the root (seed_cache.py)')
    run(p.parse_args())
