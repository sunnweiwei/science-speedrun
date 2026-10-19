"""Four-GPU SENSE/SHAPE orchestration; reconciliation/evaluation run separately on CPU.

This runner never submits jobs or resets the budget. Invoke only inside an
allocation submitted through budget.py. --plan-only validates inputs on CPU.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,device_guard,write_json,sha256
from context_storage import segmentation_path


def plan(root,output,sense_seconds,shape_seconds,max_motifs,*,image=None,assembled=None,package=None):
    if min(sense_seconds,shape_seconds,max_motifs)<=0:raise ValueError('Positive limits required')
    if any(value is None for value in (image,assembled,package)):
        raise ValueError('Explicit image, assembled predictions and model package required')
    image=Path(image);assembled=Path(assembled)
    from model_package import verify
    package=external(package);package_check=verify(package,load_extension=True)
    code=package/'runtime/pathfinder';kernel_root=package
    checkpoints={stage:package/f'weights/{stage}.pt' for stage in ('sense','shape')}
    metadata=json.loads((assembled/'metadata.json').read_text())
    if metadata['axes']!='zyx' or metadata['voxel_size_nm']!=[16]*3:raise ValueError('Current trained recipe requires16nm ZYX')
    paths=[image,assembled/'receipt.json',assembled/'instance-index.json',assembled/'coverage.json',
           segmentation_path(assembled),*checkpoints.values()]
    for path in paths:
        if not path.is_file():raise FileNotFoundError(path)
    sense=[];shape=[]
    for rank in range(4):
        sense.append([sys.executable,str(code/'infer_continuous_sense.py'),
            '--image-manifest',str(image),'--assembled',str(assembled),'--index',str(assembled/'instance-index.json'),
            '--checkpoint',str(checkpoints['sense']),'--output',str(output/f'sense/shard-{rank}'),
            '--shards','4','--shard-index',str(rank),'--batch','8','--seconds',str(sense_seconds),'--max-calls','1000000'])
        shape.append([sys.executable,str(code/'infer_shape.py'),'--root',str(kernel_root),
            '--checkpoint',str(checkpoints['shape']),'--segments',str(segmentation_path(assembled)),
            '--edges',str(output/'sense/edges.json'),'--output',str(output/f'shape/shard-{rank}.json'),
            '--instance-index',str(assembled/'instance-index.json'),'--coverage',str(assembled/'coverage.json'),
            '--metadata',str(assembled/'metadata.json'),'--voxel-size-nm',*map(str,metadata['voxel_size_nm']),
            '--origin-nm',*map(str,metadata['origin_nm']),'--shards','4','--shard-index',str(rank),
            '--seconds',str(shape_seconds),'--max-motifs',str(max_motifs),'--max-nodes','4'])
    merger=[sys.executable,str(code/'merge_inference_shards.py')]
    return dict(sense=sense,shape=shape,
        merge_sense=[*merger,'--stage','sense','--shards',*[str(output/f'sense/shard-{r}') for r in range(4)],'--output',str(output/'sense/edges.json')],
        merge_shape=[*merger,'--stage','shape','--edges',str(output/'sense/edges.json'),
            '--shards',*[str(output/f'shape/shard-{r}.json') for r in range(4)],'--output',str(output/'shape/merged.json')],
        ground_truth_used=False,paper_comparable=False,package_check=package_check,
        input_paths=dict(image_manifest=str(image),assembled=str(assembled)),
        model_seconds_per_shard=dict(sense=sense_seconds,shape=shape_seconds),
        time_limit_scope='per-stage loops; hashing/loading and in-flight batches require allocation overhead',
        assembly_receipt_sha256=sha256(assembled/'receipt.json'))


def parallel_stage(commands,masks,output,stage,*,container_image=None,readonly=None):
    def one(rank):
        begun=time.monotonic()
        command=commands[rank];env=dict(os.environ,CUDA_VISIBLE_DEVICES=masks[rank])
        if container_image:
            from isolate_inference import invocation
            command,env=invocation(command,container_image,readonly,output,gpu=masks[rank])
        with (output/f'{stage}-{rank}.log').open('w') as log:
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        return dict(rank=rank,exit_code=result.returncode,seconds=time.monotonic()-begun)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,range(4)))
    write_json(output/f'{stage}-execution.json',dict(shards=rows,slurm_job_id=os.environ['SLURM_JOB_ID']))
    if any(r['exit_code'] for r in rows):raise RuntimeError(f'{stage} shard failed; inspect logs before any retry')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sense-seconds',type=int,default=1800);p.add_argument('--shape-seconds',type=int,default=3600)
    p.add_argument('--max-motifs',type=int,default=65536);p.add_argument('--plan-only',action='store_true')
    p.add_argument('--image-manifest',type=Path,required=True);p.add_argument('--assembled',type=Path,required=True)
    p.add_argument('--model-package',type=Path,required=True)
    p.add_argument('--container-image',type=Path,help='Optional isolated GPU stages; requires explicit model package')
    a=p.parse_args();root=external(a.root);out=external(a.output)
    if root not in out.parents:raise ValueError('Output must remain under original budget root')
    steps=plan(root,out,a.sense_seconds,a.shape_seconds,a.max_motifs,
        image=a.image_manifest,assembled=a.assembled,package=a.model_package)
    readonly=None
    if a.container_image:
        if not a.model_package or not a.container_image.is_file():raise ValueError('Explicit package and existing container required')
        source=Path(steps['input_paths']['image_manifest']);assembled=Path(steps['input_paths']['assembled'])
        readonly=[a.model_package.resolve(),source.resolve(),source.parent/'raw.npy',source.parent/'image_valid_z.npy',
            segmentation_path(assembled),*[assembled/name for name in ('metadata.json','receipt.json','instance-index.json','coverage.json')]]
        steps['isolation']=dict(image=str(a.container_image.resolve()),image_sha256=sha256(a.container_image),
            readonly=[str(p) for p in readonly],writable=str(out),network='none',host_home=False,
            scope='GPU model stages only; trusted host orchestration and shard merging remain outside container')
    if a.plan_only:print(json.dumps(steps,indent=2));return
    device_guard('cuda');masks=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(masks)!=4 or not all(masks) or len(set(masks))!=4:raise ValueError('Exactly four distinct allocated GPUs required')
    ledger=json.loads((root/'budget.json').read_text());jid=os.environ['SLURM_JOB_ID']
    reserved=[j for j in ledger['jobs'] if j['job_id']==jid]
    if len(reserved)!=1 or reserved[0]['gpus']!=4 or 'actual_seconds' in reserved[0]:raise ValueError('Active four-GPU ledger reservation required')
    if a.sense_seconds+a.shape_seconds+900>reserved[0]['reserved_seconds']:
        raise ValueError('Reserve at least15min allocation overhead beyond stage limits')
    out.mkdir(parents=True,exist_ok=False);(out/'sense').mkdir();(out/'shape').mkdir()
    write_json(out/'plan.json',steps)
    for stage in ('sense','shape'):
        parallel_stage(steps[stage],masks,out,stage,container_image=a.container_image,readonly=readonly)
        subprocess.run(steps['merge_'+stage],check=True)
    write_json(out/'receipt.json',dict(slurm_job_id=jid,plan_sha256=sha256(out/'plan.json'),
        scores_sha256=sha256(out/'shape/merged.json'),ground_truth_used=False,paper_comparable=False,
        status='inference_complete; CPU reconciliation and independent evaluation still required'))


if __name__=='__main__':main()
