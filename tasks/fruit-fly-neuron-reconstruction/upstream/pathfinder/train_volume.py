"""Real-volume FFN/SENSE stages with DDP, timed stop, validation and portable weights.

FFN uses bounded detached recurrent seed updates and predicted-score moves;
this is an independent bounded training schedule, not Google's full recipe.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from ffn_model import FFN15
from sense import Sense
from volume_data import Crops
from dataset_contract import validate_manifest
from ffn_movement import FaceMaxMovement, disconnected_bias
from ffn_growth_validation import growth_validation, admit
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import device_guard,external,write_json,sha256
from execution_resources import device_guard


def loss_sum(logits,target,mask,positive_weight=1.):
    weight=logits.new_tensor(positive_weight,dtype=torch.float32)
    return (F.binary_cross_entropy_with_logits(logits.float(),target,reduction='none',pos_weight=weight)*mask).sum()


def production_windows(model,raw,target,mask,steps=8,stop_when_exhausted=False):
    """Detached production movement and freeze rules, aligned DDP backward count.

    Exhausted examples have zero loss mask; dummy forwards keep ranks aligned.
    Training may skip the remaining steps once every rank is exhausted. Leave
    this disabled for validation, whose ranks can have unequal crop counts.
    Ground truth only supplies supervision, never movement or canvas updates.
    """
    size=raw.shape[-1];batch=len(raw);seed=(size//2,)*3
    canvases=[np.full((size,)*3,-2.944439,np.float32) for _ in range(batch)]
    initialized=[np.zeros((size,)*3,bool) for _ in range(batch)]
    policies=[FaceMaxMovement(seed,(size,)*3) for _ in range(batch)]
    live=[True]*batch
    for canvas,known in zip(canvases,initialized):canvas[seed]=2.944439;known[seed]=True
    for _ in range(steps):
        centers=[]
        for i,policy in enumerate(policies):
            center=policy.next(canvases[i]) if live[i] else None
            live[i]=center is not None;centers.append(center or seed)
        if stop_when_exhausted:
            active_any=torch.tensor(int(any(live)),device=raw.device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(active_any,op=dist.ReduceOp.MAX)
            if not active_any.item():break
        slices=[tuple(slice(c-16,c+17) for c in center) for center in centers]
        def cut(t):return torch.cat([t[(slice(i,i+1),slice(None),*sl)] for i,sl in enumerate(slices)],dim=0)
        canvas=torch.from_numpy(np.stack([v[sl] for v,sl in zip(canvases,slices)])[:,None]).to(raw.device)
        logits=model(cut(raw),canvas)
        active=torch.tensor(live,device=raw.device,dtype=mask.dtype).view(batch,1,1,1,1)
        yield logits,cut(target),cut(mask)*active
        predictions=logits.detach().float().cpu().numpy()[:,0]
        for i,(sl,center) in enumerate(zip(slices,centers)):
            if not live[i]:continue
            prediction=disconnected_bias(canvases[i][sl],predictions[i],initialized[i][sl])
            canvases[i][sl]=prediction;initialized[i][sl]=True
            policies[i].update(prediction,center)


def ffn_windows(model,raw,target,mask,steps=4):
    """Independent per-example predicted moves, fixed backward count for DDP."""
    size=raw.shape[-1];batch=raw.shape[0];centers=[[size//2]*3 for _ in range(batch)]
    canvas=torch.full_like(raw,-2.944438979)
    canvas[:,:,size//2,size//2,size//2]=2.944438979
    visited=[set() for _ in range(batch)]
    for _ in range(steps):
        slices=[tuple(slice(c-16,c+17) for c in center) for center in centers]
        def cut(t):return torch.cat([t[(slice(i,i+1),slice(None),*sl)] for i,sl in enumerate(slices)],dim=0)
        logits=model(cut(raw),cut(canvas))
        yield logits,cut(target),cut(mask)
        queries=[];candidates=[]
        for i,(sl,c) in enumerate(zip(slices,centers)):
            canvas[(i,slice(None),*sl)]=logits[i].detach().float();visited[i].add(tuple(c))
            for axis in range(3):
                for delta in [-8,8]:
                    n=c.copy();n[axis]+=delta
                    if min(n)<16 or max(n)>size-17 or tuple(n) in visited[i]:continue
                    candidates.append((i,n));queries.append(canvas[i,0,n[0],n[1],n[2]])
        if queries:
            scores=torch.stack(queries).detach().cpu().tolist();best={}
            for (i,n),score in zip(candidates,scores):
                if score>=2.197224577 and score>best.get(i,(-float('inf'),None))[0]:best[i]=(score,n)
            for i,(_,n) in best.items():centers[i]=n


def initialize_ffn_head(model,gain):
    """Scale only the randomly initialized output weights, before any update."""
    if not np.isfinite(gain) or gain<=0:raise ValueError('Finite positive head initialization gain required')
    if gain!=1:
        with torch.no_grad():model.out.weight.mul_(gain)


def atomic_checkpoint(checkpoint,path):
    """Keep the preceding checkpoint intact if a node dies during serialization."""
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(checkpoint,temporary)
    os.replace(temporary,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--stage',choices=['ffn','sense'],required=True)
    p.add_argument('--seconds',type=int,required=True);p.add_argument('--steps',type=int,default=1000000)
    p.add_argument('--eval-every',type=int,default=250);p.add_argument('--batch',type=int,default=1)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--init-checkpoint',type=Path)
    p.add_argument('--ffn-policy',choices=['production','legacy'],default='production')
    p.add_argument('--ffn-recurrent-steps',type=int,default=8)
    p.add_argument('--ffn-positive-weight',type=float,default=1.)
    p.add_argument('--ffn-head-init-gain',type=float,default=1.,help='Random FFN output-weight initialization scale; no pretrained warmup')
    p.add_argument('--ffn-raw-normalization',type=float,nargs=2,metavar=('MEAN','STD'),help='Training-set statistics of raw/255, stored in FFN weights')
    p.add_argument('--ffn-validation-reference',type=Path)
    a=p.parse_args();device_guard(a.device)
    if min(a.seconds,a.steps,a.eval_every,a.batch)<=0:raise ValueError('Positive limits required')
    if a.ffn_recurrent_steps<1 or not np.isfinite(a.ffn_positive_weight) or a.ffn_positive_weight<=0:
        raise ValueError('Positive recurrence and finite positive class weight required')
    if not np.isfinite(a.ffn_head_init_gain) or a.ffn_head_init_gain<=0:
        raise ValueError('Finite positive head initialization gain required')
    if a.ffn_head_init_gain!=1 and (a.init_checkpoint or a.stage!='ffn'):
        raise ValueError('Head initialization gain requires random FFN initialization')
    if a.ffn_raw_normalization is not None:
        if a.init_checkpoint or a.stage!='ffn':raise ValueError('Normalization option requires random FFN initialization')
        if not np.isfinite(a.ffn_raw_normalization).all() or a.ffn_raw_normalization[1]<=0:
            raise ValueError('Raw normalization requires finite mean and positive std')
    production=a.stage=='ffn' and a.ffn_policy=='production'
    reference=json.loads(a.ffn_validation_reference.read_text()) if a.ffn_validation_reference else None
    if reference and not production:raise ValueError('Growth reference requires production FFN training')
    if reference and (reference['contract']['manifest_sha256']!=sha256(a.manifest)
                      or not admit(reference,reference)['passed']):
        raise ValueError('Incomplete or mismatched growth reference')
    manifest=json.loads(a.manifest.read_text())
    validate_manifest(manifest,base_dir=a.manifest.parent)
    if manifest['voxel_size_nm'] != [16,16,16]:
        raise ValueError('This model recipe requires a 16 nm ZYX grid; resample in the dataset adapter')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if world>1:dist.init_process_group('nccl' if a.device=='cuda' else 'gloo')
    if a.device=='cuda':torch.cuda.set_device(local)
    device=torch.device(f'cuda:{local}' if a.device=='cuda' else 'cpu');torch.set_num_threads(2);torch.manual_seed(a.seed)
    output=external(a.output)
    if rank==0:output.mkdir(parents=True,exist_ok=False)
    if world>1:dist.barrier(device_ids=[local] if a.device=='cuda' else None)
    shape=65 if a.stage=='ffn' else 128
    train=Crops(a.manifest,'train',shape,a.steps*a.batch*world,seed=a.seed)
    dev=Crops(a.manifest,'validation',shape,64,seed=125)
    sampler=range(rank,len(train),world) if world>1 else None
    loader=DataLoader(train,batch_size=a.batch,sampler=sampler,num_workers=a.workers,pin_memory=a.device=='cuda',drop_last=True)
    norm_options={'raw_normalization':a.ffn_raw_normalization} if a.ffn_raw_normalization is not None else {}
    model=(FFN15(activation_checkpointing=False,**norm_options) if a.stage=='ffn' else Sense()).to(device)
    if a.stage=='ffn' and not a.init_checkpoint:initialize_ffn_head(model,a.ffn_head_init_gain)
    if a.init_checkpoint:
        initial=torch.load(a.init_checkpoint,map_location='cpu',weights_only=True)
        if initial['architecture']['stage']!=a.stage or initial['manifest_sha256']!=sha256(a.manifest):
            raise ValueError('Warm start stage/training manifest differs')
        model.load_state_dict(initial['model'],strict=True)
    wrapped=DDP(model,device_ids=[local] if a.device=='cuda' else None) if world>1 else model
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    bf16=a.device=='cuda' and torch.cuda.is_bf16_supported()
    if reference and reference['contract']['bf16']!=bf16:raise ValueError('Growth reference precision differs')
    def amp():return torch.autocast('cuda',dtype=torch.bfloat16) if bf16 else nullcontext()
    architecture=dict(stage=a.stage,width=32 if a.stage=='ffn' else 8,
        residual_blocks=20 if a.stage=='ffn' else None,factors=None if a.stage=='ffn' else [1,2,4,8,8],
        input_shape=[33]*3 if a.stage=='ffn' else [128]*3,voxel_size_nm=manifest['voxel_size_nm'],
        axes='zyx',prompt_radius=16,
        ffn_recurrent_steps=a.ffn_recurrent_steps if a.stage=='ffn' else None,activation_checkpointing=False)
    if a.ffn_raw_normalization is not None:
        architecture['raw_normalization']=a.ffn_raw_normalization
    elif getattr(model,'raw_normalization',None) is not None:
        # A loaded normalized checkpoint must retain its preprocessing metadata.
        architecture['raw_normalization']=model.raw_normalization.detach().cpu().tolist()
    windows=production_windows if a.ffn_policy=='production' else ffn_windows
    if rank==0:
        write_json(output/'run.json',dict(architecture=architecture,world_size=world,batch_per_gpu=a.batch,
            params=sum(p.numel() for p in model.parameters()),dataset_id=manifest['dataset_id'],
            manifest_sha256=sha256(a.manifest),paper_comparable=False,bf16=bf16,slurm_job_id=os.getenv('SLURM_JOB_ID'),
            cpu_smoke=a.device=='cpu',seed=a.seed,optimizer='AdamW',lr=a.lr,weight_decay=.01,
            time_limit_seconds=a.seconds,sampling='uniform block then instance then center',validation_crops=64,
            ffn_policy=a.ffn_policy,ffn_positive_weight=a.ffn_positive_weight,
            ffn_head_init_gain=a.ffn_head_init_gain,
            collective_exhaustion_early_stop=production,
            init_checkpoint_sha256=sha256(a.init_checkpoint) if a.init_checkpoint else None,
            initialization='checkpoint' if a.init_checkpoint else 'random',
            checkpoint_selection='production_seed_growth' if production else 'validation_bce',
            growth_reference_sha256=sha256(a.ffn_validation_reference) if reference else None,
            code_sha256={f.name:sha256(f) for f in Path(__file__).parent.glob('*.py')}))
    start=time.monotonic();best=float('inf');timer=start;top=[]
    for step,batch in enumerate(loader,1):
        raw,prompt,target,mask=[x.to(device,non_blocking=True) for x in batch]
        wrapped.train();optimizer.zero_grad(set_to_none=True);loss_value=0.
        if a.stage=='sense':
            with amp():logits=wrapped(raw,prompt);loss=loss_sum(logits,target,mask)/mask.sum()
            if not torch.isfinite(loss):raise ValueError('Nonfinite SENSE loss')
            loss.backward();loss_value=loss.detach().item()
        else:
            # All ranks execute the same count of backward calls.
            with amp():
                for logits,y,m in windows(wrapped,raw,target,mask,steps=a.ffn_recurrent_steps,
                        **({'stop_when_exhausted':True} if production else {})):
                    loss=loss_sum(logits,y,m,a.ffn_positive_weight)/m.sum().clamp_min(1)/a.ffn_recurrent_steps
                    if not torch.isfinite(loss):raise ValueError('Nonfinite FFN loss')
                    loss.backward();loss_value+=loss.detach().item()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);optimizer.step()
        elapsed=time.monotonic()-start
        stop=torch.tensor(int(elapsed>=a.seconds or step>=a.steps),device=device)
        if world>1:dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        if step==1 or step%a.eval_every==0 or stop.item():
            model.eval();stats=torch.zeros(6,device=device,dtype=torch.float64)
            with torch.inference_mode():
                for i in range(rank,len(dev),world):
                    x,q,y,m=[t[None].to(device) for t in dev[i]]
                    with amp():
                        # Score every active corrected window, including seeds
                        # whose growth ends before the last recurrent step.
                        predictions=((model(x,q),y,m),) if a.stage=='sense' else windows(
                            model,x,y,m,steps=a.ffn_recurrent_steps)
                        for pred,yw,mw in predictions:
                            positive=pred>0;truth=yw>.5;valid=mw>.5
                            stats[:5]+=torch.stack((loss_sum(pred,yw,mw),mw.sum(),(positive&truth&valid).sum(),
                                (positive&~truth&valid).sum(),(~positive&truth&valid).sum())).double()
            stats[5]=loss_value
            if world>1:dist.all_reduce(stats)
            values=stats.cpu().tolist();bce=values[0]/max(1,values[1]);improved=bce<best;best=min(best,bce)
            rng_state=dict(torch=torch.get_rng_state(),
                           cuda=torch.cuda.get_rng_state(device) if a.device=='cuda' else None)
            rng_states=[rng_state]
            if world>1:
                rng_states=[None]*world
                dist.all_gather_object(rng_states,rng_state)
            if rank==0:
                growth=None;admission=None;selection_score=bce
                if production:
                    contract=reference['contract'] if reference else dict(samples=16,shape=97,max_calls=512)
                    growth=growth_validation(model,a.manifest,device,samples=contract['samples'],
                        shape=contract['shape'],max_calls=contract['max_calls'],bf16=bf16)
                    admission=admit(growth,reference if reference is not None else growth)
                    if reference is None:
                        admission['scope']='complete nonempty fixed validation seed growth; no baseline comparison; full-volume nERL admission still required'
                    selection_score=-growth['totals']['dice'] if admission['passed'] else float('inf')
                    write_json(output/f'growth-{step}.json',dict(**growth,admission=admission))
                row=dict(step=step,seconds=time.monotonic()-start,train_loss=values[5]/world,validation_bce=bce,
                    validation_dice=2*values[2]/max(1,2*values[2]+values[3]+values[4]),best_validation_bce=best,
                    aggregate_samples_per_second=step*a.batch*world/max(1e-9,time.monotonic()-start),
                    max_memory_allocated_gb=torch.cuda.max_memory_allocated()/1e9 if a.device=='cuda' else 0,
                    paper_comparable=False)
                if growth:row.update(growth_validation=growth['totals'],growth_admission=admission)
                with (output/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
                ckpt=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                          training_state=dict(schema='pathfinder-training-state/v1',completed_steps=step,
                              world_size=world,batch_per_gpu=a.batch,seed=a.seed,
                              next_global_sample_index=step*a.batch*world,rng_states=rng_states,
                              run_sha256=sha256(output/'run.json')),
                          architecture=architecture,step=step,validation_bce=bce,
                          manifest_sha256=sha256(a.manifest),paper_comparable=False,
                          selection_rule='production_seed_growth' if production else 'legacy_validation_bce',
                          ffn_policy=a.ffn_policy if a.stage=='ffn' else None,
                          initialization='checkpoint' if a.init_checkpoint else 'random',
                          growth_admission=admission,full_volume_admission=False)
                atomic_checkpoint(ckpt,output/'last.pt')
                eligible=admission['passed'] if admission is not None else True
                if eligible and (not top or selection_score<top[0]['selection_score']):atomic_checkpoint(ckpt,output/'best.pt')
                if eligible and (len(top)<3 or selection_score<top[-1]['selection_score']):
                    name=f'step-{step}.pt';atomic_checkpoint(ckpt,output/name)
                    top.append(dict(step=step,validation_bce=bce,selection_score=selection_score,path=name,
                                    growth_admission=admission));top.sort(key=lambda x:x['selection_score'])
                    for old in top[3:]:(output/old['path']).unlink()
                    top=top[:3];write_json(output/'top-checkpoints.json',top)
            if world>1:dist.barrier(device_ids=[local] if a.device=='cuda' else None)
            timer=time.monotonic()
        if stop.item():break
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
