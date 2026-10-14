#!/usr/bin/env python3
"""Profile or train one full WN2 curriculum stage with streamed batch 64."""
import argparse
from functools import lru_cache
import json
import os
import subprocess
import importlib.metadata
from pathlib import Path
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
import xarray_jax
from weathernext.utils import checkpoint
from weathernext.weathernext2 import fgn

from native_model import extract, stage_config, transforms
from plan_schedule import STAGES, learning_rate
from training_data import load_window, sha256, verify_case
from wn2_distributed import host_replica, load_state, save_state, replicate_from_host
from wn2_validation import ensure_deterministic_process


def draw_indices(seed,stage_index,update_index,count):
    """Deterministic shuffled passes; state depends only on recorded counters."""
    positions=np.arange(update_index*64,(update_index+1)*64)
    result=[]
    for position in positions:
        epoch,offset=divmod(int(position),count)
        permutation=np.random.default_rng(np.random.SeedSequence([seed,stage_index,epoch])).permutation(count)
        result.append(int(permutation[offset]))
    return np.asarray(result).reshape(16,4)


def optimizer():
    # Apply the stage learning rate dynamically, retaining Adam moments across
    # stage transitions without rebuilding their state structure.
    return optax.chain(optax.clip_by_global_norm(1.),optax.scale_by_adam(b2=.95),
                       optax.add_decayed_weights(.1),optax.scale(-1.))


def make_functions(loss,optim,devices,accumulation=16):
    def accumulate(params,total,key,inputs,targets,forcings):
        (value,_),grad=jax.value_and_grad(loss.apply,has_aux=True)(params,key,inputs,targets,forcings)
        finite=jnp.isfinite(value)&jnp.all(jnp.stack([jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grad)]))
        return jax.tree.map(jnp.add,total,grad),value,finite
    def update(params,state,total,lr):
        grad=jax.tree.map(lambda g:jax.lax.pmean(g/accumulation,'device'),total)
        updates,state=optim.update(grad,state,params)
        updated=optax.apply_updates(params,jax.tree.map(lambda u:u*lr,updates))
        finite=jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in jax.tree.leaves((updated,state))]))
        changed=jnp.any(jnp.stack([jnp.any(a!=b) for a,b in zip(jax.tree.leaves(params),jax.tree.leaves(updated))]))
        return updated,state,optax.global_norm(grad),finite,changed
    # xarray_jax translates structured inputs to flattened pmap arguments;
    # donation is deliberately omitted until its flattening map is verified.
    return (xarray_jax.pmap(accumulate,dim='device',devices=devices),
            jax.pmap(update,axis_name='device',devices=devices))



def device_memory_mib():
    """Per-visible-GPU memory in MiB from nvidia-smi; empty when it is unavailable."""
    try:
        out=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True)
    except (OSError,subprocess.CalledProcessError):
        return []
    return [int(float(line.strip())) for line in out.splitlines() if line.strip()]

def run(args,report,save):
    begin=time.monotonic()
    revision=subprocess.check_output(['git','-C',str(Path(fgn.__file__).resolve().parents[2]),'rev-parse','HEAD'],text=True).strip()
    if revision!='89c4b2a77a1c57b328b909c575550fd2e5aadc9c':raise ValueError('Unexpected native source revision')
    raw_flags=os.environ.get('XLA_FLAGS','')
    tuning_io=('--xla_gpu_dump_autotune_results_to=','--xla_gpu_load_autotune_results_from=',
               '--xla_gpu_require_complete_aot_autotune_results=')
    runtime=dict(weathernext_revision=revision,versions={n:importlib.metadata.version(n) for n in
                 ('jax','jaxlib','dm-haiku','optax','numpy','xarray')},
                 xla_flags=' '.join(flag for flag in raw_flags.split() if not flag.startswith(tuning_io)),
                 compilation_cache=os.environ.get('JAX_ENABLE_COMPILATION_CACHE','true'),
                 python_hash_seed=os.environ.get('PYTHONHASHSEED'),
                 cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG',''))
    device_count=1 if args.debug_single_device else 4
    accumulation=1 if args.debug_single_device else 16
    if args.debug_single_device and not args.profile:raise ValueError('Single-device mode is only an execution diagnostic')
    if jax.default_backend()!='gpu' or len(jax.local_devices())!=device_count:
        raise ValueError(f'Exactly {device_count} allocated GPUs required')
    devices=jax.local_devices()
    runtime['device_kinds']=[d.device_kind for d in devices]
    # Recorded for the stage report; the frozen schedule was measured on 96 GB
    # devices, and 80 GB H100s complete every AR stage once the training entry
    # point uses XLA's BFC allocator with preallocation instead of the
    # fragmenting platform allocator.
    runtime['device_memory_mib']=device_memory_mib()
    protocol=json.loads(args.protocol.read_text())
    statistics=json.loads((args.statistics/'manifest.json').read_text())
    if statistics['protocol_sha256']!=sha256(args.protocol):raise ValueError('Statistics belong to another training corpus')
    stage=dict(STAGES[args.stage_index])
    if args.schedule:
        schedule=json.loads(args.schedule.read_text())
        stage=schedule['stages'][args.stage_index]
        for key,value in STAGES[args.stage_index].items():
            if stage[key]!=value:raise ValueError('Frozen schedule changes native stage definition')
        updates=stage['short_updates']
    else:
        if not args.profile:raise ValueError('Training requires a measured frozen schedule')
        updates=args.debug_updates if args.debug_single_device else 2
    config=stage_config(stage,args.statistics)
    forward,loss=transforms(config)
    source='era5' if stage['name'].startswith('era5') else 'hres_fc0'
    @lru_cache(maxsize=len(protocol['initializations']))
    def batch(index):
        shard_name=f'case-{index:03d}.json'
        shard_path=args.statistics/shard_name
        if sha256(shard_path)!=statistics['case_manifests'][shard_name]:
            raise ValueError('Changed training-statistics provenance')
        shard=json.loads(shard_path.read_text())
        if shard['evidence']!=verify_case(args.data,args.protocol,index):
            raise ValueError('Weather/labels changed since normalization was computed')
        ds=load_window(args.data,protocol,index,source,stage['resolution'],stage['cadence_hours'],stage['ar_steps'])
        return extract(ds,config,stage)
    first=batch(0)
    save(phase='initializing',stage=stage,devices=device_count,effective_batch=device_count*accumulation,runtime=runtime,raw_xla_flags=raw_flags,
         debug_single_device=args.debug_single_device,unique_weather_windows=len(protocol['initializations']),
         protocol_sha256=sha256(args.protocol),statistics_manifest_sha256=sha256(args.statistics/'manifest.json'),
         pretrained_parameters_loaded=False,profile=args.profile)
    params=jax.jit(forward.init)(jax.random.PRNGKey(args.seed),first[0],xr.zeros_like(first[1]),first[2])
    jax.block_until_ready(params)
    if sum(v.size for v in jax.tree.leaves(params))!=183785575:raise ValueError('Native parameter count changed')
    save(phase='initialized',setup_and_initialization_seconds=time.monotonic()-begin,
         parameter_count=183785575)
    optim=optimizer();state=optim.init(params)
    spec=dict(schema='wn2-curriculum-state/v2',seed=args.seed,protocol_sha256=sha256(args.protocol),
              statistics_sha256=sha256(args.statistics/'manifest.json'),parameter_count=183785575,
              devices=device_count,effective_batch=device_count*accumulation,runtime=runtime,
              optimizer=dict(name='AdamW',beta1=.9,beta2=.95,epsilon=1e-8,weight_decay=.1,clip_global_norm=1.),
              training_sources={name:sha256(Path(__file__).with_name(name)) for name in
                ('train_stage.py','native_model.py','training_data.py','wn2_validation.py','wn2_distributed.py','plan_schedule.py')})
    start_update=0;global_updates=0
    if args.resume:
        with np.load(args.resume,allow_pickle=False) as checkpoint_data:
            header=json.loads(str(checkpoint_data['metadata']))
        previous_stage=header['stage_index']
        if previous_stage not in (args.stage_index,args.stage_index-1) or previous_stage<0:
            raise ValueError('Resume skips or truncates a curriculum stage')
        if previous_stage!=args.stage_index and not header['stage_complete']:
            raise ValueError('Previous native stage is incomplete')
        previous_cadence=STAGES[previous_stage]['cadence_hours']
        if header.get('cadence_hours')!=previous_cadence:
            raise ValueError('Checkpoint cadence is not explicit or differs from its stage')
        transition=(previous_cadence,stage['cadence_hours']) if previous_cadence!=stage['cadence_hours'] else None
        expected_spec=spec
        expected_schedule=sha256(args.schedule) if args.schedule else None
        if args.fork_parent_schedule:
            from ar_extension import validate_branch
            parent=json.loads(args.fork_parent_schedule.read_text())
            validate_branch(header,parent,schedule,args.stage_index)
            if not args.fork_parent_runner or header['training_sources']['train_stage.py']!=sha256(args.fork_parent_runner):
                raise ValueError('Fork parent runner differs from checkpoint')
            expected_schedule=sha256(args.fork_parent_schedule)
            expected_spec=spec|dict(training_sources=spec['training_sources']|{'train_stage.py':sha256(args.fork_parent_runner)})
            save(schedule_branch=dict(parent_schedule_sha256=expected_schedule,
                parent_checkpoint_sha256=sha256(args.resume),preserves_optimizer_state=True,
                validation_source_sha256=sha256(Path(__file__).with_name('ar_extension.py'))))
        (params,state),metadata=load_state(args.resume,(params,state),expected_spec,cadence_transition=transition)
        if metadata.get('schedule_sha256')!=expected_schedule:
            raise ValueError('Resume changes the frozen schedule')
        previous_stage=metadata['stage_index']
        if previous_stage==args.stage_index:
            start_update=metadata['stage_updates']
        elif previous_stage!=args.stage_index-1 or not metadata['stage_complete']:
            raise ValueError('Resume skips or truncates a curriculum stage')
        global_updates=metadata['global_updates']
    params,state=replicate_from_host((params,state),devices)
    accumulate,apply_update=make_functions(loss,optim,devices,accumulation)
    zero_gradients=jax.pmap(lambda p:jax.tree.map(jnp.zeros_like,p),devices=devices)
    durations=[]
    for index in range(start_update,updates):
        next_cost=max(durations[-1:] or [stage.get('estimated_update_seconds',0)])
        if args.deadline_unix and time.time()+next_cost+120>=args.deadline_unix:
            raise RuntimeError('Budget reserve reached before complete curriculum; partial state is not accepted')
        start=time.monotonic()
        total=zero_gradients(params)
        losses=[];draws=draw_indices(args.seed,args.stage_index,index,len(protocol['initializations']))[:accumulation,:device_count]
        input_seconds=0.;gradient_seconds=0.
        for micro,indices in enumerate(draws):
            input_start=time.monotonic()
            samples=[batch(int(i)) for i in indices]
            i,t,f=[xr.concat([s[k] for s in samples],dim='device',data_vars='all',coords='minimal',compat='equals') for k in range(3)]
            key=jax.random.fold_in(jax.random.PRNGKey(args.seed),args.stage_index)
            key=jax.random.fold_in(key,index*16+micro)
            keys=np.asarray(jax.random.split(key,device_count))
            input_seconds+=time.monotonic()-input_start
            if micro==0:save(phase='gradient_compile_or_execution',stage_update=index,microbatch=micro)
            gradient_start=time.monotonic()
            total,value,finite=accumulate(params,total,keys,i,t,f)
            jax.block_until_ready((total,value,finite))
            gradient_seconds+=time.monotonic()-gradient_start
            if not np.asarray(finite).all():raise ValueError('Nonfinite microbatch loss/gradient')
            losses.append(float(np.asarray(value).mean()))
            if micro==0:save(phase='accumulating',stage_update=index,microbatch=micro)
        lr=stage['peak_lr'] if args.profile else learning_rate(stage,index)
        optimizer_start=time.monotonic()
        params,state,norm,finite,changed=apply_update(params,state,total,np.full(device_count,lr,dtype=np.float32))
        jax.block_until_ready((params,state,norm,finite,changed))
        optimizer_seconds=time.monotonic()-optimizer_start
        if not np.asarray(finite).all() or not np.asarray(changed).all() or not (np.asarray(norm)>0).all():
            raise ValueError('Finite/nonzero/changed parameter gate failed')
        elapsed=time.monotonic()-start;global_updates+=1
        record=dict(stage_update=index+1,global_updates=global_updates,seconds=elapsed,
                    loss=float(np.mean(losses)),gradient_norm=float(np.asarray(norm)[0]),learning_rate=lr,
                    weather_indices=draws.tolist(),
                    input_preparation_seconds=input_seconds,
                    gradient_dispatch_and_execution_seconds=gradient_seconds,
                    optimizer_dispatch_and_execution_seconds=optimizer_seconds)
        report.setdefault('updates',[]).append(record)
        save(phase='update_complete')
        metadata=spec|dict(stage_index=args.stage_index,stage_updates=index+1,stage_complete=index+1==updates,
                           cadence_hours=stage['cadence_hours'],
                           global_updates=global_updates,elapsed_stage_seconds=time.monotonic()-begin,
                           schedule_sha256=sha256(args.schedule) if args.schedule else None)
        checkpoint_start=time.monotonic()
        save_state(args.output/'state.npz',params,state,metadata)
        record['checkpoint_seconds']=time.monotonic()-checkpoint_start
        record['seconds']=time.monotonic()-start
        record['includes_checkpoint_io']=True
        durations.append(record['seconds'])
        save(phase='checkpoint_complete')
    if args.profile:
        if len(durations)!=updates:raise ValueError('Incomplete profile updates')
        save(update_seconds=durations[-1],overhead_seconds=max(0,time.monotonic()-begin-updates*durations[-1]),
             includes_io_and_communication=not args.debug_single_device,
             cache_policy='Frozen training corpus retained in host memory after first pass; first-load and compilation charged as overhead')
    with (args.output/'weights.npz').open('wb') as stream:
        checkpoint.dump(stream,fgn.CheckPoint(params=host_replica(params),
            description='Full native WN2 random-initialized task curriculum parameters',license='Apache-2.0 code; upstream data terms apply'))
    weights_hash=sha256(args.output/'weights.npz')
    if args.profile:
        save(overhead_seconds=max(0,time.monotonic()-begin-updates*durations[-1]),
             includes_final_model_export=True)
    save(status='passed',phase='complete',wall_seconds=time.monotonic()-begin,weights_sha256=weights_hash)


def main():
    ensure_deterministic_process()
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('data','protocol','statistics','output'):
        parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--stage-index',type=int,required=True,choices=range(len(STAGES)))
    parser.add_argument('--profile',action='store_true');parser.add_argument('--schedule',type=Path)
    parser.add_argument('--debug-single-device',action='store_true')
    parser.add_argument('--debug-updates',type=int,choices=(1,2),default=1)
    parser.add_argument('--fork-parent-schedule',type=Path)
    parser.add_argument('--fork-parent-runner',type=Path)
    parser.add_argument('--resume',type=Path);parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--deadline-unix',type=float)
    args=parser.parse_args()
    if bool(args.fork_parent_schedule)!=bool(args.fork_parent_runner) or (args.fork_parent_schedule and (not args.resume or not args.schedule)):
        parser.error('Schedule branch requires resume, schedule, parent schedule and parent runner')
    args.output.mkdir(parents=True,exist_ok=True)
    report=dict(status='started',runner_sha256=sha256(Path(__file__)),
                source_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')})
    def save(**updates):
        report.update(updates);tmp=args.output/'report.json.tmp'
        tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(args.output/'report.json')
        print(json.dumps(updates),flush=True)
    try:run(args,report,save)
    except Exception as exc:
        save(status='failed',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
