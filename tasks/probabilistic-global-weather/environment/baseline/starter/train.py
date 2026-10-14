#!/usr/bin/env python3
"""Train one complete native WN2 curriculum and export a portable model."""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time



def swap_into(staged,final):
    """Replace the published package with one rename, so it is never half-written."""
    old=final.with_name(f'.{final.name}.old-{os.getpid()}')
    if final.exists():
        try:os.replace(final,old)
        except OSError as error:
            # A directory from a lower image layer cannot be renamed on overlayfs (EXDEV).
            if error.errno!=errno.EXDEV:raise
            shutil.copytree(final,old,symlinks=True);shutil.rmtree(final)
    os.replace(staged,final);shutil.rmtree(old,ignore_errors=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('/workspace/data/training'))
    parser.add_argument('--output',type=Path,default=Path('/workspace/research/training-run'))
    parser.add_argument('--submission',type=Path,default=Path('/workspace/submission'))
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    source=Path(__file__).resolve().parent
    schedule=source/'schedule.json'
    spec=json.loads(schedule.read_text())
    identity={'seed':args.seed,'schedule_sha256':hashlib.sha256(schedule.read_bytes()).hexdigest(),'data':str(args.data.resolve())}
    record=args.output/'run.json'
    if args.resume:
        state=json.loads(record.read_text())
        if any(state.get(k)!=v for k,v in identity.items()):raise ValueError('Resume changes seed, schedule or data location')
        if state.get('status')=='passed':return
    else:
        if args.output.exists():raise ValueError('Use a new run directory or --resume')
        args.output.mkdir(parents=True)
        state=identity|{'status':'running','started_unix':time.time()}
    def save(**kw):
        state.update(kw);p=record.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2)+'\n');p.replace(record)
    deadline=state['started_unix']+spec['training_cap_seconds']
    python=os.environ.get('WEATHER_MODEL_PYTHON',sys.executable)
    try:
        for index,stage in enumerate(spec['stages']):
            output=args.output/f'stage-{index:02d}'
            report=output/'report.json'
            if report.exists() and json.loads(report.read_text()).get('status')=='passed':continue
            if time.time()>=deadline:raise RuntimeError('Training time budget exhausted before complete curriculum')
            save(status='running',stage_index=index,phase='training')
            output.mkdir(exist_ok=True)
            env=os.environ|{'PYTHONHASHSEED':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8',
                'OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8','JAX_ENABLE_COMPILATION_CACHE':'true',
                'XLA_PYTHON_CLIENT_PREALLOCATE':'true','XLA_PYTHON_CLIENT_MEM_FRACTION':'0.97',
                'JAX_COMPILATION_CACHE_DIR':str((output/'jax-cache').resolve()),
                'JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS':'0','JAX_RAISE_PERSISTENT_CACHE_ERRORS':'true'}
            tune=(output/'autotune.pb').resolve()
            flags='--xla_gpu_exclude_nondeterministic_ops=true '
            flags+=(f'--xla_gpu_load_autotune_results_from={tune} --xla_gpu_require_complete_aot_autotune_results=true' if tune.exists() and tune.stat().st_size else f'--xla_gpu_dump_autotune_results_to={tune}')
            env['XLA_FLAGS']=flags
            cmd=[python,str(source/'train_stage.py'),'--data',str(args.data),
                 '--statistics',str(args.data/'statistics'),'--protocol',str(source/'training-protocol.json'),
                 '--schedule',str(schedule),'--output',str(output),'--stage-index',str(index),
                 '--seed',str(args.seed),'--deadline-unix',str(deadline)]
            previous=output/'state.npz'
            if not previous.exists() and index:previous=args.output/f'stage-{index-1:02d}'/'state.npz'
            if previous.exists():cmd+=['--resume',str(previous)]
            with (args.output/f'stage-{index:02d}.log').open('ab') as log:
                subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,
                               timeout=max(1,deadline-time.time()))
        save(phase='export')
        remaining=deadline+spec['checkpoint_reserve_seconds']-time.time()
        if remaining<=0:raise RuntimeError('Packaging time budget exhausted')
        staged=args.submission.with_name(f'.{args.submission.name}.stage-{os.getpid()}');shutil.rmtree(staged,ignore_errors=True)
        subprocess.run([python,str(source/'package_model.py'),'--stage',str(output),
                        '--statistics',str(args.data/'statistics'),'--output',str(staged)],check=True,timeout=remaining)
        swap_into(staged,args.submission)
        save(status='passed',phase='complete',submission=str(args.submission),elapsed_seconds=time.time()-state['started_unix'])
    except Exception as error:
        save(status='failed',error=repr(error));raise


if __name__=='__main__':main()
