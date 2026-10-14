#!/usr/bin/env python3
"""Execute the measured shortened native curriculum and export its submission."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from plan_schedule import STAGES,plan
from training_data import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','plan','image','eval-python','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args();begin=time.monotonic()
    if args.output.exists():raise ValueError('A curriculum run must use a new output directory')
    args.output.mkdir(parents=True)
    source=Path(__file__).resolve().parent
    schedule_path=args.plan/'schedule.json';schedule=json.loads(schedule_path.read_text())
    profiles=json.loads((args.plan/'profiles.json').read_text())
    expected=plan(profiles)
    if schedule['stages']!=expected['stages'] or schedule['profile_sha256']!=sha256(args.plan/'profiles.json'):
        raise ValueError('Schedule is not the measured frozen curriculum')
    code=args.output/'code';code.mkdir()
    for path in source.glob('*.py'):shutil.copyfile(path,code/path.name)
    for path in source.glob('*protocol.json'):shutil.copyfile(path,code/path.name)
    for name in ('wn2_validation.py','wn2_distributed.py'):
        path=source/name if (source/name).exists() else source.parent/'slurm'/name
        shutil.copyfile(path,code/name)
    namespace=source/'namespace_inference.py'
    shutil.copyfile(namespace,code/namespace.name)
    license_path=source/'LICENSE.weathernext-code'
    shutil.copyfile(license_path,code/license_path.name)
    shutil.copyfile(schedule_path,args.output/'schedule.json')
    provenance=json.loads((args.plan/'profile-provenance.json').read_text())
    image_hash=sha256(args.image)
    by_stage={}
    for entry in provenance:
        path=Path(entry['path'])
        if sha256(path)!=entry['sha256']:raise ValueError('Changed timing evidence')
        if image_hash!=entry['image_sha256']:raise ValueError('Runtime image changed after profiling')
        if sha256(path.parent.parent/'autotune.pb')!=entry['autotune_sha256']:raise ValueError('Changed frozen autotuning results')
        report=json.loads(path.read_text());by_stage[report['stage']['name']]=path
        for name in ('train_stage.py','native_model.py','training_data.py','wn2_validation.py','wn2_distributed.py','plan_schedule.py'):
            if sha256(code/name)!=report['source_sha256'][name]:raise ValueError('Training implementation changed after profiling')
    wall_cap=schedule['training_cap_seconds']+schedule['checkpoint_reserve_seconds']
    report=dict(status='started',seed=args.seed,schedule_sha256=sha256(schedule_path),image_sha256=image_hash,
                stages=[],pretrained_initialization=False,total_wall_cap_seconds=wall_cap)
    def save(**updates):
        report.update(updates);tmp=args.output/'run.json.tmp'
        tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(args.output/'run.json')
    def run_bounded(command, deadline, **kwargs):
        remaining=deadline-time.time()
        if remaining<=0:raise RuntimeError('Curriculum phase budget exhausted')
        process=subprocess.Popen(command,start_new_session=True,**kwargs)
        try:
            code=process.wait(timeout=remaining)
            if code:raise subprocess.CalledProcessError(code,command)
        finally:
            # Apptainer and inference may spawn descendants. Terminate the
            # whole phase on a deadline, including during compilation.
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()
    total_deadline=time.time()+wall_cap-(time.monotonic()-begin)
    training_deadline=time.time()+schedule['training_cap_seconds']-(time.monotonic()-begin)
    try:
        for index,stage in enumerate(STAGES):
            profile_runtime=json.loads(by_stage[stage['name']].read_text())['runtime']
            tuning=args.output/f'autotune-{index:02d}.pb'
            shutil.copyfile(by_stage[stage['name']].parent.parent/'autotune.pb',tuning)
            stage_path=args.output/f'stage-{index:02d}'
            cache=args.output/f'cache-{index:02d}'
            profile_cache=by_stage[stage['name']].parent.parent/'jax-cache'
            if profile_cache.exists():shutil.copytree(profile_cache,cache)
            else:cache.mkdir()
            cmd=['apptainer','exec','--nv','--containall','--cleanenv',
                '--env',f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}",
                '--env','PYTHONUNBUFFERED=1','--env','OMP_NUM_THREADS=8','--env','OPENBLAS_NUM_THREADS=8',
                '--env','XLA_PYTHON_CLIENT_PREALLOCATE=false','--env','XLA_PYTHON_CLIENT_ALLOCATOR=platform',
                '--env',f"XLA_FLAGS={profile_runtime['xla_flags']} --xla_gpu_load_autotune_results_from=/run/autotune-{index:02d}.pb --xla_gpu_require_complete_aot_autotune_results=true",'--env',f"CUBLAS_WORKSPACE_CONFIG={profile_runtime['cublas_workspace_config']}",
                '--env',f'JAX_COMPILATION_CACHE_DIR=/run/cache-{index:02d}',
                '--env','JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0','--env',f"JAX_ENABLE_COMPILATION_CACHE={profile_runtime['compilation_cache']}",
                '--env','JAX_RAISE_PERSISTENT_CACHE_ERRORS=true',
                '--bind',f'{args.data.resolve()}:/data:ro','--bind',f'{args.output.resolve()}:/run',
                str(args.image),'/opt/weather-python/bin/python','/run/code/train_stage.py',
                '--data','/data','--statistics','/data/statistics','--protocol','/run/code/training-protocol.json',
                '--output',f'/run/stage-{index:02d}','--stage-index',str(index),'--schedule','/run/schedule.json',
                '--deadline-unix',str(training_deadline),'--seed',str(args.seed)]
            if index:cmd+=['--resume',f'/run/stage-{index-1:02d}/state.npz']
            save(phase='training',stage_index=index,elapsed_seconds=time.monotonic()-begin)
            with (args.output/f'stage-{index:02d}.log').open('wb') as log:
                run_bounded(cmd,training_deadline,stdout=log,stderr=subprocess.STDOUT)
            result=json.loads((stage_path/'report.json').read_text())
            if result['status']!='passed':raise ValueError('Native stage did not complete')
            report['stages'].append(dict(stage=stage['name'],report_sha256=sha256(stage_path/'report.json'),
                                        wall_seconds=result['wall_seconds'],updates=len(result['updates'])))
        save(phase='packaging',training_wall_seconds=time.monotonic()-begin)
        cold_package=args.output/'submission-cold'
        run_bounded([str(args.eval_python),str(code/'package_model.py'),'--stage',str(stage_path),
                        '--statistics',str(args.data/'statistics'),'--output',str(cold_package)],total_deadline)
        elapsed=time.monotonic()-begin
        if elapsed>wall_cap:raise RuntimeError('Training/package wall cap exceeded')
        save(status='passed',phase='complete',total_wall_seconds=elapsed,allocated_gpu_hours=4*elapsed/3600,
             package=str(cold_package),evaluation_status='not_run',
             scope='Complete curriculum and exported submission. Run unified forecast and CPU score phases separately; this report does not establish evaluation acceptance.')
    except Exception as exc:
        save(status='failed',error=repr(exc),elapsed_seconds=time.monotonic()-begin);raise


if __name__=='__main__':main()
