"""One allocation: complete profiles, freeze shared schedule, train and export submissions."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root=Path(os.environ['WEATHER_PAIR_ROOT'])
task=root/'source'
code=task/'wn2'
sys.path.insert(0,str(code))
from plan_schedule import STAGES
from profile_reuse import invalidation_reasons
scratch=Path('/usr/project/xtmp/sc829/weather-track/wn2-pilot')
job=os.environ['SLURM_JOB_ID']
seed=int(os.environ['WEATHER_TRAIN_SEED'])
out=root/f'pipeline-{seed}'
out.mkdir(exist_ok=False)
status={'job':job,'seed':seed,'status':'running','profiles':[]}
def save(**kw):
    status.update(kw)
    p=out/'status.tmp';p.write_text(json.dumps(status,indent=2)+'\n');p.replace(out/'status.json')
def run(cmd,env=None):
    subprocess.run(cmd,env=env,check=True)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
    shared=root/'shared-plan'
    reports=[]
    # Only use the explicitly selected canonical array's immutable reports.
    candidates=[Path(p) for p in json.loads((root/'reusable-reports.json').read_text())]
    for i,stage in enumerate(STAGES):
        if (shared/'schedule.json').exists():break
        chosen=None
        reuse_decisions=[]
        for p in candidates:
            if not p.exists():continue
            d=json.loads(p.read_text())
            if d.get('stage')!=stage:continue
            reasons=invalidation_reasons(d,stage,code)
            reuse_decisions.append({'report':str(p),'invalidation_reasons':reasons})
            if not reasons:
                chosen=p;break
        (out/f'profile-reuse-{i:02d}.json').write_text(json.dumps(reuse_decisions,indent=2)+'\n')
        if chosen is None:
            # Never silently spend hours profiling when reuse unexpectedly fails.
            authorization=root/'profile-authorization.json'
            allowed=json.loads(authorization.read_text()) if authorization.exists() else {}
            if not allowed.get(str(i),{}).get('reason'):
                raise RuntimeError(f'Stage {i} has no reusable profile and no recorded measurement authorization; inspect profile-reuse-{i:02d}.json')
            save(phase='profiling',stage_index=i)
            profile_id=f'{job}-profile-{i:02d}'
            env=os.environ|{'WEATHER_TASK_ROOT':str(task),'WEATHER_STAGE_INDEX':str(i),'SLURM_JOB_ID':profile_id}
            run(['bash',str(task/'slurm/wn2_stage.sbatch')],env)
            chosen=scratch/'stage-runs'/profile_id/'result/report.json'
        reports.append(chosen)
        status['profiles'].append(str(chosen));save()
    with (root/'plan.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (shared/'schedule.json').exists():
            candidate=out/'candidate-plan'
            run([sys.executable,str(code/'collect_profiles.py'),'--reports',*map(str,reports),'--output',str(candidate)])
            shutil.copytree(candidate,root/'shared-plan.tmp')
            (root/'shared-plan.tmp').rename(shared)
    save(phase='curriculum',shared_schedule_sha256=digest(shared/'schedule.json'))
    run([sys.executable,str(code/'train.py'),'--seed',str(seed),
        '--data',str(scratch/'training-2017-2020'),'--plan',str(shared),
        '--image','/usr/project/xtmp/sc829/weather-track/images-final/environment.sif',
        '--eval-python',sys.executable,'--output',str(out/'curriculum')])
    save(status='passed',phase='complete',evaluation_status='not_run',
         package=str(out/'curriculum/submission-cold'))
except Exception as error:
    save(status='failed',error=repr(error));raise
