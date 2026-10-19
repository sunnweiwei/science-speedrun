"""Serialize submissions and reserve remaining whole-experiment Slurm budget.

Only jobs submitted through this ledger count as experiment jobs. Unknown or
unfinished accounting keeps the full time reservation; never silently frees it.
"""
import os
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
from fetch_axonem import usage,LIMIT,storage_limit
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,write_json,sha256

TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def allocation(jobs):
    charges=[j.get('actual_seconds',j['reserved_seconds']+j.get('prior_attempt_seconds',0)) for j in jobs]
    seconds=sum(charges)
    gpu_seconds=sum(seconds*j['gpus'] for seconds,j in zip(charges,jobs))
    return seconds,gpu_seconds


def settle_attempts(job,accounting):
    """Charge every allocation attempt, including failed attempts before requeue.

    sacct must include duplicates (-D). An unfinished retry retains its full
    reservation in addition to the elapsed time of earlier terminal attempts.
    Empty accounting leaves the existing charge/reservation untouched.
    """
    attempts=[]
    for line in accounting.splitlines():
        fields=line.split('|')
        if len(fields)<3 or fields[0]!=job['job_id']:continue
        state=fields[1].split()[0].rstrip('+')
        seconds=int(fields[2])
        if seconds<0:raise ValueError('Negative Slurm elapsed time')
        attempts.append(dict(state=state,elapsed_seconds=seconds))
    if not attempts:return
    active=[a for a in attempts if a['state'] not in TERMINAL]
    job['accounting_attempts']=attempts
    job['accounting_basis']='sum all sacct -D allocation attempts; unfinished retry keeps reservation'
    job['state']=(active or attempts)[-1]['state']
    if active:
        job.pop('actual_seconds',None)
        job['prior_attempt_seconds']=sum(a['elapsed_seconds'] for a in attempts if a['state'] in TERMINAL)
    else:
        job['actual_seconds']=sum(a['elapsed_seconds'] for a in attempts)
        job.pop('prior_attempt_seconds',None)


def within_budget(ledger,disk,seconds,gpu_seconds,root=None):
    limits=ledger["limits"]
    if any(type(limits[k]) is not int or limits[k]<=0 for k in ("bytes","allocation_seconds","gpu_seconds")):
        raise ValueError("Positive integer ledger limits required")
    ceiling=min(LIMIT,limits["bytes"]) if root is None else storage_limit(root,limits["bytes"])
    return (disk<=ceiling and seconds<=limits["allocation_seconds"]
            and gpu_seconds<=limits["gpu_seconds"])


def concurrency_allowed(jobs, requested_gpus, concurrent_limit=None):
    """Default remains serial; explicit bounded opt-in allows independent jobs."""
    active=[j for j in jobs if 'actual_seconds' not in j]
    if concurrent_limit is None:return not active
    if type(concurrent_limit) is not int or concurrent_limit<1:raise ValueError('Positive concurrent GPU limit required')
    return sum(j['gpus'] for j in active)+requested_gpus<=concurrent_limit


def observed_disk_usage(root,ledger,skip_waived=False):
    ceiling=storage_limit(root,ledger['limits']['bytes'])
    if skip_waived and ceiling==float('inf'):
        return ledger.get('bytes_on_disk',0),'previous observation; development waiver validated, not a current disk measurement'
    return usage(root),'fresh filesystem traversal'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--script',type=Path)
    p.add_argument('--minutes',type=int);p.add_argument('--gpus',type=int,default=4)
    p.add_argument('--name',default='fly-pathfinder')
    p.add_argument('--gpu-type',choices=['a5000','a6000'],default='a5000',help='Explicit GPU model within gpu; record hardware in comparisons')
    p.add_argument('--concurrent-gpus',type=int,help='Explicit concurrent GPU ceiling; must also be authorized in ledger limits')
    p.add_argument('--cached-settlements',action='store_true',help='Preserve already settled historical charges without repeat scheduler queries')
    p.add_argument('--skip-waived-disk-scan',action='store_true',help='Skip tree scan only under valid explicit development storage waiver; label prior disk observation')
    a=p.parse_args()
    root=external(a.root);root.mkdir(parents=True,exist_ok=True)
    with (root/'budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=root/'budget.json';ledger=json.loads(path.read_text()) if path.exists() else dict(jobs=[],limits=dict(bytes=LIMIT,allocation_seconds=86400,gpu_seconds=345600))
        for job in ledger['jobs']:
            if a.cached_settlements and 'actual_seconds' in job:continue
            result=subprocess.run(['sacct','-X','-D','-j',job['job_id'],'--noheader','--parsable2','--format=JobIDRaw,State,ElapsedRaw'],capture_output=True,text=True,check=True)
            settle_attempts(job,result.stdout)
        seconds,gpu_seconds=allocation(ledger['jobs'])
        disk,disk_scope=observed_disk_usage(root,ledger,a.skip_waived_disk_scan)
        ledger.update(bytes_on_disk=disk,bytes_on_disk_scope=disk_scope,charged_seconds=seconds,charged_gpu_seconds=gpu_seconds)
        write_json(path,ledger)
        if a.script:
            if not a.minutes or a.minutes<1 or not 1<=a.gpus<=4:raise ValueError('Positive minutes and 1..4 GPUs required')
            if a.concurrent_gpus is not None and a.concurrent_gpus>ledger['limits'].get('concurrent_gpus',4):
                raise ValueError('Requested concurrency exceeds ledger authorization')
            if not concurrency_allowed(ledger['jobs'],a.gpus,a.concurrent_gpus):raise ValueError('An existing experiment allocation is unfinished or concurrent GPU ceiling exceeded')
            if not within_budget(ledger,disk,seconds+a.minutes*60,gpu_seconds+a.minutes*60*a.gpus,root=root):
                raise ValueError('Whole-experiment storage/time/GPU budget exceeded')
            script=a.script.resolve()
            # Persist intent BEFORE sbatch: an uncertain submission must be
            # reconciled manually instead of allowing a duplicate allocation.
            intent=root/'submission-intent.json'
            if intent.exists():raise ValueError('Unreconciled submission intent')
            write_json(intent,dict(script=str(script),sha256=sha256(script),minutes=a.minutes,gpus=a.gpus,gpu_type=a.gpu_type))
            result=subprocess.run(['sbatch','--parsable','--no-requeue','--partition='+os.environ.get('PATHFINDER_GPU_PARTITION', 'gpu'),'--account='+os.environ.get('PATHFINDER_SLURM_ACCOUNT', 'research'),'--nodes=1',
                '--ntasks=1',f'--gres=gpu:{a.gpu_type}:{a.gpus}',f'--time={a.minutes}',f'--job-name={a.name}',
                f'--output={root}/slurm-%j.log',str(script)],text=True,capture_output=True)
            if result.returncode:
                intent.unlink();raise RuntimeError(result.stderr)
            job_id=result.stdout.strip().split(';')[0]
            if not job_id.isdigit():raise RuntimeError('Ambiguous submission output; intent retained')
            ledger['jobs'].append(dict(job_id=job_id,reserved_seconds=a.minutes*60,gpus=a.gpus,
                script=str(script),script_sha256=sha256(script),gpu_type=a.gpu_type,state='SUBMITTED'))
            seconds,gpu_seconds=allocation(ledger['jobs'])
            ledger.update(charged_seconds=seconds,charged_gpu_seconds=gpu_seconds)
            write_json(path,ledger);intent.unlink();print('Submitted '+job_id,flush=True)
        print(json.dumps(ledger,indent=2))

if __name__=='__main__':main()
