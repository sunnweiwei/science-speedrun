#!/usr/bin/env python3
"""Submit paper inference when each paired curriculum exports its final package."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def verify_final_package(package):
    manifest=package/'package.json'
    if not manifest.exists():return None
    r=json.loads(manifest.read_text())
    if (r.get('schema')!='wn2-starter-package/v1' or r.get('profile_artifact') is not False
            or r.get('parameter_count')!=183785575
            or r.get('source_revision')!='89c4b2a77a1c57b328b909c575550fd2e5aadc9c'):
        raise ValueError('Not a final native WN2 package')
    if not {'weights.npz','training-report.json','native_model.py','predict.py'}.issubset(r['files']):
        raise ValueError('Incomplete package inventory')
    for name,expected in r['files'].items():
        path=(package/name).resolve()
        if not path.is_relative_to(package.resolve()) or digest(path)!=expected:raise ValueError('Changed package content')
    report=json.loads((package/'training-report.json').read_text())
    if report.get('status')!='passed' or report.get('profile') or report.get('stage',{}).get('name')!='hres_025deg_ar8':
        raise ValueError('Final HRES AR8 training has not completed')
    if report['weights_sha256']!=r['files']['weights.npz']:raise ValueError('Weights differ from final training report')
    return digest(manifest)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',required=True,type=Path);ap.add_argument('--source',required=True,type=Path)
    ap.add_argument('--pair',required=True,type=Path);ap.add_argument('--submit',action='store_true')
    ap.add_argument('--partition', required=True)
    ap.add_argument('--account', required=True)
    ap.add_argument('--gpu-type', default='a6000')
    ap.add_argument('--gpus', type=int, default=2)
    ap.add_argument('--cpus', type=int, default=16)
    ap.add_argument('--memory', default='128G')
    ap.add_argument('--time-limit', default='06:00:00')
    ap.add_argument('--exclude', default='')
    ap.add_argument('--seeds', type=int, nargs='+', choices=(0,1), default=[0,1])
    ap.add_argument('--worker-base', type=int, default=100)
    ap.add_argument('--dispatch-directory', type=Path)
    args=ap.parse_args();root=args.root.resolve();source=args.source.resolve()
    dispatch=args.dispatch_directory.resolve() if args.dispatch_directory else root/'wn2-dispatch'
    dispatch.mkdir(parents=True,exist_ok=True)
    if not 1 <= args.gpus <= 8 or not 0 <= args.worker_base <= 1024-args.gpus:
        ap.error('GPU count or worker range invalid')
    with (dispatch/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for seed in args.seeds:
            partition=args.partition
            ticket=dispatch/f'seed-{seed}.json'
            if ticket.exists():continue
            package=args.pair.resolve()/f'pipeline-{seed}/curriculum/submission-cold'
            package_hash=verify_final_package(package)
            if package_hash is None:continue
            print(json.dumps(dict(seed=seed,package_sha256=package_hash,partition=partition,eligible=True,submit=args.submit)),flush=True)
            if not args.submit:continue
            name=f'paper-wn2-seed-{seed}'
            live=subprocess.check_output(['squeue','--noheader','--me','--name',name,'--format=%i'],text=True).split()
            if live:
                if len(live)!=1 or not re.fullmatch(r'\d+',live[0]):raise ValueError('Ambiguous WN2 inference allocation')
                job=live[0]
            else:
                env=os.environ|dict(PAPER_ROOT=str(root),PAPER_SOURCE=str(source),PAPER_PACKAGE=str(package),
                    PAPER_TRAINING_SEED=str(seed),PAPER_WORKER_BASE=str(args.worker_base),PAPER_GPUS=str(args.gpus))
                job=subprocess.check_output(['sbatch','--parsable',f'--account={args.account}',f'--partition={partition}',
                    f'--gres=gpu:{args.gpu_type}:{args.gpus}',f'--cpus-per-task={args.cpus}',
                    f'--mem={args.memory}',f'--time={args.time_limit}',
                    *(['--exclude='+args.exclude] if args.exclude else []),
                    '--job-name='+name,'--output='+str(dispatch/f'seed-{seed}-%j.log'),
                    str(source/'paper-comparison/wn2.sbatch')],env=env,text=True).strip().split(';')[0]
                if not re.fullmatch(r'\d+',job):raise ValueError('Invalid WN2 job ID')
            temp=ticket.with_suffix('.tmp')
            temp.write_text(json.dumps(dict(job_id=job,package=str(package),package_sha256=package_hash,source=str(source),
                partition=partition,account=args.account,gpus=args.gpus,gpu_type=args.gpu_type,
                excluded_nodes=args.exclude,worker_base=args.worker_base,recovered=bool(live),submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                scope='Final-model 56-member, 15-day, eight-date paper inference; first GPU capacity validation gates remaining workers'),indent=2)+'\n')
            temp.replace(ticket)


if __name__=='__main__':main()
