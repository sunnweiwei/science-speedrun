#!/usr/bin/env python3
"""Submit tracking as members finish, then score each fully tracked ensemble."""
import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


def submit(dispatch,key,commands,execute,cpus=2,memory='8G',prefix='paper-cyclone-',time_limit='06:00:00'):
    ticket=dispatch/f'{key}.json'
    if ticket.exists():return
    print(json.dumps(dict(task=key,eligible=True,submit=execute)),flush=True)
    if not execute:return
    name=prefix+key
    live=subprocess.check_output(['squeue','--noheader','--me','--name',name,'--format=%i'],text=True).split()
    if live:
        if len(live)!=1 or not re.fullmatch(r'\d+',live[0]):raise ValueError('Ambiguous existing cyclone job')
        job=live[0]
    else:
        script=dispatch/f'{key}.sbatch'
        script.write_text('#!/bin/bash\nset -euo pipefail\nexport OPENBLAS_NUM_THREADS=1\n'+
            '\n'.join(shlex.join(command) for command in commands)+'\n')
        job=subprocess.check_output(['sbatch','--parsable','--partition=compsci','--account=users',
            '--nodes=1','--ntasks=1',f'--cpus-per-task={cpus}',f'--mem={memory}',f'--time={time_limit}',
            '--job-name',name,'--output',str(dispatch/f'{key}-%j.log'),str(script)],text=True).strip().split(';')[0]
        if not re.fullmatch(r'\d+',job):raise ValueError('Invalid cyclone job ID')
    temp=ticket.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(job_id=job,recovered=bool(live),commands=commands,
        submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n');temp.replace(ticket)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',required=True,type=Path);ap.add_argument('--source',required=True,type=Path)
    ap.add_argument('--python',required=True,type=Path);ap.add_argument('--bin',required=True,type=Path)
    ap.add_argument('--submit',action='store_true');args=ap.parse_args()
    root=args.root.resolve();source=args.source.resolve()/'paper-comparison'
    protocol=source/'weather-protocol.json';p=json.loads(protocol.read_text())
    digest=hashlib.sha256(protocol.read_bytes()).hexdigest()
    dispatch=root/'cyclone-dispatch';dispatch.mkdir(exist_ok=True)
    with (dispatch/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for model in ('gencast','wn2-seed-0','wn2-seed-1'):
            predictions=root/'predictions'/model;tracks=root/'tracker'/model
            for case,date in enumerate(p['evaluation_initializations']):
                all_tracked=True
                for member in range(p['members']):
                    stem=f'case-{case:03d}-member-{member:03d}';directory=tracks/stem
                    complete=directory/'tracking/complete.json'
                    if complete.exists():
                        if json.loads(complete.read_text()).get('status')!='passed':raise ValueError('Invalid tracking completion')
                        continue
                    all_tracked=False
                    manifest=predictions/f'{stem}.complete.json'
                    if not manifest.exists():continue
                    r=json.loads(manifest.read_text())
                    if r['status']!='passed' or r['protocol_sha256']!=digest or r['member']!=member:
                        raise ValueError('Invalid forecast completion')
                    output=directory/'input.nc'
                    commands=[[str(args.python),str(source/'export_tracker_forecast.py'),'--predictions',str(predictions),
                        '--history',str(root/'tracker/truth'),'--case',str(case),'--member',str(member),'--output',str(output)],
                        [str(args.python),str(source/'run_tracker.py'),'--bin',str(args.bin),'--input',str(output),
                         '--initialization',date,'--output',str(directory/'tracking')]]
                    submit(dispatch,f'track-{model}-{case:03d}-{member:03d}',commands,args.submit)
                if not all_tracked:continue
                output=root/'cyclone-scores'/model/f'case-{case:03d}.json'
                if output.exists():
                    if json.loads(output.read_text()).get('status')!='passed':raise ValueError('Invalid cyclone score')
                    continue
                command=[str(args.python),str(source/'score_cyclones.py'),'--predictions',str(predictions),
                    '--tracks',str(tracks),'--truth-root',str(root/'tracker'),'--case',str(case),'--output',str(output)]
                submit(dispatch,f'score-{model}-{case:03d}',[command],args.submit)


if __name__=='__main__':main()
