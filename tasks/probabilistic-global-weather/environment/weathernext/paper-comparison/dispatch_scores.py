#!/usr/bin/env python3
"""Run every 30 minutes: submit one CPU score job per complete ensemble case."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess


def ready(directory,case,members,protocol_hash):
    for member in range(members):
        name=f'case-{case:03d}-member-{member:03d}'
        path=directory/f'{name}.complete.json'
        if not path.exists(): return False
        record=json.loads(path.read_text())
        if record.get('status')!='passed' or record.get('member')!=member or record.get('protocol_sha256')!=protocol_hash:
            raise ValueError(f'Invalid forecast completion: {path}')
        if not (directory/f'{name}.zarr').is_dir(): raise ValueError(f'Missing forecast: {path}')
    return True


def main():
    import hashlib
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',required=True,type=Path)
    ap.add_argument('--source',type=Path,help='Frozen scorer snapshot; defaults to ROOT/scoring-source')
    ap.add_argument('--python',type=Path,default=Path('/usr/project/xtmp/sc829/weather-track/wn2-pilot/eval-py312/bin/python'))
    ap.add_argument('--submit',action='store_true',help='Otherwise only report eligible score jobs')
    args=ap.parse_args()
    root=args.root.resolve()
    source=args.source.resolve() if args.source else root/'scoring-source'
    if args.submit:
        subprocess.run([str(args.python),'-c','import dask.array, xarray, numpy'],check=True)
    protocol=source/'paper-comparison/weather-protocol.json'
    p=json.loads(protocol.read_text())
    digest=hashlib.sha256(protocol.read_bytes()).hexdigest()
    if json.loads((root/'data/ready.json').read_text())['weather_protocol_sha256']!=digest:
        raise ValueError('Data protocol differs from scorer snapshot')
    dispatch=root/'score-dispatch';dispatch.mkdir(exist_ok=True)
    with (dispatch/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for model in ('gencast','wn2-seed-0','wn2-seed-1'):
            directory=root/'predictions'/model
            for case in range(len(p['evaluation_initializations'])):
                output=root/'scores'/model/f'case-{case:03d}.json'
                ticket=dispatch/f'{model}-{case:03d}.json'
                if output.exists():
                    if json.loads(output.read_text()).get('status')!='passed': raise ValueError('Invalid completed score')
                    continue
                if ticket.exists(): continue  # Failed jobs require investigation, not unbounded automatic retries.
                if not ready(directory,case,p['members'],digest): continue
                if not (root/'rev-climatology/ready.json').exists(): continue
                if not (root/'seeps/climatology/ready.json').exists(): continue
                name=f'paper-score-{model}-{case:03d}'
                # Recover a submission whose process ended before saving the receipt.
                live=subprocess.check_output(['squeue','--noheader','--me','--name',name,'--format=%i'],text=True).split()
                if live:
                    if len(live)!=1 or not re.fullmatch(r'\d+',live[0]): raise ValueError('Ambiguous existing score job')
                    ticket.write_text(json.dumps(dict(job_id=live[0],recovered=True))+'\n')
                    continue
                print(json.dumps(dict(model=model,case=case,eligible=True,submit=args.submit)),flush=True)
                if not args.submit: continue
                output.parent.mkdir(parents=True,exist_ok=True)
                command=[str(args.python),
                    str(source/'paper-comparison/score_marginals.py'),'--predictions',str(directory),
                    '--data',str(root/'data'),'--case',str(case),'--output',str(output),
                    '--rev-climatology',str(root/'rev-climatology'),
                    '--seeps-climatology',str(root/'seeps/climatology')]
                script=dispatch/f'{model}-{case:03d}.sbatch'
                script.write_text('#!/bin/bash\nset -euo pipefail\n'
                    'export OPENBLAS_NUM_THREADS=2\n'
                    'export PYTHONPATH=/usr/project/xtmp/sc829/weather-track/wn2-pilot/weatherbenchX\n'
                    'exec '+shlex.join(command)+'\n')
                result=subprocess.check_output(['sbatch','--parsable','--partition=compsci','--account=users',
                    '--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=32G','--time=12:00:00',
                    '--job-name',name,'--output',str(dispatch/f'{model}-{case:03d}-%j.log'),str(script)],text=True).strip()
                job=result.split(';')[0]
                if not re.fullmatch(r'\d+',job): raise ValueError('Invalid submission receipt')
                temporary=ticket.with_suffix('.tmp')
                temporary.write_text(json.dumps(dict(job_id=job,submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    scope='marginals, 12h/24h precipitation including SEEPS, derived quantities and REV; spherical pooling scored separately'))+'\n')
                temporary.replace(ticket)


if __name__=='__main__': main()
