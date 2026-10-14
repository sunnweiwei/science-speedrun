#!/usr/bin/env python3
"""Bounded CPU scoring and complete-date comparisons for persistence/dry references."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
from dispatch_cyclones import submit
from dispatch_comparisons import ready


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source','python'):
        ap.add_argument('--'+name,required=True,type=Path)
    ap.add_argument('--reference-scorer',type=Path)
    ap.add_argument('--max-active',type=int,default=2)
    ap.add_argument('--memory',default='32G',help='CPU job memory; larger contexts need more than 16G')
    ap.add_argument('--candidates',nargs='+',default=['wn2-seed-0','gencast','wn2-seed-1'])
    ap.add_argument('--submit',action='store_true')
    a=ap.parse_args()
    if a.max_active<1:ap.error('--max-active must be positive')
    root=a.root.resolve();source=a.source.resolve()/'paper-comparison'
    protocol=source/'weather-protocol.json';p=json.loads(protocol.read_text());digest=hashlib.sha256(protocol.read_bytes()).hexdigest()
    if json.loads((root/'data/ready.json').read_text())['weather_protocol_sha256']!=digest:
        raise ValueError('Baseline and data protocols differ')
    dispatch=root/'simple-baseline-dispatch';dispatch.mkdir(exist_ok=True)
    with (dispatch/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        live=subprocess.check_output(['squeue','--noheader','--me','--format=%j'],text=True).split()
        active=sum(name.startswith('paper-simple-') for name in live)
        # Core dates first; no baseline forecast arrays or GPU allocation.
        for endpoint in ('scores','pooling-scores'):
            if endpoint == 'pooling-scores' and not ready(root/'scores'/'persistence-dry',len(p['evaluation_initializations']),digest):
                break
            for case in range(len(p['evaluation_initializations'])):
                key=f'{endpoint}-{case:03d}';output=root/endpoint/'persistence-dry'/f'case-{case:03d}.json'
                if output.exists():
                    r=json.loads(output.read_text())
                    if r.get('status')!='passed' or r.get('protocol_sha256')!=digest or r.get('baseline')!='persistence-dry':
                        raise ValueError('Invalid baseline report')
                    continue
                if (dispatch/f'{key}.json').exists():continue
                if active>=a.max_active:continue
                if not (root/'data/inputs'/f'inputs-{case:03d}.nc').exists():continue
                command=['env',f'PYTHONPATH={root.parent}/weatherbenchX',str(a.python),str(source/'score_marginals.py'),
                    '--baseline','persistence-dry','--data',str(root/'data'),'--case',str(case),'--output',str(output)]
                if a.reference_scorer:command+=['--reference-scorer',str(a.reference_scorer.resolve())]
                if endpoint=='scores':
                    if not all((root/d/'ready.json').exists() for d in ['rev-climatology','seeps/climatology']):continue
                    command+=['--rev-climatology',str(root/'rev-climatology'),'--seeps-climatology',str(root/'seeps/climatology')]
                else:command+=['--pooling',str(root/'pooling'),'--pooling-only']
                submit(dispatch,key,[command],a.submit,cpus=4,memory=a.memory,prefix='paper-simple-',time_limit='12:00:00')
                active+=1
            baseline=root/endpoint/'persistence-dry'
            if not ready(baseline,len(p['evaluation_initializations']),digest):continue
            for model in a.candidates:
                candidate=root/endpoint/model
                if not ready(candidate,len(p['evaluation_initializations']),digest):continue
                key=f'compare-{endpoint}-{model}'
                output=root/'comparisons'/f'{endpoint}-{model}-vs-persistence-dry.json'
                if output.exists():continue
                if (dispatch/f'{key}.json').exists() or active>=a.max_active:continue
                command=[str(a.python),str(source/'compare.py'),'--baseline',str(baseline),'--candidate',str(candidate),'--output',str(output)]
                submit(dispatch,key,[command],a.submit,cpus=2,memory='8G',prefix='paper-simple-')
                active+=1


if __name__=='__main__':main()
