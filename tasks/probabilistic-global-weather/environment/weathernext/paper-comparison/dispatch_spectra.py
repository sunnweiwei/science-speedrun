#!/usr/bin/env python3
"""Submit full-case CPU spectral scoring on the half-hour completion sweep."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
from dispatch_scores import ready
from dispatch_cyclones import submit


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',required=True,type=Path);ap.add_argument('--source',required=True,type=Path)
    ap.add_argument('--image',required=True,type=Path);ap.add_argument('--submit',action='store_true')
    args=ap.parse_args();root=args.root.resolve();source=args.source.resolve()/'paper-comparison'
    protocol=source/'weather-protocol.json';p=json.loads(protocol.read_text())
    digest=hashlib.sha256(protocol.read_bytes()).hexdigest()
    dispatch=root/'spectra-dispatch';dispatch.mkdir(exist_ok=True)
    with (dispatch/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for model in ('gencast','wn2-seed-0','wn2-seed-1'):
            predictions=root/'predictions'/model
            for case in range(len(p['evaluation_initializations'])):
                output=root/'spectra'/model/f'case-{case:03d}.json'
                if output.exists():
                    if json.loads(output.read_text()).get('status')!='passed':raise ValueError('Invalid completed spectra')
                    continue
                truth=root/'spectra/truth'/f'case-{case:03d}.json'
                if not truth.exists() or not ready(predictions,case,p['members'],digest):continue
                command=['apptainer','exec','--cleanenv','--bind',str(root.parent)+':'+str(root.parent),
                    '--env','JAX_PLATFORMS=cpu,OPENBLAS_NUM_THREADS=8',str(args.image),'/opt/weather-python/bin/python',
                    str(source/'score_spectra.py'),'--predictions',str(predictions),'--truth',str(truth),
                    '--case',str(case),'--output',str(output)]
                submit(dispatch,f'{model}-{case:03d}',[command],args.submit,cpus=8,memory='64G',prefix='paper-spectra-',time_limit='12:00:00')


if __name__=='__main__':main()
