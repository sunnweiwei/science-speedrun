#!/usr/bin/env python3
"""Submit independent spherical pooling scores when all case members are ready."""
import argparse
import fcntl
import hashlib
import json
import subprocess
from pathlib import Path
from dispatch_scores import ready
from dispatch_cyclones import submit


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',required=True,type=Path)
    ap.add_argument('--source',required=True,type=Path)
    ap.add_argument('--python',required=True,type=Path)
    ap.add_argument('--submit',action='store_true')
    args=ap.parse_args();root=args.root.resolve();source=args.source.resolve()/'paper-comparison'
    if args.submit:
        subprocess.run([str(args.python),'-c','import dask.array, xarray, numpy'],check=True)
    protocol=source/'weather-protocol.json';p=json.loads(protocol.read_text())
    digest=hashlib.sha256(protocol.read_bytes()).hexdigest()
    if json.loads((root/'data/ready.json').read_text())['weather_protocol_sha256']!=digest:
        raise ValueError('Data protocol differs from pooling snapshot')
    geometry=root/'pooling'
    for stride in (1,4):
        for k in range(2,8 if stride==1 else 7):
            for suffix in ('.npz','-metadata.npz'):
                if not (geometry/f'stride-{stride}-mesh-{k}{suffix}').is_file():
                    raise ValueError('Missing spherical pooling geometry')
    dispatch=root/'pooling-dispatch';dispatch.mkdir(exist_ok=True)
    with (dispatch/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for model in ('gencast','wn2-seed-0','wn2-seed-1'):
            predictions=root/'predictions'/model
            for case in range(len(p['evaluation_initializations'])):
                output=root/'pooling-scores'/model/f'case-{case:03d}.json'
                if output.exists():
                    if json.loads(output.read_text()).get('status')!='passed':
                        raise ValueError('Invalid completed pooling score')
                    continue
                if not ready(predictions,case,p['members'],digest):continue
                command=['env',f'PYTHONPATH={root.parent}/weatherbenchX',str(args.python),
                    str(source/'score_marginals.py'),'--predictions',str(predictions),
                    '--data',str(root/'data'),'--case',str(case),'--pooling',str(geometry),
                    '--pooling-only','--output',str(output)]
                submit(dispatch,f'{model}-{case:03d}',[command],args.submit,
                    cpus=8,memory='64G',prefix='paper-pooling-',time_limit='12:00:00')


if __name__=='__main__':main()
