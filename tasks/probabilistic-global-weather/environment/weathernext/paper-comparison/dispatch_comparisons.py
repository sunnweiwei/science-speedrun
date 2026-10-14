#!/usr/bin/env python3
"""Aggregate complete fixed-date score sets on the half-hour sweep."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
from dispatch_cyclones import submit

ENDPOINTS={'scores':'compare.py','pooling-scores':'compare.py',
           'spectra':'compare_spectra.py','cyclone-scores':'compare_cyclones.py'}
PAIRS=(('gencast','wn2-seed-0'),('gencast','wn2-seed-1'),('wn2-seed-0','wn2-seed-1'))


def ready(directory,cases,protocol_hash):
    for case in range(cases):
        path=directory/f'case-{case:03d}.json'
        if not path.is_file():return False
        report=json.loads(path.read_text())
        if report.get('status')!='passed' or report.get('protocol_sha256')!=protocol_hash:
            raise ValueError(f'Invalid score report: {path}')
    return True


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',required=True,type=Path)
    ap.add_argument('--source',required=True,type=Path)
    ap.add_argument('--python',required=True,type=Path)
    ap.add_argument('--submit',action='store_true')
    args=ap.parse_args();root=args.root.resolve();source=args.source.resolve()/'paper-comparison'
    protocol=source/'weather-protocol.json';p=json.loads(protocol.read_text())
    digest=hashlib.sha256(protocol.read_bytes()).hexdigest()
    dispatch=root/'comparison-dispatch';dispatch.mkdir(exist_ok=True)
    with (dispatch/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for endpoint,script in ENDPOINTS.items():
            for baseline,candidate in PAIRS:
                key=f'{endpoint}-{candidate}-vs-{baseline}'
                output=root/'comparisons'/f'{key}.json'
                if output.exists():
                    if json.loads(output.read_text()).get('status')!='passed':
                        raise ValueError(f'Invalid completed comparison: {output}')
                    continue
                directories=[root/endpoint/model for model in (baseline,candidate)]
                if not all(ready(d,len(p['evaluation_initializations']),digest) for d in directories):continue
                command=[str(args.python),str(source/script),'--baseline',str(directories[0]),
                         '--candidate',str(directories[1]),'--output',str(output)]
                submit(dispatch,key,[command],args.submit,prefix='paper-compare-')


if __name__=='__main__':main()
