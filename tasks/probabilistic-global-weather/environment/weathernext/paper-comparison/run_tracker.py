#!/usr/bin/env python3
"""Execute the frozen TE 2.1 commands and verify outputs, including silent errors."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'wn2'))
import numpy as np
import xarray as xr
from training_data import sha256


def run(args):
    protocol=Path(__file__).parent/'tracker-protocol.json'
    p=json.loads(protocol.read_text())
    identity=dict(input_sha256=sha256(args.input),protocol_sha256=sha256(protocol),
        initialization=str(np.datetime64(args.initialization,'ns')),
        binaries_sha256={n:sha256(args.bin/n) for n in ('DetectNodes','StitchNodes')})
    args.output.mkdir(parents=True,exist_ok=True)
    receipt=args.output/'complete.json'
    if receipt.exists():
        previous=json.loads(receipt.read_text())
        if previous.get('status')!='passed' or any(previous.get(k)!=v for k,v in identity.items()):raise ValueError('Changed completed tracker run')
        for name,digest in previous['files'].items():
            if sha256(args.output/name)!=digest:raise ValueError('Corrupt tracking output')
        return
    init=np.datetime64(args.initialization,'ns')
    with xr.open_dataset(args.input) as ds:
        expected=init+np.arange(-240,361,12).astype('timedelta64[h]')
        np.testing.assert_array_equal(ds.time,expected)
        for name in ('msl','thickness','wind','zs'):
            if not np.isfinite(ds[name]).all():raise ValueError(f'Invalid tracker input {name}')
        np.testing.assert_array_equal(ds.lat,np.linspace(-90,90,721))
        np.testing.assert_array_equal(ds.lon,np.arange(1440)/4)
    d=p['detection'];s=p['stitching']
    commands=[('detect',[str(args.bin/'DetectNodes'),'--in_data',str(args.input),'--out',str(args.output/'candidates.txt'),
        '--latname','lat','--lonname','lon','--searchbymin',d['searchbymin'],'--closedcontourcmd',d['closedcontourcmd'],
        '--mergedist',str(d['mergedist_degrees']),'--outputcmd',d['outputcmd']]),
        ('stitch',[str(args.bin/'StitchNodes'),'--in',str(args.output/'candidates.txt'),'--out',str(args.output/'tracks.csv'),
        '--in_fmt',s['in_fmt'],'--range',str(s['range_degrees']),'--mintime',s['mintime'],'--maxgap',s['maxgap'],
        '--threshold',s['threshold'],'--out_file_format','csv'])]
    for label,command in commands:
        log=args.output/f'{label}.log'
        step_receipt=args.output/f'{label}.execution.json'
        product=args.output/('candidates.txt' if label=='detect' else 'tracks.csv')
        step_identity=dict(**identity,command=command,
            candidates_sha256=sha256(args.output/'candidates.txt') if label=='stitch' else None)
        returncode=0
        if step_receipt.exists():
            previous=json.loads(step_receipt.read_text())
            if (previous['identity']!=step_identity or previous['log_sha256']!=sha256(log)
                    or previous['output_sha256']!=sha256(product)):
                raise ValueError('Changed tracker execution; use a fresh output directory')
        else:
            with log.open('w') as stream:
                returncode=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=6*3600).returncode
        # TE 2.1 catches C++ exceptions and can still return zero.
        if returncode or 'EXCEPTION' in log.read_text().upper() or 'Done' not in log.read_text():
            raise RuntimeError(f'Tracker failed; inspect {log}')
        if not product.is_file():raise ValueError('Tracker output missing')
        if not step_receipt.exists():
            temp=step_receipt.with_suffix('.tmp')
            temp.write_text(json.dumps(dict(identity=step_identity,log_sha256=sha256(log),output_sha256=sha256(product)))+'\n')
            temp.replace(step_receipt)
    candidates=args.output/'candidates.txt'
    tracks=args.output/'tracks.csv'
    if not candidates.is_file() or not tracks.is_file():raise ValueError('Tracker output missing')
    with tracks.open() as stream:
        reader=csv.DictReader(stream,skipinitialspace=True)
        if not reader.fieldnames or 'track_id' not in reader.fieldnames or 'lon' not in reader.fieldnames:raise ValueError('Invalid track CSV header')
        rows=list(reader)
    for row in rows:
        if not -90<=float(row['lat'])<=90 or not np.isfinite(float(row['lon'])):raise ValueError('Invalid track coordinates')
    record=dict(status='passed',**identity,track_count=len({r['track_id'] for r in rows}),
        track_points=len(rows),commands=[cmd for _,cmd in commands],
        files={name:sha256(args.output/name) for name in ('candidates.txt','tracks.csv','detect.log','stitch.log')},
        scope='Tracker execution only; named-storm matching and model comparison scored separately')
    temp=receipt.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(receipt)
    print(json.dumps({k:record[k] for k in ('status','track_count','track_points')}),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bin',required=True,type=Path);ap.add_argument('--input',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path);ap.add_argument('--initialization',required=True)
    run(ap.parse_args())


if __name__=='__main__':main()
