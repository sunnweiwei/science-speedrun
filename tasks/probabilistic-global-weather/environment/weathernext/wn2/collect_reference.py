#!/usr/bin/env python3
"""Aggregate independently replayed reference cases with exact package identity."""
import argparse
import json
from pathlib import Path

import xarray as xr

from merge_scores import merge, reward
from training_data import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('package','runs','output','protocol','precipitation-protocol','scales'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    identity={str(f.relative_to(args.package)):sha256(f) for f in args.package.rglob('*') if f.is_file()}
    selected=[]
    for path in args.runs.glob('*/verification/replay.json'):
        record=json.loads(path.read_text())
        if (record.get('status')=='passed' and record.get('relocated_files')==identity
                and record.get('protocol_sha256')==sha256(args.protocol)
                and all((path.parent/f'{name}-scores.json').is_file() for name in ('weather','precipitation'))):
            selected.append(path.parent)
    results={}
    for name,protocol_path in (('weather',args.protocol),('precipitation',args.precipitation_protocol)):
        protocol=json.loads(protocol_path.read_text())
        shards=[json.loads((directory/f'{name}-scores.json').read_text()) for directory in selected]
        report=merge(shards,protocol,sha256(protocol_path))
        with xr.open_dataset(args.scales) as scales:
            report['reward_metrics']=reward(report,scales)
        report['reward_scales_sha256']=sha256(args.scales)
        report['independent_replay_provenance']=[dict(path=str(d.resolve()),
            replay_sha256=sha256(d/'replay.json'),scores_sha256=sha256(d/f'{name}-scores.json')) for d in selected]
        results[name]=report
    if args.output.exists():raise ValueError('Reference aggregation output must be new')
    args.output.mkdir(parents=True)
    for name,report in results.items():
        (args.output/f'{name}-scores.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    metrics={name:r['reward_metrics']['reward'] for name,r in results.items()}
    (args.output/'reward.json').write_text(json.dumps(dict(reward=metrics['weather'],precipitation_reward=metrics['precipitation']),indent=2)+'\n')
    (args.output/'status.json').write_text(json.dumps(dict(status='passed',cases=len(selected),
        package_files_sha256=identity,scope='Complete independent reference replay and requested score panels'),indent=2)+'\n')


if __name__=='__main__':main()
