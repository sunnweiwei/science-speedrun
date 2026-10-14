#!/usr/bin/env python3
"""Compare completed cold/warm native forecasts; needs no GPU allocation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from training_data import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();root=args.run
    reports=[json.loads((root/phase/'replay.json').read_text()) for phase in ('cold','warm')]
    if any(r['status']!='passed' for r in reports):raise ValueError('Both independent replays must have passed')
    for key in ('input_sha256','protocol_sha256','case_index'):
        if reports[0][key]!=reports[1][key]:raise ValueError('Replay inputs/protocol differ')
    paths=[root/phase/'untrusted-output'/r['forecast_store'] for phase,r in zip(('cold','warm'),reports)]
    names=[sorted(str(p.relative_to(d)) for p in d.rglob('*') if p.is_file()) for d in paths]
    if not names[0] or names[0]!=names[1]:raise ValueError('Missing or different forecast files')
    def compare(name):
        a,b=(sha256(d/name) for d in paths)
        if a!=b:raise ValueError(f'Fresh-process forecast differs: {name}')
        return name,a
    with ThreadPoolExecutor(max_workers=4) as pool:
        files=dict(pool.map(compare,names[0]))
    inventory=root/'exact-file-hashes.json'
    inventory.write_text(json.dumps(files,sort_keys=True)+'\n')
    request=json.loads((root/'cold/public/request.json').read_text())
    result=dict(status='passed',members=request['members'],lead_hours=request['lead_hours'],
        exact_store_files=len(files),file_hashes_sha256=sha256(inventory),
        cold_seconds=reports[0]['seconds'],warm_seconds=reports[1]['seconds'],
        replay_reports_sha256={phase:sha256(root/phase/'replay.json') for phase in ('cold','warm')},
        scope='Complete requested native forecasts, fresh IDs/processes and packaged compiler cache; execution fixture weights/statistics, not research-trained starter')
    (root/'exact-replay.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
