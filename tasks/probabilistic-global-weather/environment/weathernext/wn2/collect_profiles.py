#!/usr/bin/env python3
"""Require complete, consistent measured native stage profiles before planning."""
import argparse
import json
from pathlib import Path

from plan_schedule import STAGES,plan
from training_data import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();profiles={};provenance=[];identities=[]
    for path in args.reports:
        report=json.loads(path.read_text())
        if report['status']!='passed' or not report['profile'] or len(report['updates'])!=2:
            raise ValueError('Each stage requires a passing two-update timing profile')
        name=report['stage']['name']
        if name in profiles:raise ValueError('Duplicate profile stage')
        for key,value in next(s for s in STAGES if s['name']==name).items():
            if report['stage'][key]!=value:raise ValueError('Profile stage differs from native curriculum')
        if any(not u.get('includes_checkpoint_io') for u in report['updates']):raise ValueError('Checkpoint I/O missing from update timings')
        if not report.get('includes_final_model_export'):raise ValueError('Final model export missing from timing overhead')
        profiles[name]={key:report[key] for key in ('status','devices','effective_batch','includes_io_and_communication','update_seconds','overhead_seconds')}
        image_hash=(path.parent.parent/'image.sha256').read_text().split()[0]
        identities.append((image_hash,json.dumps(report['runtime'],sort_keys=True),report['protocol_sha256'],report['statistics_manifest_sha256'],
                           json.dumps({k:report['source_sha256'][k] for k in
                            ('train_stage.py','native_model.py','training_data.py','wn2_validation.py','wn2_distributed.py','plan_schedule.py')},sort_keys=True)))
        autotune=path.parent.parent/'autotune.pb'
        if not autotune.is_file() or autotune.stat().st_size==0:raise ValueError('Frozen autotuning results missing')
        provenance.append(dict(path=str(path.resolve()),sha256=sha256(path),image_sha256=image_hash,
                               autotune_sha256=sha256(autotune)))
    if len(set(identities))!=1:raise ValueError('Data, normalization or training implementation changed across profiles')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'profiles.json').write_text(json.dumps(profiles,indent=2)+'\n')
    (args.output/'profile-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    schedule=plan(profiles)
    schedule.update(profile_sha256=sha256(args.output/'profiles.json'),evaluation_reserve_seconds=21600,
                    checkpoint_reserve_seconds=1800,total_wall_cap_seconds=84600)
    (args.output/'schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')


if __name__=='__main__':main()
