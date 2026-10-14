#!/usr/bin/env python3
"""Score one fully tracked ensemble, preserving paired-comparison statistics."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'wn2'))
from training_data import sha256
from cyclone_metrics import read_tracks,position_rows,cell_confusion
from prepare_named_cyclones import IBTRACS_SHA256


def verified_tracks(directory,protocol_hash,initialization):
    receipt=directory/'complete.json'
    r=json.loads(receipt.read_text())
    if (r['status']!='passed' or r['protocol_sha256']!=protocol_hash
            or np.datetime64(r['initialization'])!=np.datetime64(initialization)):
        raise ValueError('Tracker execution differs from protocol/case')
    for name,digest in r['files'].items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path)!=digest:
            raise ValueError('Corrupt tracker output')
    tracks=read_tracks(directory/'tracks.csv')
    if len(tracks)!=r['track_count'] or sum(map(len,tracks.values()))!=r['track_points']:
        raise ValueError('Track inventory mismatch')
    return tracks,r,sha256(receipt)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions',type=Path,required=True)
    ap.add_argument('--tracks',type=Path,required=True,help='Model directory of case-XXX-member-YYY directories')
    ap.add_argument('--truth-root',type=Path,required=True,help='Contains named-truth.json, truth/, observed-case-XXX/')
    ap.add_argument('--case',type=int,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();here=Path(__file__).resolve().parent
    protocol=here/'weather-protocol.json';p=json.loads(protocol.read_text());digest=sha256(protocol)
    tracker_hash=sha256(here/'tracker-protocol.json')
    date=p['evaluation_initializations'][args.case]
    named_path=args.truth_root/'named-truth.json';named=json.loads(named_path.read_text())
    if named['status']!='passed' or named['protocol_sha256']!=digest or named['ibtracs_sha256']!=IBTRACS_SHA256:
        raise ValueError('Wrong named-storm truth')
    case=named['cases'][args.case]
    if np.datetime64(case['initialization'])!=np.datetime64(date):raise ValueError('Wrong named-storm case')
    truth={sid:{np.datetime64(t,'ns'):point for t,point in storm['points'].items()} for sid,storm in case['storms'].items()}
    observed,observed_record,observed_hash=verified_tracks(args.truth_root/f'observed-case-{args.case:03d}',tracker_hash,date)
    history_receipt=args.truth_root/'truth'/f'case-{args.case:03d}.json'
    history=json.loads(history_receipt.read_text())
    if history['status']!='passed' or observed_record['input_sha256']!=history['sha256'] or history['protocol_sha256']!=digest:
        raise ValueError('Wrong HRES tracker truth')
    members=[];receipts=[];prediction_receipts=[];lineage=None;exporter=None
    for member in range(p['members']):
        stem=f'case-{args.case:03d}-member-{member:03d}'
        manifest=args.predictions/f'{stem}.complete.json';prediction=json.loads(manifest.read_text())
        if (prediction['status']!='passed' or prediction['protocol_sha256']!=digest
                or prediction['member']!=member or np.datetime64(prediction['initialization'])!=np.datetime64(date)):
            raise ValueError('Wrong forecast member identity')
        identity={k:prediction.get(k) for k in ('assets_lock_sha256','package_sha256','runner_sha256','noise_grid_cache_sha256','seed','context_sha256')}
        if lineage is not None and lineage!=identity:raise ValueError('Mixed forecast lineage')
        lineage=identity
        directory=args.tracks/stem
        exported=json.loads((directory/'input.json').read_text());e=exported['identity']
        if (exported['status']!='passed' or e['prediction_manifest_sha256']!=sha256(manifest)
                or e['history_receipt_sha256']!=sha256(history_receipt) or e['protocol_sha256']!=digest
                or e['member']!=member or e['case']!=args.case):
            raise ValueError('Tracker input not bound to forecast/shared history')
        transform={k:e[k] for k in ('exporter_sha256','field_transform_sha256')}
        if exporter is not None and exporter!=transform:raise ValueError('Mixed tracker export implementations')
        exporter=transform
        tracks,r,receipt_hash=verified_tracks(directory/'tracking',tracker_hash,date)
        if r['input_sha256']!=exported['sha256'] or r['binaries_sha256']!=observed_record['binaries_sha256']:
            raise ValueError('Tracker input or executable changed')
        members.append(tracks);receipts.append(receipt_hash);prediction_receipts.append(sha256(manifest))
    leads=[lead for lead in p['lead_hours'] if lead<=216]
    rows=position_rows(members,truth,date,leads)
    events=[dict(lead_hours=lead,confusion=cell_confusion(members,observed,np.datetime64(date)+np.timedelta64(lead,'h')).tolist()) for lead in leads]
    report=dict(status='passed',case=date,members=p['members'],protocol_sha256=digest,
        tracker_protocol_sha256=tracker_hash,score_protocol_sha256=sha256(here/'cyclone-score-protocol.json'),
        evaluator_sha256=sha256(Path(__file__)),metrics_sha256=sha256(here/'cyclone_metrics.py'),
        rev_sha256=sha256(here/'rev.py'),named_truth_sha256=sha256(named_path),
        observed_tracks_receipt_sha256=observed_hash,binaries_sha256=observed_record['binaries_sha256'],
        export_implementation=exporter,lineage=lineage,tracker_receipts_sha256=receipts,
        prediction_receipts_sha256=prediction_receipts,named_storms=len(truth),position_rows=rows,rev_rows=events,
        scope='Per-case statistics; paired position support and cross-date REV aggregation required before reporting')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(args.output)
    print(json.dumps(dict(status='passed',case=date,position_pairs=len(rows),rev_leads=len(events))),flush=True)


if __name__=='__main__':main()
