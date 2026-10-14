#!/usr/bin/env python3
"""Compare all frozen dates using common storm/lead support and pooled REV."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from cyclone_metrics import paired_position_rows
from rev import potential_value

MATCH=('members','protocol_sha256','tracker_protocol_sha256','score_protocol_sha256',
       'evaluator_sha256','metrics_sha256','rev_sha256','named_truth_sha256',
       'binaries_sha256','export_implementation')


def compare(baselines,candidates,dates):
    def index(reports):
        result={r['case']:r for r in reports}
        if len(result)!=len(reports) or set(result)!=set(dates):raise ValueError('Incomplete/duplicate cyclone dates')
        identity=None
        for r in reports:
            if r['status']!='passed':raise ValueError('Unverified cyclone score')
            current=({k:r[k] for k in MATCH},{k:v for k,v in r['lineage'].items() if k!='context_sha256'})
            if identity is not None and current!=identity:raise ValueError('Cyclone model/evaluator lineage changed across dates')
            identity=current
        return result
    baseline=index(baselines);candidate=index(candidates)
    positions=defaultdict(list);events=defaultdict(list)
    for date in dates:
        b=baseline[date];c=candidate[date]
        for key in MATCH+('observed_tracks_receipt_sha256',):
            if b[key]!=c[key]:raise ValueError(f'Incompatible cyclone comparison: {key}')
        for row in paired_position_rows(b['position_rows'],c['position_rows']):
            positions[row['lead_hours']].append(dict(case=date,**row))
        def event_index(report):
            rows={r['lead_hours']:r['confusion'] for r in report['rev_rows']}
            if len(rows)!=len(report['rev_rows']) or set(rows)!=set(range(12,217,12)):
                raise ValueError('Missing or duplicate cyclone REV lead')
            if any(np.asarray(matrix).shape!=(report['members']+2,4) for matrix in rows.values()):
                raise ValueError('Wrong cyclone ensemble decision thresholds')
            return rows
        br=event_index(b);cr=event_index(c)
        for lead in br:events[lead].append((br[lead],cr[lead]))
    position_report=[]
    for lead in range(12,217,12):
        rows=positions[lead]
        position_report.append(dict(lead_hours=lead,paired_storm_initializations=len(rows),
            baseline_mean_error_km=float(np.mean([r['baseline_km'] for r in rows])) if rows else None,
            candidate_mean_error_km=float(np.mean([r['candidate_km'] for r in rows])) if rows else None,
            paired_rows=rows))
    costs=np.geomspace(1e-6,.5,101);rev=[]
    clean=lambda values:[float(v) if np.isfinite(v) else None for v in values]
    for lead,pairs in sorted(events.items()):
        b=potential_value(np.mean([r[0] for r in pairs],axis=0),costs)
        c=potential_value(np.mean([r[1] for r in pairs],axis=0),costs)
        rev.append(dict(lead_hours=lead,cases=len(pairs),cost_loss_ratios=costs.tolist(),
            baseline_potential_rev=clean(b),candidate_potential_rev=clean(c),candidate_minus_baseline=clean(c-b)))
    return dict(status='passed',dates=dates,position=position_report,rev=rev,
        scope='Common-12-hour cyclone comparison; no full-year significance claim')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',required=True,type=Path);ap.add_argument('--candidate',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    import hashlib
    here=Path(__file__).resolve().parent
    protocol=json.loads((here/'weather-protocol.json').read_text());dates=protocol['evaluation_initializations']
    files=[[root/f'case-{i:03d}.json' for i in range(len(dates))] for root in (args.baseline,args.candidate)]
    reports=[[json.loads(f.read_text()) for f in group] for group in files]
    expected=hashlib.sha256((here/'weather-protocol.json').read_bytes()).hexdigest()
    if any(r['protocol_sha256']!=expected for group in reports for r in group):raise ValueError('Wrong frozen weather protocol')
    result=compare(*reports,dates)
    result['score_receipts_sha256']=[[hashlib.sha256(f.read_bytes()).hexdigest() for f in group] for group in files]
    result['comparator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temp.replace(args.output)


if __name__=='__main__':main()
