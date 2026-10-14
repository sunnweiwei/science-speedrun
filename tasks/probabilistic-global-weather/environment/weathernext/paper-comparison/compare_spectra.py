#!/usr/bin/env python3
"""Average spectra across the full frozen subset with strict shared provenance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

MATCH=('members','protocol_sha256','spectra_protocol_sha256','evaluator_sha256','transform','total_wavenumber')


def compare(baselines,candidates,dates,panels):
    def index(reports):
        result={r['case']:r for r in reports}
        if len(result)!=len(reports) or set(result)!=set(dates):raise ValueError('Missing/duplicate spectral dates')
        identity=None
        for r in reports:
            if r['status']!='passed':raise ValueError('Unverified spectra')
            current=({k:r[k] for k in MATCH},{k:v for k,v in r['lineage'].items() if k!='context_sha256'})
            if identity is not None and current!=identity:raise ValueError('Changed spectral model/evaluator lineage')
            identity=current
        return result
    def rows(r):
        output={(v['lead_hours'],v['variable'],v['level_hpa']):v for v in r['rows']}
        if len(output)!=len(r['rows']) or set(output)!=panels:raise ValueError('Wrong spectral panels')
        for v in output.values():
            for name in ('member_mean_power','ensemble_mean_field_power','truth_power'):
                if name=='truth_power' and v[name] is None:
                    if v.get('truth_status')!='missing_truth' or v.get('missing_truth_grid_points',0)<=0:
                        raise ValueError('Unexplained missing truth spectrum')
                    continue
                a=np.asarray(v[name])
                if a.shape!=(len(r['total_wavenumber']),) or not np.isfinite(a).all() or np.any(a<0):
                    raise ValueError('Invalid spectral power')
        return output
    b=index(baselines);c=index(candidates);accum={key:[] for key in panels}
    for date in dates:
        for key in MATCH+('truth_receipt_sha256','truth_values_sha256'):
            if b[date][key]!=c[date][key]:raise ValueError(f'Incompatible spectra: {key}')
        br=rows(b[date]);cr=rows(c[date])
        for key in panels:
            if br[key]['truth_power']!=cr[key]['truth_power']:raise ValueError('Changed spectral truth values')
            if br[key].get('truth_status')!=cr[key].get('truth_status') or br[key].get('missing_truth_grid_points')!=cr[key].get('missing_truth_grid_points'):
                raise ValueError('Mismatched spectral truth availability')
            accum[key].append((date,br[key],cr[key]))
    result=[]
    for key in sorted(panels,key=lambda x:(x[0],x[1],x[2] or 0)):
        all_pairs=accum[key]
        pairs=[(b,c) for _,b,c in all_pairs if b['truth_power'] is not None]
        entry=dict(lead_hours=key[0],variable=key[1],level_hpa=key[2],cases=len(pairs),requested_cases=len(dates),
            excluded_truth_cases=[dict(case=date,missing_grid_points=b['missing_truth_grid_points'])
                                  for date,b,c in all_pairs if b['truth_power'] is None])
        for i,model in enumerate(('baseline','candidate')):
            for name in ('member_mean_power','ensemble_mean_field_power'):
                entry[model+'_'+name]=np.mean([r[i][name] for r in pairs],axis=0).tolist() if pairs else None
        entry['truth_power']=np.mean([r[0]['truth_power'] for r in pairs],axis=0).tolist() if pairs else None
        result.append(entry)
    return dict(status='passed',dates=dates,total_wavenumber=baselines[0]['total_wavenumber'],rows=result,
        normalization=baselines[0]['transform']['normalization'],scope='Equal-date paired spectral means on complete truth fields; all eight case reports required, missing truth support disclosed; no statistical significance claim')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',required=True,type=Path);ap.add_argument('--candidate',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    here=Path(__file__).resolve().parent
    p=json.loads((here/'weather-protocol.json').read_text());s=json.loads((here/'spectra-protocol.json').read_text())
    dates=p['evaluation_initializations']
    files=[[root/f'case-{i:03d}.json' for i in range(len(dates))] for root in (args.baseline,args.candidate)]
    reports=[[json.loads(f.read_text()) for f in group] for group in files]
    for name,file in [('protocol_sha256','weather-protocol.json'),('spectra_protocol_sha256','spectra-protocol.json')]:
        expected=hashlib.sha256((here/file).read_bytes()).hexdigest()
        if any(r[name]!=expected for group in reports for r in group):raise ValueError('Wrong frozen spectral protocol')
    panels={(lead,name,level) for lead in p['lead_hours'] for name,levels in s['variables'].items() for level in (levels or [None])}
    result=compare(*reports,dates,panels)
    result['score_receipts_sha256']=[[hashlib.sha256(f.read_bytes()).hexdigest() for f in group] for group in files]
    result['comparator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(result,allow_nan=False)+'\n');temp.replace(args.output)


if __name__=='__main__':main()
