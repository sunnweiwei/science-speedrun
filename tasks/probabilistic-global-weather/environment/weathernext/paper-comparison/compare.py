#!/usr/bin/env python3
"""Paired subset comparison from recomputed scores, never paper-table numbers."""
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
from rev import potential_value

MATCH = ('protocol_sha256','precipitation_protocol_sha256','evaluator_sha256',
         'panel_evaluator_sha256','weatherbenchx_revision','estimator','members',
         'pooling_geometry_sha256','pooling_evaluator_sha256','precipitation_evaluator_sha256','truth_values_sha256',
         'rev_evaluator_sha256','rev_climatology_source_sha256','rev_threshold_receipts',
         'seeps_evaluator_sha256','seeps_protocol_sha256','seeps_climatology_source_sha256','seeps_climatology_receipts')


def indexed(report, section):
    result={}
    for row in report[section]:
        key=(row['case'],row['variable'],row['level_hpa'],row['lead_hours'],
             row.get('pooling','native'),row.get('mesh_refinement'))
        if key in result: raise ValueError('Duplicate score panel')
        result[key]=row
    return result


def compare(baselines,candidates,expected_dates):
    if len(baselines)!=len(expected_dates) or len(candidates)!=len(expected_dates):
        raise ValueError('Missing cases')
    groups=defaultdict(list)
    # Context differs by date, but model, RNG and evaluator lineage must not.
    for reports in (baselines,candidates):
        identity = None
        for report in reports:
            current = ({k:v for k,v in report['lineage'].items() if k!='context_sha256'},
                       {k:report[k] for k in MATCH if k not in ('truth_values_sha256','rev_threshold_receipts','seeps_climatology_receipts')})
            if identity is not None and current!=identity:
                raise ValueError('Model or evaluation lineage changed across dates')
            identity=current
    seen=set()
    for baseline,candidate in zip(baselines,candidates):
        if baseline['status']!='passed' or candidate['status']!='passed': raise ValueError('Unverified score')
        for key in MATCH:
            # A new baseline adapter may score an already archived model run.
            # All metric implementations, truth, protocols and settings still match.
            if (key == 'evaluator_sha256' and baseline.get('baseline') == 'persistence-dry'
                    and baseline.get('baseline_reference_scorer_sha256') == candidate.get(key)):
                continue
            if key not in baseline or key not in candidate or baseline[key]!=candidate[key]:
                raise ValueError(f'Incompatible score provenance: {key}')
        dates={r['case'] for section in ('per_case','pooled_per_case') for r in baseline[section]}
        if len(dates)!=1 or seen.intersection(dates): raise ValueError('Wrong/duplicate case')
        seen.update(dates)
        for section in ('per_case','pooled_per_case'):
            b,c=indexed(baseline,section),indexed(candidate,section)
            if b.keys()!=c.keys(): raise ValueError('Mismatched paired panels')
            for key in b:
                groups[key[1:]].append((key[0],b[key],c[key]))
    if seen!=set(expected_dates): raise ValueError('Dates differ from frozen subset')
    output=[]
    for (variable,level,lead,pooling,mesh),pairs in groups.items():
        if len(pairs)!=len(expected_dates): raise ValueError('Incomplete score panel across dates')
        n=len(pairs)
        b=sum(x[1]['crps'] for x in pairs)/n
        c=sum(x[2]['crps'] for x in pairs)/n
        bm=sum(x[1]['mse'] for x in pairs)/n
        cm=sum(x[2]['mse'] for x in pairs)/n
        output.append(dict(variable=variable,level_hpa=level,lead_hours=lead,pooling=pooling,mesh_refinement=mesh,
            cases=n,baseline_crps=b,candidate_crps=c,crps_skill_vs_baseline=1-c/b if b>0 else None,
            baseline_rmse=math.sqrt(bm),candidate_rmse=math.sqrt(cm),
            rmse_skill_vs_baseline=1-math.sqrt(cm/bm) if bm>0 else None,
            paired_crps_differences=[dict(case=date,candidate_minus_baseline=cr['crps']-br['crps']) for date,br,cr in pairs]))
    return output


def compare_rev(baselines,candidates,expected_dates):
    groups=defaultdict(list)
    for baseline,candidate in zip(baselines,candidates):
        def rows(report):
            result={}
            for row in report['rev_per_case']:
                key=(row['case'],row['variable'],row['lead_hours'],row['quantile'],row['direction'])
                if key in result:raise ValueError('Duplicate REV panel')
                result[key]=np.asarray(row['confusion'],dtype=float)
            return result
        b,c=rows(baseline),rows(candidate)
        if b.keys()!=c.keys():raise ValueError('Mismatched paired REV panels')
        for key in b:groups[key[1:]].append((key[0],b[key],c[key]))
    costs=np.geomspace(.0001,.5,81)
    output=[]
    for (variable,lead,quantile,direction),pairs in groups.items():
        if len(pairs)!=len(expected_dates) or {p[0] for p in pairs}!=set(expected_dates):
            raise ValueError('Incomplete REV date coverage')
        b=potential_value(np.mean([p[1] for p in pairs],axis=0),costs)
        c=potential_value(np.mean([p[2] for p in pairs],axis=0),costs)
        clean=lambda values:[float(v) if np.isfinite(v) else None for v in values]
        output.append(dict(variable=variable,lead_hours=lead,quantile=quantile,direction=direction,
            cases=len(pairs),cost_loss_ratios=costs.tolist(),baseline_potential_rev=clean(b),
            candidate_potential_rev=clean(c),candidate_minus_baseline=clean(c-b)))
    return output


def compare_seeps(baselines,candidates,dates):
    if all(not r.get('seeps_per_case') for r in baselines+candidates):return []
    expected={(hours,lead,dry) for hours in (12,24) for lead in range(hours,361,12) for dry in (.1,.25)}
    groups=defaultdict(list)
    for date,b,c in zip(dates,baselines,candidates):
        def index(report):
            rows=report['seeps_per_case']
            if any(r['case']!=date for r in rows):raise ValueError('Wrong SEEPS date')
            values={(r['accumulation_hours'],r['lead_hours'],r['dry_threshold_mm']):r for r in rows}
            if len(values)!=len(rows) or values.keys()!=expected:raise ValueError('Incomplete SEEPS panels')
            return values
        br=index(b);cr=index(c)
        for key in expected:
            if br[key]['valid_area_fraction']!=cr[key]['valid_area_fraction']:
                raise ValueError('Different SEEPS verification support')
            if (br[key]['seeps'] is None)!=(cr[key]['seeps'] is None):raise ValueError('Different SEEPS missing support')
            groups[key].append((date,br[key],cr[key]))
    result=[]
    for (hours,lead,dry),pairs in sorted(groups.items()):
        if len(pairs)!=len(dates):raise ValueError('Missing SEEPS dates')
        valid=[(date,b,c) for date,b,c in pairs if b['seeps'] is not None]
        b=float(np.mean([b['seeps'] for _,b,c in valid])) if valid else None
        c=float(np.mean([c['seeps'] for _,b,c in valid])) if valid else None
        result.append(dict(accumulation_hours=hours,lead_hours=lead,dry_threshold_mm=dry,cases=len(valid),
            baseline_seeps=b,candidate_seeps=c,baseline_one_minus_seeps=1-b if b is not None else None,
            candidate_one_minus_seeps=1-c if c is not None else None,
            paired_differences=[dict(case=date,candidate_minus_baseline=c['seeps']-b['seeps'],
                valid_area_fraction=b['valid_area_fraction']) for date,b,c in valid]))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',required=True,type=Path)
    ap.add_argument('--candidate',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args()
    protocol=json.loads((Path(__file__).parent/'weather-protocol.json').read_text())
    dates=protocol['evaluation_initializations']
    import hashlib
    files=[[directory/f'case-{i:03d}.json' for i in range(len(dates))]
           for directory in (args.baseline,args.candidate)]
    baseline,candidate=[[json.loads(path.read_text()) for path in group] for group in files]
    expected=hashlib.sha256((Path(__file__).parent/'weather-protocol.json').read_bytes()).hexdigest()
    if any(r['protocol_sha256']!=expected for r in baseline+candidate):
        raise ValueError('Wrong frozen weather protocol')
    panels=compare(baseline,candidate,dates)
    report=dict(status='passed',scope='paired 2022 subset; not a reproduction of published 2023 scores',
        dates=dates,members=protocol['members'],baseline_lineage=baseline[0]['lineage'],
        score_receipts_sha256=[[hashlib.sha256(path.read_bytes()).hexdigest() for path in group] for group in files],
        comparator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        candidate_lineage=candidate[0]['lineage'],panels=panels,
        rev_panels=compare_rev(baseline,candidate,dates),
        seeps_panels=compare_seeps(baseline,candidate,dates),
        statistical_significance='Not asserted from eight initialization dates',
        full_paper_evaluation=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(args.output)


if __name__=='__main__':main()
