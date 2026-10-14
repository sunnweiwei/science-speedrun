#!/usr/bin/env python3
"""Merge complete case shards and compute a dimensionless, quality-based reward."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

from evaluate import sha256,summarize


def merge(reports,protocol,protocol_hash):
    indices=[r['case_index'] for r in reports]
    if sorted(indices)!=list(range(len(protocol['evaluation_initializations']))):
        raise ValueError('Missing or duplicated case shard')
    ordered=sorted(reports,key=lambda r:r['case_index'])
    for field in ('evaluator_sha256','shard_scorer_sha256','weatherbenchx_revision','estimator'):
        if len({r[field] for r in ordered})!=1:raise ValueError('Scorer changed between shards')
    rows=[]
    for i,r in enumerate(ordered):
        if r['status']!='passed' or r['protocol_sha256']!=protocol_hash:raise ValueError('Incomplete or mismatched shard')
        case=protocol['evaluation_initializations'][i]
        expected={(case,name,level,lead,factor) for name,spec in protocol['variables'].items()
                  for level in (protocol['pressure_levels_hpa'] if spec['pressure_levels'] else [None])
                  for lead in protocol['lead_hours'] for factor in protocol['pool_cells']}
        actual=[(row['case'],row['variable'],row['level_hpa'],row['lead_hours'],row['pool_cells']) for row in r['per_case']]
        if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError('Incomplete or duplicated metric panel')
        for row in r['per_case']:
            if any(not np.isfinite(row[k]) or row[k]<0 for k in ('crps','mse','ensemble_variance')):
                raise ValueError('Invalid metric statistic')
        rows.extend(r['per_case'])
    digest=lambda field:hashlib.sha256(json.dumps([r[field] for r in ordered]).encode()).hexdigest()
    return dict(status='passed',protocol_sha256=protocol_hash,protocol_status=protocol['status'],
                **{k:ordered[0][k] for k in ('evaluator_sha256','shard_scorer_sha256','weatherbenchx_revision','estimator')},
                scored_input_values_sha256=digest('scored_input_values_sha256'),truth_values_sha256=digest('truth_values_sha256'),
                digest_definition='SHA256 of ordered per-case digest JSON list',training_provenance_verified=False,
                per_case=rows,aggregate=summarize(rows))


def reward(report,scales):
    """Equal variable weights; equal levels/leads within each variable."""
    groups={}
    for row in report['aggregate']:
        if row['pool_cells']!=1:continue
        scale=scales[row['variable']]
        if row['level_hpa'] is not None:scale=scale.sel(level=row['level_hpa'])
        value=float(scale)
        if not np.isfinite(value) or value<=0:raise ValueError('Invalid frozen normalization scale')
        groups.setdefault(row['variable'],[]).append(row['crps']/value)
    if not groups:raise ValueError('No primary metric panels')
    per_variable={name:float(np.mean(values)) for name,values in groups.items()}
    normalized=float(np.mean(list(per_variable.values())))
    return dict(reward=1/(1+normalized),normalized_crps=normalized,normalized_crps_by_variable=per_variable,
                definition='1/(1+mean normalized CRPS); equal variables, then equal pressure levels and leads; native cells only; scales frozen from training data; larger reward is better')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True,type=Path);parser.add_argument('--shards',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--scales',type=Path)
    args=parser.parse_args();protocol=json.loads(args.protocol.read_text())
    reports=[json.loads(p.read_text()) for p in sorted(args.shards.glob('case-*.json'))]
    result=merge(reports,protocol,sha256(args.protocol))
    if args.scales:
        with xr.open_dataset(args.scales) as scales:result['reward_metrics']=reward(result,scales)
        result['reward_scales_sha256']=sha256(args.scales)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(args.output)


if __name__=='__main__':main()
