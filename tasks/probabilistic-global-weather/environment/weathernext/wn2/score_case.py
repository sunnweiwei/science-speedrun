#!/usr/bin/env python3
"""Independently score one case shard against the full frozen protocol."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate import evaluate, open_dataset, sha256, summarize, verify_metric_source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prediction','truth','protocol','output'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--case-index',required=True,type=int)
    args=p.parse_args()
    revision=verify_metric_source()
    protocol=json.loads(args.protocol.read_text())
    case=protocol['evaluation_initializations'][args.case_index]
    subset=protocol|dict(evaluation_initializations=[case])
    inputs=hashlib.sha256();truth_digest=hashlib.sha256()
    with open_dataset(args.prediction) as prediction,open_dataset(args.truth) as truth:
        # The native package emits weather plus precipitation in a single
        # store; panels deliberately select only their own declared variables.
        names=list(protocol['variables'])
        predicted=prediction[names].assign_coords(level=prediction.level)
        observed=truth[names].assign_coords(level=truth.level).isel(case=[args.case_index])
        observed=observed.sel(lead=np.asarray(protocol['lead_hours'],dtype='timedelta64[h]'))
        rows=list(evaluate(predicted,observed,subset,inputs,truth_digest))
    report=dict(status='passed',case_index=args.case_index,initialization=case,
                protocol_sha256=sha256(args.protocol),protocol_status=protocol['status'],
                weatherbenchx_revision=revision,evaluator_sha256=sha256(Path(__file__).with_name('evaluate.py')),
                shard_scorer_sha256=sha256(Path(__file__)),estimator='empirical CRPS; fair=False',
                scored_input_values_sha256=inputs.hexdigest(),truth_values_sha256=truth_digest.hexdigest(),
                training_provenance_verified=False,per_case=rows,aggregate=summarize(rows))
    args.output.parent.mkdir(parents=True,exist_ok=True);tmp=args.output.with_suffix('.tmp')
    tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(args.output)


if __name__=='__main__':main()
