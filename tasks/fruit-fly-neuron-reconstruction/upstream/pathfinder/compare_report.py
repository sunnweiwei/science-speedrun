"""Display pinned paper numbers beside local metrics without a false SOTA gap."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import external, sha256, write_json


def local_change(before,after):
    if before.get('adapter_sha256')!=after.get('adapter_sha256'):raise ValueError('Different local evaluator adapters')
    for name in ('skeletons','metadata'):
        if before['inputs'][name]!=after['inputs'][name]:raise ValueError('Different local evaluation input: '+name)
    for name in ('evaluator_sha256','protocol','boundary_margin_voxels','maximum_erl_nm','total_path_length_nm'):
        if before[name]!=after[name]:raise ValueError('Different local evaluation protocol: '+name)
    for name in ('core_evaluator_adapter_sha256','topology_sha256'):
        if before.get(name)!=after.get(name):raise ValueError('Different local evaluator component: '+name)
    changes={}
    for key in ('nerl','erl_nm','split_errors','merge_errors','omitted_node_fraction'):
        a,b=before[key],after[key]
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (a,b)):
            raise ValueError('Finite local metric required: '+key)
        changes[key]=dict(before=a,after=b,delta=b-a)
    return dict(metrics=changes,identical_prediction_bytes=before['inputs']['segmentation']==after['inputs']['segmentation'],
        scope='Same frozen local skeletons/ROI/evaluator only; not a paper gap')


def compare(reference, result, before=None):
    target = next(r for r in reference['table1'] if r['stage'] == 'PATHFINDER')
    return dict(reference=target, local={k:result.get(k) for k in ['nerl','split_errors','merge_errors']},
        directly_comparable=False, numeric_gap=None,
        reasons=['Paper dataset bytes, ROI and GT membership not pinned',
                 'Official skeleton/evaluation numerical parity not established'],
        local_protocol=result.get('protocol'), local_dataset=result.get('dataset_id'),
        local_change=local_change(before,result) if before is not None else None,
        interpretation='Paper reference only; local numbers are not a measured PATHFINDER gap')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--before',type=Path,help='Optional FFN baseline scored on exactly the same frozen protocol')
    a=p.parse_args(); ref=Path(__file__).with_name('paper_reference.json')
    result=compare(json.loads(ref.read_text()),json.loads(a.result.read_text()),
        json.loads(a.before.read_text()) if a.before else None)
    result.update(reference_sha256=sha256(ref),result_sha256=sha256(a.result))
    if a.before:result['before_sha256']=sha256(a.before)
    write_json(external(a.output),result); print(json.dumps(result,indent=2))

if __name__=='__main__':main()
