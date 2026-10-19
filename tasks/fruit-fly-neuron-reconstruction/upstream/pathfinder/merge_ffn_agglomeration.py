"""Validate all FFN score shards and globally union accepted prediction pairs.

Stores a compact global ID map. Applying this map to a scoring crop after the
union is identical to cropping a full remapped volume, without another 22GB
array. No ground truth is needed to construct the map.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from dataset_contract import sha256


def union_mapping(pairs):
    parent={}
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    for a,b in pairs:
        if type(a)!=int or type(b)!=int or a<=0 or b<=0 or a==b:
            raise ValueError('Positive distinct integer pair required')
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    return {sid:find(sid) for sid in sorted(parent) if sid!=find(sid)}


def remap(array, mapping):
    if array.dtype.kind!='u':raise ValueError('Unsigned labels required')
    if not mapping:return np.array(array,copy=True)
    keys=np.array(sorted(mapping),dtype=np.uint64)
    values=np.array([mapping[int(k)] for k in keys],dtype=np.uint64)
    if int(values.max())>np.iinfo(array.dtype).max:raise ValueError('Output dtype overflow')
    index=np.searchsorted(keys,array);safe=np.minimum(index,len(keys)-1)
    hit=(index<len(keys)) & (keys[safe]==array)
    result=np.array(array,copy=True);result[hit]=values[safe[hit]]
    return result


def scoring_rule(name):
    from ffn_agglomeration import accept_pair,accept_pathfinder_pair
    if name=='manual':return accept_pair
    if name=='pathfinder2025':return accept_pathfinder_pair
    raise ValueError('Unknown FFN agglomeration scoring rule')


def combine(plan, shards, decisions_sha256, *, allow_partial=False, rule='manual'):
    selected_rule=scoring_rule(rule)
    if not shards:raise ValueError('No score shards')
    count=shards[0]['shards']
    if type(count)!=int or count<1 or any(s['shards']!=count for s in shards):
        raise ValueError('Invalid shard count')
    if len(shards)!=count or sorted(s['shard_index'] for s in shards)!=list(range(count)):
        raise ValueError('Missing or duplicate shards')
    signatures={(s['checkpoint_sha256'],s['raw_sha256'],s['base_sha256'],s['decisions_sha256']) for s in shards}
    if len(signatures)!=1:raise ValueError('Inconsistent shard sources')
    signature=next(iter(signatures))
    if signature[2]!=plan['base_sha256'] or signature[3]!=decisions_sha256:
        raise ValueError('Decisions/base mismatch')
    from decision_table import DecisionRows
    disk_rows=plan['decisions'] if isinstance(plan['decisions'],DecisionRows) else None
    expected=disk_rows if disk_rows is not None else [tuple(d['pair']) for d in plan['decisions']]
    if disk_rows is None and len(set(expected))!=len(expected):raise ValueError('Duplicate decision pairs')
    pairs=[];seen=bytearray(len(expected)) if disk_rows is not None else set();incomplete=0;seen_count=0
    from ffn_agglomeration import accept_pair
    for shard in shards:
        if shard['ground_truth_used'] is not False:raise ValueError('Invalid prediction-only declaration')
        assigned=None if disk_rows is not None else set(expected[shard['shard_index']::count])
        for row in shard['results']:
            pair=tuple(row['pair'])
            if disk_rows is not None:
                ordinal=disk_rows.pair_index(pair)
                if ordinal%count!=shard['shard_index'] or seen[ordinal]:
                    raise ValueError('Unexpected or duplicate candidate result')
                seen[ordinal]=1
            else:
                if pair not in assigned or pair in seen:raise ValueError('Unexpected or duplicate candidate result')
                seen.add(pair)
            seen_count+=1
            if not row['complete']:
                incomplete+=1
                if row['accepted']:raise ValueError('Incomplete candidate accepted')
                continue
            accepted=accept_pair(row,complete=True)
            if accepted!=row['accepted']:raise ValueError('Stored decision disagrees with frozen scoring rule')
            if selected_rule(row,complete=True):pairs.append(list(pair))
    missing=len(expected)-seen_count
    if (missing or incomplete) and not allow_partial:
        raise ValueError('Candidate evaluation incomplete; explicit partial mode required')
    mapping=union_mapping(pairs)
    return dict(schema='ffn-global-agglomeration/v1',mapping={str(k):v for k,v in mapping.items()},
        accepted_pairs=len(pairs),requested_pairs=len(expected),missing_pairs=missing,
        incomplete_pairs=incomplete,complete=missing==0 and incomplete==0,
        checkpoint_sha256=signature[0],raw_sha256=signature[1],base_sha256=signature[2],
        decisions_sha256=signature[3],ground_truth_used=False,score_rule=rule,
        partial_policy='Unscored or incomplete pairs are not merged; coverage explicitly reported')


def combine_ensemble(plan, shards, decisions_sha256, *, rule='pathfinder2025', expected_checkpoints=None):
    """Require all three checkpoint decisions, before computing global unions."""
    groups={}
    for shard in shards:groups.setdefault(shard['checkpoint_sha256'],[]).append(shard)
    if len(groups)!=3:raise ValueError('Exactly three distinct checkpoint files required')
    if expected_checkpoints is not None:
        expected=list(expected_checkpoints)
        if len(expected)!=3 or len(set(expected))!=3 or set(groups)!=set(expected):
            raise ValueError('Agglomeration checkpoints differ from verified model package')
    reports=[combine(plan,group,decisions_sha256,rule=rule) for group in groups.values()]
    signatures={(r['raw_sha256'],r['base_sha256'],r['decisions_sha256']) for r in reports}
    if len(signatures)!=1:raise ValueError('Ensemble inputs differ')
    accepted=None
    gate=scoring_rule(rule)
    for group in groups.values():
        passing={tuple(row['pair']) for shard in group for row in shard['results']
                 if gate(row,complete=row['complete'])}
        if accepted is None:accepted=passing
        else:accepted.intersection_update(passing)
    mapping=union_mapping(sorted(accepted));raw,base,decisions=next(iter(signatures))
    return dict(schema='ffn-ensemble-agglomeration/v1',mapping={str(k):v for k,v in mapping.items()},
                accepted_pairs=len(accepted),requested_pairs=len(plan['decisions']),
                missing_pairs=0,incomplete_pairs=0,complete=True,raw_sha256=raw,base_sha256=base,
                decisions_sha256=decisions,checkpoint_sha256s=sorted(groups),score_rule=rule,
                ground_truth_used=False,checkpoint_summaries=reports,
                ensemble_policy='All three checkpoint-specific gates pass on each pair before global union',
                paper_comparable=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decisions',required=True,type=Path)
    parser.add_argument('--shards',required=True,nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--rule',choices=['manual','pathfinder2025'],default='manual')
    parser.add_argument('--ensemble',action='store_true',help='Require three complete checkpoint groups')
    parser.add_argument('--model-package',type=Path,help='Bind ensemble scores to three verified full FFN assets')
    a=parser.parse_args()
    if a.output.exists():raise ValueError('Output exists')
    pins={str(p):sha256(p) for p in [a.decisions,*a.shards]}
    expected=None;package_check=None
    if a.model_package:
        from model_package import verify
        package=a.model_package.resolve();package_check=verify(package,require_ensemble=a.ensemble)
        manifest=json.loads((package/'package.json').read_text())
        if a.ensemble:expected=[r['checkpoint_sha256'] for r in package_check['ffn_ensemble']['ordered_runs'][::2]]
        else:
            if 'ffn_ensemble' in manifest:raise ValueError('Explicit single-checkpoint model package required')
            expected=[manifest['files']['weights/ffn.pt']]
        pins[str(package/'package.json')]=package_check['package_manifest_sha256']
        pins.update({str(package/name):digest for name,digest in manifest['files'].items()})
    plan=json.loads(a.decisions.read_text())
    from decision_table import load_rows
    rows=load_rows(plan,a.decisions)
    plan={**plan,'decisions':rows}
    from score_stream import load_shard
    shards=[load_shard(p) for p in a.shards]
    if a.model_package and not a.ensemble and any(s['checkpoint_sha256']!=expected[0] for s in shards):
        raise ValueError('Agglomeration checkpoint differs from single-checkpoint model package')
    for path,shard in zip(a.shards,shards):
        if 'result_stream' in shard:
            pins[str(path.parent/shard['result_stream']['file'])]=shard['result_stream']['sha256']
    if a.ensemble:
        if a.allow_partial:raise ValueError('Ensemble acceptance requires complete evaluation')
        result=combine_ensemble(plan,shards,pins[str(a.decisions)],rule=a.rule,expected_checkpoints=expected)
    else:
        result=combine(plan,shards,pins[str(a.decisions)],allow_partial=a.allow_partial,rule=a.rule)
    for path,digest in pins.items():
        if sha256(path)!=digest:raise ValueError('Agglomeration input changed during merge')
    result['shard_files']={str(p):pins[str(p)] for p in a.shards}
    if package_check is not None:
        result['model_package_check']=package_check
        result['input_sha256']=pins
    result['adapter_sha256']=sha256(__file__)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
