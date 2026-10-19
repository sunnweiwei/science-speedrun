"""Summarize same-reference regional comparisons; never infer a direct paper gap."""
import argparse
import json
import math
from pathlib import Path
import statistics

from compare_report import local_change
from common import external, sha256, write_json


def summarize(plan, records, native_check):
    if native_check.get('investigation_required') is not False:
        raise ValueError('Native baseline reproduction must pass its discrepancy gate')
    if type(native_check.get('nerl')) not in (int,float) or not math.isfinite(native_check['nerl']):
        raise ValueError('Finite native baseline score required')
    reference_path=Path(__file__).with_name('baseline_reference.json')
    gate=json.loads(reference_path.read_text())['native_reproduction_gate']
    if abs(native_check['nerl']-gate['target_nerl'])>gate['absolute_difference_investigation_threshold']:
        raise ValueError('Native score contradicts passed discrepancy gate')
    references=set()
    expected={(r['region_id'],s) for r in plan['regions'] for s in plan['nested_scale_voxels']}
    seen=set();rows=[]
    for record in records:
        key=(record['region_id'],record['core_voxels'])
        if key not in expected or key in seen:raise ValueError('Unexpected or duplicate region/scale')
        seen.add(key)
        if record.get('status')=='invalid_reference':
            if not record.get('reason'):raise ValueError('Invalid reference requires an explicit reason')
            rows.append(dict(region_id=key[0],core_voxels=key[1],status='invalid_reference',reason=record['reason']))
            continue
        ffn,full,baseline=(record[k] for k in ('ffn','full','baseline'))
        for metric in (ffn,full,baseline):
            for field in ('adapter_sha256','core_evaluator_adapter_sha256','topology_sha256','suite_sha256'):
                if not metric.get(field):raise ValueError('Missing physical evaluator provenance: '+field)
            if not 0<=metric['nerl']<=1+1e-12:raise ValueError('Invalid nERL')
        if len({m['suite_sha256'] for m in (ffn,full,baseline)})!=1:
            raise ValueError('Different frozen reference suites')
        identity=(key[1],full['suite_sha256'])
        if identity in references:raise ValueError('Same reference reused as separate spatial regions')
        references.add(identity)
        refinement=local_change(ffn,full)
        comparison=local_change(baseline,full)
        rows.append(dict(region_id=key[0],core_voxels=key[1],status='scored',
            suite_sha256=full['suite_sha256'],
            scores={k:{n:m[n] for n in ('nerl','erl_nm','split_errors','merge_errors','omitted_node_fraction')}
                for k,m in [('ffn',ffn),('full',full),('baseline',baseline)]},
            refinement=refinement,baseline_to_full=comparison))
    aggregate=[]
    for scale in plan['nested_scale_voxels']:
        selected=[r for r in rows if r['core_voxels']==scale and r['status']=='scored']
        if not selected:continue
        differences=[r['baseline_to_full']['metrics']['nerl']['delta'] for r in selected]
        aggregate.append(dict(core_voxels=scale,regions=len(selected),
            mean_nerl_full_minus_baseline=statistics.mean(differences),
            min_nerl_difference=min(differences),max_nerl_difference=max(differences),
            weighting='Equal weight per scored region; descriptive only',confidence_interval=None))
    missing=[dict(region_id=r,core_voxels=s) for r,s in sorted(expected-seen)]
    invalid=sum(r['status']=='invalid_reference' for r in rows)
    return dict(schema='pathfinder-regional-comparison/v1',regions=rows,aggregates=aggregate,
        missing=missing,invalid_reference_count=invalid,all_expected_regions_scored=not missing and not invalid,
        baseline_reference_sha256=sha256(reference_path),native_reproduction=native_check,training_exposure=plan['training_exposure'],
        statistics=plan['statistics'],direct_pathfinder_gap=None,global_sota_claim=False,
        interpretation='Measured transfer gap on these references only; negative full-minus-baseline means ours is lower. Nested scales are correlated.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','records','native-check','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();paths={};rows=json.loads(a.records.read_text())
    for row in rows:
        if row.get('status')=='invalid_reference':continue
        for name in ('ffn','full','baseline'):
            path=Path(row[name]);path=path if path.is_absolute() else a.records.parent/path
            paths[str(path.resolve())]=sha256(path);row[name]=json.loads(path.read_text())
    result=summarize(json.loads(a.plan.read_text()),rows,json.loads(a.native_check.read_text()))
    result['inputs']={str(p.resolve()):sha256(p) for p in (a.plan,a.records,a.native_check)}|paths
    write_json(external(a.output),result)
