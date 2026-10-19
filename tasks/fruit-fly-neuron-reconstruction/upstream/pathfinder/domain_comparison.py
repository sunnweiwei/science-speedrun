"""Compare three methods on frozen MaleCNS references without cross-data ranking.

Public-number reproduction is reported independently; a failed NISB check must
not prevent honestly reporting a same-data FFN agglomeration comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
from compare_report import local_change
from dataset_contract import sha256


def reconstruction_coverage(assembled):
    """Check full-context coverage independently of whether scores exist."""
    from assemble_tiles import validate_shards
    assembled=Path(assembled)
    receipt=json.loads((assembled/'receipt.json').read_text())
    if receipt.get('schema')=='prepared-prediction-context/v1':
        from prepare_prediction_context import audit
        return audit(assembled)
    coverage=json.loads((assembled/'coverage.json').read_text())
    shards=[]
    for item in receipt['inputs']:
        path=Path(item['path'])
        if sha256(path)!=item['sha256']:raise ValueError('Assembly shard receipt changed')
        shards.append(json.loads(path.read_text()))
    first,expected,tiles=validate_shards(shards)
    observed={row['index']:row for row in coverage['tiles']}
    if len(observed)!=len(coverage['tiles']) or set(observed)!=set(expected):
        raise ValueError('Coverage grid missing or duplicated')
    complete=0
    for index,tile in expected.items():
        row=observed[index];found=tiles.get(index)
        if found is not None and type(found.get('budget_exhausted')) is not bool:
            raise ValueError('Unknown tile completion state')
        status='unstarted' if found is None else 'partial' if found['budget_exhausted'] else 'complete'
        if row['status']!=status or any(row[k]!=tile[k] for k in ('start','stop')):
            raise ValueError('Coverage disagrees with shard evidence')
        complete+=status=='complete'
    full=complete==len(expected)
    if (coverage['total_tiles']!=len(expected) or coverage['complete_tiles']!=complete
        or coverage['all_tiles_complete'] is not full or receipt['all_tiles_complete'] is not full):
        raise ValueError('Coverage summary disagrees with tile evidence')
    if first['metadata']!=receipt['metadata'] or json.loads((assembled/'metadata.json').read_text())!=first['metadata']:
        raise ValueError('Assembly metadata changed')
    if sha256(assembled/'segmentation.npy')!=receipt['segmentation_sha256']:
        raise ValueError('Assembled prediction changed')
    return dict(full_context_reconstruction_complete=full,complete_tiles=complete,total_tiles=len(expected),
        assembly_receipt_sha256=sha256(assembled/'receipt.json'),coverage_sha256=sha256(assembled/'coverage.json'),
        segmentation_sha256=receipt['segmentation_sha256'],
        scope='Complete compute coverage only; does not establish biological accuracy or baseline candidate completeness')


def summarize(expected_suites, records, native_check):
    seen=set();rows=[]
    for record in records:
        name=record['region']
        if name not in expected_suites or name in seen:raise ValueError('Unexpected/duplicate region')
        seen.add(name)
        methods={k:record[k] for k in ('ffn','baseline','full')}
        for metric in methods.values():
            if metric.get('suite_sha256')!=expected_suites[name]:raise ValueError('Wrong frozen reference')
            for field in ('adapter_sha256','core_evaluator_adapter_sha256','topology_sha256'):
                if not metric.get(field):raise ValueError('Missing scoring provenance')
            if not 0<=metric['nerl']<=1:raise ValueError('Invalid nERL')
        comparison=local_change(methods['baseline'],methods['full'])
        refinement=local_change(methods['ffn'],methods['full'])
        before,after=methods['baseline']['nerl'],methods['full']['nerl']
        rows.append(dict(region=name,suite_sha256=expected_suites[name],
            baseline_to_full=comparison,ffn_to_full=refinement,
            relative_nerl_change=(after-before)/before if before else None,
            direction='improved' if after>before else 'worse' if after<before else 'unchanged'))
    missing=sorted(set(expected_suites)-seen)
    primary=next((r for r in rows if r['region']=='full-core'),None)
    return dict(schema='malecns-domain-comparison/v1',regions=rows,primary=primary,
        missing_regions=missing,all_regions_scored=not missing,
        native_public_number_check=native_check,
        native_check_independent_of_domain_comparison=True,
        statistical_claim='Descriptive fixed spatial subsets; no independent-sample confidence interval',
        direct_pathfinder_paper_gap=None,global_sota_claim=False)


def resource_costs(ledger, job_roles):
    """Charge each allocation once, even when FFN/full share the same job."""
    jobs={j['job_id']:j for j in ledger['jobs']}
    if len(jobs)!=len(ledger['jobs']):raise ValueError('Duplicate ledger job')
    if not job_roles:raise ValueError('Explicit campaign jobs required')
    rows=[]
    for job_id,roles in job_roles.items():
        if job_id not in jobs or not isinstance(roles,list) or not roles or not all(isinstance(r,str) and r for r in roles):
            raise ValueError('Unknown job or missing accounting role')
        job=jobs[job_id];actual=job.get('actual_seconds');reserved=job['reserved_seconds']
        if type(job['gpus'])!=int or not 1<=job['gpus']<=4:
            raise ValueError('Invalid GPU allocation')
        if type(reserved)!=int or reserved<=0 or (actual is not None and (type(actual)!=int or actual<0)):
            raise ValueError('Invalid allocation duration')
        seconds=actual if actual is not None else reserved
        rows.append(dict(job_id=job_id,roles=sorted(set(roles)),gpus=job['gpus'],
            charged_seconds=seconds,charged_gpu_seconds=seconds*job['gpus'],
            actual_seconds=actual,accounting_complete=actual is not None))
    return dict(jobs=rows,accounting_complete=all(r['accounting_complete'] for r in rows),
        charged_allocation_hours=sum(r['charged_seconds'] for r in rows)/3600,
        charged_gpu_hours=sum(r['charged_gpu_seconds'] for r in rows)/3600,
        resident_bytes_at_ledger_measurement=ledger.get('bytes_on_disk'),
        measured_peak_disk_bytes=None,
        scope='Each listed GPU allocation charged once; joint-stage jobs are not attributed to individual methods. CPU-only jobs and unlisted historical jobs excluded. Unfinished allocations retain full reservations. Residency is a ledger snapshot, not a measured peak.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('expected','records','native_check','output'):
        p.add_argument('--'+name.replace('_','-'),required=True,type=Path)
    p.add_argument("--ledger",type=Path);p.add_argument("--job-roles",type=Path)
    p.add_argument('--assembled',type=Path,help='Full-context FFN assembly for independent completion audit')
    a=p.parse_args()
    if bool(a.ledger)!=bool(a.job_roles):raise ValueError("Provide both ledger and explicit job roles")
    if a.output.exists():raise ValueError('Output exists')
    expected=json.loads(a.expected.read_text());records=json.loads(a.records.read_text());inputs={}
    for record in records:
        for method in ('ffn','baseline','full'):
            path=Path(record[method]);path=path if path.is_absolute() else a.records.parent/path
            inputs[str(path.resolve())]=sha256(path);record[method]=json.loads(path.read_text())
    result=summarize(expected,records,json.loads(a.native_check.read_text()))
    result['reconstruction_coverage']=reconstruction_coverage(a.assembled) if a.assembled else None
    result['full_context_reconstruction_verified']=bool(result['reconstruction_coverage'] and
        result['reconstruction_coverage']['full_context_reconstruction_complete'])
    result['resources']=None
    if a.ledger:
        ledger_bytes=a.ledger.read_bytes();roles_bytes=a.job_roles.read_bytes()
        result['resources']=resource_costs(json.loads(ledger_bytes),json.loads(roles_bytes))
        result['ledger_sha256']=hashlib.sha256(ledger_bytes).hexdigest()
        result['job_roles_sha256']=hashlib.sha256(roles_bytes).hexdigest()
    result.update(metric_files=inputs,expected_references_sha256=sha256(a.expected),
                  native_check_sha256=sha256(a.native_check),adapter_sha256=sha256(__file__))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
