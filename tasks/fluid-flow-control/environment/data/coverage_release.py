"""Build an explicitly reviewed coverage expansion; no universal speed cutoff."""
import argparse
import json
import math
from pathlib import Path

import public_release as public

ROOT=Path(__file__).resolve().parent
RELEASE_ID='hydrogym-public-coverage-v1'

def read(name):
    return json.loads((ROOT/name).read_text())

def build(selection,evidence):
    if selection['release_id']!=RELEASE_ID or selection['status']!='frozen':
        raise ValueError('Unknown/unfrozen coverage selection')
    if selection['max_source_bytes']!=500_000_000_000:
        raise ValueError('Wrong fixed source-data budget')
    if selection['policy'].get('global_seconds_per_control_cutoff') is not None:
        raise ValueError('Coverage selection must not impose a universal speed cutoff')
    if selection['cost_evidence_sha256']!=public.digest(evidence):
        raise ValueError('Coverage evidence has changed')
    if evidence['source_revision']!=public.REVISION or evidence['code_revision']!=public.CODE_REVISION:
        raise ValueError('Coverage source/code identity mismatch')
    decisions=selection['included']
    ids=[c['case'] for c in decisions]
    if not ids or ids!=sorted(set(ids)):
        raise ValueError('Expected sorted unique selected cases')
    previous=set(read('selection-speed-v1.json')['case_ids'])
    if not previous < set(ids):
        raise ValueError('Coverage must expand and preserve the first tested set')
    records={r['case']:r for r in evidence['cases']}
    if len(records)!=len(evidence['cases']):
        raise ValueError('Duplicate cost evidence')
    for decision in decisions:
        if not decision.get('scientific_value') or not decision.get('cost_review'):
            raise ValueError('Every included case requires value and cost review')
        r=records.get(decision['case'],{})
        if r.get('observed_field_response') is False:
            raise ValueError('An unresolved zero-field-response probe cannot qualify a case')
        if r.get('backend')=='maia' and r.get('hardware')=='2 H100':
            bindings=r.get('gpu_bindings',[])
            if (len(bindings)!=2 or len({b.get('cuda_device_uuid') for b in bindings})!=2
                    or any(b.get('visible_cuda_devices')!=1 for b in bindings)):
                raise ValueError('Two-GPU MAIA evidence requires distinct verified CUDA rank bindings')
        samples=r.get('timing_samples_seconds',[])
        if (r.get('status')!='complete' or r.get('container_exit_code')!=0
                or len(samples)<2 or any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in samples)):
            raise ValueError('No clean, sufficiently sampled native short loop')
        reset=r.get('reset_seconds')
        if type(reset) not in (int,float) or not math.isfinite(reset) or reset<=0:
            raise ValueError('Missing completed second reset')
        if (r.get('measured_controls')!=len(samples) or not math.isclose(
                sum(samples)/len(samples),r['seconds_per_control'],rel_tol=1e-10)):
            raise ValueError('Timing sample/summary mismatch')
        if not decision.get('runtime_profile'):
            raise ValueError('Missing explicit native runtime profile')
    result=public.build(read('resource-audit.json'),read('source-files.json'),
                        read('native-configs.json'),selected_case_ids=ids)
    excluded={r['case']:r for r in selection['excluded_complete_resources']}
    actual_excluded={r['environment'] for r in result['exclusions'] if r['reason']=='not_selected_500gb_scope'}
    if (len(excluded)!=len(selection['excluded_complete_resources']) or set(excluded)!=actual_excluded
            or selection['candidate_inventory_count']!=len(ids)+len(excluded)):
        raise ValueError('Incomplete or ambiguous inventory accounting')
    for r in result['exclusions']:
        if r['reason']=='not_selected_500gb_scope':
            decision=excluded[r['environment']]
            r.update(reason=decision['reason'],detail=decision['detail'])
    by_decision={r['case']:r for r in decisions}
    for c in result['cases']:
        record=records[c['id']]
        if any(record[k]!=c[v] for k,v in [('data_bytes','bytes'),('backend','backend'),('dimension','dimension')]):
            raise ValueError('Cost measurement/source case mismatch')
        c.update(cost_screening=record,selection_review=by_decision[c['id']],
                 execution_status='short_loop_passed_full_evaluation_pending')
    result.update(release_id=RELEASE_ID,created=selection['created'],
        supersedes='hydrogym-public-speed-v1',selection_spec_sha256=public.digest(selection),
        cost_evidence_sha256=public.digest(evidence),selection_policy=selection['policy'],
        scope='Value- and cost-reviewed known-task coverage expansion within 500 GB; not the paper full benchmark.',
        reconstruction_policy='All source bytes unchanged. Runtime profiles explicitly distinguish native YAML, official code examples and organizer-defined bindings. Full evaluator acceptance remains separate.')
    return result

def build_current():
    return build(read('selection-coverage-v1.json'),read('cost-evidence-coverage-v1.json'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args()
    result=build_current()
    if args.write:
        import speed_release
        old=speed_release.build_current()
        archive=ROOT/'release-speed-v1.json'
        if archive.exists() and read(archive.name)!=old:
            raise ValueError('Historical first-tested manifest has changed')
        if read('release.json') not in (old,result):
            raise ValueError('Refusing to overwrite an unrelated release')
        if not archive.exists():
            archive.write_text(json.dumps(old,indent=2)+'\n')
        (ROOT/'release.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'release_id':result['release_id'],'case_count':result['case_count'],
                      'files':result['files'],'sha256':public.digest(result)}))

if __name__=='__main__':
    main()
