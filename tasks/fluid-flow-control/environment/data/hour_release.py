"""Rebuild the small, hour-scale data release from frozen native training evidence.

Budgets are reference experiments, not convergence promises or agent stop rules.
Every data byte is counted without compression or cross-release hard-link discounts.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import public_release as public

ROOT = Path(__file__).resolve().parent
RELEASE_ID = 'hydrogym-public-hour-v1'
CASE_IDS = frozenset('''
Channel_3D_Retau180
Nozzle2D_coarse
Cylinder_2D_Re200
Cylinder_2D_Re1000
Cylinder_2D_Re3900
SquareCylinder_2D_Re200
SquareCylinder_2D_Re1000
Pinball_2D_Re30
Pinball_2D_Re75
Pinball_2D_Re100
Pinball_2D_Re150
NACA0012Gust_2D_Re100_AOA20
NACA0012Gust_2D_Re100_AOA40
NACA0012Gust_2D_Re1000_AOA20
NACA0012Gust_2D_Re1000_AOA40
NACA0012_2D_Re100_AOA20
NACA0012_2D_Re100_AOA40
NACA0012_2D_Re1000_AOA20
NACA0012_2D_Re1000_AOA40
'''.split())


def read(name):
    return json.loads((ROOT/name).read_text())


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def estimate(record, controls):
    """Measured PPO sampling/update rate plus measured fixed cold/replay cost.

Use the slower of the old step-only probe and the new actual learning rate.
The 25% reserve is explicitly an estimate, not a confidence bound or timeout.
"""
    run = record['training']
    rate = max(record['previous_seconds_per_control'],
               run['training_seconds']/run['training_controls'])
    fixed = max(0.0, record['container_wall_seconds'] - run['training_seconds'])
    base = controls * rate + fixed
    return {'controls': controls, 'seconds_per_control_with_updates': rate,
            'measured_fixed_overhead_seconds': fixed,
            'estimated_seconds': base, 'planning_seconds_with_25pct_reserve': base * 1.25,
            'status': 'extrapolated_from_short_real_training_not_measured_full_budget'}


def build(evidence):
    if (evidence.get('release_id') != RELEASE_ID or evidence.get('source_revision') != public.REVISION
            or evidence.get('code_revision') != public.CODE_REVISION):
        raise ValueError('Hour release evidence identity mismatch')
    records = {r['case']: r for r in evidence['cases']}
    if len(records) != len(evidence['cases']) or set(records) != CASE_IDS:
        raise ValueError('Hour release requires the exact frozen 19-case membership')
    clock=evidence.get('channel_clock_check',{})
    if (clock.get('status')!='passed' or clock.get('outer_CFD_clock')!=5000
            or clock.get('controls_per_episode')!=100
            or clock.get('zero_returns',[None,1])[0]!=clock.get('zero_returns',[0,None])[1]
            or any(clock.get(key)!={'U':0.0,'V':0.0,'W':0.0} for key in [
                'zero_final_field_max_abs_differences','nonzero_first_window_max_abs_differences'])):
        raise ValueError('Channel clock repair lacks native invariance checks')
    profile=records['Channel_3D_Retau180']['runtime_profile']
    if (profile.get('actuator_clock_profile')!='channel-72cube-local-actuator-clock-v1'
            or profile.get('adapter_sha256')!=hashlib.sha256((ROOT.parent/'evaluation/native_channel.py').read_bytes()).hexdigest()):
        raise ValueError('Channel adapter differs from the validated runtime repair')
    result = public.build(read('resource-audit.json'), read('source-files.json'),
                          read('native-configs.json'), selected_case_ids=CASE_IDS)
    total_short = total_research = total_eval = 0.0
    for c in result['cases']:
        r = records[c['id']]
        run, replay = r['training'], r['replay']
        if r['data_bytes'] != c['bytes'] or r['hardware'] != '1 H100' or r['container_exit_code'] != 0:
            raise ValueError('Native learning evidence does not match one-GPU source case')
        if run.get('status') != 'passed' or replay.get('status') != 'passed':
            raise ValueError('Unsuccessful training/replay cannot qualify')
        if replay.get('protocol')!='hydrogym-policy/v1' or replay.get('relocated') is not True:
            raise ValueError('Missing fresh relocated delivery-protocol replay')
        if (type(run['training_controls']) is not int or run['training_controls'] < 400
                or run['ppo_epochs'] < 20 or run['changed_parameter_tensors'] <= 0):
            raise ValueError('Missing actual iterative policy learning')
        if any(not positive(v) for v in [run['training_seconds'], r['previous_seconds_per_control'],
                                         r['container_wall_seconds']]):
            raise ValueError('Invalid measured training cost')
        horizon = 100 if c['backend'] == 'jax' else 667 if c['backend'] == 'jax-fluids' else 201
        if run['episode_controls'] != horizon:
            raise ValueError('Changed native episode horizon')
        for ep in [run['zero'], run['trained'], replay]:
            n = ep.get('controls', ep.get('steps'))
            if n != horizon or not math.isfinite(ep['return']):
                raise ValueError('Full finite native episodes required')
        short, research = estimate(r, 4000), estimate(r, 20000)
        if research['planning_seconds_with_25pct_reserve'] > 10*3600:
            raise ValueError('Case exceeds ten-hour reference-training admission')
        # A known-action rollout is not a measured training budget. Keep all
        # components and the extrapolation status visible in the manifest.
        c.update(execution_status='short_training_and_frozen_full_episode_replay_passed',
                 runtime_profile=r['runtime_profile'], training_cost_evidence=r,
                 reference_training_budgets={'short_multitask_phase': short, 'single_case_research': research})
        total_short += short['planning_seconds_with_25pct_reserve']
        total_research += research['planning_seconds_with_25pct_reserve']
        total_eval += replay['wall_seconds']
    if total_short > 10*3600:
        raise ValueError('Entire serial short research round exceeds ten-hour planning budget')
    for c in result['exclusions']:
        if c['reason'] == 'not_selected_500gb_scope':
            c.update(reason='outside_hour_scale_validated_runtime_selection',
                detail='Excluded from BOTH training and scoring: slower native interaction budget or no sufficiently validated native training profile. Unknown runtime is not claimed slow; no hidden/optional test-only slow tasks.')
        if c.get('environment')=='Cavity_2D_Re7500':
            c.update(reason='nonfinite_native_uncontrolled_full_episode',
                     detail=evidence['excluded_runtime_failure']['reason'])
    result.update(release_id=RELEASE_ID, created=evidence['created'],
        supersedes='hydrogym-public-coverage-v1', cost_evidence_sha256=public.digest(evidence),
        scope='Fixed 19-case known-task hour-scale subset: 17 MAIA 2D cases, one JAX-Fluids 2D nozzle and one JAX 3D channel. Not the full paper benchmark.',
        reconstruction_policy='No changes to source bytes, grids, solver precision, physical time steps or native horizons. Explicit organizer sensors and Channel class defaults PLUS the documented local actuator-clock repair are frozen separately from paper/YAML claims.',
        reference_training_budget={
            'algorithm': 'shared task-conditioned PPO, 256x256 MLP; optimizer on CPU, native CFD on one H100',
            'short_round_controls_per_case': 4000, 'short_round_total_controls': 4000*len(CASE_IDS),
            'short_round_serial_single_h100_planning_seconds': total_short,
            'larger_round_controls_per_case': 20000, 'larger_round_total_controls': 20000*len(CASE_IDS),
            'larger_round_serial_single_h100_planning_seconds': total_research,
            'full_evaluation_measured_case_replays_sum_seconds': total_eval,
            'full_evaluation_scope': 'Sum of individual cold frozen-policy native replays; not a measured single shared checkpoint suite run',
            'time_status': 'Longer training times extrapolated from measured short PPO updates and full native replays, with 25% planning reserve. Not a guaranteed learning outcome or wall-clock upper bound.',
            'one_checkpoint_for_all_cases': True,
            'paper_convergence_claim': False, 'agent_training_limit': False,
        })
    result['test']['episode_protocol'] = {'seeds': [123], 'maia_controls': 201,
        'channel_controls': 100, 'nozzle_controls': 667, 'termination': 'native',
        'initialization': 'Pinned native default restart; no fabricated random flow state',
        'seed_note': 'Different API seed does not imply a hidden or distinct physical initial state'}
    result['source']['organizer_runtime_adapters']={
        'Channel_3D_Retau180':{'profile':profile['actuator_clock_profile'],
            'file':'environment/evaluation/native_channel.py','sha256':profile['adapter_sha256']}}
    return result


def build_current():
    return build(read('cost-evidence-hour-v1.json'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write', action='store_true')
    args = p.parse_args()
    value = build_current()
    if args.write:
        import coverage_release
        old = coverage_release.build_current()
        archive = ROOT/'release-coverage-v1.json'
        if archive.exists() and read(archive.name) != old:
            raise ValueError('Historical coverage archive differs')
        if read('release.json') not in (old, value):
            raise ValueError('Refusing to replace another release')
        if not archive.exists():
            archive.write_text(json.dumps(old, indent=2)+'\n')
        (ROOT/'release.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps({'release_id': value['release_id'], 'case_count': value['case_count'],
                      'files': value['files'], 'sha256': public.digest(value),
                      'budgets': value['reference_training_budget']}, indent=2))


if __name__ == '__main__':
    main()
