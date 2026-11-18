"""Harbor bridge: independently replay one package and export physical metrics.

The fixed 26 verifier cases also determine a dimensionless v2 task reward.
Development subsets keep their own physical results.
"""
import argparse
import math
from pathlib import Path

from aggregate import task_score
from case_spec import select_cases, VERIFIER_CASES
from evaluate import write_json
from evaluate_native import evaluate_native
from metrics import definition
from protocol import InfrastructureError, PolicyError


def physical_rewards(result, names):
    _, cases = select_cases(names)
    expected = {c['id'] for c in cases}
    if set(result.get('cases', {})) != expected:
        raise InfrastructureError('Missing/extra cases: never grade a partial table as the requested suite')
    infrastructure = [name for name, row in result['cases'].items()
                      if row.get('status') not in ('ok', 'policy_error')]
    if infrastructure:
        raise InfrastructureError(f'Native infrastructure errors: {infrastructure}')
    reward = {}
    failed = []
    for name, row in sorted(result['cases'].items()):
        metric = row.get('primary_metric', {})
        if row.get('status') == 'policy_error':
            failed.append(name)
            continue
        if any(metric.get(k) != v for k, v in definition(name).items()):
            raise InfrastructureError(f'Missing/changed primary metric definition: {name}')
        if metric.get('status') == 'invalid_operating_regime':
            failed.append(name)
            continue
        value = metric.get('value')
        if metric.get('status') != 'ok' or type(value) not in (int, float) or not math.isfinite(value):
            raise InfrastructureError(f'Missing/nonfinite physical metric: {name}')
        # Preserve the user's physical metric, its units and sign. Directions are
        # stored in results.json and METRICS.md; no synthetic cross-unit sum.
        reward[f'{name}/{metric["id"]}'] = value
    if not reward:
        raise PolicyError(f'No case produced a valid result: {failed}')
    # A failed case (policy error, diverged simulation, invalid operating regime) has no physical
    # metric; it is flagged and counted as a failure while every other case keeps its value.
    for name in failed:
        reward[f'{name}/failed'] = 1
    reward['failed_cases'] = len(failed)
    return reward


def verify(data, submission, output, names=None, evaluator=evaluate_native):
    names = ['verifier'] if names is None else names
    _, selected = select_cases(names)
    full_panel = set(VERIFIER_CASES).issubset(c['id'] for c in selected)
    try:
        result = evaluator(data, submission, output, names)
        reward = physical_rewards(result, names)
    except (InfrastructureError, PolicyError) as error:
        write_json(Path(output)/'grading-error.json', {
            'status': 'policy_error' if isinstance(error, PolicyError) else 'infrastructure_error',
            'error': str(error), 'accepted': False})
        if isinstance(error, PolicyError):
            # The package is invalid: Harbor needs a numeric reward.json; there are no physical metrics.
            invalid = {'invalid_submission': 1}
            if full_panel:
                invalid['score'] = 0.0
                invalid['reward'] = 0.0
            write_json(Path(output)/'reward.json', invalid)
            return invalid
        raise
    metric_count = sum('/' in key and not key.endswith('/failed') for key in reward)
    summary = task_score(reward, sorted(VERIFIER_CASES)) if full_panel else None
    if summary is not None:
        reward['score'] = summary['score']
        reward['reward'] = summary['score']
    write_json(Path(output)/'reward.json', reward)
    grading_payload = {
        'format': 'hydrogym-harbor-physical/v3', 'accepted': True,
        'scope': {('verifier',): 'verifier-suite', ('all',): 'full-39'}.get(tuple(names), 'development-subset'),
        'case_count': metric_count,
        'failed_cases': sorted(k[:-len('/failed')] for k in reward if k.endswith('/failed')),
        'score': summary['score'] if summary else None,
        'reward': summary['score'] if summary else None,
        'aggregate_score': summary['score'] if summary else None,
        'signed_score': summary['signed_score'] if summary else None,
        'signed_improvement': summary,
        'note': ('The full-panel score groups the 26 verifier cases into 10 equal-weight flow families '
                 '(by prefix case.split("_")[0]), averages per-case clipped signed relative improvements '
                 'against paired zero controls within each family, and clips the 10-family mean to [0, 1]. '
                 'Physical measurements retain their original units.')}
    write_json(Path(output)/'grading.json', grading_payload)
    write_json(Path(output)/'report.json', grading_payload)
    return reward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('/data'))
    parser.add_argument('--submission', type=Path, default=Path('/workspace/submission'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', action='append',
                        help='Development subset (repeat), or all for the complete 39; default: the scored verifier suite')
    args = parser.parse_args()
    verify(args.data, args.submission, args.output, args.case)


if __name__ == '__main__':
    main()
