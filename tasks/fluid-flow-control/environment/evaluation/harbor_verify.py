"""Harbor bridge: independently replay one package and export physical metrics.

No reward anchors, suite.json, candidate-provided scores, or mixed-unit aggregate.
The default is all 39 cases. Explicit subsets are development checks only.
"""
import argparse
import math
from pathlib import Path

from case_spec import select_cases
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
    result = evaluator(data, submission, output, names)
    try:
        reward = physical_rewards(result, names)
    except (InfrastructureError, PolicyError) as error:
        write_json(Path(output)/'grading-error.json', {
            'status': 'policy_error' if isinstance(error, PolicyError) else 'infrastructure_error',
            'error': str(error), 'accepted': False})
        if isinstance(error, PolicyError):
            # The package is invalid: Harbor needs a numeric reward.json; there are no physical metrics.
            write_json(Path(output)/'reward.json', {'invalid_submission': 1})
            return {'invalid_submission': 1}
        raise
    write_json(Path(output)/'reward.json', reward)
    write_json(Path(output)/'grading.json', {
        'format': 'hydrogym-harbor-physical/v1', 'accepted': True,
        'scope': {('verifier',): 'verifier-suite', ('all',): 'full-39'}.get(tuple(names), 'development-subset'),
        'case_count': len(reward) - 1 - sum(k.endswith('/failed') for k in reward),
        'failed_cases': sorted(k[:-len('/failed')] for k in reward if k.endswith('/failed')),
        'aggregate_score': None,
        'note': 'Raw per-case primary metrics. Follow each metric direction; do not average unlike units.'})
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
