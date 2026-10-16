"""Compare complete recorded episodes, without declaring paper equivalence.

This quantifies repeatability; it does not decide which differences are
acceptable for a chaotic CFD benchmark. Field selection must be explicit when
comparing runs with different sensor protocols (for example force-only audits).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


FIELDS = ('reward', 'drag', 'lift', 'normalized_action', 'physical_control',
          'observation', 'next_observation', 'force_coefficients', 'raw_deviation_l2',
          'raw_probe_velocity')


def shape(value):
    if not isinstance(value, list):
        return ()
    children = [shape(child) for child in value]
    if children and any(child != children[0] for child in children):
        raise ValueError('Ragged physical arrays are not comparable')
    return (len(value),) + (children[0] if children else ())


def flatten(value):
    if isinstance(value, list):
        return tuple(item for child in value for item in flatten(child))
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError('Expected finite numeric trajectory values')
    return (float(value),)


def compare(left, right, expected_steps, fields=None):
    if expected_steps <= 0 or len(left) != expected_steps or len(right) != expected_steps:
        raise ValueError('Both selected episodes must be complete at the expected horizon')
    for rows in (left, right):
        if [row['step'] for row in rows] != list(range(1, expected_steps + 1)):
            raise ValueError('Episode steps must be unique, contiguous and ordered')
    if fields is None:
        fields = [name for name in FIELDS if name in left[0] or name in right[0]]
    if not fields:
        raise ValueError('No physical or action fields selected')
    result = {}
    for field in fields:
        differences, per_step = [], []
        for old, new in zip(left, right, strict=True):
            if field not in old or field not in new:
                raise ValueError(f'Missing field {field}; specify a justified common field set')
            if shape(old[field]) != shape(new[field]):
                raise ValueError(f'Incompatible field dimensions: {field}')
            a, b = flatten(old[field]), flatten(new[field])
            if not a or len(a) != len(b):
                raise ValueError(f'Incompatible field dimensions: {field}')
            delta = [abs(x - y) for x, y in zip(a, b, strict=True)]
            if not all(math.isfinite(x) for x in delta):
                raise ValueError(f'Nonfinite difference: {field}')
            differences.extend(delta)
            per_step.append(max(delta))
        result[field] = dict(max_abs_error=max(differences),
            rms_error=math.hypot(*differences) / math.sqrt(len(differences)),
            differing_values=sum(value != 0 for value in differences),
            first_exact_difference_step=next((i + 1 for i, value in enumerate(per_step) if value), None),
            first_error_above_1e_6_step=next((i + 1 for i, value in enumerate(per_step) if value > 1e-6), None))
    return dict(status='complete-trajectory-comparison-not-paper-certification',
        steps=expected_steps, fields=result,
        exact_numeric_equality=all(row['differing_values'] == 0 for row in result.values()),
        left_raw_return=math.fsum(flatten(row['reward'])[0] for row in left),
        right_raw_return=math.fsum(flatten(row['reward'])[0] for row in right))


def selected(path, episode, training_step):
    payload = path.read_bytes()
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    rows = [row for row in rows if (episode is None or row['episode'] == episode)
            and (training_step is None or row['training_step'] == training_step)]
    return rows, dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(),
                      episode=episode, training_step=training_step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    for side in ('left', 'right'):
        parser.add_argument(f'--{side}-episode', type=int)
        parser.add_argument(f'--{side}-training-step', type=int)
    parser.add_argument('--expected-steps', type=int, required=True)
    parser.add_argument('--fields', nargs='+', choices=FIELDS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    left, a = selected(args.left, args.left_episode, args.left_training_step)
    right, b = selected(args.right, args.right_episode, args.right_training_step)
    result = compare(left, right, args.expected_steps, args.fields)
    result.update(left=a, right=b)
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(payload)
    print(payload, end='')


if __name__ == '__main__':
    main()
