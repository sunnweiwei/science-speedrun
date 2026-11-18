"""Task score for fluid-flow-control (SSI V3/V4): 10-family prefix mean of clipped signed relative improvement.

The verifier preserves each physical metric. This module defines the V3/V4 task score from
those values and the paired zero-control anchors (`environment/references/specialists/verified/results.json`,
or calibrated blind-seed anchors when `HYDROGYM_BLIND_SEEDS_FILE` carries `"calibrated": true`).

Per-case relative improvement against zero control `z`:
  gain = clip(((value - z) / |z|) * (-1 if lower_is_better else +1), -1.0, 1.0)
  a failed or nonfinite case counts -1.0
Cases are grouped by prefix `case.split("_")[0]` (10 flow families across the 26 verifier cases,
equal 1/10 weight per family):
  signed_score = mean_family(mean_case_in_family(gain))
  score = clip(signed_score, 0.0, 1.0)
"""
import json
import math
import os
from pathlib import Path
from statistics import mean

ANCHORS = Path(__file__).resolve().parents[1] / 'references/specialists/verified/results.json'
BLIND_SEEDS_ENV = 'HYDROGYM_BLIND_SEEDS_FILE'


def anchors(path=None):
    if path is None:
        path = ANCHORS if ANCHORS.is_file() else Path(__file__).resolve().parents[1]/'references/results/results.json'
    rows = {row['case']: dict(row) for row in json.loads(Path(path).read_text())['rows']}
    blind_file = os.environ.get(BLIND_SEEDS_ENV, '').strip()
    if blind_file and Path(blind_file).is_file():
        blind = json.loads(Path(blind_file).read_text())
        if blind.get('calibrated') is True and isinstance(blind.get('zero_control_anchors'), dict):
            for case, z in blind['zero_control_anchors'].items():
                if case in rows:
                    rows[case] = dict(rows[case], zero=float(z))
    return rows


def improvement(value, zero, direction):
    if not all(type(x) in (int, float) and math.isfinite(x) for x in (value, zero)) or zero == 0:
        raise ValueError('Finite value and nonzero zero-control anchor required')
    if direction not in ('lower_is_better', 'higher_is_better'):
        raise ValueError(f'Unknown physical metric direction: {direction}')
    gain = (float(value) - float(zero)) / abs(float(zero))
    if direction == 'lower_is_better':
        gain = -gain
    return max(-1., min(1., gain))


def task_score(rewards, cases, rows=None):
    """rewards: flat verifier reward.json; cases: scored case names. Returns V3 10-family score and diagnostics."""
    if not cases:
        raise ValueError('At least one scored case is required')
    rows = rows or anchors()
    per_case = {}
    groups = {}
    for case in cases:
        metric = rows[case]['primary_metric']
        key = f"{case}/{metric['id']}"
        val = rewards.get(key)
        if (rewards.get(f'{case}/failed') or val is None
                or type(val) not in (int, float) or not math.isfinite(float(val))):
            gain = -1.
        else:
            gain = improvement(val, rows[case]['zero'], metric['direction'])
        per_case[case] = gain
        groups.setdefault(case.split('_')[0], []).append(gain)
    by_family = {family: mean(values) for family, values in groups.items()}
    signed_score = mean(by_family.values())
    flat_mean = mean(per_case.values())
    score = max(0., min(1., signed_score))
    return {
        'score': score,
        'signed_score': signed_score,
        'raw_mean_improvement': signed_score,
        'flat_case_mean_improvement': flat_mean,
        'families': len(by_family),
        'by_family': by_family,
        'cases': len(per_case),
        'per_case': per_case,
    }
