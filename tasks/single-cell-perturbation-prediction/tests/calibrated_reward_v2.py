"""Task reward calibration for single-cell-perturbation-prediction.

V3 primary score (`score_context_v3` / `aggregate_contexts_v3`, emitted by `grader.py` as
`reward.json["score"]` and `reward.json["reward"]`):
  - Per context `c`:
      amplitude_skill      = (negative_loss - WMSE) / (negative_loss - repeat_loss)
      identification_skill = (NIR - 0.5) / (repeat_NIR - 0.5)
      signed_context       = 0.5 * amplitude_skill + 0.5 * identification_skill
      context_score        = max(0.0, signed_context)
  - Overall score = clip01(mean_c(context_score))

Underlying cross-fitted WMSE and negative/positive repeat calibration follow Shift Bioscience,
Nature Biotechnology 2026 (DOI: 10.1038/s41587-026-03307-w); perturbation-specific identification
via NIR addresses cross-perturbation mean collapse documented by Ahlmann-Eltze et al.,
Nature Methods 2025 (DOI: 10.1038/s41592-025-02772-6).
"""
import math

VERSION = 'ssi-perturbation-weighted-error-skill/2.0-rc1'
V3_VERSION = 'ssi-vcc-dual-skill-v3/3.0'


def _loss(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Loss must be finite and nonnegative')
    return float(value)


def _unit_interval(value, name='Metric'):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not (0.0 <= float(value) <= 1.0):
        raise ValueError(f'{name} must be finite in [0, 1]')
    return float(value)


def score_context(*, loss, negative_loss, repeat_loss, gap_lower95, status='valid'):
    """Trusted statuses only (v2 WMSE-only skill retained for report continuity).

    A data/infrastructure error produces no numeric reward. An invalid submission
    receives the floor, never a better value than any valid bounded reward.
    """
    negative_loss = _loss(negative_loss)
    repeat_loss = _loss(repeat_loss)
    gap = negative_loss - repeat_loss
    if gap <= 0 or not math.isfinite(gap):
        raise ValueError('Reference lacks a positive recoverable signal gap')
    if _loss(gap_lower95) <= 0 or gap_lower95 > gap:
        raise ValueError('Reference calibration lacks a positive lower confidence bound')
    if status == 'infrastructure_error':
        raise RuntimeError('Infrastructure failure: no reward should be emitted')
    if status == 'invalid_submission':
        return {'version': VERSION, 'status': status, 'reward': 0., 'raw_skill': None}
    if status != 'valid':
        raise ValueError('Unknown trusted status')
    loss = _loss(loss)
    raw = (negative_loss - loss) / gap
    if not math.isfinite(raw):
        raise ValueError('Nonfinite calibrated result; no numeric reward is valid')
    return {'version': VERSION, 'status': status, 'loss': loss,
            'negative_loss': negative_loss, 'repeat_loss': repeat_loss,
            'gap_lower95': gap_lower95,
            'raw_skill': raw, 'reward': min(1., max(0., raw))}


def aggregate_contexts(contexts, *, expected_contexts):
    expected = tuple(expected_contexts)
    if not expected or len(set(expected)) != len(expected) or set(contexts) != set(expected):
        raise ValueError('Exact frozen context set required; no available-case averaging')
    scored = {key: score_context(**contexts[key]) for key in expected}
    return {'version': VERSION, 'reward': sum(row['reward'] for row in scored.values()) / len(expected),
            'contexts': scored, 'aggregation': 'equal weight over the frozen context set'}


def score_context_v3(
    *,
    loss,
    negative_loss,
    repeat_loss,
    gap_lower95,
    nir,
    repeat_nir,
    targets=None,
    status='valid',
):
    """Exact SSI V3 context score: pre-clip mean of signed amplitude & identification skills,
    followed by a zero floor `context_score = max(0.0, signed_context)`."""
    negative_loss = _loss(negative_loss)
    repeat_loss = _loss(repeat_loss)
    gap = negative_loss - repeat_loss
    if gap <= 0 or not math.isfinite(gap):
        raise ValueError('Reference lacks a positive recoverable signal gap')
    if _loss(gap_lower95) <= 0 or gap_lower95 > gap:
        raise ValueError('Reference calibration lacks a positive lower confidence bound')
    repeat_nir_val = _unit_interval(repeat_nir, 'repeat_nir')
    if repeat_nir_val <= 0.5:
        raise ValueError('Reference repeat_nir must exceed chance anchor 0.5')
    if status == 'infrastructure_error':
        raise RuntimeError('Infrastructure failure: no reward should be emitted')
    if status == 'invalid_submission':
        return {
            'version': V3_VERSION,
            'status': status,
            'WMSE': None,
            'amplitude_skill': None,
            'NIR': None,
            'repeat_NIR': repeat_nir_val,
            'identification_skill': None,
            'signed_context': None,
            'context_score': 0.0,
            'targets': targets,
        }
    if status != 'valid':
        raise ValueError('Unknown trusted status')
    loss_val = _loss(loss)
    nir_val = _unit_interval(nir, 'nir')
    amplitude_skill = (negative_loss - loss_val) / gap
    identification_skill = (nir_val - 0.5) / (repeat_nir_val - 0.5)
    # CRITICAL: average signed skills BEFORE clipping at zero!
    signed_context = 0.5 * (amplitude_skill + identification_skill)
    if not all(math.isfinite(x) for x in (amplitude_skill, identification_skill, signed_context)):
        raise ValueError('Nonfinite V3 calibrated context result')
    context_score = max(0.0, signed_context)
    return {
        'version': V3_VERSION,
        'status': status,
        'WMSE': loss_val,
        'negative_loss': negative_loss,
        'repeat_loss': repeat_loss,
        'gap_lower95': gap_lower95,
        'amplitude_skill': amplitude_skill,
        'NIR': nir_val,
        'repeat_NIR': repeat_nir_val,
        'identification_skill': identification_skill,
        'signed_context': signed_context,
        'context_score': context_score,
        'targets': targets,
    }


def aggregate_contexts_v3(contexts, *, expected_contexts):
    expected = tuple(expected_contexts)
    if not expected or len(set(expected)) != len(expected) or set(contexts) != set(expected):
        raise ValueError('Exact frozen context set required; no available-case averaging')
    scored = {key: score_context_v3(**contexts[key]) for key in expected}
    signed_mean = sum(row['context_score'] for row in scored.values()) / len(expected)
    score = min(1.0, max(0.0, signed_mean))
    return {
        'version': V3_VERSION,
        'score': score,
        'reward': score,
        'signed_score': signed_mean,
        'contexts': scored,
        'aggregation': 'equal weight over the frozen context set; per-context max(0, 0.5*(amplitude_skill + identification_skill))',
    }
