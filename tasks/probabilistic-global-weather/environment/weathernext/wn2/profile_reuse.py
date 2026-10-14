"""Explain profile invalidation using the same stage sources as the collector."""
import hashlib

TIMING_SOURCES = ('train_stage.py', 'native_model.py', 'training_data.py',
                  'wn2_validation.py', 'wn2_distributed.py', 'plan_schedule.py')


def invalidation_reasons(report, stage, code):
    reasons = []
    if report.get('status') != 'passed' or not report.get('profile'):
        reasons.append('not a passed timing profile')
    if report.get('stage') != stage:
        reasons.append('different curriculum stage')
    for name in TIMING_SOURCES:
        path = code / name
        if not path.is_file():
            reasons.append(f'missing source: {name}')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != report.get('source_sha256', {}).get(name):
            reasons.append(f'computation source changed: {name}')
    return reasons
