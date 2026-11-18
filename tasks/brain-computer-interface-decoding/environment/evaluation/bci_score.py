"""Shared development/verifier metrics for the 4-track V4 BCI benchmark. No model code is imported here."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

# V4 scores ONLY the 4 subject/patient-disjoint EEG tracks; EOG-confounded/stimulus-leaking FACED is removed.
CLASSES = {'tuev': 6, 'ad65': 3, 'openbmi': 2, 'longitudinal': 2}
CHANCE = {'tuev': 1.0 / 6.0, 'ad65': 1.0 / 3.0, 'openbmi': 0.5, 'longitudinal': 0.5}
AGGREGATE_TRACKS = ('tuev', 'ad65', 'openbmi', 'longitudinal')
DEPRECATED_TRACKS = {
    'faced': 'Removed in V4: recurring video stimuli across subjects leak clip identity via EOG/visual artifacts.'
}


def chance_adjusted(track, reward):
    if track not in CHANCE:
        raise ValueError(f'unsupported or deprecated BCI track: {track}')
    c = CHANCE[track]
    return (float(reward) - c) / (1.0 - c)


def aggregate(rewards):
    """Unclipped signed mean of chance-adjusted subject/session balanced accuracy over the 4 tracks."""
    if any(t not in rewards or isinstance(rewards[t], bool) or not np.isfinite(rewards[t]) for t in AGGREGATE_TRACKS):
        return None
    return float(np.mean([chance_adjusted(t, rewards[t]) for t in AGGREGATE_TRACKS]))


def composite_score(rewards):
    """V3/V4 primary composite score matching harness/ssi_index_v3.py:bci()."""
    signed = aggregate(rewards)
    if signed is None:
        return None
    clipped = float(np.clip(signed, 0.0, 1.0))
    terms = {t: chance_adjusted(t, rewards[t]) for t in AGGREGATE_TRACKS}
    return {'score': clipped, 'reward': clipped, 'signed_score': signed, 'terms': terms}


def read_predictions(path, truth, classes):
    expected = {r['id'] for r in truth}
    if len(expected) != len(truth):
        raise ValueError('duplicate truth IDs')
    values = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if not isinstance(r, dict) or not isinstance(r.get('id'), str) or not isinstance(r.get('probabilities'), list):
                raise ValueError('invalid row schema')
            sid = r['id']
            if sid in values or sid not in expected:
                raise ValueError('duplicate or unexpected ID')
            p = np.asarray(r['probabilities'], dtype=float)
            if p.shape != (classes,) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.isclose(p.sum(), 1, atol=1e-5):
                raise ValueError('invalid probabilities')
            values[sid] = p
    if set(values) != expected:
        raise ValueError('missing predictions')
    return np.stack([values[r['id']] for r in truth])


def metrics(y, p):
    pred = p.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=range(p.shape[1]))
    return {
        'class_counts': cm.sum(1).tolist(),
        'predicted_class_counts': cm.sum(0).tolist(),
        'confusion_matrix': cm.tolist(),
        'per_class_recall': np.divide(
            cm.diagonal(), cm.sum(1), out=np.zeros(p.shape[1], dtype=float), where=cm.sum(1) != 0
        ).tolist(),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'accuracy': float(accuracy_score(y, pred)),
        'weighted_f1': float(f1_score(y, pred, average='weighted', zero_division=0)),
        'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'kappa': float(cohen_kappa_score(y, pred)) if len(np.unique(y)) > 1 else None,
    }


def score(track, rows, p):
    if track not in CLASSES:
        raise ValueError(f'unsupported or deprecated BCI track: {track}')
    if not rows:
        raise ValueError('empty truth')
    y = np.array([r['label'] for r in rows], dtype=int)
    if p.shape != (len(rows), CLASSES[track]):
        raise ValueError('shape mismatch')
    result = metrics(y, p)
    result['samples'] = len(rows)
    result['window_balanced_accuracy'] = result['balanced_accuracy']
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r.get('subject', 'unknown')].append(i)
    if track == 'ad65':
        sy = []
        sp = []
        for idx in groups.values():
            if len(set(y[idx])) != 1:
                raise ValueError('inconsistent patient label')
            sy.append(y[idx[0]])
            sp.append(p[idx].mean(0))
        result['subject_metrics'] = metrics(np.array(sy), np.array(sp))
        result['reward'] = result['subject_metrics']['balanced_accuracy']
    elif track in ('openbmi', 'longitudinal'):
        by_subject = {}
        for subject, idx in groups.items():
            if track == 'openbmi':
                by_subject[subject] = float(balanced_accuracy_score(y[idx], p[idx].argmax(1)))
            else:
                sessions = defaultdict(list)
                for i in idx:
                    sessions[rows[i]['session']].append(i)
                vals = []
                for ii in sessions.values():
                    if set(y[ii]) != {0, 1}:
                        raise ValueError('session lacks an eye condition')
                    vals.append(float(balanced_accuracy_score(y[ii], p[ii].argmax(1))))
                by_subject[subject] = float(np.mean(vals))
        result['subject_scores'] = by_subject
        result['reward'] = float(np.mean(list(by_subject.values())))
    else:
        result['reward'] = result['balanced_accuracy']
    result['primary_balanced_accuracy'] = result['reward']
    result['chance_adjusted_skill'] = chance_adjusted(track, result['reward'])
    result['subjects'] = len(groups)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', choices=CLASSES, required=True)
    ap.add_argument('--truth', type=Path, required=True)
    ap.add_argument('--predictions', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.truth.read_text().splitlines()]
    p = read_predictions(a.predictions, rows, CLASSES[a.track])
    a.output.write_text(json.dumps(score(a.track, rows, p), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
