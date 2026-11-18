"""Summarize verified five-seed full runs without treating partial runs as results."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

MODELS = ('eegnet', 'contrawr', 'cbramod', 'codebrain')
TRACKS = ('tuev', 'ad65', 'openbmi', 'longitudinal')
SEEDS = range(5)
SUMMARY_KEYS = frozenset({'score', 'reward', 'signed_score'})
# TUEV uses one official-test metadata group; this is not a patient count.
TEST_SUBJECT_COUNTS = {'tuev': 1, 'ad65': 18,
                       'openbmi': 11, 'longitudinal': 9}
TEST_COUNTS = {'tuev': 29421, 'ad65': 1376,
               'openbmi': 4400, 'longitudinal': 10028}


def package_hash(source):
    source = source.resolve()
    h = hashlib.sha256()
    for p in sorted(source.rglob('*')):
        if p.is_symlink():
            raise ValueError('package symlink is not allowed')
        if p.is_file():
            h.update(str(p.relative_to(source)).encode()); h.update(b'\0')
            with p.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
    return h.hexdigest()


def checked_score(root, jobs, model, track, seed):
    key = f'{model}-{track}-s{seed}'
    job = jobs.get(key)
    if not job or job.get('status') != 'SUCCEED' or job.get('verified') is not True:
        return None
    run = root / key
    receipt = json.loads((run / 'submission' / f'{track}-full-run.json').read_text())
    if receipt.get('status') != 'complete' or receipt.get('completed_epochs') != 50:
        raise ValueError(f'{key}: missing 50-epoch completion receipt')
    if any(receipt.get(field) != value for field, value in
           (('model', model), ('track', track), ('seed', seed))):
        raise ValueError(f'{key}: training identity mismatch')
    package = run / 'submission' / track
    config = json.loads((package / 'config.json').read_text())
    training = json.loads((package / 'training.json').read_text())
    if (config.get('model') != model or config.get('track') != track
            or training.get('seed') != seed or training.get('scope') != 'full-training'):
        raise ValueError(f'{key}: package metadata mismatch')
    if (run / 'score.exit').read_text().strip() != '0':
        raise ValueError(f'{key}: verifier failed')
    reward = json.loads((run / 'score' / 'reward.json').read_text())
    report = json.loads((run / 'score' / 'report.json').read_text())
    if (report.get('scope') != 'full'
            or (set(reward) - SUMMARY_KEYS) != {track} or set(report.get('tracks', {})) != {track}):
        raise ValueError(f'{key}: smoke/subset score is not a full result')
    score = reward.get(track)
    detail = report.get('tracks', {}).get(track, {})
    if (not isinstance(score, (float, int)) or isinstance(score, bool)
            or not math.isfinite(score) or not 0 <= score <= 1
            or detail.get('status') != 'ok'
            or not math.isclose(score, detail.get('reward', float('nan')), abs_tol=1e-12)
            or detail.get('sample_count') != TEST_COUNTS[track]
            or not isinstance(detail.get('subject_count'), int)
            or isinstance(detail['subject_count'], bool)
            or detail['subject_count'] != TEST_SUBJECT_COUNTS[track]
            or not isinstance(detail.get('data_identity'), str)
            or len(detail['data_identity']) != 64
            or detail.get('data_identity') != job.get('test_data_identity')
            or any(c not in '0123456789abcdef' for c in detail['data_identity'])
            or detail.get('package_sha256') != package_hash(package)
            or report.get('package_sha256') != detail.get('package_sha256')):
        raise ValueError(f'{key}: invalid or inconsistent private score')
    return float(score)


def summarize(root, models=MODELS):
    jobs = json.loads((root / 'campaign.json').read_text())['jobs']
    result = {'schema': 'bci-five-seed-summary-v1', 'expected_jobs': len(models) * len(TRACKS) * len(SEEDS),
              'verified_jobs': 0, 'groups': {}}
    for track in TRACKS:
        result['groups'][track] = {}
        for model in models:
            scores = {}
            pending = []
            for seed in SEEDS:
                key = f'{model}-{track}-s{seed}'
                try:
                    score = checked_score(root, jobs, model, track, seed)
                except FileNotFoundError:
                    score = None
                if score is None:
                    pending.append(key)
                else:
                    scores[str(seed)] = score
                    result['verified_jobs'] += 1
            row = {'status': 'complete' if not pending else 'pending',
                   'scores': scores, 'pending': pending}
            if not pending:
                values = [scores[str(seed)] for seed in SEEDS]
                row['mean'] = statistics.mean(values)
                row['sample_std'] = statistics.stdev(values)
            result['groups'][track][model] = row
    result['complete_groups'] = sum(row['status'] == 'complete'
                                    for models in result['groups'].values()
                                    for row in models.values())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):parser.error('duplicate models')
    result = summarize(args.root, tuple(args.models))
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end='')
    if args.require_complete and result['complete_groups'] != len(args.models) * len(TRACKS):
        raise SystemExit(f"only {result['complete_groups']}/{len(args.models) * len(TRACKS)} model-track groups complete")


if __name__ == '__main__':
    main()
