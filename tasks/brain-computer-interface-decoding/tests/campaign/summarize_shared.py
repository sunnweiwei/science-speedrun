"""Accept only complete, bound, five-track shared-model scoring receipts."""
import argparse
import json
import math
import statistics
from pathlib import Path

from summarize_full import MODELS, SEEDS, SUMMARY_KEYS, TRACKS, TEST_COUNTS, package_hash


def checked_run(root, campaign, family, seed, expected_epochs=50):
    if type(expected_epochs) is not int or expected_epochs not in (50, 100):
        raise ValueError('expected epochs must be 50 or 100')
    key = f'{family}-s{seed}'
    run = root / key
    receipt = json.loads((run / 'submission-full-run.json').read_text())
    package = run / 'submission'
    config = json.loads((package / 'config.json').read_text())
    training = json.loads((package / 'training.json').read_text())
    report = json.loads((run / 'score' / 'report.json').read_text())
    reward = json.loads((run / 'score' / 'reward.json').read_text())
    if (receipt.get('status') != 'complete' or receipt.get('completed_epochs') != expected_epochs
            or receipt.get('model') != family or receipt.get('seed') != seed
            or config.get('schema') != 'bci-shared-package-v2' or config.get('family') != family
            or config.get('pipeline_version') != 'shared-v2.1-channel-preserving'
            or training.get('pipeline_version') != config.get('pipeline_version')
            or receipt.get('pipeline_version') != config.get('pipeline_version')
            or training.get('model') != family or training.get('seed') != seed
            or training.get('scope') != 'full'
            or receipt.get('best_epoch') != training.get('best_epoch')
            or receipt.get('best_validation_mean') != training.get('best_validation_mean')
            or training.get('data_identity') != campaign.get('development_identity')
            or report.get('scope') != 'full' or set(report.get('tracks', {})) != set(TRACKS)
            or (set(reward) - SUMMARY_KEYS) != set(TRACKS)
            or (run / 'score.exit').read_text().strip() != '0'):
        raise ValueError(f'{key}: incomplete or mismatched full run')
    if (set(receipt.get('training_coverage', {})) != set(TRACKS)
            or any(receipt['training_coverage'][t].get('complete_passes', 0) < 1 for t in TRACKS)):
        raise ValueError(f'{key}: not all training examples were visited')
    actual_hash = package_hash(package)
    if report.get('package_sha256') != actual_hash:
        raise ValueError(f'{key}: package hash mismatch')
    scores = {}
    for track in TRACKS:
        detail = report['tracks'][track]
        value = reward[track]
        if (detail.get('status') != 'ok'
                or isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1
                or not math.isclose(value, detail.get('reward', float('nan')), abs_tol=1e-12)
                or detail.get('sample_count') != TEST_COUNTS[track]
                or detail.get('subject_count') != campaign.get('test_subject_counts', {}).get(track)
                or detail.get('data_identity') != campaign.get('test_data_identity', {}).get(track)
                or detail.get('package_sha256') != actual_hash):
            raise ValueError(f'{key}/{track}: incomplete or inconsistent private score')
        scores[track] = float(value)
    return scores


def summarize(root, models=MODELS, expected_epochs=50):
    campaign = json.loads((root / 'campaign.json').read_text())
    if set(campaign.get('test_data_identity', {})) != set(TRACKS):
        raise ValueError('missing frozen private data identity')
    result = {'schema': 'bci-shared-five-seed-summary-v1', 'expected_epochs': expected_epochs, 'groups': {}}
    for family in models:
        scores, pending = {}, []
        for seed in SEEDS:
            try:
                scores[str(seed)] = checked_run(root, campaign, family, seed, expected_epochs)
            except FileNotFoundError:
                pending.append(seed)
        row = {'status': 'complete' if not pending else 'pending',
               'scores': scores, 'pending': pending}
        if not pending:
            row['mean'] = {track: statistics.mean(scores[str(seed)][track] for seed in SEEDS)
                           for track in TRACKS}
            row['sample_std'] = {track: statistics.stdev(scores[str(seed)][track] for seed in SEEDS)
                                 for track in TRACKS}
        result['groups'][family] = row
    result['complete_families'] = sum(row['status'] == 'complete' for row in result['groups'].values())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--expected-epochs', type=int, choices=(50, 100), default=50)
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error('duplicate models')
    result = summarize(args.root, tuple(args.models), args.expected_epochs)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if args.require_complete and result['complete_families'] != len(args.models):
        raise SystemExit('shared baseline campaign incomplete')


if __name__ == '__main__':
    main()
