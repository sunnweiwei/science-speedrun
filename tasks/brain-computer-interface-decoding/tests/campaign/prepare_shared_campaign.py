"""Freeze development and private identities for an operator-side campaign."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'environment/evaluation'))
from data_identity import split_identity

TRACKS = ('faced', 'tuev', 'ad65', 'openbmi', 'longitudinal')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    development = {t: {split: split_identity(args.data / t / split)
                       for split in ('train', 'val')} for t in TRACKS}
    test_identity, subject_counts = {}, {}
    for track in TRACKS:
        truth = args.private / track / 'truth.jsonl'
        manifest = args.private / track / 'public/manifest.json'
        test_identity[track] = split_identity(manifest.parent, truth)
        rows = [json.loads(line) for line in truth.read_text().splitlines()]
        subject_counts[track] = len({row.get('subject') for row in rows})
    campaign = {'schema': 'bci-shared-campaign-v1', 'development_identity': development,
                'test_data_identity': test_identity, 'test_subject_counts': subject_counts}
    if args.output.exists() and json.loads(args.output.read_text()) != campaign:
        raise ValueError('frozen campaign identity changed; use a fresh output directory')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(campaign, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, args.output)
    print(json.dumps({'test_subject_counts': subject_counts}))


if __name__ == '__main__':
    main()
