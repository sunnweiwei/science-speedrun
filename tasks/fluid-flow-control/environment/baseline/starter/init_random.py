"""Publish a genuine random shared checkpoint for end-to-end evaluator testing."""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'evaluation'))
from case_spec import specifications, digest
from evaluate import package_hash
from publish import publish
from shared import build_policy, dimensions


def initialize(submission, seed=0):
    submission = Path(submission)
    if submission.exists() and any(submission.iterdir()):
        raise ValueError('Random initialization requires an empty submission directory')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed must be an integer in [0, 2**32)')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    profiles = specifications()
    config = dimensions(profiles)
    config['initialization'] = {'seed': seed, 'method': 'SB3 ActorCriticPolicy defaults',
        'profiles_sha256': digest(profiles), 'pretrained_weights': False,
        'optimizer_updates': 0}
    policy = build_policy(config)
    policy.eval()
    policy.requires_grad_(False)
    publish(policy, config, submission, training_steps=0)
    (submission / 'README.md').write_text(
        '# Random shared controller — evaluator integration fixture\n\n'
        f'Random initialization seed: {seed}. No training interactions, optimizer '
        'updates, imported checkpoint or baseline weights. One shared task-conditioned '
        'MLP actor/critic with hidden widths [256, 256], vocabulary covering all '
        '39 Research v2 cases; capacities 128 observation and 24 action components. '
        'Standard SB3 initialization, without searching for a favorable seed.\n\n'
        'policy.pyz contains checkpoint.pt, config.json, inference code and fixed '
        'clipping/padding/action conversion. infer.sh works from any directory. '
        'Dependencies: declared task runtime (PyTorch, SB3, Gymnasium, NumPy). '
        'Inference is deterministic and frozen; zero task conditioning for unknown '
        'names. Parameter count and initialization provenance are in config.json. '
        'No optimizer state or training paths are needed.\n\n'
        'This is an untrained interface test, not a jointly trained benchmark '
        'baseline, stability guarantee or competitive performance claim.\n')
    return {'initialization_seed': seed, 'training_steps': 0,
        'parameter_count': sum(p.numel() for p in policy.parameters()),
        'submission_sha256': package_hash(submission),
        'case_capacity': len(config['environments'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(initialize(args.submission, args.seed), indent=2))


if __name__ == '__main__':
    main()
