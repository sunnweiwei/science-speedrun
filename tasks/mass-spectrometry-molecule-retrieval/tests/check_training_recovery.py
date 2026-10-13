#!/usr/bin/env python3
"""Real native-model recovery integration check; run in the GLACIER runtime."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['data-root', 'fallback', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=Path('/opt/ms-pred'))
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve().parents[1] / 'environment/baseline/starter/glacier/train.py'
    for name, steps, resume in [('resumed', 2, False), ('resumed', 4, True), ('continuous', 4, False)]:
        command = [sys.executable, str(script), '--data-root', str(args.data_root),
                   '--source-root', str(args.source_root), '--work-dir', str(args.output / name),
                   '--fallback', str(args.fallback), '--submission-dir', str(args.output / (name + '-submission')),
                   '--max-steps', str(steps), '--device', 'cpu', '--workers', str(args.workers), '--batch-size', '1',
                   '--accumulate-grad-batches', '1', '--validate-every', '1', '--checkpoint-every', '500',
                   '--max-seconds', '600', '--trace-batches']
        if resume:
            command += ['--resume', str(args.output / name / 'last.ckpt')]
        with (args.output / f'{name}-{steps}.log').open('w') as log:
            subprocess.run(command, check=True, env=dict(os.environ, OMP_NUM_THREADS='2'), stdout=log, stderr=subprocess.STDOUT)
    resumed = torch.load(args.output / 'resumed/last.ckpt', map_location='cpu', weights_only=False)
    continuous = torch.load(args.output / 'continuous/last.ckpt', map_location='cpu', weights_only=False)
    different = [key for key, value in resumed['state_dict'].items() if not torch.equal(value, continuous['state_dict'][key])]
    same_batches = ((args.output / 'resumed/batch-trace.jsonl').read_bytes() ==
                    (args.output / 'continuous/batch-trace.jsonl').read_bytes())
    report = {'global_steps': [resumed['global_step'], continuous['global_step']],
              'magma_steps': [resumed['task_magma_step'], continuous['task_magma_step']],
              'differing_parameter_tensors': different, 'identical_batch_order_and_rng': same_batches,
              'scope': f'CPU, loader workers={args.workers}, same pinned runtime; not cross-hardware determinism'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if different or not same_batches or report['global_steps'] != [4, 4] or report['magma_steps'] != [4, 4]:
        raise AssertionError(report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
