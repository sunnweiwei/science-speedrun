#!/usr/bin/env python3
"""Run a portable fixed hybrid on arbitrary conforming public input."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input_root', type=Path)
    p.add_argument('output_root', type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    config = json.loads((root / 'model_config.json').read_text())
    manifest = json.loads((root / 'model_manifest.json').read_text())
    if digest(root / 'glacier.ckpt') != manifest['checkpoint']['sha256']:
        raise ValueError('GLACIER checkpoint checksum mismatch')
    fallback_manifest = json.loads((root / 'fallback' / 'model_manifest.json').read_text())
    if digest(root / 'fallback' / fallback_manifest['checkpoint']['path']) != config['fallback_checkpoint_sha256']:
        raise ValueError('fallback checkpoint checksum mismatch')
    env = dict(os.environ, PYTHONPATH=str(root / 'vendor'), PYTHONDONTWRITEBYTECODE='1',
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', ''))

    def run(script, *arguments):
        subprocess.run([sys.executable, str(script), *map(str, arguments)], check=True, env=env)

    args.output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='massspecgym-infer-') as temporary:
        tmp = Path(temporary)
        run(root / 'stage_input.py', '--input-root', args.input_root, '--output-root', tmp / 'partition')
        glacier_input = tmp / 'partition' / 'glacier'
        fallback_input = tmp / 'partition' / 'fallback'
        (tmp / 'fallback').mkdir()
        if (fallback_input / 'queries.jsonl').stat().st_size:
            subprocess.run(['bash', str(root / 'fallback' / 'infer.sh'),
                            str(fallback_input), str(tmp / 'fallback')], check=True, env=env)
        else:
            (tmp / 'fallback' / 'rankings.jsonl').touch()
        candidates = tmp / 'candidates.tsv'
        run(root / 'prepare_glacier_candidates.py', '--input-root', glacier_input,
            '--output', candidates, '--eligible-only')
        with candidates.open() as f:
            has_candidates = next(csv.DictReader(f, delimiter='\t'), None) is not None
        if has_candidates:
            import torch
            gpu_flag = ['--gpu'] if torch.cuda.is_available() else []
            run(root / 'vendor' / 'ms_pred' / 'glacier' / 'predict_smis_joint.py',
                '--dataset-labels', candidates, '--checkpoint', root / 'glacier.ckpt',
                '--save-dir', tmp / 'predictions', '--out-name', 'preds.hdf5',
                '--sparse-out', '--sparse-k', 100, '--batch-size', config['batch_size'],
                '--num-cpu-workers', config['workers'], '--feat-cache-size', config['feature_cache_size'],
                '--prefetch-factor', 1, *gpu_flag)
            run(root / 'score_glacier_predictions.py', '--input-root', glacier_input,
                '--pred-file', tmp / 'predictions' / 'preds.hdf5', '--output-root', tmp / 'glacier',
                '--distance', config['distance'])
        else:
            (tmp / 'glacier').mkdir()
            (tmp / 'glacier' / 'rankings.jsonl').touch()
        run(root / 'merge_glacier_fallback.py', '--input-root', args.input_root,
            '--glacier-query-id-file', tmp / 'partition' / 'glacier-query-ids.txt',
            '--glacier', tmp / 'glacier' / 'rankings.jsonl',
            '--fallback', tmp / 'fallback' / 'rankings.jsonl', '--output-root', args.output_root)
        # The adapter also prints this diagnostic to stdout. The model output
        # directory itself must contain only the contracted rankings file.
        (args.output_root / 'runtime.json').unlink()


if __name__ == '__main__':
    main()
