#!/usr/bin/env python3
"""Export a trained native GLACIER plus a trained all-condition fallback."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from publication import publish_directory, snapshot_package


DEFAULT_INFERENCE_BATCH_SIZE = 64


def positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('inference batch size must be a positive integer') from error
    if result <= 0:
        raise argparse.ArgumentTypeError('inference batch size must be a positive integer')
    return result


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def publish(checkpoint, fallback, source_root, destination, reference_scripts=None, *,
            inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE) -> None:
    if type(inference_batch_size) is not int or inference_batch_size <= 0:
        raise ValueError('inference batch size must be a positive integer')
    here = Path(__file__).resolve().parent
    reference_scripts = reference_scripts or here.parents[2] / 'references' / 'scripts'
    checkpoint, fallback, source_root = map(Path, [checkpoint, fallback, source_root])
    if Path(destination).resolve().is_relative_to(fallback.resolve()):
        raise ValueError('freeze the fallback outside the native publication directory')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    training = state.get('task_training', {})
    if training.get('initialization') != 'random' or state.get('global_step', 0) <= 0:
        raise ValueError('expected a task-trained random-initialized checkpoint, not released weights')
    fallback_manifest = json.loads((fallback / 'model_manifest.json').read_text())
    if fallback_manifest['training_provenance']['pretrained_assets']:
        raise ValueError('fallback must obey the no-pretrained-assets rule')
    fallback_checkpoint = fallback / fallback_manifest['checkpoint']['path']
    if sha256(fallback_checkpoint) != fallback_manifest['checkpoint']['sha256']:
        raise ValueError('fallback checkpoint hash mismatch')

    def build(stage):
        export = {k: state[k] for k in ['state_dict', 'hyper_parameters', 'pytorch-lightning_version', 'global_step', 'epoch']}
        export['task_training'] = training
        torch.save(export, stage / 'glacier.ckpt')
        snapshot_package(fallback, stage / 'fallback')
        shutil.copytree(source_root / 'src' / 'ms_pred', stage / 'vendor' / 'ms_pred',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copyfile(source_root / 'LICENSE', stage / 'LICENSE.ms-pred')
        for name in ['glacier_contract.py', 'prepare_glacier_candidates.py', 'stage_glacier_fallback.py',
                     'score_glacier_predictions.py', 'merge_glacier_fallback.py']:
            shutil.copyfile(reference_scripts / name, stage / name)
        shutil.copyfile(here / 'infer.py', stage / 'infer.py')
        shutil.copyfile(here / 'stage_input.py', stage / 'stage_input.py')
        (stage / 'infer.sh').write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\n'
            'package_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
            'exec "${MASSSPECGYM_GLACIER_PYTHON:-/opt/glacier/bin/python}" "$package_root/infer.py" "$@"\n'
        )
        (stage / 'infer.sh').chmod(0o755)
        config = {'method': 'native-glacier-with-trained-deepsets-fallback',
                  'routing': 'released_conditions_with_usable_observation_and_supported_pool', 'distance': 'entropy',
                  'batch_size': inference_batch_size, 'workers': 2, 'feature_cache_size': 64,
                  'fallback_checkpoint_sha256': sha256(fallback_checkpoint),
                  'inference_runtime': '/opt/glacier/bin/python',
                  'training': training}
        (stage / 'model_config.json').write_text(json.dumps(config, indent=2) + '\n')
        manifest = {'schema': 'massspecgym-retrieval-model/v1',
                    'checkpoint': {'path': 'glacier.ckpt', 'sha256': sha256(stage / 'glacier.ckpt'), 'format': 'pytorch'},
                    'model_config': 'model_config.json',
                    'training_provenance': {'data': 'MassSpecGym v1.5 train; validation for selection only', 'pretrained_assets': []}}
        (stage / 'model_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (stage / 'README.md').write_text(
            '# Task-trained GLACIER retrieval package\n\n'
            'One fixed package uses native GLACIER on the published condition domain and '
            'a trained DeepSets model elsewhere. Both are initialized from scratch. '
            'The pinned /opt/glacier runtime is supplied by the verifier; model source '
            'is copied into this package. Inference performs no fitting or downloads.\n'
        )
        shutil.copytree(here, stage / 'training', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copyfile(here.parent / 'publication.py', stage / 'publication.py')

    publish_directory(destination, build)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--fallback', type=Path, required=True)
    p.add_argument('--source-root', type=Path, default=Path('/opt/ms-pred'))
    p.add_argument('--output-dir', type=Path, default=Path('/workspace/submission'))
    p.add_argument('--reference-scripts', type=Path)
    p.add_argument('--inference-batch-size', type=positive_integer, default=DEFAULT_INFERENCE_BATCH_SIZE,
                   help='native inference batch size; lower for limited GPU memory (default: 64); does not change training')
    args = p.parse_args()
    publish(args.checkpoint, args.fallback, args.source_root, args.output_dir, args.reference_scripts,
            inference_batch_size=args.inference_batch_size)


if __name__ == '__main__':
    main()
