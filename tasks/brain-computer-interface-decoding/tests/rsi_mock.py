"""Two-candidate shared-model controller mock; no agent or private-label selection."""
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / 'campaign'))
from check_receipt import checked_rewards

TRACKS = ('tuev', 'ad65', 'openbmi', 'longitudinal')


def run(command, log):
    with log.open('a') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def validation(package):
    record = json.loads((package.resolve() / 'training.json').read_text())
    if (record.get('scope') != 'smoke' or record.get('model') != 'eegnet'
            or record.get('pipeline_version') != 'shared-v2.1-channel-preserving'):
        raise ValueError('unexpected candidate metadata')
    value = float(record['best_validation_mean'])
    if not math.isfinite(value):
        raise ValueError('non-finite validation selection')
    return value


def main():
    data = Path(os.environ['TASK_DATA_DIR']).resolve()
    private = Path(os.environ['TASK_PRIVATE_DIR']).resolve()
    research = Path(os.environ['TASK_RESEARCH_DIR']).resolve()
    root = research / 'rsi_shared_mock'
    root.mkdir(parents=True, exist_ok=True)
    log = root / 'run.log'
    started = time.monotonic()
    candidate_a = research / 'shared-smoke' / 'eegnet' / 'submission'
    candidate_b = root / 'candidate-b'
    if not candidate_a.exists():
        raise FileNotFoundError('run the shared baseline smoke matrix first')
    if os.environ.get('BCI_PYTHON') and os.environ.get('TASK_TRAIN_DATA_DIR'):
        task = Path(__file__).resolve().parents[1]
        command = [os.environ['BCI_PYTHON'],
                   str(task / 'environment/baseline/braindecode/train_shared.py'),
                   '--model', 'eegnet', '--data', os.environ['TASK_TRAIN_DATA_DIR'],
                   '--output', str(candidate_b)]
    else:
        command = ['docker', 'run', '--rm', '--network', 'none', '--gpus', 'all',
                   '-v', f'{data}:/workspace/data:ro',
                   '-v', f'{root}:/workspace/research', os.environ.get('BCI_DEVELOPMENT_IMAGE', 'bci-development:0.2'),
                   'python', '/workspace/baseline/braindecode/train_shared.py',
                   '--model', 'eegnet', '--data', '/workspace/data',
                   '--output', '/workspace/research/candidate-b']
    run(command + ['--seed', '3407', '--epochs', '1', '--steps-per-epoch', '100',
                   '--batch-size', '8', '--train-limit', '1024', '--val-limit', '512',
                   '--lr', '0.0003'], log)
    values = {'a': validation(candidate_a), 'b': validation(candidate_b)}
    selected = 'b' if values['b'] > values['a'] else 'a'
    source = candidate_b if selected == 'b' else candidate_a
    target = root / 'selected'
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source.resolve(), target)
    run(['docker', 'run', '--rm', '--network', 'none', '--gpus', 'all',
         '-v', f'{private}:/private/data:ro',
         '-v', f'{target}:/workspace/submission:ro',
         '-v', f'{root}:/logs/verifier', os.environ.get('BCI_VERIFIER_IMAGE', 'bci-verifier:0.2'),
         'bash', '/tests/test.sh', '--smoke-limit', '16'], log)
    checked = checked_rewards(root, TRACKS)
    result = {'schema': 'bci-shared-rsi-mock-v1', 'real_agent': False,
              'real_rsi_controller': False, 'selection_split': 'validation',
              'candidate_validation': values, 'selected': selected,
              'verifier_rewards': checked, 'seconds': time.monotonic() - started}
    (root / 'status.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
