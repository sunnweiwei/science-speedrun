"""Balanced five-track supervised training of one shared EEG backbone."""
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import braindecode
import numpy as np
import torch

from data import Dataset
from model import MODELS
from pretrained import load_backbone
from shared import SharedModel, TRACKS, PIPELINE_VERSION
from sampling import CyclingBatchSampler

CLASS_NAMES = {'faced': [str(i) for i in range(9)],
               'tuev': ['SPSW', 'GPED', 'PLED', 'EYEM', 'ARTF', 'BCKG'],
               'ad65': ['Alzheimer', 'frontotemporal_dementia', 'healthy'],
               'openbmi': ['left', 'right'], 'longitudinal': ['closed', 'open']}

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'evaluation'))
from bci_score import score
from data_identity import split_identity


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(model, output, config, training):
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + '.staging-' + str(time.time_ns()))
    stage.mkdir()
    here = Path(__file__).resolve().parent
    for name in ('model.py', 'shared.py', 'data.py', 'shared_infer.py'):
        shutil.copy2(here / name, stage / name)
    shutil.copy2(here / 'shared_infer.sh', stage / 'infer.sh')
    torch.save(model.state_dict(), stage / 'weights.pt')
    (stage / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')
    public_training = {k: v for k, v in training.items() if k != '_device'}
    (stage / 'training.json').write_text(json.dumps(public_training, indent=2, allow_nan=False) + '\n')
    version = output.parent / (output.name + '-version-' + str(time.time_ns()))
    stage.rename(version)
    link = output.with_name(output.name + '.next')
    link.symlink_to(version.name)
    if output.exists() and not output.is_symlink():
        raise ValueError('output must be absent or a version symlink')
    os.replace(link, output)


def validate(model, datasets, loaders, device):
    model.eval()
    metrics = {}
    with torch.inference_mode():
        for track in TRACKS:
            probabilities = []
            for x, _, _ in loaders[track]:
                probabilities.append(model(track, x.to(device)).softmax(-1).cpu().numpy())
            result = score(track, datasets[track].rows, np.concatenate(probabilities))
            value = float(result['reward'])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{track}: non-finite validation reward')
            metrics[track] = result
    return metrics, sum(metrics[t]['reward'] for t in TRACKS) / len(TRACKS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--steps-per-epoch', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--train-limit', type=int, default=0)
    parser.add_argument('--val-limit', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--cache-data', action='store_true')
    parser.add_argument('--extend-from', type=Path,
                        help='Completed resume checkpoint; extend at its final LR in a new output directory')
    args = parser.parse_args()
    if (args.epochs < 1 or args.steps_per_epoch < 5 or args.steps_per_epoch % 5
            or args.batch_size < 1 or args.num_workers < 0):
        raise ValueError('invalid epoch, step, or batch configuration')
    if args.model in ('cbramod', 'biot', 'codebrain') and args.checkpoint_dir is None:
        raise ValueError('pinned pretrained checkpoint directory required')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed); torch.backends.cudnn.deterministic = True
    torch.set_num_threads(8)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SharedModel(args.model)
    initialization = load_backbone(model, args.model, args.checkpoint_dir)
    model.to(device)
    train = {t: Dataset(args.data / t / 'train', args.train_limit or None, cache=args.cache_data) for t in TRACKS}
    val = {t: Dataset(args.data / t / 'val', args.val_limit if t in ('faced', 'tuev') else None, cache=args.cache_data)
           for t in TRACKS}
    if any(not train[t] or not val[t] for t in TRACKS):
        raise ValueError('empty training or validation split')
    val_loaders = {t: torch.utils.data.DataLoader(val[t], batch_size=args.batch_size,
                   num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
                   generator=torch.Generator().manual_seed(args.seed * 100000 + 100 + i))
                   for i, t in enumerate(TRACKS)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    identities = {t: {split: split_identity(args.data / t / split)
                      for split in ('train', 'val')} for t in TRACKS}
    data_config = {t: json.loads((args.data / t / 'train' / 'manifest.json').read_text())['config']
                   for t in TRACKS}
    config = {'schema': 'bci-shared-package-v2', 'family': args.model,
              'pipeline_version': PIPELINE_VERSION,
              'braindecode': braindecode.__version__, 'initialization': initialization,
              'input_units': 'microvolts', 'loader_scale': 0.01,
              'tracks': list(TRACKS), 'classes': CLASS_NAMES, 'data': data_config}
    history, best, best_epoch, completed = [], -math.inf, 0, 0
    resume = args.output.parent / (args.output.name + '-resume.pt')
    settings = {'model': args.model, 'seed': args.seed, 'epochs': args.epochs,
                'pipeline_version': PIPELINE_VERSION,
                'steps_per_epoch': args.steps_per_epoch, 'batch_size': args.batch_size,
                'lr': args.lr, 'weight_decay': args.weight_decay,
                'train_limit': args.train_limit, 'val_limit': args.val_limit,
                'num_workers': args.num_workers,
                'data_identity': identities, 'initialization': initialization}
    extension_state = None
    if args.extend_from:
        if args.extend_from.resolve() == resume.resolve():
            raise ValueError('extension must use a new output directory')
        extension_state = torch.load(args.extend_from, map_location='cpu', weights_only=False)
        parent_settings = extension_state['settings']
        if (extension_state['completed'] != parent_settings['epochs']
                or args.epochs <= extension_state['completed']):
            raise ValueError('extension requires a completed shorter parent run')
        for key, value in parent_settings.items():
            if key not in ('epochs', 'num_workers') and settings.get(key) != value:
                raise ValueError('extension setting mismatch: ' + key)
        settings['extension'] = {
            'parent_resume_sha256': sha256(args.extend_from),
            'from_epoch': extension_state['completed'],
            'schedule': 'constant parent final learning rate',
            'learning_rates': [g['lr'] for g in extension_state['optimizer']['param_groups']]}
    if resume.exists() or extension_state is not None:
        state = torch.load(resume, map_location='cpu', weights_only=False) if resume.exists() else extension_state
        if not resume.exists():
            state = dict(state)
            state['settings'] = settings
            parent_package = args.extend_from.parent / args.extend_from.name.removesuffix('-resume.pt')
            if args.output.exists():
                raise FileExistsError('fresh extension output already exists')
            args.output.parent.mkdir(parents=True, exist_ok=True)
            copied = args.output.with_name(args.output.name + '-parent-best')
            shutil.copytree(parent_package.resolve(), copied, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            args.output.symlink_to(copied.name)
        if state['settings'] != settings:
            raise ValueError('resume settings mismatch')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        if args.extend_from:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            for group, rate in zip(optimizer.param_groups, settings['extension']['learning_rates']):
                group['lr'] = rate
            scheduler.base_lrs = list(settings['extension']['learning_rates'])
            if resume.exists():scheduler.load_state_dict(state['scheduler'])
        else:
            scheduler.load_state_dict(state['scheduler'])
        history, best, best_epoch, completed = (state[k] for k in ('history', 'best', 'best_epoch', 'completed'))
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
    started = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Carry each task's shuffled iterator across reporting epochs. With the
    # default 50×1000 steps, even the largest track is visited in full once.
    consumed_batches = completed * args.steps_per_epoch // len(TRACKS)
    loaders = {t: torch.utils.data.DataLoader(train[t],
               batch_sampler=CyclingBatchSampler(len(train[t]), args.batch_size,
                                                 args.seed * 100000 + i * 1000, consumed_batches),
               num_workers=args.num_workers,
               persistent_workers=args.num_workers > 0,
               generator=torch.Generator().manual_seed(args.seed * 100000 + i))
               for i, t in enumerate(TRACKS)}
    iterators = {t: iter(loaders[t]) for t in TRACKS}
    for epoch in range(completed + 1, args.epochs + 1):
        model.train(); losses = []; task_losses = {t: [] for t in TRACKS}
        for step in range(args.steps_per_epoch):
            track = TRACKS[step % len(TRACKS)]
            try:
                x, y, _ = next(iterators[track])
            except StopIteration:
                iterators[track] = iter(loaders[track]); x, y, _ = next(iterators[track])
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(track, x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise ValueError(f'non-finite loss at epoch {epoch}, step {step}')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); losses.append(float(loss.detach())); task_losses[track].append(float(loss.detach()))
        metrics, selection = validate(model, val, val_loaders, device)
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)),
                        'training_loss_by_track': {t: float(np.mean(v)) for t,v in task_losses.items()},
                        'validation': metrics, 'selection': selection,
                        'elapsed_seconds': time.monotonic() - started})
        if selection > best:
            best, best_epoch = selection, epoch
            training = {**settings, 'scope': 'smoke' if args.train_limit or args.val_limit else 'full',
                        'best_epoch': epoch, 'best_validation_mean': selection,
                        'parameters': sum(p.numel() for p in model.parameters()),
                        'history': history, '_device': device}
            publish(model, args.output, config, training)
        receipt = {**settings, 'status': 'complete' if epoch == args.epochs else 'running',
                   'completed_epochs': epoch, 'best_epoch': best_epoch,
                   'best_validation_mean': best, 'history': history,
                   'training_coverage': {t: {'available_samples': len(train[t]),
                       'complete_passes': (epoch * args.steps_per_epoch // len(TRACKS))
                           // math.ceil(len(train[t]) / args.batch_size)} for t in TRACKS}}
        (args.output.parent / (args.output.name + '-full-run.json')).write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + '\n')
        stage = resume.with_suffix('.tmp')
        torch.save({'settings': settings, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'history': history, 'best': best, 'best_epoch': best_epoch,
                    'completed': epoch, 'python_rng': random.getstate(),
                    'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all()}, stage)
        os.replace(stage, resume)
        print(json.dumps({'epoch': epoch, 'selection': selection, 'best_epoch': best_epoch}), flush=True)


if __name__ == '__main__':
    main()
