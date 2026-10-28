"""Resumable 50-epoch full-data training; test data are never opened here."""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import braindecode
import numpy as np
import torch

from data import Dataset
from model import MODELS, Model, SPECS
from train import publish

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'evaluation'))
from bci_score import score


def atomic_torch_save(payload, path):
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def validate(model, loader, rows, track, device):
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for x, _, _ in loader:
            probabilities.append(model(x.to(device)).softmax(-1).cpu().numpy())
    result = score(track, rows, np.concatenate(probabilities))
    key = result['kappa'] if track in ('faced', 'tuev') else result['reward']
    if key is None or not math.isfinite(key):
        raise ValueError('non-finite validation selection metric')
    return result, float(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', choices=SPECS, required=True)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=range(5), required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--input-normalization', choices=['none','per-channel-zscore'], default='none')
    parser.add_argument('--normalization-layer', choices=['batch','group'], default='batch')
    parser.add_argument('--class-balanced', action='store_true')
    parser.add_argument('--cache-data', action='store_true')
    parser.add_argument('--extend-from', type=Path, help='Completed resume checkpoint; continue at its final LR in a new directory')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.num_workers < 0:
        raise ValueError('invalid training configuration')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state_path = args.output.parent / f'{args.output.name}-resume.pt'
    receipt_path = args.output.parent / f'{args.output.name}-full-run.json'
    settings = {'model': args.model, 'track': args.track, 'seed': args.seed,
                'epochs': args.epochs, 'batch_size': args.batch_size,
                'lr': args.lr, 'weight_decay': args.weight_decay,
                'braindecode': braindecode.__version__, 'input_normalization': args.input_normalization,
                'class_balanced': args.class_balanced, 'normalization_layer': args.normalization_layer}
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(8)
    train = Dataset(args.data / args.track / 'train', cache=args.cache_data)
    validation = Dataset(args.data / args.track / 'val', cache=args.cache_data)
    if not len(train) or not len(validation):
        raise ValueError('empty development split')
    data_config = json.loads((args.data / args.track / 'train' / 'manifest.json').read_text()).get('config', {})
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        raise RuntimeError('full training requires an allocated GPU')
    model = Model(args.track, args.model, args.input_normalization, args.normalization_layer).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    batches_per_epoch = math.ceil(len(train) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * batches_per_epoch, eta_min=1e-6)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    val_loader = torch.utils.data.DataLoader(validation, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers)
    history, best, best_epoch, completed = [], -float('inf'), 0, 0
    extension_state = None
    if args.extend_from:
        import hashlib, shutil
        if args.extend_from.resolve() == state_path.resolve():raise ValueError('extension requires a new output')
        extension_state = torch.load(args.extend_from, map_location='cpu', weights_only=False)
        parent_settings = {'input_normalization':'none','class_balanced':False,'normalization_layer':'batch',**extension_state['settings']}
        if extension_state['completed'] != parent_settings['epochs'] or args.epochs <= extension_state['completed']:
            raise ValueError('extension requires a completed shorter parent')
        if any(settings.get(k)!=v for k,v in parent_settings.items() if k!='epochs'):
            raise ValueError('extension settings mismatch')
        settings['extension']={'parent_resume_sha256':hashlib.sha256(args.extend_from.read_bytes()).hexdigest(),
                               'from_epoch':extension_state['completed'],'schedule':'constant parent final learning rate',
                               'learning_rates':[g['lr'] for g in extension_state['optimizer']['param_groups']]}
    if state_path.exists() or extension_state is not None:
        state = torch.load(state_path, map_location='cpu', weights_only=False) if state_path.exists() else extension_state
        if not state_path.exists():
            state = dict(state);state['settings'] = settings
            if args.output.exists():raise FileExistsError('fresh extension output exists')
            parent_package=args.extend_from.parent/args.extend_from.name.removesuffix('-resume.pt')
            copied=args.output.with_name(args.output.name+'-parent-best')
            shutil.copytree(parent_package.resolve(),copied,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            args.output.symlink_to(copied.name)
        saved_settings = {'input_normalization': 'none', 'class_balanced': False, 'normalization_layer': 'batch', **state['settings']}
        if saved_settings != settings:
            raise ValueError('resume settings do not match the existing run')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if args.extend_from:
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.0)
            scheduler.base_lrs=list(settings['extension']['learning_rates'])
            for group,rate in zip(optimizer.param_groups,scheduler.base_lrs):group['lr']=rate
            if state_path.exists():scheduler.load_state_dict(state['scheduler'])
        else:scheduler.load_state_dict(state['scheduler'])
        history, best, best_epoch, completed = (state[k] for k in ('history', 'best', 'best_epoch', 'completed'))
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        if completed and (not args.output.exists() or not args.output.resolve().is_dir()):
            raise FileNotFoundError('resume state exists but best model package is missing')
    started = time.monotonic()
    for epoch in range(completed + 1, args.epochs + 1):
        generator = torch.Generator().manual_seed(args.seed * 100000 + epoch)
        sampler = None
        if args.class_balanced:
            labels=np.array([r['label'] for r in train.rows])
            counts=np.bincount(labels)
            sampler=torch.utils.data.WeightedRandomSampler(
                torch.as_tensor(1.0/counts[labels]),len(train),replacement=True,generator=generator)
        loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=sampler is None,
                                              sampler=sampler, generator=generator, num_workers=args.num_workers)
        model.train()
        total_loss = 0.0
        seen = 0
        for x, y, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise ValueError(f'non-finite training loss at epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step()
            total_loss += loss.detach().item() * len(y)
            seen += len(y)
        result, value = validate(model, val_loader, validation.rows, args.track, device)
        row = {'epoch': epoch, 'training_loss': total_loss / seen, 'validation': result,
               'seconds_since_resume': time.monotonic() - started}
        history.append(row)
        if value > best:
            previous = args.output.resolve() if args.output.is_symlink() else None
            best, best_epoch = value, epoch
            record = {**settings, 'scope': 'full-training', 'best_epoch': best_epoch,
                      'best_validation': best, 'train_samples': len(train),
                      'val_samples': len(validation), 'parameters': sum(p.numel() for p in model.parameters()),
                      'history': history}
            publish(model, args, record, data_config)
            if previous is not None and previous.exists() and previous.parent == args.output.parent:
                import shutil
                shutil.rmtree(previous)
        atomic_torch_save({'settings': settings, 'model': model.state_dict(),
                           'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                           'history': history, 'best': best, 'best_epoch': best_epoch,
                           'completed': epoch, 'python_rng': random.getstate(),
                           'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                           'cuda_rng': torch.cuda.get_rng_state_all()}, state_path)
        receipt_path.write_text(json.dumps({**settings, 'status': 'running' if epoch < args.epochs else 'complete',
                                            'completed_epochs': epoch, 'best_epoch': best_epoch,
                                            'best_validation': best, 'history': history},
                                           indent=2, allow_nan=False) + '\n')
        print(json.dumps({'epoch': epoch, 'best_epoch': best_epoch, 'best_validation': best,
                          'training_loss': row['training_loss'], 'validation': result}), flush=True)


if __name__ == '__main__':
    main()
