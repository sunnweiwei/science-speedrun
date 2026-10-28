"""Real-data short training for one Braindecode 1.8.1 model and EEG track."""
import argparse
import json
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
from model import Model, MODELS, SPECS

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'evaluation'))
from bci_score import score


def publish(model, args, record, data_config):
    destination = args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(destination.name + '.staging-' + str(time.time_ns()))
    stage.mkdir()
    here = Path(__file__).resolve().parent
    for name in ('model.py', 'data.py', 'infer.py', 'infer.sh'):
        shutil.copy2(here / name, stage / name)
    torch.save(model.state_dict(), stage / 'weights.pt')
    classes = {'faced': [str(i) for i in range(9)],
               'tuev': ['SPSW', 'GPED', 'PLED', 'EYEM', 'ARTF', 'BCKG'],
               'ad65': ['Alzheimer', 'frontotemporal_dementia', 'healthy'],
               'openbmi': ['left', 'right'], 'longitudinal': ['closed', 'open']}
    config = {'track': args.track, 'model': args.model, 'braindecode': braindecode.__version__,
              'initialization': 'random', 'external_weights': False, 'input_units': 'microvolts',
              'input_normalization': getattr(args, 'input_normalization', 'none'),
              'normalization_layer': getattr(args, 'normalization_layer', 'batch'),
              'scale': 0.01, 'shape': list(SPECS[args.track][:2]) + [200],
              'data': data_config, 'classes': classes[args.track]}
    (stage / 'config.json').write_text(json.dumps(config, indent=2))
    (stage / 'training.json').write_text(json.dumps(record, indent=2, allow_nan=False))
    version = destination.parent / (destination.name + '-version-' + str(time.time_ns()))
    stage.rename(version)
    link = destination.with_name(destination.name + '.next')
    link.symlink_to(version.name)
    if destination.exists() and not destination.is_symlink():
        raise ValueError('publish destination must be absent or a version symlink')
    os.replace(link, destination)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', choices=SPECS, required=True)
    ap.add_argument('--model', choices=MODELS, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--seed', type=int, default=3407)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=0.05)
    ap.add_argument('--train-limit', type=int, default=1024)
    ap.add_argument('--val-limit', type=int, default=0)
    ap.add_argument('--eval-every', type=int, default=100)
    args = ap.parse_args()
    if min(args.steps, args.batch_size, args.eval_every) < 1:
        raise ValueError('steps, batch-size and eval-every must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(8)
    train = Dataset(args.data / args.track / 'train', args.train_limit)
    val = Dataset(args.data / args.track / 'val', args.val_limit)
    if not len(train) or not len(val):
        raise ValueError('empty development split')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(args.track, args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-6)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    iterator = iter(loader)
    best = -float('inf')
    history = []
    started = time.monotonic()
    data_config = json.loads((args.data / args.track / 'train' / 'manifest.json').read_text()).get('config', {})
    for step in range(1, args.steps + 1):
        try:
            x, y, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y, _ = next(iterator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x.to(device)), y.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()
        if step % 10 == 0:
            print(json.dumps({'step': step, 'loss': loss.item(), 'seconds': time.monotonic()-started}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            probabilities = []
            with torch.inference_mode():
                for batch, _, _ in validation:
                    probabilities.append(model(batch.to(device)).softmax(-1).cpu().numpy())
            result = score(args.track, val.rows, np.concatenate(probabilities))
            key = result['kappa'] if args.track in ('faced', 'tuev') else result['reward']
            key = -1 if key is None else key
            history.append({'step': step, 'loss': loss.item(), 'validation': result})
            if key > best:
                best = key
                record = {'seed': args.seed, 'steps': step, 'batch_size': args.batch_size,
                          'train_samples': len(train), 'val_samples': len(val),
                          'seconds': time.monotonic()-started, 'history': history,
                          'scope': 'smoke', 'parameters': sum(p.numel() for p in model.parameters()),
                          'torch': torch.__version__, 'braindecode': braindecode.__version__,
                          'device': torch.cuda.get_device_name() if device == 'cuda' else 'cpu',
                          'lr': args.lr}
                publish(model, args, record, data_config)
            print(json.dumps(history[-1]), flush=True)


if __name__ == '__main__':
    main()
