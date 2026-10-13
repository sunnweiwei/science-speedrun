#!/usr/bin/env python3
"""Train native GLACIER from scratch; resume and publish without test access."""

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def charged_elapsed(work_dir, checkpoint_elapsed):
    """Do not refund completed validation/export when resuming older fit state."""
    result = Path(work_dir) / 'result.json'
    completed = json.loads(result.read_text())['elapsed_seconds'] if result.exists() else 0.0
    ledger = Path(work_dir) / 'budget-ledger.json'
    charged = json.loads(ledger.read_text())['elapsed_seconds'] if ledger.exists() else 0.0
    return max(float(checkpoint_elapsed), float(completed), float(charged))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('/workspace/data/glacier'))
    p.add_argument('--source-root', type=Path, default=Path('/opt/ms-pred'))
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--fallback', type=Path, required=True)
    p.add_argument('--submission-dir', type=Path, default=Path('/workspace/submission'))
    p.add_argument('--max-epochs', type=int, default=200)
    p.add_argument('--max-steps', type=int, default=-1)
    p.add_argument('--max-seconds', type=float, default=21600)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--devices', type=int, choices=[1], default=1,
                   help='The supplied reproducible recipe uses one GPU; run independent trials on other GPUs.')
    p.add_argument('--accumulate-grad-batches', type=int, default=4)
    p.add_argument('--validate-every', type=int, default=1000)
    p.add_argument('--checkpoint-every', type=int, default=500)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--resume', type=Path)
    p.add_argument('--trace-batches', action='store_true', help='Record batch identities and RNG hashes for recovery audits')
    p.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default='auto')
    args = p.parse_args()
    if min(args.batch_size, args.devices, args.max_seconds, args.validate_every,
           args.checkpoint_every, args.accumulate_grad_batches, args.max_epochs) <= 0 or args.workers < 0:
        p.error('invalid resource or training limits')
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    source = args.source_root.resolve() / 'src'
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader
    from torchdata.stateful_dataloader import StatefulDataLoader
    from torchdata.stateful_dataloader.sampler import RandomSampler
    from ms_pred.glacier.joint_model import JointModel
    from dataset import PreparedDataset
    from publish import publish

    config = json.loads(args.config.read_text())
    manifest_path = args.data_root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest['files'].items():
        if sha256(args.data_root / name) != expected:
            raise ValueError(f'training resource checksum mismatch: {name}')
    source_digest = hashlib.sha256()
    for path in sorted((source / 'ms_pred').rglob('*.py')):
        source_digest.update(str(path.relative_to(source)).encode())
        source_digest.update(path.read_bytes())
    identity = {'initialization': 'random', 'data_manifest_sha256': sha256(manifest_path),
                'config': config, 'source_sha256': source_digest.hexdigest(), 'seed': args.seed,
                'batch_size': args.batch_size, 'devices': args.devices,
                'accumulate_grad_batches': args.accumulate_grad_batches,
                'loader': 'torchdata-stateful-separate-generators-v2'}
    previous = args.work_dir / 'recipe.json'
    if previous.exists() and (not args.resume or json.loads(previous.read_text()) != identity):
        raise ValueError('use a new work directory, or --resume with the same recipe')
    (args.work_dir / 'recipe.json').write_text(json.dumps(identity, indent=2) + '\n')
    # Preserve the editable recipe and source snapshot independently of submission.
    import shutil
    if not (args.work_dir / 'training-source').exists():
        shutil.copytree(Path(__file__).parent, args.work_dir / 'training-source',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    pl.seed_everything(args.seed, workers=True)

    class TaskModel(JointModel):
        # The pinned native method uses the older Lightning scheduler hook
        # signature. Adapt only its API, preserving the native step schedule.
        def lr_scheduler_step(self, scheduler, metric):
            scheduler.step()

        def training_step(self, batch, batch_idx):
            if args.trace_batches:
                with (args.work_dir / 'batch-trace.jsonl').open('a') as stream:
                    stream.write(json.dumps({'step': self.global_step, 'batch_index': batch_idx,
                        'names': batch['names'], 'magma_step': self.step,
                        'torch_rng': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}) + '\n')
            return super().training_step(batch, batch_idx)

        def on_save_checkpoint(self, checkpoint):
            checkpoint['task_training'] = dict(identity, elapsed_seconds=clock.elapsed())
            checkpoint['task_magma_step'] = self.step
            checkpoint['task_rng'] = {'python': random.getstate(), 'numpy': np.random.get_state(),
                                      'torch': torch.get_rng_state(),
                                      'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

        def on_load_checkpoint(self, checkpoint):
            saved = checkpoint.get('task_training', {})
            if {k: saved.get(k) for k in identity} != identity:
                raise ValueError('resume data/model/seed/batch configuration changed')
            clock.prior = charged_elapsed(args.work_dir, saved['elapsed_seconds'])
            if 'task_magma_step' not in checkpoint:
                raise ValueError('checkpoint predates native MAGMa schedule state; start a fresh run')
            self.step = checkpoint['task_magma_step']
            self.resume_rng = checkpoint.get('task_rng')

        def on_train_start(self):
            # Restore before fetching: native graph features also use randomness.
            rng = getattr(self, 'resume_rng', None)
            if rng:
                random.setstate(rng['python'])
                np.random.set_state(rng['numpy'])
                torch.set_rng_state(rng['torch'].cpu())
                if rng['cuda'] and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all([s.cpu() for s in rng['cuda']])
                self.resume_rng = None

    class ClockAndPublication(pl.Callback):
        def __init__(self):
            self.started = time.monotonic()
            self.prior = 0.0
            self.last_published = None

        def elapsed(self):
            return self.prior + time.monotonic() - self.started

        def publish_best(self, trainer):
            path = best.best_model_path
            if trainer.is_global_zero and path and Path(path).is_file():
                signature = (path, Path(path).stat().st_mtime_ns)
                if signature != self.last_published:
                    publish(path, args.fallback, args.source_root, args.submission_dir)
                    self.last_published = signature
                    print(json.dumps({'event': 'published', 'step': trainer.global_step,
                                      'checkpoint': path, 'elapsed_seconds': self.elapsed()}), flush=True)

        def on_train_batch_start(self, trainer, model, batch, batch_idx):
            self.publish_best(trainer)

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            # All DDP ranks make the same stop decision, including a slow rank.
            stop = torch.tensor(int(self.elapsed() >= args.max_seconds), device=model.device)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(stop, op=torch.distributed.ReduceOp.MAX)
            if stop.item() and (batch_idx + 1) % args.accumulate_grad_batches == 0:
                trainer.should_stop = True
            if trainer.is_global_zero and trainer.global_step % 25 == 0:
                metrics = {k: float(v.detach().cpu()) for k, v in trainer.callback_metrics.items()
                           if torch.is_tensor(v) and v.numel() == 1}
                record = {'step': trainer.global_step, 'elapsed_seconds': self.elapsed(), 'metrics': metrics}
                with (args.work_dir / 'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')

        def on_train_end(self, trainer, model):
            # Save state while the fit loop is active (ModelCheckpoint), not
            # after Lightning has advanced its end-of-fit epoch bookkeeping.
            if trainer.is_global_zero and latest.last_model_path:
                temporary = args.work_dir / 'last.ckpt.tmp'
                shutil.copyfile(latest.last_model_path, temporary)
                os.replace(temporary, args.work_dir / 'last.ckpt')
            self.publish_best(trainer)

    class ResumeCheckpoint(ModelCheckpoint):
        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            super().on_train_batch_end(trainer, model, outputs, batch, batch_idx)
            at_limit = args.max_steps > 0 and trainer.global_step >= args.max_steps
            final_batch = (trainer.current_epoch + 1 >= args.max_epochs and
                           batch_idx + 1 == trainer.num_training_batches)
            optimizer_boundary = ((batch_idx + 1) % args.accumulate_grad_batches == 0 or
                                  batch_idx + 1 == trainer.num_training_batches)
            if optimizer_boundary and (trainer.should_stop or at_limit or final_batch):
                if self._last_global_step_saved != trainer.global_step:
                    metrics = self._monitor_candidates(trainer)
                    self._save_topk_checkpoint(trainer, metrics)
                    self._save_last_checkpoint(trainer, metrics)

    clock = ClockAndPublication()
    train = PreparedDataset(args.data_root, 'train', config)
    validation = PreparedDataset(args.data_root, 'validation', config)
    model = TaskModel(**dict(config, node_feats=train.processor.get_node_feats(),
                            edge_feats=train.processor.get_edge_feats()))
    device = 'gpu' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if device == 'auto':
        device = 'cpu'
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        prior = charged_elapsed(args.work_dir, saved.get('task_training', {}).get('elapsed_seconds', 0))
        if prior >= args.max_seconds:
            raise ValueError('resume has already exhausted the cumulative training budget')
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.workers,
                         collate_fn=train.collate_fn, persistent_workers=args.workers > 0)
    if args.workers:
        loader_kwargs.update(multiprocessing_context='spawn', prefetch_factor=1)
    # Keep sampler and worker-base generators separate. Sharing one lets worker
    # initialization advance the saved sampler state and changes order on resume.
    sampler = RandomSampler(train, generator=torch.Generator().manual_seed(args.seed))
    train_loader = StatefulDataLoader(train, sampler=sampler,
                                     generator=torch.Generator().manual_seed(args.seed + 2), **loader_kwargs)
    val_loader = DataLoader(validation, shuffle=False,
                          generator=torch.Generator().manual_seed(args.seed + 1), **loader_kwargs)
    best = ModelCheckpoint(dirpath=args.work_dir / 'checkpoints', filename='best',
                           monitor='val_loss', save_top_k=1, save_last=False,
                           enable_version_counter=False)
    latest = ResumeCheckpoint(dirpath=args.work_dir / 'resume', filename='latest', save_top_k=1,
                             every_n_train_steps=args.checkpoint_every, save_last=True,
                             enable_version_counter=False)
    trainer = pl.Trainer(
        accelerator=device, devices=args.devices, logger=pl.loggers.CSVLogger(args.work_dir, name='lightning'),
        strategy='ddp' if args.devices > 1 else 'auto', callbacks=[clock, best, latest],
        default_root_dir=args.work_dir, max_epochs=args.max_epochs, max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulate_grad_batches, gradient_clip_val=5,
        gradient_clip_algorithm='value', num_sanity_val_steps=0,
        val_check_interval=min(args.validate_every, len(train_loader)),
        log_every_n_steps=1, enable_progress_bar=False,
        max_time=timedelta(seconds=max(1, args.max_seconds - (prior if args.resume else 0))),
    )
    print(json.dumps({'parameters': sum(p.numel() for p in model.parameters()),
                      'train_rows': len(train), 'validation_rows': len(validation),
                      'initialization': 'random', 'max_seconds': args.max_seconds}), flush=True)
    trainer.fit(model, train_loader, val_loader, ckpt_path=str(args.resume) if args.resume else None,
                weights_only=False)
    # A very short smoke may stop before scheduled validation. Run validation
    # on the trained parameters, then save/publish; never select on test data.
    if not best.best_model_path:
        trainer.validate(model, val_loader)
        path = args.work_dir / 'validated-final.ckpt'
        trainer.save_checkpoint(path)
        if trainer.is_global_zero:
            publish(path, args.fallback, args.source_root, args.submission_dir)
    if trainer.is_global_zero:
        (args.work_dir / 'result.json').write_text(json.dumps({
            'global_step': trainer.global_step, 'elapsed_seconds': clock.elapsed(),
            'selected_checkpoint': best.best_model_path or str(args.work_dir / 'validated-final.ckpt'),
            'validation_loss': float(best.best_model_score) if best.best_model_score is not None else float(trainer.callback_metrics['val_loss']),
            'last_validation_loss': float(trainer.callback_metrics['val_loss']),
            'source_sha256': source_digest.hexdigest(), 'data_manifest_sha256': sha256(manifest_path),
            'hardware': torch.cuda.get_device_name() if device == 'gpu' else 'CPU',
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'gpu' else 0,
            'selected_on': 'validation spectrum loss; retrieval must be measured separately',
        }, indent=2) + '\n')


if __name__ == '__main__':
    main()
