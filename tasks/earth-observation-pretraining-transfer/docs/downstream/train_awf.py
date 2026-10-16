"""Official rslearn AWF head, data transforms, optimizer and unfreezing loop."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
import yaml
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from jsonargparse import ArgumentParser
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.lightning_module import RslearnLightningModule
from rslearn.train.callbacks.freeze_unfreeze import FreezeUnfreeze


def parameter_digest(module):
    h = hashlib.sha256()
    for name, p in module.state_dict().items():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Audit(Callback):
    def __init__(self, output):
        self.output = output
        self.rows = []

    def on_fit_start(self, trainer, module):
        counts = {k: len(v) for k, v in trainer.datamodule.datasets.items()}
        if counts.get("train") != 1115 or counts.get("val") != 344:
            raise RuntimeError(f"Unexpected released AWF coverage: {counts}")
        (self.output / "data-coverage.json").write_text(json.dumps(counts, indent=2))
        self.initial_encoder = parameter_digest(module.model.encoder[0].model)
        self.initial_head = parameter_digest(module.model.decoders)
        (self.output / "initialization.json").write_text(
            json.dumps(
                {
                    "encoder_state_digest": self.initial_encoder,
                    "head_state_digest": self.initial_head,
                    "encoder_file_sha256": module.model.encoder[0].encoder_sha256,
                },
                indent=2,
            )
        )

    def on_validation_epoch_start(self, trainer, module):
        # Replay identical random crops each validation epoch, without consuming
        # training RNG. num_workers=0 keeps this contract explicit.
        self.rng = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all(),
        )
        pl.seed_everything(73491, workers=True, verbose=False)

    def on_validation_epoch_end(self, trainer, module):
        random.setstate(self.rng[0])
        np.random.set_state(self.rng[1])
        torch.set_rng_state(self.rng[2])
        torch.cuda.set_rng_state_all(self.rng[3])

    def on_after_backward(self, trainer, module):
        encoder = module.model.encoder[0].model
        grads = [p.grad for p in encoder.parameters() if p.grad is not None]
        if trainer.current_epoch < 20 and grads:
            raise RuntimeError("Frozen encoder has gradients")
        if trainer.current_epoch >= 20 and (
            not grads or not any(torch.any(g != 0) for g in grads)
        ):
            raise RuntimeError("Unfrozen encoder has no gradients")
        if any(not torch.isfinite(g).all() for g in grads):
            raise RuntimeError("Nonfinite encoder gradient")

    def on_train_epoch_start(self, trainer, module):
        if trainer.current_epoch == 20:
            unchanged = (
                parameter_digest(module.model.encoder[0].model) == self.initial_encoder
            )
            if not unchanged:
                raise RuntimeError("Encoder changed during frozen epochs")
            (self.output / "unfreeze-boundary.json").write_text(
                json.dumps({"epoch": 20, "frozen_weights_unchanged": unchanged})
            )

    def on_train_epoch_end(self, trainer, module):
        row = {
            "epoch": trainer.current_epoch,
            "encoder_trainable_parameters": sum(
                p.numel()
                for p in module.model.encoder[0].parameters()
                if p.requires_grad
            ),
            "lr": [g["lr"] for g in trainer.optimizers[0].param_groups],
            "metrics": {
                k: float(v)
                for k, v in trainer.callback_metrics.items()
                if torch.is_tensor(v) and v.numel() == 1
            },
        }
        self.rows.append(row)
        (self.output / "epochs.json").write_text(json.dumps(self.rows, indent=2))

    def on_fit_end(self, trainer, module):
        final = parameter_digest(module.model.encoder[0].model)
        (self.output / "completion.json").write_text(
            json.dumps(
                {
                    "epochs": trainer.current_epoch,
                    "encoder_changed": final != self.initial_encoder,
                    "final_encoder_state_digest": final,
                    "test_used": False,
                },
                indent=2,
            )
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=8123)
    p.add_argument("--pilot", action="store_true")
    a = p.parse_args()
    if (
        os.environ.get("EARTH_SLURM_PARTITION")
        and os.environ.get("SLURM_JOB_PARTITION") != os.environ["EARTH_SLURM_PARTITION"]
    ):
        raise RuntimeError("Requires configured-gpu-partition")
    if a.output.exists():
        raise ValueError("Use a fresh output directory")
    a.output.mkdir(parents=True)
    pl.seed_everything(a.seed, workers=True)
    cfg = yaml.safe_load(a.config.read_text())
    data = copy.deepcopy(cfg["data"]["init_args"])
    data.update(path=a.dataset, num_workers=0)
    parser = ArgumentParser()
    parser.add_class_arguments(RslearnDataModule, "data")
    dm = parser.instantiate_classes(parser.parse_object({"data": data})).data
    model = copy.deepcopy(cfg["model"]["init_args"])
    model["task"] = data["task"]
    model["model"]["init_args"]["encoder"] = [
        {
            "class_path": "portable_encoder.PortableAWFEncoder",
            "init_args": {"package": str(a.package)},
        }
    ]
    parser = ArgumentParser()
    parser.add_class_arguments(RslearnLightningModule, "model")
    module = parser.instantiate_classes(parser.parse_object({"model": model})).model
    # Reset after architecture construction so loader and dropout streams match.
    pl.seed_everything(a.seed, workers=True)
    checkpoint = ModelCheckpoint(
        dirpath=a.output / "checkpoints",
        monitor="val_segment/accuracy",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    callbacks = [
        FreezeUnfreeze(
            module_selector=["model", "encoder", 0],
            unfreeze_at_epoch=20,
            unfreeze_lr_factor=10,
        ),
        Audit(a.output),
        checkpoint,
    ]
    settings = {
        "seed": a.seed,
        "pilot": a.pilot,
        "package": str(a.package),
        "dataset": a.dataset,
        "max_epochs": 22 if a.pilot else 100,
        "limit_train_batches": 2 if a.pilot else 1.0,
        "limit_val_batches": 2 if a.pilot else 1.0,
        "validation_rng_seed": 73491,
        "num_workers": 0,
        "test_used": False,
        "config_sha256": hashlib.sha256(a.config.read_bytes()).hexdigest(),
        "train_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "adapter_source_sha256": hashlib.sha256(
            Path(__file__).with_name("portable_encoder.py").read_bytes()
        ).hexdigest(),
    }
    (a.output / "settings.json").write_text(json.dumps(settings, indent=2))
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=settings["max_epochs"],
        limit_train_batches=settings["limit_train_batches"],
        limit_val_batches=settings["limit_val_batches"],
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=CSVLogger(a.output, name="metrics"),
        enable_progress_bar=False,
        log_every_n_steps=1,
        default_root_dir=a.output,
    )
    trainer.fit(module, datamodule=dm)


if __name__ == "__main__":
    main()
