"""Official builders and training loop, with schedule checks and local logging."""


def entry():
    import hashlib
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    upstream = Path(os.environ["OLMO_UPSTREAM"])
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != "0497dfbb6711ded4e6bf10cf089fc1e4d58c186b":
        raise RuntimeError("Wrong official implementation revision")
    sys.path.insert(0, str(upstream / "scripts/official/v1"))
    import importlib

    variant = os.environ.get("OLMO_VARIANT", "tiny")
    if variant not in {"nano", "tiny", "base", "large"}:
        raise ValueError(f"Unsupported official v1 variant: {variant}")
    model_builder = importlib.import_module(variant)
    import script
    from olmo_core.train.callbacks.console_logger import ConsoleLoggerCallback
    from olmoearth_pretrain.internal.experiment import build_config, main

    activation_checkpointing = os.environ.get("OLMO_ACTIVATION_CHECKPOINT", "0") == "1"
    implementation = {
        "train_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "recomputation_version": 0,
    }
    if activation_checkpointing:
        import recomputation
        from recomputation import wrap_block
        from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

        official_build = LatentMIMConfig.build

        def build_with_recomputation(self, *args, **kwargs):
            model = official_build(self, *args, **kwargs)
            for module in (model.encoder, model.decoder):
                for index, block in enumerate(module.blocks):
                    module.blocks[index] = wrap_block(block)
            return model

        LatentMIMConfig.build = build_with_recomputation
        implementation.update(
            recomputation_version=recomputation.RECOMPUTATION_VERSION,
            recomputation_sha256=hashlib.sha256(
                Path(recomputation.__file__).read_bytes()
            ).hexdigest(),
        )

    def trainer_config(common):
        config = script.build_trainer_config(common)
        config = config.with_callback(
            "console_logger",
            ConsoleLoggerCallback(
                log_interval=10,
                metrics_log_interval=10,
                metrics=[
                    "train/*",
                    "optim/*",
                    "gpu_memory/*",
                    "throughput/*",
                    "checkpoint/*",
                ],
            ),
        )
        return config

    if __name__ == "__main__":
        if (
            sys.argv[1] == "train"
            and os.environ.get("EARTH_SLURM_PARTITION")
            and os.environ.get("SLURM_JOB_PARTITION")
            != os.environ["EARTH_SLURM_PARTITION"]
        ):
            raise RuntimeError("GPU training requires Slurm configured-gpu-partition")
        builders = dict(
            model_config_builder=model_builder.build_model_config,
            train_module_config_builder=script.build_train_module_config,
            dataset_config_builder=script.build_dataset_config,
            dataloader_config_builder=script.build_dataloader_config,
            trainer_config_builder=trainer_config,
        )
        _, cmd, name, cluster, *overrides = sys.argv
        common = script.build_common_components(
            sys.argv[0], cmd, name, cluster, overrides
        )
        config = build_config(common=common, overrides=overrides, **builders)
        scheduler = config.train_module.scheduler
        steps = config.trainer.max_duration.value
        lr = config.train_module.optim_config.lr
        if (
            config.trainer.max_duration.unit != "steps"
            or not 0 < scheduler.warmup < steps
        ):
            raise ValueError(
                f"Effective warmup {scheduler.warmup} must be shorter than {steps} steps"
            )
        if scheduler.warmup_steps is not None:
            raise ValueError(
                "Use canonical scheduler.warmup; deprecated alias can silently fail to override"
            )
        effective = {
            "upstream": revision,
            "variant": f"v1/{variant}",
            "activation_checkpointing": activation_checkpointing,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "training_implementation": implementation,
            "steps": steps,
            "warmup": scheduler.warmup,
            "lr_step1": scheduler.get_lr(lr, 1, steps),
            "lr_warmup_end": scheduler.get_lr(lr, scheduler.warmup, steps),
            "lr_final": scheduler.get_lr(lr, steps, steps),
            "global_batch": config.data_loader.global_batch_size,
            "microbatch": config.train_module.rank_microbatch_size,
            "initialization_seed": config.init_seed,
            "data_seed": config.data_loader.seed,
            "weight_decay": config.train_module.optim_config.weight_decay,
            "pretraining_data": str(config.dataset.h5py_dir),
        }
        assert abs(effective["lr_warmup_end"] - lr) < 1e-12
        out = Path(config.trainer.save_folder)
        out.mkdir(parents=True, exist_ok=True)
        if int(os.environ.get("RANK", "0")) == 0:
            (out / "preflight.json").write_text(json.dumps(effective, indent=2))
        print("EFFECTIVE_TRAINING_CONFIG", json.dumps(effective), flush=True)
        main(common_components_builder=script.build_common_components, **builders)


if __name__ == "__main__":
    entry()
