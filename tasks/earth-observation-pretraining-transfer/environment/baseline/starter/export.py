"""Export initialization or our own trained checkpoint, never public weights."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import torch
from olmo_core.distributed.checkpoint import load_model_and_optim_state


def checkpoint_recipe(config, preflight, checkpoint_step):
    scheduler = dict(config["train_module"]["scheduler"])
    horizon = scheduler.pop("t_max", None) or config["trainer"]["max_duration"]["value"]
    duration = dict(config["trainer"]["max_duration"], value=checkpoint_step)
    return {
        "dataset": config["dataset"],
        "optimizer": config["train_module"]["optim_config"],
        "scheduler": scheduler,
        "schedule_horizon_steps": horizon,
        "global_batch": config["data_loader"]["global_batch_size"],
        "microbatch": config["train_module"]["rank_microbatch_size"],
        "duration": duration,
        "checkpoint_step": f"step{checkpoint_step}",
        "repeats": config["data_loader"]["num_dataset_repeats_per_epoch"],
        "world_size": preflight.get("world_size", 1),
        "training_implementation": preflight.get("training_implementation"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--variant", choices=["nano", "tiny", "base", "large"], default="tiny"
    )
    p.add_argument("--seed", type=int, default=12536)
    a = p.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(a.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != "0497dfbb6711ded4e6bf10cf089fc1e4d58c186b":
        raise RuntimeError("Wrong official implementation revision")
    script_dir = a.upstream / "scripts/official/v1"
    sys.path.insert(0, str(script_dir))
    model_builder = importlib.import_module(a.variant)
    import script

    # Canonical modality list from the official builder.
    common = script.build_common_components(
        str(script_dir / f"{a.variant}.py"), "dry_run", "export", "local", []
    )
    config = model_builder.build_model_config(common)
    training_recipe = None
    implementation = None
    if a.checkpoint:
        preflight_path = a.checkpoint.parent.parent / "preflight.json"
        preflight = json.loads(preflight_path.read_text())
        implementation = preflight.get("training_implementation")
        if (
            preflight.get("activation_checkpointing")
            and (implementation or {}).get("recomputation_version", 1) < 2
            and a.checkpoint.parent.name != "step0"
        ):
            raise ValueError(
                "Checkpoint used invalid keyword-CUDA-RNG recomputation v1"
            )
        training_config = json.loads((a.checkpoint.parent / "config.json").read_text())
        if training_config["init_seed"] != a.seed:
            raise ValueError("Export seed does not match the training checkpoint")
        expected = config.encoder_config.as_config_dict()
        if training_config["model"]["encoder_config"] != expected:
            raise ValueError(
                "Export architecture does not match the training checkpoint"
            )
        training_recipe = checkpoint_recipe(
            training_config,
            preflight,
            int(a.checkpoint.parent.name.removeprefix("step")),
        )
    torch.manual_seed(a.seed)
    model = config.build()
    if a.checkpoint:
        load_model_and_optim_state(a.checkpoint, model)
    a.output.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.state_dict(), a.output / "encoder.pt")
    (a.output / "config.json").write_text(
        json.dumps(config.encoder_config.as_config_dict(), indent=2)
    )
    (a.output / "provenance.json").write_text(
        json.dumps(
            {
                "upstream": revision,
                "origin": "own_from_scratch_training"
                if a.checkpoint
                else "random_initialization",
                "variant": f"v1/{a.variant}",
                "inference_adapter_version": 3,
                "infer_sha256": hashlib.sha256(
                    Path(__file__).with_name("infer.py").read_bytes()
                ).hexdigest(),
                "initialization_seed": a.seed,
                "checkpoint": str(a.checkpoint) if a.checkpoint else None,
                "parameters": sum(p.numel() for p in model.encoder.parameters()),
                "training_recipe": training_recipe,
                "planned_training_duration": training_config["trainer"]["max_duration"]
                if a.checkpoint
                else None,
                "training_implementation": implementation,
                "training_validity": "not_flagged_invalid",
                "data_seed": training_config["data_loader"]["seed"]
                if a.checkpoint
                else None,
                "encoder_sha256": hashlib.sha256(
                    (a.output / "encoder.pt").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )
    shutil.copyfile(Path(__file__).with_name("infer.py"), a.output / "infer.py")
    shutil.copyfile(a.upstream / "LICENSE", a.output / "LICENSE-OlmoEarth")
    print(a.output)


if __name__ == "__main__":
    main()
