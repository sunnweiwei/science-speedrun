#!/usr/bin/env python3
"""Publish one current replay package from a native starter checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile


INFERENCE_MODULES = (
    "__init__.py",
    "adapter.py",
    "architecture.py",
    "audit.py",
    "config.py",
    "features.py",
    "recipe.py",
    "runtime.py",
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--submission", type=Path, default=Path("/workspace/submission"))
    parser.add_argument("--loops", type=int, default=3)
    parser.add_argument("--sampling-steps", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.loops < 0 or args.sampling_steps < 1:
        raise ValueError("loops must be non-negative and sampling steps positive")

    import torch
    from safetensors.torch import save_file
    from esmfold2_50m.architecture import build_model, persistent_state_dict
    from esmfold2_50m.config import EXPECTED_PARAMETER_RANGE, MODEL_NAME_50M
    from esmfold2_50m.recipe import parameter_count

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "foldbench-esmfold2-final-hidden-checkpoint/v1":
        raise ValueError("unsupported native checkpoint")
    state = persistent_state_dict(payload["model"])
    unknown = set(payload["ema"]) - set(state)
    if unknown:
        raise ValueError(f"EMA contains unknown keys: {sorted(unknown)[:3]}")
    state.update(persistent_state_dict(payload["ema"]))
    model = build_model()
    model.load_state_dict(state, strict=True)
    count = parameter_count(model)
    if not EXPECTED_PARAMETER_RANGE[0] <= count <= EXPECTED_PARAMETER_RANGE[1]:
        raise ValueError(f"published graph has {count:,} parameters")

    source = Path(__file__).resolve().parent
    args.submission.mkdir(parents=True, exist_ok=True)
    # The submission root is commonly a dedicated bind mount, so keep the
    # complete stage on that filesystem and expose it through one symlink swap.
    stage = Path(tempfile.mkdtemp(prefix=".model-", dir=args.submission))
    os.chmod(stage, 0o755)
    exposed = False
    try:
        shutil.copy2(source / "infer.sh", stage / "infer.sh")
        shutil.copy2(source / "predict.py", stage / "predict.py")
        shutil.copy2(source / "model.py", stage / "model.py")
        package = stage / "esmfold2_50m"
        package.mkdir()
        for name in INFERENCE_MODULES:
            shutil.copy2(source / "esmfold2_50m" / name, package / name)
        tensors = {
            name: value.detach().cpu().contiguous()
            for name, value in persistent_state_dict(model.state_dict()).items()
        }
        save_file(
            tensors,
            str(stage / "model.safetensors"),
            metadata={
                "schema_version": "foldbench-from-scratch-esmfold2-50m/v1",
                "parameter_count": str(count),
                "training_step": str(int(payload["step"])),
                "pretrained_parameters": "false",
            },
        )
        os.chmod(stage / "model.safetensors", 0o644)
        config = {
            "schema_version": "foldbench-esmfold2-final-hidden-config/v1",
            "architecture": MODEL_NAME_50M,
            "parameter_count": count,
            "training_step": int(payload["step"]),
            "loops": args.loops,
            "sampling_steps": args.sampling_steps,
            "ccd_dir": "/opt/esmfold2-assets",
            "input_feature": "esmc6b-final-ln-int8-g64-45b0fa5d-ptsdpa-v1",
            "pretrained_parameters": False,
        }
        (stage / "model_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n"
        )
        readme = (
            "# From-scratch ESMFold2-50M candidate\n\n"
            f"One fixed {count:,}-parameter Biohub ESMFold2-derived graph for all nine "
            "tracks. It consumes evaluator-supplied ESMC-6B final-hidden INT8 data "
            "through a learned projection, then uses the all-atom input embedder, pair "
            "trunk, PARCAE/CODA recurrence, diffusion structure module, distogram and "
            "confidence heads. Every packaged parameter was randomly initialized and "
            f"trained in this session; this package is from step {int(payload['step'])}. "
            "No ESMC, ESMFold2, ESMFold2-Fast, MSA, template, optimizer, or historical "
            "checkpoint parameters are included.\n\n"
            "Parameter audit: `cd current && PYTHONPATH=. python -m "
            "esmfold2_50m.audit`. Known "
            "limitations and measured FoldBench results must be updated as research "
            "proceeds.\n"
        )
        (stage / "README.md").write_text(readme)
        os.chmod(stage / "infer.sh", 0o755)

        # The bind-mount root cannot itself be renamed. A stable root entry
        # point follows `current`, while replacing that relative symlink is one
        # atomic metadata operation. A concurrent snapshot therefore sees
        # either the complete previous package or the complete new package.
        root_infer = args.submission / ".infer.next"
        root_infer.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
            'exec bash "$root/current/infer.sh" "$@"\n'
        )
        os.chmod(root_infer, 0o755)
        os.replace(root_infer, args.submission / "infer.sh")
        for name in ("README.md", "model_config.json"):
            temporary_link = args.submission / f".{name}.next"
            temporary_link.unlink(missing_ok=True)
            os.symlink(f"current/{name}", temporary_link)
            os.replace(temporary_link, args.submission / name)
        next_link = args.submission / ".current.next"
        next_link.unlink(missing_ok=True)
        os.symlink(stage.name, next_link)
        os.replace(next_link, args.submission / "current")
        exposed = True

        keep = {stage.name, "current", "infer.sh", "README.md", "model_config.json"}
        for child in args.submission.iterdir():
            if child.name in keep:
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    finally:
        if not exposed:
            shutil.rmtree(stage, ignore_errors=True)
    print(json.dumps(config, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
