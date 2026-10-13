"""Turn a frozen AlphaQubit package into a `surface-code-decoding-model/v1` submission.

A frozen package is what `python -m baselines.alphaqubit_artifact --models-root M --stage finetune
--output PKG` writes after `baselines.alphaqubit_train` has trained every group; then

    python -m qecbench.package --package PKG --output /workspace/submission

writes infer.sh, the ten specialists' weights, sources, lock files, model_config.json,
training_provenance.json and model_manifest.json. `write_template` writes the same package without
weights (the development release ships it under reference/package-template).
"""
from __future__ import annotations

import argparse
import time
import os
import errno
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

TASK = Path(__file__).resolve().parents[1]
INFER_SH = ('#!/usr/bin/env bash\nset -euo pipefail\nROOT="$(cd -- "$(dirname -- "$0")" && pwd)"\n'
            'export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"\n'
            'exec "${QEC_PYTHON:-python}" -m qecbench.inference --submission "$ROOT" --input "$1" --output "$2" '
            '--batch-size "${QEC_BATCH_SIZE:-256}"\n')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _common(output: Path, baselines_source: Path) -> None:
    shutil.copytree(baselines_source, output / "src/baselines", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(TASK / "baselines/metrics.py", output / "src/baselines/metrics.py")
    shutil.copytree(TASK / "qecbench", output / "src/qecbench", ignore=shutil.ignore_patterns("__pycache__"))
    for filename in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(TASK / "alphaqubit-runtime" / filename, output / filename)
    shutil.copyfile(TASK / "alphaqubit-runtime/uv.lock", output / "requirements.lock")
    (output / "infer.sh").write_text(INFER_SH)
    (output / "infer.sh").chmod(0o755)


def package_submission(package: Path, output: Path, provenance: dict[str, Any], readme: str) -> None:
    """Frozen AlphaQubit package (all ten finetune specialists) -> relocatable submission directory."""
    from baselines.alphaqubit_artifact import EXPECTED_GROUPS, verify_package
    source = verify_package(package)
    if set(source["groups"]) != set(EXPECTED_GROUPS) or "finetune" not in source["stages"]:
        raise ValueError("Require all ten finetune specialists")
    output.mkdir(parents=True)
    models, checkpoints = {}, []
    for group in EXPECTED_GROUPS:
        record = source["models"][f"{group}/finetune"]
        directory = output / "weights" / group
        directory.mkdir(parents=True)
        for field, filename in (("model", "model.npz"), ("checkpoint_metadata", "best.json")):
            shutil.copyfile(package / record[field], directory / filename)
        weights, metadata = f"weights/{group}/model.npz", f"weights/{group}/best.json"
        models[group] = {"weights": weights, "metadata": metadata}
        checkpoints.append({"path": weights, "sha256": _sha256(output / weights), "format": "npz"})
    # The semantic source files remain byte-identical to the frozen model.
    _common(output, package / "baselines")
    (output / "README.md").write_text(readme)
    _write_json(output / "model_config.json", {"schema": "surface-code-decoding-model-config/v1", "models": models})
    _write_json(output / "training_provenance.json", {**provenance, "source_package_manifest_sha256": _sha256(package / "manifest.json")})
    _write_json(output / "model_manifest.json", {"schema": "surface-code-decoding-model/v1", "checkpoints": checkpoints,
                                                 "model_config": "model_config.json", "training_provenance": "training_provenance.json"})


def write_template(output: Path) -> None:
    """The submission layout without weights: what `package_submission` writes, minus weights/ and the manifest."""
    from baselines.alphaqubit_artifact import EXPECTED_GROUPS, INFERENCE_SOURCES
    output.mkdir(parents=True)
    staging = output / ".baselines"
    staging.mkdir()
    for name in INFERENCE_SOURCES:
        shutil.copyfile(TASK / "baselines" / name, staging / name)
    (staging / "__init__.py").write_text('"""Portable AlphaQubit inference package."""\n')
    _common(output, staging)
    shutil.rmtree(staging)
    _write_json(output / "model_config.json", {"schema": "surface-code-decoding-model-config/v1", "models": {
        group: {"weights": f"weights/{group}/model.npz", "metadata": f"weights/{group}/best.json"} for group in EXPECTED_GROUPS}})
    (output / "README.md").write_text(
        "# AlphaQubit submission template (no weights)\n\n"
        "The layout `python -m qecbench.package` produces, without `weights/` and `model_manifest.json`. "
        "Train every group with `python -m baselines.alphaqubit_train` (pretrain, then finetune), freeze with "
        "`python -m baselines.alphaqubit_artifact --models-root MODELS --stage finetune --output FROZEN`, then run "
        "`python -m qecbench.package --package FROZEN --output /workspace/submission`. Any other model is equally "
        "valid if it follows /workspace/contract.md.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True, help="frozen AlphaQubit package (baselines.alphaqubit_artifact)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialization", default="random (trained from scratch in this session)")
    parser.add_argument("--training", default="first19880even hardware rows and pij_from_even_for_odd simulations")
    args = parser.parse_args()
    # Assemble next to --output and swap it in with one rename, so a reader of --output (for example
    # /workspace/submission, which may be snapshotted at any moment) never sees a half-written package.
    final = args.output.resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".{final.name}.staging-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    package_submission(args.package, staging, {
        "initialization": args.initialization, "training": args.training, "validation": "last5120even",
        "test_selection": False,
        "calibration_exception": "pij DEM was fitted to all even syndromes, including validation syndromes; no odd syndromes or logical labels"},
        "# AlphaQubit submission\n\nTen specialists packaged by `python -m qecbench.package`. "
        "Run `bash infer.sh INPUT_ROOT OUTPUT_ROOT`; see training_provenance.json.\n")
    previous = final.parent / f".{final.name}.previous-{int(time.time())}"
    if final.exists():
        try:
            os.replace(final, previous)
        except OSError as error:        # a directory from a lower image layer cannot be renamed on overlayfs
            if error.errno != errno.EXDEV:
                raise
            shutil.copytree(final, previous, symlinks=True)
            shutil.rmtree(final)
    os.replace(staging, final)
    shutil.rmtree(previous, ignore_errors=True)
    print(f"wrote submission {final}")


if __name__ == "__main__":
    main()
