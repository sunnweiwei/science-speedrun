"""Freeze and verify portable, independently implemented AlphaQubit weights.

The NPZ contains plain FP32 Haiku leaves, never Python pickles. Package hashes
detect corruption and source drift; they are not a cryptographic signature.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import fields
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np


TASK_ROOT = Path(__file__).resolve().parents[1]
LABEL_MODE = "official_path_observable_flip"
INFERENCE_SOURCES = ("alphaqubit_artifact.py", "alphaqubit_predict.py",
                     "alphaqubit_model.py", "alphaqubit_data.py", "data.py")
SEMANTIC_SOURCES = ("alphaqubit_model.py", "alphaqubit_data.py", "data.py")
GEOMETRY_KEYS = ("distance", "stabilizer_positions", "stabilizer_types", "on_basis",
                 "data_positions", "logical_axis", "logical_index", "logical_data_indices")
EXPECTED_GROUPS = tuple(sorted(
    f"d{distance}_b{basis}_center_{row}_{col}"
    for distance, centers in ((3, ((3, 5), (5, 3), (5, 7), (7, 5))), (5, ((5, 5),)))
    for row, col in centers for basis in ("X", "Z")
))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, _json_bytes(value))


def _payload_files(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for directory, dirs, names in os.walk(root):
        parent = Path(directory)
        # A fresh uv environment and Python bytecode are execution byproducts.
        dirs[:] = [name for name in dirs if name != "__pycache__" and not (parent == root and name == ".venv")]
        for name in [*dirs, *names]:
            if (parent / name).is_symlink():
                raise ValueError(f"Package payload cannot contain symbolic links: {parent / name}")
        for name in names:
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if relative != "manifest.json":
                result[relative] = path
    return result


def verify_package(root: Path) -> dict[str, Any]:
    """Check exact payload coverage, hashes and checkpoint/manifest identity."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir() or (root / "manifest.json").is_symlink():
        raise ValueError("Package root and manifest must not be symbolic links")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("format") != "alphaqubit-haiku-fp32-npz":
        raise ValueError("Unsupported AlphaQubit package manifest")
    declared = manifest.get("files")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("Package has no declared payloads")
    for relative in declared:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
            raise ValueError(f"Unsafe/noncanonical package path: {relative}")
    actual = _payload_files(root)
    if set(actual) != set(declared):
        raise ValueError("Package file coverage mismatch")
    required = {f"baselines/{name}" for name in INFERENCE_SOURCES} | {"baselines/__init__.py", "README.md", "pyproject.toml", "uv.lock"}
    if not required.issubset(actual):
        raise ValueError("Package is missing required inference/runtime sources")
    for relative, record in declared.items():
        if actual[relative].stat().st_size != record.get("bytes") or sha256_file(actual[relative]) != record.get("sha256"):
            raise ValueError(f"Package hash/size mismatch: {relative}")
    records = manifest.get("models", {})
    if (not set(manifest.get("groups", [])).issubset(EXPECTED_GROUPS)
            or not set(manifest.get("stages", [])).issubset({"pretrain", "finetune"})):
        raise ValueError("Package contains unsupported groups/stages")
    expected = {f"{group}/{stage}" for group in manifest.get("groups", []) for stage in manifest.get("stages", [])}
    if not expected or set(records) != expected:
        raise ValueError("Package model coverage disagrees with groups/stages")
    for key, record in records.items():
        for field in ("model", "checkpoint_metadata", "training_metadata", "progress"):
            if record.get(field) not in declared:
                raise ValueError(f"Undeclared model payload: {key}/{field}")
        metadata = json.loads(actual[record["checkpoint_metadata"]].read_text())
        if (f"{metadata.get('group')}/{metadata.get('stage')}" != key
                or metadata.get("label_mode") != LABEL_MODE
                or metadata.get("identity_sha256") != record.get("identity_sha256")
                or metadata.get("checkpoint_sha256") != declared[record["model"]]["sha256"]
                or record.get("checkpoint_sha256") != metadata["checkpoint_sha256"]):
            raise ValueError(f"Package checkpoint identity mismatch: {key}")
        group, stage = key.split("/")
        prefix = f"models/{group}/{stage}"
        if record["model"] != f"{prefix}/model.npz" or record["checkpoint_metadata"] != f"{prefix}/best.json":
            raise ValueError(f"Noncanonical packaged checkpoint location: {key}")
        _, verified_record = _checkpoint_record(root / prefix, group, stage,
            {name: actual[name].read_bytes() for name in required}, model_filename="model.npz")
        if record != verified_record:
            raise ValueError(f"Manifest selection/provenance differs from checkpoint: {key}")
    return manifest


def read_parameters(source: Path | bytes) -> dict[str, dict[str, np.ndarray]]:
    """Restore flat NPZ keys split at the final slash, with strict FP32 leaves."""
    result: dict[str, dict[str, np.ndarray]] = {}
    with np.load(io.BytesIO(source) if isinstance(source, bytes) else source, allow_pickle=False) as archive:
        if not archive.files or len(set(archive.files)) != len(archive.files):
            raise ValueError("Empty/duplicate checkpoint parameters")
        for key in archive.files:
            if "/" not in key or not all(key.rsplit("/", 1)):
                raise ValueError(f"Malformed Haiku parameter name: {key}")
            module, name = key.rsplit("/", 1)
            value = archive[key]
            if value.dtype != np.dtype("float32") or not np.isfinite(value).all():
                raise ValueError(f"Checkpoint leaf must be finite FP32: {key}")
            result.setdefault(module, {})[name] = value
    return result


def load_model(model_path: Path, metadata_path: Path | None = None) -> tuple[Any, Any, dict[str, Any]]:
    """Load ordinary Haiku parameters with the package's pinned model source."""
    from .alphaqubit_model import AlphaQubitConfig, transformed_model

    path = metadata_path or model_path.parent / "best.json"
    metadata = json.loads(path.read_text())
    if metadata.get("schema_version") != 1 or metadata.get("label_mode") != LABEL_MODE:
        raise ValueError("Unsupported checkpoint metadata/label convention")
    if metadata.get("checkpoint_sha256") != sha256_file(model_path):
        raise ValueError("Checkpoint does not match metadata SHA256")
    for name in SEMANTIC_SOURCES:
        values = [value for key, value in metadata.get("source_sha256", {}).items() if Path(key).name == name]
        if values != [sha256_file(Path(__file__).with_name(name))]:
            raise ValueError(f"Checkpoint/inference source mismatch: {name}")
    raw_config = metadata["model_config"]
    config = AlphaQubitConfig(**{field.name: raw_config[field.name] for field in fields(AlphaQubitConfig)})
    if config.to_dict() != raw_config:
        raise ValueError("Checkpoint model configuration or implementation choices disagree with inference source")
    if config.distance != metadata.get("distance") or metadata["geometry"].get("distance") != config.distance:
        raise ValueError("Checkpoint model and geometry distances disagree")
    return transformed_model(config, metadata["geometry"]), read_parameters(model_path), metadata


def _checkpoint_record(folder: Path, group: str, stage: str, sources: dict[str, bytes],
                       model_filename: str = "best.npz") -> tuple[dict[str, bytes], dict[str, Any]]:
    paths = {name: folder / (model_filename if name == "best.npz" else name)
             for name in ("best.npz", "best.json", "metadata.json", "progress.json")}
    payloads = {name: path.read_bytes() for name, path in paths.items()}
    best, metadata, progress = (json.loads(payloads[name]) for name in ("best.json", "metadata.json", "progress.json"))
    identity = metadata.get("identity", {})
    identity_sha = _digest(json.dumps(identity, sort_keys=True, allow_nan=False).encode())
    if not identity or metadata.get("identity_sha256") != identity_sha or best.get("identity_sha256") != identity_sha:
        raise ValueError(f"{group}: checkpoint training identity mismatch")
    for record in (best, identity, metadata):
        if (record.get("group"), record.get("stage"), record.get("distance"), record.get("label_mode")) != (group, stage, int(group[1]), LABEL_MODE):
            raise ValueError(f"{group}: checkpoint group/stage/distance/label mode disagree")
    sessions = metadata.get("sessions", [])
    if not sessions or not sessions[-1].get("ended_at") or progress.get("status") != "stopped":
        raise ValueError(f"{group}: training stage is not stopped and finalized")
    if progress.get("identity_sha256") != identity_sha:
        raise ValueError(f"{group}: progress identity differs from checkpoint")
    steps, ler = best.get("global_steps"), best.get("selected_LER")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0 or progress.get("global_steps", -1) < steps:
        raise ValueError(f"{group}: selected checkpoint is not a valid completed training step")
    if isinstance(ler, bool) or not isinstance(ler, (int, float)) or not math.isfinite(ler) or not 0 <= ler < 0.5:
        raise ValueError(f"{group}: checkpoint has no accepted validation LER")
    if not isinstance(best.get("validation"), dict) or not best["validation"]:
        raise ValueError(f"{group}: checkpoint lacks validation selection evidence")
    validation = best["validation"]
    if (progress.get("best_step") != steps or progress.get("best_LER") != ler
            or validation.get("global_steps") != steps or validation.get("seen_samples") != best.get("seen_samples")
            or validation.get("complete") is not True or validation.get("selected_best") is not True
            or validation.get("fit", {}).get("accepted") is not True
            or validation["fit"].get("logical_error_per_round") != ler):
        raise ValueError(f"{group}: selected step/LER disagrees with validation or final progress")
    for key in ("model_config", "geometry", "source_sha256"):
        if best.get(key) != identity.get(key):
            raise ValueError(f"{group}: checkpoint {key} differs from training identity")
    checkpoint_sha = _digest(payloads["best.npz"])
    if best.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError(f"{group}: checkpoint SHA256 differs from best.json")
    read_parameters(payloads["best.npz"])
    source_hashes = best.get("source_sha256", {})
    for name in SEMANTIC_SOURCES:
        matches = [value for key, value in source_hashes.items() if Path(key).name == name]
        if matches != [_digest(sources[f"baselines/{name}"])]:
            raise ValueError(f"{group}: packaged {name} differs from checkpoint source")
    prefix = f"models/{group}/{stage}"
    record = {
        "model": f"{prefix}/model.npz", "checkpoint_metadata": f"{prefix}/best.json",
        "training_metadata": f"{prefix}/metadata.json", "progress": f"{prefix}/progress.json",
        "checkpoint_sha256": checkpoint_sha, "identity_sha256": identity_sha,
        "group": group, "stage": stage, "distance": best["distance"], "label_mode": LABEL_MODE,
        "selection": {key: best.get(key) for key in ("selection", "global_steps", "seen_samples", "selected_LER", "validation")},
        "stop_reason": progress.get("stop_reason"), "final_global_steps": progress["global_steps"],
        "data_provenance": identity.get("data"), "source_sha256": source_hashes,
        "limitations": metadata.get("limitations", []), "protocol": identity.get("protocol", {}),
        "convergence_established": False,
    }
    if any(paths[name].read_bytes() != payload for name, payload in payloads.items()):
        raise ValueError(f"{group}: training checkpoint changed during snapshot")
    copied = {f"{prefix}/{'model.npz' if name == 'best.npz' else name}": payload for name, payload in payloads.items()}
    return copied, record


def freeze_package(models_root: Path, stage: str, output: Path, group: str | None = None) -> dict[str, Any]:
    """Atomically freeze selected stages; refuse nonidentical existing packages."""
    if stage not in ("pretrain", "finetune", "both") or (group is not None and group not in EXPECTED_GROUPS):
        raise ValueError("Invalid stage or Sycamore group")
    stages = ["pretrain", "finetune"] if stage == "both" else [stage]
    groups = [group] if group else list(EXPECTED_GROUPS)
    sources = {f"baselines/{name}": (TASK_ROOT / "baselines" / name).read_bytes() for name in INFERENCE_SOURCES}
    sources["baselines/__init__.py"] = b'"""Portable AlphaQubit inference package."""\n'
    for name in ("pyproject.toml", "uv.lock"):
        sources[name] = (TASK_ROOT / "alphaqubit-runtime" / name).read_bytes()
    sources["README.md"] = (
        "# Portable AlphaQubit reproduction\n\n"
        "Independent implementation of Bausch et al., Learning high-accuracy error decoding for quantum processors, Nature 635, 834–840 (2024). "
        "https://www.nature.com/articles/s41586-024-08148-8\n\n"
        "These are independently trained weights, not official Google code or weights. "
        "Checkpoint best.json records model dimensions and explicit implementation choices; "
        "metadata/progress preserve training data hashes, validation selection and budget stopping. "
        "A stopped run does not establish convergence or the paper's reported accuracy. "
        "Inputs are hard detector-derived measurements/events, without analogue readout or leakage.\n\n"
        "Install the pinned environment with `uv sync --frozen`. From this relocated directory run "
        "`.venv/bin/python -m baselines.alphaqubit_predict --package . --data-root /path/to/detector-data "
        "--split validation --output /path/out --require-gpu`. Choose `--stage` if both stages are packaged. "
        "CPU replay is permitted by omitting --require-gpu. Input directories need only "
        "circuit_ideal.stim and detection_events.b8. Historical absolute paths are provenance only. "
        "Write predictions outside this package; verification rejects undeclared payload files.\n"
    ).encode()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=TASK_ROOT, capture_output=True, text=True, check=False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        with ExitStack() as stack:
            # Training owns the same lock. The finalized metadata checks remain
            # mandatory even when an older training run has no lock file.
            for selected_group in groups:
                for selected_stage in stages:
                    lock = Path(models_root) / selected_group / selected_stage / ".training.lock"
                    if lock.exists():
                        handle = stack.enter_context(lock.open("rb"))
                        try:
                            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                        except BlockingIOError as error:
                            raise ValueError(f"Training still holds checkpoint lock: {lock}") from error
            payloads, records = dict(sources), {}
            for selected_group in groups:
                for selected_stage in stages:
                    copied, record = _checkpoint_record(Path(models_root) / selected_group / selected_stage, selected_group, selected_stage, sources)
                    payloads.update(copied)
                    records[f"{selected_group}/{selected_stage}"] = record
            for relative, payload in payloads.items():
                atomic_bytes(temporary / relative, payload)
            manifest = {
                "schema_version": 1, "format": "alphaqubit-haiku-fp32-npz", "groups": groups, "stages": stages,
                "models": records, "source_pin": {"git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
                    "sha256": {name: _digest(value) for name, value in sorted(sources.items())}},
                "files": {name: {"bytes": len(value), "sha256": _digest(value)} for name, value in sorted(payloads.items())},
            }
            atomic_json(temporary / "manifest.json", manifest)
            verify_package(temporary)
            if output.exists() or output.is_symlink():
                existing = verify_package(output)
                if existing != manifest:
                    raise ValueError("Existing package is not identical; refusing replacement")
                return existing
            os.rename(temporary, output)
            return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("pretrain", "finetune", "both"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group")
    args = parser.parse_args()
    result = freeze_package(args.models_root, args.stage, args.output, args.group)
    print(json.dumps({"status": "verified", "groups": result["groups"], "stages": result["stages"], "files": len(result["files"])}))


if __name__ == "__main__":
    main()
