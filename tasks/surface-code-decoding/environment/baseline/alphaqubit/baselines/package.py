"""Freeze completed native decoder checkpoints into a relocatable replay package."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import zipfile


TASK_ROOT = Path(__file__).resolve().parents[1]
LABEL_MODE = "qrennd_global_data_parity"
CHECKPOINT_STATE = "reproduction_state.json"
INFERENCE_SOURCES = ("predict.py", "data.py", "observables.py")
RUNTIME_FILES = ("pyproject.toml", "uv.lock")
EXPECTED_GROUPS = tuple(sorted(
    f"d{distance}_b{basis}_center_{row}_{col}"
    for distance, centers in ((3, ((3, 5), (5, 3), (5, 7), (7, 5))), (5, ((5, 5),)))
    for row, col in centers for basis in ("X", "Z")
))


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _identity_digest(identity: dict[str, Any]) -> str:
    # Match the trainer's exact serialization, not the pretty-printed sidecar.
    return _digest(json.dumps(identity, sort_keys=True).encode())


def _atomic_bytes(path: Path, payload: bytes) -> None:
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


def _payload_files(root: Path) -> dict[str, Path]:
    """Ignore only Python/uv runtime byproducts, not extra package payloads."""
    files: dict[str, Path] = {}
    for directory, dirs, names in os.walk(root):
        parent = Path(directory)
        for name in [*dirs, *names]:
            if (parent / name).is_symlink():
                raise ValueError(f"Package cannot contain symbolic links: {parent / name}")
        dirs[:] = [name for name in dirs if name != "__pycache__" and not (parent == root and name == ".venv")]
        for name in names:
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if relative != "manifest.json":
                files[relative] = path
    return files


def verify_package(root: Path) -> dict[str, Any]:
    """Verify all declared payloads, coverage and relative-path containment."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Package root must be a real directory: {root}")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Package manifest cannot be a symbolic link")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported package manifest")
    actual = _payload_files(root)
    declared = manifest["files"]
    for relative in declared:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
            raise ValueError(f"Manifest contains an unsafe/noncanonical relative path: {relative}")
    if set(actual) != set(declared):
        raise ValueError(f"Package file coverage mismatch: missing={sorted(set(declared) - set(actual))}, unexpected={sorted(set(actual) - set(declared))}")
    for relative, record in declared.items():
        payload = actual[relative].read_bytes()
        if len(payload) != record.get("bytes") or _digest(payload) != record.get("sha256"):
            raise ValueError(f"Package payload hash/size mismatch: {relative}")
    return manifest


def _checkpoint_record(
    folder: Path, group: str, stage: str, source_payloads: dict[str, bytes],
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    metadata_bytes = (folder / "metadata.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    identity = metadata.get("identity", {})
    distance = int(group[1])
    for record in (metadata, identity):
        if (record.get("group"), record.get("stage"), record.get("distance"), record.get("label_mode")) != (group, stage, distance, LABEL_MODE):
            raise ValueError(f"{group}: checkpoint metadata group/stage/distance/label mode disagree")
    identity_sha = _identity_digest(identity)
    if metadata.get("identity_sha256") != identity_sha:
        raise ValueError(f"{group}: training identity digest mismatch")
    sessions = metadata.get("sessions", [])
    if not sessions or not sessions[-1].get("ended_at"):
        raise ValueError(f"{group}: training session has not completed")
    if sessions[-1].get("stop_reason") not in {"epochs", "max_steps", "walltime"}:
        raise ValueError(f"{group}: training session was interrupted or failed")
    for name in ("data.py", "observables.py"):
        if identity.get("semantic_source_sha256", {}).get(name) != _digest(source_payloads[f"baselines/{name}"]):
            raise ValueError(f"{group}: packaged {name} differs from the checkpoint's semantic source")

    model_bytes = (folder / "best.keras").read_bytes()
    model_sha = _digest(model_bytes)
    with zipfile.ZipFile(io.BytesIO(model_bytes)) as archive:
        required = {"config.json", "metadata.json", "model.weights.h5", CHECKPOINT_STATE}
        if not required.issubset(archive.namelist()):
            raise ValueError(f"{group}: not a complete standard Keras checkpoint with reproduction state")
        state = json.loads(archive.read(CHECKPOINT_STATE))
    if state.get("identity_sha256") != identity_sha:
        raise ValueError(f"{group}: embedded checkpoint identity does not match metadata")
    step = state.get("global_steps")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0 or state.get("best_step") != step:
        raise ValueError(f"{group}: best_step must equal this selected checkpoint's positive global_steps")
    validation = state.get("latest_validation", {})
    loss = state.get("best_validation_loss")
    if not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0:
        raise ValueError(f"{group}: invalid selected validation loss")
    if validation.get("complete") is not True or validation.get("selected_best") is not True:
        raise ValueError(f"{group}: checkpoint was not selected on a complete validation population")
    if validation.get("global_steps") != step or state.get("last_validation_step") != step or validation.get("loss") != loss:
        raise ValueError(f"{group}: selected step/loss do not match embedded validation evidence")
    progress_path = folder / "progress.json"
    progress_bytes = progress_path.read_bytes() if progress_path.is_file() else None
    progress = json.loads(progress_bytes) if progress_bytes is not None else None
    if progress is not None:
        if progress.get("status") != "stopped" or progress.get("identity_sha256") != identity_sha:
            raise ValueError(f"{group}: stage is not stopped under the selected checkpoint identity")
        if progress.get("best_step") != step or progress.get("best_validation_loss") != loss or progress.get("global_steps", -1) < step:
            raise ValueError(f"{group}: selected checkpoint disagrees with final stage progress")
    # Preserve identity verbatim, including historical path strings, so its
    # digest remains independently auditable. Replay never resolves these paths.
    packaged_metadata = {**metadata, "checkpoint_sha256": model_sha,
                         "package_selection": {"checkpoint": "best.keras", "best_step": step, "best_validation_loss": loss}}
    record = {
        "model": f"models/{group}/model.keras", "metadata": f"models/{group}/metadata.json",
        "checkpoint_sha256": model_sha, "original_metadata_sha256": _digest(metadata_bytes),
        "identity_sha256": identity_sha, "stage": stage, "group": group, "distance": distance,
        "label_mode": LABEL_MODE,
        "selection": {
            "rule": metadata.get("protocol", {}).get("selection"), "best_step": step,
            "best_validation_loss": loss, "validation": validation,
            "final_global_steps": progress.get("global_steps") if progress else None,
            "stop_reason": sessions[-1]["stop_reason"], "session_ended_at": sessions[-1]["ended_at"],
        },
        "data_provenance": identity.get("data"),
        "source_sha256": metadata.get("source_sha256", {}),
        "semantic_source_sha256": identity.get("semantic_source_sha256", {}),
        "native_source": {"repository": identity.get("native_repository"), "commit": identity.get("native_commit")},
        "limitations": {
            "inherited": metadata.get("limitations", []),
            "protocol": metadata.get("protocol", {}),
            "convergence_established": state.get("convergence_established", False),
        },
    }
    # Holding the trainer's lock normally ensures stability. These checks also
    # detect a changed source when the original run had no lock file.
    if (folder / "metadata.json").read_bytes() != metadata_bytes or _digest((folder / "best.keras").read_bytes()) != model_sha:
        raise ValueError(f"{group}: source changed while freezing")
    if progress_bytes is not None and progress_path.read_bytes() != progress_bytes:
        raise ValueError(f"{group}: stage progress changed while freezing")
    return model_bytes, packaged_metadata, record


def freeze_package(models_root: Path, stage: str, output: Path, group: str | None = None) -> dict[str, Any]:
    """Atomically publish one deterministic package; never replace other content."""
    if stage not in {"pretrain", "finetune"}:
        raise ValueError("stage must be pretrain or finetune")
    if group is not None and group not in EXPECTED_GROUPS:
        raise ValueError(f"Unknown group: {group}")
    groups = [group] if group is not None else list(EXPECTED_GROUPS)
    if not models_root.is_dir():
        raise FileNotFoundError(models_root)
    discovered = {path.name for path in models_root.iterdir() if path.is_dir() and (path / stage).is_dir()}
    if group is None and discovered != set(groups):
        raise ValueError(f"Expected exactly 10 completed model groups; missing={sorted(set(groups) - discovered)}, unexpected={sorted(discovered - set(groups))}")
    source_payloads = {f"baselines/{name}": (TASK_ROOT / "baselines" / name).read_bytes() for name in INFERENCE_SOURCES}
    source_payloads.update({name: (TASK_ROOT / name).read_bytes() for name in RUNTIME_FILES})
    source_payloads["UPSTREAM_LICENSE"] = (TASK_ROOT / "baselines" / "reference" / "LICENSE").read_bytes()
    output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        package_lock = stack.enter_context((output.parent / f".{output.name}.freeze.lock").open("a"))
        fcntl.flock(package_lock, fcntl.LOCK_EX)
        for selected in groups:
            lock_path = models_root / selected / stage / ".training.lock"
            if lock_path.is_file():
                handle = stack.enter_context(lock_path.open("r"))
                try:
                    fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ValueError(f"{selected}: training is still active; cannot freeze") from None
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
        try:
            for relative, payload in source_payloads.items():
                _atomic_bytes(temporary / relative, payload)
            records: dict[str, Any] = {}
            for selected in groups:
                folder = models_root / selected / stage
                model, metadata, record = _checkpoint_record(folder, selected, stage, source_payloads)
                _atomic_bytes(temporary / record["model"], model)
                _atomic_bytes(temporary / record["metadata"], _json_bytes(metadata))
                records[selected] = record
            pins = {(record["native_source"]["repository"], record["native_source"]["commit"]) for record in records.values()}
            if len(pins) != 1 or any(not value for value in next(iter(pins))):
                raise ValueError("All packaged models must have one explicit native source pin")
            repository, commit = next(iter(pins))
            readme = (
                "# Frozen surface-code decoder replay\n\n"
                "These checkpoints use the QRENND architecture by Boris M. Varbanov. "
                "The original project is https://github.com/BorisVarbanov/qrennd; "
                f"the audited mirror is {repository}, pinned at `{commit}`. "
                "The upstream Apache-2.0 license is preserved verbatim in `UPSTREAM_LICENSE`.\n\n"
                "Run from this directory after `uv sync --frozen` with Python 3.11.\n\n"
                "```sh\nuv run --no-sync python -m baselines.predict \\\n"
                f"  --model models/{groups[0]}/model.keras --group {groups[0]} \\\n"
                "  --data-root /path/to/sycamore --split test --output /path/to/predictions --require-gpu\n```\n\n"
                "The caller supplies detector data; inference reads no hardware labels. Models output native global parity; "
                "the bundled circuit-derived transform writes official-path probabilities.\n\n"
                "Training paths preserved inside provenance are historical strings, never replay dependencies. "
                "There is no scorer, training implementation, experiment cache, or label dataset in this package. "
                "Validation-selected checkpoints from bounded runs do not establish convergence or a published-score reproduction.\n"
            )
            _atomic_bytes(temporary / "README.md", readme.encode())
            manifest = {
                "schema_version": 1, "stage": stage, "groups": groups,
                "native_source": {"repository": repository, "commit": commit,
                                  "author": "Boris M. Varbanov", "license": "Apache-2.0", "license_file": "UPSTREAM_LICENSE"},
                "models": records,
                "file_scope": "all packaged payloads except manifest.json itself; .venv and __pycache__ are runtime byproducts",
                "portability": "standard Keras archives and bundled inference; historical absolute paths in provenance are never resolved",
                "files": {relative: {"bytes": path.stat().st_size, "sha256": _digest(path.read_bytes())}
                          for relative, path in sorted(_payload_files(temporary).items())},
            }
            _atomic_bytes(temporary / "manifest.json", _json_bytes(manifest))
            verify_package(temporary)
            if output.exists() or output.is_symlink():
                existing = verify_package(output)
                if existing != manifest:
                    raise ValueError("Existing package is not identical to the verified candidate; refusing replacement")
                return existing
            os.rename(temporary, output)
            return manifest
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("pretrain", "finetune"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group")
    args = parser.parse_args()
    manifest = freeze_package(args.models_root, args.stage, args.output, args.group)
    print(json.dumps({"status": "verified", "output": str(args.output), "groups": manifest["groups"], "files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
