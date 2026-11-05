"""Small publication and integrity helpers shared by training and inference."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def compact_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    import torch
    torch.save(value, temporary)
    os.replace(temporary, path)


def publish_directory(destination: Path, build) -> None:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        build(stage)
        backup = destination.with_name(destination.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        if destination.exists():
            os.replace(destination, backup)
        os.replace(stage, destination)
        if backup.exists():
            shutil.rmtree(backup)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("manifest path escapes model package")
    return path
