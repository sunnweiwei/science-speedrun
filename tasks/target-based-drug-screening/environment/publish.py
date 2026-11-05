#!/usr/bin/env python3
"""Validate an owned inference package and publish it with an atomic directory swap."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def validate_package(root: Path) -> dict:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("submission must be a real directory")
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"package contains a link or special file: {path}")
    for name in ("infer.sh", "model_manifest.json"):
        if not (root / name).is_file():
            raise ValueError(f"missing package file: {name}")
    manifest = json.loads((root / "model_manifest.json").read_text())
    for relative in (manifest["checkpoint"]["path"], manifest["model_config"]):
        path = root / relative
        if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("manifest asset escapes package")
        if not path.is_file():
            raise ValueError(f"missing manifest asset: {relative}")
    digest = hashlib.sha256()
    with (root / manifest["checkpoint"]["path"]).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != manifest["checkpoint"]["sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    if int(manifest["parameter_count"]) <= 0:
        raise ValueError("parameter_count must be positive")
    return manifest


def publish(source: Path, destination: Path) -> None:
    source, destination = source.resolve(), destination.absolute()
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("source and destination must be separate directories")
    validate_package(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".publish-", dir=destination.parent))
    try:
        shutil.copytree(source, stage, dirs_exist_ok=True)
        if not (stage / "README.md").is_file():
            (stage / "README.md").write_text(
                "# Candidate\n\nSee model_manifest.json, model_config.json and "
                "training_summary.json for method, parameter count and provenance.\n"
                "This is a short-run candidate; full-suite performance is unverified.\n")
        validate_package(stage)
        if destination.exists():
            if destination.is_symlink() or not destination.is_dir():
                raise ValueError("destination must be a real directory")
            libc = ctypes.CDLL(None, use_errno=True)
            renameat2 = libc.renameat2
            renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                                 ctypes.c_char_p, ctypes.c_uint]
            renameat2.restype = ctypes.c_int
            # RENAME_EXCHANGE=2 has no missing-directory window. Do not fall
            # back to renaming the live submission out of the way first.
            if renameat2(-100, os.fsencode(stage), -100, os.fsencode(destination), 2):
                code = ctypes.get_errno()
                if code != errno.EXDEV:
                    raise OSError(code, os.strerror(code))
                # OverlayFS can reject exchange with an empty image-layer directory.
                # rename replaces an empty directory atomically and refuses a nonempty one.
                os.replace(stage, destination)
        else:
            os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--destination", type=Path, default=Path("/workspace/submission"))
    args = parser.parse_args()
    publish(args.source, args.destination)
