#!/usr/bin/env python3
"""Fetch immutable pre-materialized Hugging Face archives with strict checks."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_role(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Role root must be an ordinary directory")
    manifest = json.loads((root / "inventory.json").read_text())
    if manifest.get("schema") != "surface-code-decoding-inventory/v1":
        raise ValueError("Unsupported role inventory")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("Materialized payload must contain ordinary files and directories")
        if path.is_file() and path.stat().st_nlink != 1:
            raise ValueError("Materialized payload cannot contain hard links")
        if path.is_file() and path != root / "inventory.json":
            actual.add(path.relative_to(root).as_posix())
    if actual != set(manifest["files"]):
        raise ValueError("Role payload inventory mismatch")
    for relative, record in manifest["files"].items():
        path = root / relative
        if (not path.resolve().is_relative_to(root.resolve()) or path.stat().st_size != record["bytes"]
                or sha256(path) != record["sha256"]):
            raise ValueError(f"Materialized payload checksum mismatch: {relative}")


def validated_members(tar: tarfile.TarFile, role: str, expanded_bytes: int) -> list[tarfile.TarInfo]:
    """Apply identical path/type/coverage bounds to fresh and reused archives."""
    members = tar.getmembers()
    names = [item.name for item in members]
    if len(names) != len(set(names)) or sum(item.size for item in members) > expanded_bytes + 1024 * 1024:
        raise ValueError("Archive expanded size or coverage mismatch")
    for item in members:
        path = Path(item.name)
        if (path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != role
                or not (item.isfile() or item.isdir()) or path.as_posix() != item.name):
            raise ValueError(f"Unsafe archive member: {item.name}")
    return members


def archive_inventory(archive: Path, role: str, expanded_bytes: int) -> bytes:
    """Read the trusted inventory directly without creating temporary payloads.

    The caller must verify the complete pinned archive SHA-256 and size first.
    Reuse still validates all tar member paths/types and the expanded-size bound.
    """
    with tarfile.open(archive, "r:gz") as tar:
        members = validated_members(tar, role, expanded_bytes)
        selected = [item for item in members if item.name == f"{role}/inventory.json"]
        if len(selected) != 1 or not selected[0].isfile():
            raise ValueError("Archive requires one ordinary root inventory.json")
        item = selected[0]
        limit = min(expanded_bytes, 16 * 1024 * 1024)
        if not 0 < item.size <= limit:
            raise ValueError("Archive inventory exceeds its bounded size")
        stream = tar.extractfile(item)
        if stream is None:
            raise ValueError("Archive inventory is unreadable")
        with stream:
            payload = stream.read(limit + 1)
        if len(payload) != item.size:
            raise ValueError("Archive inventory size mismatch")
        return payload


def extract(archive: Path, destination: Path, role: str, expanded_bytes: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = validated_members(tar, role, expanded_bytes)
        # Paths/types are checked before any write. tarfile's data filter adds
        # defense in depth when available, but is not required on Python 3.11.
        tar.extractall(destination, members=members, filter="data")
    verify_role(destination / role)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.from_sources:
        if not args.source or not args.reference_results:
            raise ValueError("--from-sources requires --source and --reference-results")
        subprocess.run([sys.executable, str(HERE / "build_release.py"), "--source", str(args.source),
                        "--reference-results", str(args.reference_results), "--output", str(args.output)], check=True)
        return {"status": "rebuilt_from_sources", "directory": str(args.output / "materialized")}
    lock = json.loads(args.lock.read_text())
    if lock.get("schema") != "surface-code-decoding-hf-release/v1":
        raise ValueError("Unsupported HF release lock")
    args.output.mkdir(parents=True, exist_ok=True)
    completed = []
    for role in args.roles:
        record = lock["artifacts"][role]
        if not re.fullmatch(r"[0-9a-f]{40}", record["revision"]):
            raise ValueError("HF revision must be an immutable 40-character commit SHA")
        if not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
            raise ValueError("Invalid archive checksum")
        if args.archive_dir:
            source = args.archive_dir / record["filename"]
        else:
            from huggingface_hub import hf_hub_download
            source = Path(hf_hub_download(repo_id=record["repo_id"], repo_type="dataset", filename=record["filename"],
                                          revision=record["revision"], cache_dir=args.cache_dir))
        if source.stat().st_size != record["bytes"] or sha256(source) != record["sha256"]:
            raise ValueError(f"HF archive checksum/size mismatch: {role}")
        target = args.output / role
        if target.exists() or target.is_symlink():
            verify_role(target)
            expected_inventory = archive_inventory(source, role, record["expanded_bytes"])
            if (target / "inventory.json").read_bytes() != expected_inventory:
                raise ValueError("Existing role belongs to another release; choose a new output directory")
            completed.append(role)
            continue
        with tempfile.TemporaryDirectory(prefix=".qec-materialize-", dir=args.output) as temporary:
            staging = Path(temporary)
            extract(source, staging, role, record["expanded_bytes"])
            # Same-filesystem rename publishes the role atomically and refuses
            # to merge into a nonempty role created by a concurrent materializer.
            (staging / role).rename(target)
        completed.append(role)
    return {"status": "verified", "release": lock["release"], "roles": completed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roles", nargs="+", choices=("development", "verifier", "grader", "reference-model"), default=["development"],
                        help="reference-model: the frozen reference submission, operator-only (known-answer replays); never mount it for agents")
    parser.add_argument("--lock", type=Path, default=HERE / "hf_release.json")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--archive-dir", type=Path, help="Offline mirror; archive hashes still required")
    parser.add_argument("--from-sources", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--reference-results", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
