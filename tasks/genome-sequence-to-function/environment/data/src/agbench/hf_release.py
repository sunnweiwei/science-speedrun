"""Install a checksummed sharded release from a Hugging Face dataset repo."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

from .errors import ContractError
from .validate import validate_release


HF_RELEASE_SCHEMA = "genome-sequence-to-function-hf/v4"
HF_RECEIPT_SCHEMA = "genome-sequence-to-function-hf-materialization/v4"
MANIFEST_NAME = "release-manifest.json"


class HuggingFaceUnavailable(RuntimeError):
    """The configured Hub release could not be reached or downloaded."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_identity(repo_id: str, revision: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id):
        raise ValueError("HF dataset must be an owner/repository ID")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", revision):
        raise ValueError("HF revision must be a branch, tag, or commit")


def _hf(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    if not shutil.which("hf"):
        raise HuggingFaceUnavailable("`hf` CLI is unavailable")
    try:
        return subprocess.run(["hf", *command], check=True, **kwargs)
    except subprocess.CalledProcessError as exc:
        raise HuggingFaceUnavailable(f"hf {' '.join(command[:2])} failed") from exc


def _resolved_revision(repo_id: str, revision: str) -> str:
    result = _hf(
        ["datasets", "info", repo_id, "--revision", revision, "--format", "json"],
        text=True,
        stdout=subprocess.PIPE,
    )
    try:
        resolved = str(json.loads(result.stdout)["sha"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise HuggingFaceUnavailable("HF dataset info did not contain a commit SHA") from exc
    if not re.fullmatch(r"[0-9a-f]{40,64}", resolved):
        raise HuggingFaceUnavailable(f"HF returned an invalid commit SHA: {resolved!r}")
    return resolved


def _load_release(download: Path, repo_id: str, manifest_sha256: str | None = None) -> dict:
    path = download / MANIFEST_NAME
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"HF {MANIFEST_NAME} is missing or invalid")
    if manifest_sha256 is not None and _sha256(path) != manifest_sha256:
        raise ContractError(f"HF {MANIFEST_NAME} differs from the pinned SHA-256")
    release = json.loads(path.read_text())
    if release.get("schema") != HF_RELEASE_SCHEMA:
        raise ContractError("Unsupported HF release schema")
    if release.get("repo_id") != repo_id:
        raise ContractError("HF release repo identity differs from download-source.json")
    artifacts = release.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ContractError("HF release contains no artifacts")
    groups = [entry.get("group") for entry in artifacts]
    if not groups or groups[0] != "support" or set(groups) != {
        "support", "train", "valid", "test"
    }:
        raise ContractError("HF release must contain support/train/valid/test archives")
    return release


def _archive_path(download: Path, entry: dict) -> Path:
    relative = entry.get("path")
    if not isinstance(relative, str):
        raise ContractError("HF archive path is missing")
    path = (download / relative).resolve()
    if not path.is_relative_to(download.resolve()) or not path.is_file() or path.is_symlink():
        raise ContractError(f"HF archive path is invalid: {relative}")
    expected_size = entry.get("size")
    expected_hash = entry.get("sha256")
    if path.stat().st_size != expected_size:
        raise ContractError(f"HF archive size mismatch: {relative}")
    if not isinstance(expected_hash, str) or _sha256(path) != expected_hash:
        raise ContractError(f"HF archive checksum mismatch: {relative}")
    return path


def _extract_archive(archive_path: Path, destination: Path) -> None:
    """Extract with Python's traversal/link/device-rejecting data filter."""
    with tarfile.open(archive_path, mode="r:") as archive:
        while member := archive.next():
            try:
                archive.extract(member, path=destination, filter="data")
            except (tarfile.TarError, OSError) as exc:
                raise ContractError(
                    f"Unsafe or invalid HF archive member in {archive_path.name}: {member.name}"
                ) from exc


def install_huggingface_release(
    data_dir: Path,
    *,
    repo_id: str,
    revision: str,
    manifest_sha256: str | None = None,
) -> dict:
    """Verify and extract every archive one at a time, validate, then install atomically.

    Each archive is downloaded, checked against the manifest, extracted (or moved) and deleted
    before the next one, so peak disk use is the release plus one archive.
    """
    _validate_identity(repo_id, revision)
    data_dir = data_dir.resolve()
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    resolved = _resolved_revision(repo_id, revision)
    with tempfile.TemporaryDirectory(
        prefix=f".{data_dir.name}-hf-", dir=data_dir.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        download = temporary / "download"
        prepared = temporary / "prepared"
        backup = temporary / "previous"
        prepared.mkdir()

        def fetch(*names: str) -> None:
            _hf([
                "download", repo_id, *names, "--type", "dataset",
                "--revision", resolved, "--local-dir", str(download),
            ])

        fetch(MANIFEST_NAME)
        release = _load_release(download, repo_id, manifest_sha256)
        artifact_names = [entry.get("path") for entry in release["artifacts"]]
        if any(not isinstance(name, str) for name in artifact_names):
            raise ContractError("HF release contains an invalid artifact path")
        for entry in release["artifacts"]:
            fetch(entry["path"])
            artifact_path = _archive_path(download, entry)
            kind = entry.get("kind", "archive")
            if kind == "archive":
                _extract_archive(artifact_path, prepared)
                artifact_path.unlink()
            elif kind == "file":
                relative = entry.get("install_path")
                if not isinstance(relative, str):
                    raise ContractError("HF file artifact install path is missing")
                destination = (prepared / relative).resolve()
                if not destination.is_relative_to(prepared.resolve()):
                    raise ContractError(f"HF artifact install path is invalid: {relative}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                artifact_path.replace(destination)
            else:
                raise ContractError(f"Unsupported HF artifact kind: {kind!r}")
            shutil.rmtree(download / ".cache", ignore_errors=True)
        report = validate_release(
            prepared, splits=["train", "valid", "test"], require_complete=True,
            require_release=True,
        )
        receipt = {
            "schema": HF_RECEIPT_SCHEMA,
            "status": "PASS",
            "repo_id": repo_id,
            "requested_revision": revision,
            "resolved_revision": resolved,
            "release_id": release.get("release_id"),
            "release_manifest_sha256": _sha256(download / MANIFEST_NAME),
            "validation": report,
        }
        (prepared / "HF_MATERIALIZATION.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        had_previous = data_dir.exists()
        if had_previous:
            data_dir.rename(backup)
        try:
            prepared.rename(data_dir)
        except BaseException:
            if had_previous:
                backup.rename(data_dir)
            raise
    return receipt
