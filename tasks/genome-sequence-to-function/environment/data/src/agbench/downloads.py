"""Small, checksum-verifying downloads and explicitly requested reference fetches."""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import urllib.request

from .errors import ContractError


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_base64_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def download_verified(
    url: str,
    destination: Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    expected_md5_base64: str | None = None,
) -> Path:
    """Download atomically and verify before replacing the destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _verify(destination, expected_size, expected_sha256, expected_md5_base64)
        return destination
    partial = destination.with_name(destination.name + ".partial")
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as output:
            while chunk := response.read(8 << 20):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        _verify(partial, expected_size, expected_sha256, expected_md5_base64)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return destination


def _verify(
    path: Path,
    expected_size: int | None,
    expected_sha256: str | None,
    expected_md5_base64: str | None,
) -> None:
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ContractError(
            f"Size mismatch for {path}: {path.stat().st_size} != {expected_size}"
        )
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ContractError(f"SHA-256 mismatch for {path}")
    if expected_md5_base64 is not None and md5_base64_file(path) != expected_md5_base64:
        raise ContractError(f"GCS MD5 mismatch for {path}")
