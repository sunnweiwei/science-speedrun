from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Iterator

import lmdb
import requests
from tqdm import tqdm


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_immutable_text(path: Path, content: str) -> None:
    """Create a protocol artifact once; refuse silent split drift."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"immutable artifact would change: {path}; use a new root or --overwrite")
        return
    path.write_text(content)


def download_resumable(url: str, destination: Path, expected_size: int | None = None,
                       expected_sha256: str | None = None,
                       expected_etag: str | None = None) -> dict:
    """Resume into .part, verify, then atomically publish destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        size = destination.stat().st_size
        digest = sha256_file(destination)
        if (expected_size is None or size == expected_size) and (
            expected_sha256 is None or digest == expected_sha256
        ):
            return {"url": url, "filename": destination.name, "size": size, "sha256": digest}
        raise RuntimeError(f"existing download does not match lock: {destination}")
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists() and expected_size is not None and partial.stat().st_size == expected_size:
        digest = sha256_file(partial)
        if expected_sha256 is None or digest == expected_sha256:
            os.replace(partial, destination)
            return {"url": url, "filename": destination.name, "size": expected_size,
                    "sha256": digest, "etag": expected_etag or ""}
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 300),
                      allow_redirects=True) as response:
        response.raise_for_status()
        response_etag = response.headers.get("ETag", "").strip('"')
        if expected_etag and response_etag and response_etag != expected_etag.strip('"'):
            raise RuntimeError(f"ETag mismatch for {destination.name}: {response_etag}")
        if offset and response.status_code != 206:
            partial.unlink()
            offset = 0
        mode = "ab" if offset else "wb"
        progress = tqdm(total=expected_size, initial=offset,
                        desc=f"Downloading {destination.name}", unit="B",
                        unit_scale=True, unit_divisor=1024)
        with partial.open(mode) as handle, progress:
            for chunk in response.iter_content(8 << 20):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
    size = partial.stat().st_size
    if expected_size is not None and size != expected_size:
        raise RuntimeError(f"size mismatch for {destination.name}: {size} != {expected_size}")
    digest = sha256_file(partial)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"SHA256 mismatch for {destination.name}: {digest}")
    os.replace(partial, destination)
    return {"url": url, "filename": destination.name, "size": size, "sha256": digest,
            "etag": response_etag}


def _safe_member(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if root.resolve() not in (target, *target.parents):
        raise RuntimeError(f"unsafe archive path: {name}")
    return target


def extract_zip(archive: Path, destination: Path, strip_prefix: str) -> None:
    marker = destination / ".extracted.json"
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = [m for m in bundle.infolist() if m.filename.startswith(strip_prefix + "/")]
        if not members:
            raise RuntimeError(f"{strip_prefix!r} not found in {archive}")
        for member in members:
            relative = member.filename[len(strip_prefix) + 1:]
            if not relative:
                continue
            target = _safe_member(destination, relative)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, 8 << 20)
    write_json(marker, {"archive_sha256": sha256_file(archive), "prefix": strip_prefix})


def extract_tar(archive: Path, destination: Path) -> None:
    marker = destination / ".extracted.json"
    digest = sha256_file(archive)
    if marker.exists() and json.loads(marker.read_text()).get("archive_sha256") == digest:
        return
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            _safe_member(destination, member.name)
            bundle.extract(member, destination, filter="data")
    write_json(marker, {"archive_sha256": digest})


def iter_lmdb(path: Path) -> Iterator[tuple[bytes, dict]]:
    env = lmdb.open(str(path), subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False, max_readers=256)
    try:
        with env.begin() as txn:
            for key, value in txn.cursor():
                yield key, pickle.loads(value)
    finally:
        env.close()


def write_lmdb(path: Path, records: Iterable[dict], map_size: int = 1 << 40) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    env = lmdb.open(str(temporary), subdir=False, map_size=map_size, lock=True,
                    readahead=False, meminit=False, max_readers=1)
    count = 0
    try:
        txn = env.begin(write=True)
        for count, record in enumerate(records, 1):
            txn.put(str(count - 1).encode("ascii"), pickle.dumps(record, protocol=-1))
            if count % 10_000 == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.commit()
        env.sync()
    finally:
        env.close()
    os.replace(temporary, path)
    temporary_lock = Path(str(temporary) + "-lock")
    if temporary_lock.exists():
        temporary_lock.unlink()
    return count


def require_executable(name: str) -> str:
    executable = shutil.which(name)
    if not executable:
        raise RuntimeError(f"required executable is missing: {name}")
    return executable


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, text=True, **kwargs)


def atomic_directory(target: Path):
    """Return an adjacent temporary directory for caller-managed publication."""
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
