#!/usr/bin/env python3
"""Download immutable prepared releases, verify them, and publish atomically."""
from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable

from release_manifest import safe_path, sha256, validate_manifest, verify


def validate_pin(pin: dict[str, Any]) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", pin.get("repo_id", "")):
        raise ValueError("invalid Hugging Face dataset ID")
    if not re.fullmatch(r"[0-9a-f]{40}", pin.get("revision", "")):
        raise ValueError("prepared dataset release requires a full 40-character commit SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", pin.get("manifest_sha256", "")):
        raise ValueError("prepared dataset release requires a pinned manifest SHA-256")



def verify_download_layout(root: Path, manifest: dict[str, Any]) -> None:
    expected = set(manifest["files"]) | {"release-manifest.json"}
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("prepared release cannot contain symlinks")
    if {str(path.relative_to(root)) for path in paths if path.is_file()} != expected:
        raise ValueError("prepared release contains undeclared or missing files")


def fetch_release(pin: dict[str, Any], destination: Path, component: str,
                  download: Callable[..., str] | None = None) -> dict[str, Any]:
    validate_pin(pin)
    if destination.is_symlink():
        raise ValueError("destination must not be a symlink")
    destination = destination.absolute()
    for path in destination.parents:
        if path.is_symlink():
            raise ValueError("destination ancestors must not be symlinks")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A verified complete local release is reusable offline without credentials.
    if destination.exists():
        local_manifest = destination / "release-manifest.json"
        if not local_manifest.is_file() or local_manifest.is_symlink() or sha256(local_manifest) != pin["manifest_sha256"]:
            raise FileExistsError("destination exists without the pinned release manifest; use a fresh destination")
        manifest = json.loads(local_manifest.read_text())
        validate_manifest(manifest, component)
        verify_download_layout(destination, manifest)
        return verify(destination, manifest, component)
    if download is None:
        from huggingface_hub import hf_hub_download
        download = hf_hub_download
    # Hugging Face may retain open metadata/lock files, which cannot be unlinked
    # reliably on NFS. Keep its persistent cache outside the transactional data
    # tree and mount neither this directory nor operator credentials to models.
    cache = destination.parent / ".massspecgym-hf-cache"
    if cache.is_symlink():
        raise ValueError("operator download cache must not be a symlink")
    cache.mkdir(mode=0o700, exist_ok=True)
    cache.chmod(0o700)
    with tempfile.TemporaryDirectory(prefix=f".massspecgym-{component}-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "payload"
        stage.mkdir(mode=0o700)
        kwargs = {"repo_id": pin["repo_id"], "repo_type": "dataset", "revision": pin["revision"], "cache_dir": cache}

        def stage_download(name: str) -> Path:
            source = Path(download(filename=name, **kwargs)).resolve(strict=True)
            if not source.is_relative_to(cache.resolve()) or not source.is_file():
                raise ValueError("HF download result must be a file inside the operator cache")
            target = safe_path(stage, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                # The cache and stage are siblings: normally one physical copy,
                # without importing snapshot symlinks or HF metadata into data.
                os.link(source, target)
            except OSError as error:
                if error.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.EMLINK):
                    raise
                if shutil.disk_usage(stage).free < source.stat().st_size + 64 * 1024**2:
                    raise OSError("insufficient disk space for cache-to-stage copy") from error
                shutil.copyfile(source, target)
            return target

        manifest_path = stage_download("release-manifest.json")
        if manifest_path.stat().st_size > 1024 * 1024 or sha256(manifest_path) != pin["manifest_sha256"]:
            raise ValueError("downloaded release manifest is not the pinned manifest")
        manifest = json.loads(manifest_path.read_text())
        validate_manifest(manifest, component)
        if shutil.disk_usage(destination.parent).free < manifest["total_bytes"] + 64 * 1024**2:
            raise OSError("insufficient free disk space for prepared release")
        for name in manifest["files"]:
            stage_download(name)
        result = verify(stage, manifest, component)
        # Keep only explicitly declared data; no source caches, credentials or model weights.
        verify_download_layout(stage, manifest)
        for path in stage.rglob("*"):
            path.chmod((0o700 if path.is_dir() else 0o600) if component == "private" else (0o755 if path.is_dir() else 0o644))
        stage.chmod(0o700 if component == "private" else 0o755)
        if destination.exists():
            raise FileExistsError("destination appeared during download")
        os.rename(stage, destination)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--private-label-root", type=Path, required=True)
    parser.add_argument("--release-config", type=Path, default=Path(__file__).with_name("hf_release.json"))
    args = parser.parse_args()
    public, private = args.public_root.absolute(), args.private_label_root.absolute()
    public_resolved, private_resolved = public.resolve(), private.resolve()
    if (public_resolved == private_resolved or public_resolved.is_relative_to(private_resolved)
            or private_resolved.is_relative_to(public_resolved)):
        raise ValueError("public and grader-label destinations must be separate, nonnested directories")
    config = json.loads(args.release_config.read_text())
    if config.get("schema") != "massspecgym-hf-release-pins/v1":
        raise ValueError("invalid prepared release config")
    # Validate both pins before starting a potentially large download.
    for key in ("public", "private"):
        validate_pin(config[key])
    results = [fetch_release(config["public"], public, "public"),
               fetch_release(config["private"], private, "private")]
    print(json.dumps({"status": "ready", "components": results}, indent=2))


if __name__ == "__main__":
    main()
