from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest

from agbench import hf_release
from agbench.errors import ContractError


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _release(tmp_path: Path) -> Path:
    release = tmp_path / "release"
    artifacts = []
    for index, group in enumerate(("support", "train", "valid", "test")):
        payload = tmp_path / f"payload-{group}"
        payload.mkdir()
        (payload / f"{group}.txt").write_text(group)
        archive = release / "archives" / f"{group}.tar"
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w") as stream:
            stream.add(payload / f"{group}.txt", arcname=f"installed/{group}.txt")
        artifacts.append({
            "group": group,
            "path": str(archive.relative_to(release)),
            "size": archive.stat().st_size,
            "sha256": _sha256(archive),
        })
    (release / hf_release.MANIFEST_NAME).write_text(json.dumps({
        "schema": hf_release.HF_RELEASE_SCHEMA,
        "repo_id": "owner/repo",
        "release_id": "example",
        "artifacts": artifacts,
    }))
    return release


def _install(monkeypatch, release: Path, output: Path, manifest_sha256: str | None = None):
    monkeypatch.setattr(hf_release, "_resolved_revision", lambda *_: "a" * 40)

    def download(command, **_):
        target = Path(command[command.index("--local-dir") + 1])
        target.mkdir(parents=True, exist_ok=True)
        requested = command[2:command.index("--type")]
        for relative in requested:
            source = release / relative
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(hf_release, "_hf", download)
    monkeypatch.setattr(
        hf_release, "validate_release",
        lambda root, **_: {"status": "PASS", "root": str(root)},
    )
    return hf_release.install_huggingface_release(
        output, repo_id="owner/repo", revision="main", manifest_sha256=manifest_sha256
    )


def test_hf_release_is_verified_extracted_and_atomically_installed(tmp_path, monkeypatch):
    release = _release(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "old.txt").write_text("old")
    receipt = _install(monkeypatch, release, output)
    assert receipt["resolved_revision"] == "a" * 40
    assert not (output / "old.txt").exists()
    for group in ("support", "train", "valid", "test"):
        assert (output / "installed" / f"{group}.txt").read_text() == group
    assert json.loads((output / "HF_MATERIALIZATION.json").read_text())["status"] == "PASS"


def test_bad_archive_does_not_replace_existing_data(tmp_path, monkeypatch):
    release = _release(tmp_path)
    manifest = json.loads((release / hf_release.MANIFEST_NAME).read_text())
    manifest["artifacts"][1]["sha256"] = "0" * 64
    (release / hf_release.MANIFEST_NAME).write_text(json.dumps(manifest))
    output = tmp_path / "output"
    output.mkdir()
    (output / "old.txt").write_text("old")
    with pytest.raises(ContractError, match="checksum mismatch"):
        _install(monkeypatch, release, output)
    assert (output / "old.txt").read_text() == "old"


def test_file_artifact_is_moved_to_checked_install_path(tmp_path, monkeypatch):
    release = _release(tmp_path)
    manifest_path = release / hf_release.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    train = manifest["artifacts"][1]
    train.update({"kind": "file", "install_path": "installed/train.zarr.zip"})
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "output"
    _install(monkeypatch, release, output)
    assert (output / "installed/train.zarr.zip").is_file()


def test_installer_ignores_files_not_listed_by_release(tmp_path, monkeypatch):
    release = _release(tmp_path)
    (release / "obsolete-secret.bin").write_bytes(b"not part of this release")
    output = tmp_path / "output"
    _install(monkeypatch, release, output)
    assert not (output / "obsolete-secret.bin").exists()


def test_pinned_manifest_hash_is_enforced(tmp_path, monkeypatch):
    release = _release(tmp_path)
    manifest = release / hf_release.MANIFEST_NAME
    output = tmp_path / "installed"
    receipt = _install(monkeypatch, release, output, manifest_sha256=_sha256(manifest))
    assert receipt["release_manifest_sha256"] == _sha256(manifest)
    with pytest.raises(ContractError, match="pinned SHA-256"):
        _install(monkeypatch, release, tmp_path / "other", manifest_sha256="0" * 64)
