from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil

from .io import extract_zip
from .validate import validate_dataset


HERE = Path(__file__).resolve().parents[1]
HF_SOURCE = HERE / "hf-source.json"
RELEASE_ROOTS = {
    "canonical", "metadata", "processed", "smoke", "splits", "test",
    "test_archives",
}
RELEASE_FILES = {
    ".complete.json", ".gitattributes", "DATASET_REPORT.md",
    "FILE_INVENTORY.tsv", "README.md", "VALIDATION.json",
}
REQUIRED_FILES = {
    ".complete.json",
    "DATASET_REPORT.md",
    "VALIDATION.json",
    "canonical/biolip2.parquet",
    "canonical/profsadb.parquet",
    "processed/biolip_dev/data.lmdb",
    "processed/biolip_train/data.lmdb",
    "processed/profsa_pretrain_500k/data.lmdb",
    "processed/manifest.json",
    "splits/dev.txt",
    "splits/train.txt",
    "splits/profsa_pretrain_500k.txt",
    "test/manifest.json",
    "test/test_ligands.parquet",
    "test/test_targets.fasta",
    "test/test_targets.parquet",
    "test_archives/DUD-E.zip",
    "test_archives/LIT-PCBA.zip",
}


class HubUnavailable(RuntimeError):
    """The configured Hub release could not be accessed or downloaded."""


def huggingface_source() -> tuple[str, str]:
    config = json.loads(HF_SOURCE.read_text(encoding="utf-8"))
    repo = os.environ.get("TARGET_BASED_DRUG_SCREENING_HF_REPO") or config["repo_id"]
    revision = (
        os.environ.get("TARGET_BASED_DRUG_SCREENING_HF_REVISION")
        or config["revision"]
    )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("HF dataset must be an owner/repository ID")
    if not revision or not re.fullmatch(r"[A-Za-z0-9_./-]+", revision):
        raise ValueError("HF revision must be a branch, tag, or commit")
    return repo, revision


def data_source() -> str:
    source = os.environ.get("TARGET_BASED_DRUG_SCREENING_DATA_SOURCE", "huggingface")
    if source not in {"huggingface", "upstream"}:
        raise ValueError(
            "TARGET_BASED_DRUG_SCREENING_DATA_SOURCE must be huggingface or upstream"
        )
    return source


def _release_files(root: Path) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symlink in HF release: {path.relative_to(root)}")
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if ".cache" in relative.parts:
            continue
        if relative.parts[0] not in RELEASE_ROOTS and str(relative) not in RELEASE_FILES:
            raise ValueError(f"unexpected file in HF release: {relative}")
        files.add(relative.as_posix())
    missing = REQUIRED_FILES - files
    if missing:
        raise ValueError(f"HF release is missing required files: {sorted(missing)}")
    return files


def _download_release(stage: Path, repo: str, revision: str) -> str:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise HubUnavailable("install huggingface_hub to download the HF release") from error
    try:
        resolved = HfApi().repo_info(
            repo_id=repo, repo_type="dataset", revision=revision
        ).sha
        if not resolved or not re.fullmatch(r"[0-9a-f]{40}", resolved):
            raise ValueError("HF did not resolve a complete commit SHA")
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            revision=resolved,
            local_dir=str(stage),
        )
    except ValueError:
        raise
    except Exception as error:
        raise HubUnavailable(f"could not download {repo}@{revision}: {error}") from error
    cache = stage / ".cache"
    if cache.exists():
        shutil.rmtree(cache)
    return resolved


def _prepare_layout(stage: Path) -> None:
    _release_files(stage)
    archive_root = stage / "raw" / "test_archives"
    archive_root.mkdir(parents=True)
    source_archives = stage / "test_archives"
    for name in ("DUD-E.zip", "LIT-PCBA.zip"):
        (source_archives / name).replace(archive_root / name)
    source_archives.rmdir()
    extract_zip(archive_root / "DUD-E.zip", stage / "test" / "dude", "DUD-E")
    extract_zip(
        archive_root / "LIT-PCBA.zip", stage / "test" / "lit_pcba", "lit_pcba"
    )


def _publish(stage: Path, output_root: Path) -> None:
    backup = output_root / ".hf-previous"
    backup.mkdir()
    previous = [path for path in output_root.iterdir() if path not in {stage, backup}]
    installed: list[Path] = []
    try:
        for path in previous:
            path.replace(backup / path.name)
        for path in list(stage.iterdir()):
            destination = output_root / path.name
            path.replace(destination)
            installed.append(destination)
        stage.rmdir()
    except Exception:
        for path in reversed(installed):
            if path.exists():
                path.replace(stage / path.name)
        for path in list(backup.iterdir()):
            path.replace(output_root / path.name)
        backup.rmdir()
        raise
    shutil.rmtree(backup)


def _receipt_matches(output_root: Path, repo: str, resolved: str) -> bool:
    path = output_root / "HF_MATERIALIZATION.json"
    if not path.is_file():
        return False
    receipt = json.loads(path.read_text(encoding="utf-8"))
    return (
        receipt.get("schema") == "target-based-drug-screening-hf/v1"
        and receipt.get("repo_id") == repo
        and receipt.get("resolved_commit") == resolved
        and (output_root / ".complete.json").is_file()
    )


def materialize_huggingface(output_root: Path) -> dict:
    repo, revision = huggingface_source()
    try:
        from huggingface_hub import HfApi
        resolved = HfApi().repo_info(
            repo_id=repo, repo_type="dataset", revision=revision
        ).sha
    except Exception as error:
        raise HubUnavailable(f"could not access {repo}@{revision}: {error}") from error
    if not resolved or not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ValueError("HF did not resolve a complete commit SHA")
    if _receipt_matches(output_root, repo, resolved):
        validate_dataset(output_root)
        print(f"reused HF release {repo}@{resolved} at {output_root}")
        return json.loads((output_root / "HF_MATERIALIZATION.json").read_text())

    output_root.mkdir(parents=True, exist_ok=True)
    stage = output_root / f".hf-release-{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    try:
        downloaded_commit = _download_release(stage, repo, resolved)
        if downloaded_commit != resolved:
            raise ValueError("HF release commit changed during download")
        _prepare_layout(stage)
        validate_dataset(stage)
        receipt = {
            "schema": "target-based-drug-screening-hf/v1",
            "repo_id": repo,
            "requested_revision": revision,
            "resolved_commit": resolved,
        }
        (stage / "HF_MATERIALIZATION.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _publish(stage, output_root)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f"materialized HF release {repo}@{resolved} at {output_root}")
    return receipt
