#!/usr/bin/env python3
"""Controller-only: stage disjoint allowlisted development and private views."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "evaluation"))
from input_view import discover_targets

DEVELOPMENT_FILES = (
    "processed/profsa_pretrain_500k/data.lmdb",
    "processed/biolip_train/data.lmdb", "processed/biolip_dev/data.lmdb",
    "processed/manifest.json", "splits/train.txt", "splits/dev.txt",
    "splits/profsa_pretrain_500k.txt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"missing or unsafe source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def write_view_receipt(root: Path, provenance: dict, view: str) -> None:
    """Fingerprint the bytes actually staged, independently of mutable source paths."""
    files = [
        {"path": path.relative_to(root).as_posix(), "size": path.stat().st_size,
         "sha256": sha256(path)}
        for path in sorted(root.rglob("*")) if path.is_file()
    ]
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    receipt = dict(provenance, view=view, files=files,
                   file_manifest_sha256=hashlib.sha256(encoded).hexdigest())
    (root / ".complete.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def stage(data: Path, pretrained: Path, development: Path, private: Path) -> None:
    roots = [path.resolve() for path in (data, pretrained, development, private)]
    data, pretrained, development, private = roots
    for destination in (development, private):
        for other in (data, pretrained):
            if destination.is_relative_to(other) or other.is_relative_to(destination):
                raise ValueError("source and destination directories must be disjoint")
    if (development.is_relative_to(private) or private.is_relative_to(development)
            or development.exists() or private.exists()):
        raise ValueError("use new, separate development and private destinations")
    if not (data / ".complete.json").is_file():
        raise ValueError("materialized data have no completion receipt")
    validation = json.loads((data / "VALIDATION.json").read_text())
    if validation.get("status") != "PASS" or validation.get("smoke", True):
        raise ValueError("materialized data validation did not pass")
    targets = discover_targets(data, full_suite=True)
    manifest = json.loads((pretrained / "manifest.json").read_text())
    expected = json.loads((HERE.parent / "baseline/drugclip/pretrained_manifest.example.json").read_text())
    if manifest != expected:
        raise ValueError("pretrained manifest differs from the pinned generic Uni-Mol assets")
    for key in ("molecule", "molecule_dictionary", "pocket_dictionary"):
        asset = manifest[key]
        path = pretrained / asset["path"]
        if path.is_symlink() or path.stat().st_size != asset["size"] or sha256(path) != asset["sha256"]:
            raise ValueError(f"pretrained asset integrity failure: {key}")
    provenance = {"schema": "target-based-drug-screening-task-data/v1"}
    source_receipt = data / "HF_MATERIALIZATION.json"
    if source_receipt.is_file():
        hub = json.loads(source_receipt.read_text())
        if not re.fullmatch(r"[0-9a-f]{40}", hub["resolved_commit"]):
            raise ValueError("HF materialization has no immutable resolved commit")
        provenance.update(repo_id=hub["repo_id"], resolved_commit=hub["resolved_commit"])
    stages = []
    try:
        for destination in (development, private):
            destination.parent.mkdir(parents=True, exist_ok=True)
            stages.append(Path(tempfile.mkdtemp(prefix=".task-data-", dir=destination.parent)))
        dev_stage, private_stage = stages
        for name in DEVELOPMENT_FILES:
            copy_file(data / name, dev_stage / name)
        for name in ("manifest.json", "mol_pre_no_h_220816.pt", "dict_mol.txt", "dict_pkt.txt"):
            copy_file(pretrained / name, dev_stage / "pretrained" / name)
        for (benchmark, target), (molecules, pocket) in targets.items():
            root = private_stage / "test" / benchmark / target
            copy_file(molecules, root / "mols.lmdb")
            copy_file(pocket, root / ("pocket.lmdb" if benchmark == "dude" else "pockets.lmdb"))
        for view, destination, temporary in zip(
                ("development", "private"), (development, private), stages, strict=True):
            write_view_receipt(temporary, provenance, view)
            os.replace(temporary, destination)
    finally:
        for temporary in stages:
            if temporary.exists():
                shutil.rmtree(temporary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained-root", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    args = parser.parse_args()
    stage(args.data_root, args.pretrained_root, args.development_root, args.private_root)
