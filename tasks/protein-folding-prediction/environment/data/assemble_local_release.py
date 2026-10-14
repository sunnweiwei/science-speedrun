#!/usr/bin/env python3
"""Assemble a clean Agent-visible release tree from organizer-side local data."""

import errno
import json
import os
from pathlib import Path
import shutil
import sys


FEATURE_VERSION = "esmc6b-final-ln-int8-g64-45b0fa5d-ptsdpa-v1"


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES}:
            raise
        shutil.copy2(source, destination)


def copy_tree(source, destination):
    count = 0
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            link_or_copy(path, target)
            count += 1
        else:
            raise ValueError(f"Unsupported source entry: {path}")
    return count


def read_jsonl(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: assemble_local_release.py ORGANIZER_ROOT RELEASE_ROOT")
    source = Path(sys.argv[1]).resolve()
    destination = Path(sys.argv[2]).resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError("Release root must be separate from the organizer root")
    if destination.exists():
        raise FileExistsError(f"Release destination already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary destination already exists: {temporary}")
    temporary.mkdir(parents=True)

    direct_files = (
        "manifests/foldingtrain-100k-v1/foldingtrain_100k.jsonl",
        "manifests/foldingtrain-100k-v1/features.structures.jsonl",
        "manifests/foldingtrain-100k-v1/features.esmc6b_final_int8.jsonl",
        "foldbench/features.esmc6b_final_int8.jsonl",
    )
    for relative in direct_files:
        link_or_copy(source / relative, temporary / relative)

    copied = {}
    for relative in (
        "objects/openfold3",
        "foldbench/suite",
        "foldbench/ground_truths",
        "models/esmfold2-fast-b28d8ace",
    ):
        copied[relative] = copy_tree(source / relative, temporary / relative)

    training_manifest = source / direct_files[2]
    training_store = temporary / f"features/{FEATURE_VERSION}"
    training_features = 0
    for record in read_jsonl(training_manifest):
        digest = record["sequence_sha256"]
        link_or_copy(
            source / f"features/{FEATURE_VERSION}/{digest}.safetensors",
            training_store / f"{digest}.safetensors",
        )
        training_features += 1

    suite_manifest = source / direct_files[3]
    suite_store = temporary / f"foldbench/features/{FEATURE_VERSION}"
    suite_features = 0
    for record in read_jsonl(suite_manifest):
        digest = record["sequence_sha256"]
        link_or_copy(
            source / f"foldbench/features/{FEATURE_VERSION}/{digest}.safetensors",
            suite_store / f"{digest}.safetensors",
        )
        suite_features += 1

    temporary.replace(destination)
    print(json.dumps({
        "release_root": str(destination),
        "training_features": training_features,
        "foldbench_features": suite_features,
        "copied_files_by_tree": copied,
        "contains_organizer_data": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
