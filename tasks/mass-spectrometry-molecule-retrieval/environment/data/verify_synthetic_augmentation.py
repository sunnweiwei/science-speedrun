#!/usr/bin/env python3
"""Fail-closed verification for an existing clean MIST augmentation view."""

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "massspecgym-retrieval-synthetic-augmentation/v1"
ROWS = 2_333_485
SOURCE = {
    "zenodo_record": "20671842",
    "file": "msg_aug_filt.tar.gz",
    "md5": "32d8813d84d06364fc318e81f9cb4416",
    "license": "MIT",
}
FILES = {
    "labels.tsv",
    "subformulae.sqlite",
    "sanitation_report.json",
    "training_data_audit.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def verify(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"augmentation root is missing or unsafe: {root}")
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unexpected augmentation manifest schema")
    if manifest.get("source") != SOURCE or manifest.get("rows") != ROWS:
        raise ValueError("unexpected augmentation provenance or row count")
    entries = manifest.get("files")
    if not isinstance(entries, dict) or set(entries) != FILES:
        raise ValueError("unexpected augmentation file set")
    for name in sorted(FILES):
        entry = entries[name]
        path = root / name
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("bytes"), int)
            or not isinstance(entry.get("sha256"), str)
            or not path.is_file()
            or path.is_symlink()
        ):
            raise ValueError(f"invalid augmentation entry: {name}")
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"size mismatch for {path}")
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path}: {actual}")

    sanitation = read_json(root / "sanitation_report.json")
    audit = read_json(root / "training_data_audit.json")
    if sanitation.get("retained_rows") != ROWS:
        raise ValueError("sanitation report row count mismatch")
    if sanitation.get("database_integrity") != "ok":
        raise ValueError("synthetic database failed its recorded integrity check")
    if audit.get("synthetic_spectra") != ROWS:
        raise ValueError("training audit row count mismatch")
    if audit.get("synthetic_overlap_with_held_out") != {"val": 0, "test": 0}:
        raise ValueError("training audit reports held-out overlap")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    verify(args.root.resolve())
    print(f"verified {ROWS} clean synthetic rows and {len(FILES)} file digests")


if __name__ == "__main__":
    main()
