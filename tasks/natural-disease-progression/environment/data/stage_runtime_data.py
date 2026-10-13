"""Stage disjoint development and private verifier trees from a materialization.

The development tree is the only tree that may be mounted into an agent.  The
private tree must remain controller/verifier-only.  MIMIC-derived artifacts are
credentialed and must not be uploaded to a public release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


DEVELOPMENT = (
    "manifest.json",
    "scored_diseases.parquet",
    "train",
    "public/validation",
    "trusted/validation",
    "trusted/risk_normalization.parquet",
)
PRIVATE = (
    "public/test",
    "trusted/test",
    "trusted/risk_normalization.parquet",
)
FORBIDDEN_DEVELOPMENT_NAMES = {
    "working.duckdb", "subject_split.parquet", "first_disease.parquet",
    "controller_patient_splits.parquet",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    # A runtime/release tree must be an immutable snapshot. Hard links would
    # allow later in-place edits of the controller materialization to mutate a
    # previously inventoried package.
    # Content-addressed inventory supplies integrity/provenance. Avoid copy2's
    # timestamp update, which is unsupported on common WSL/NTFS bind mounts.
    shutil.copyfile(source, target)


def copy_entry(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_symlink():
        raise ValueError(f"Runtime data source may not be a symlink: {source}")
    if source.is_file():
        copy_file(source, target)
        return
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Runtime data source may not contain symlinks: {path}")
        relative = path.relative_to(source)
        if path.is_dir():
            (target / relative).mkdir(parents=True, exist_ok=True)
        else:
            copy_file(path, target / relative)


def write_inventory(root: Path, kind: str) -> None:
    files = []
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        if path.name == "runtime-manifest.json":
            continue
        files.append({"path": path.relative_to(root).as_posix(),
                      "bytes": path.stat().st_size, "sha256": digest(path)})
    (root / "runtime-manifest.json").write_text(json.dumps({
        "schema": "natural-disease-progression-runtime-data/v1",
        "kind": kind,
        "files": files,
    }, indent=2) + "\n")


def stage(materialized: Path, destination: Path) -> tuple[Path, Path]:
    development = destination / "development"
    private = destination / "private"
    if destination.exists():
        raise FileExistsError("Runtime destination must be new; refusing a mixed/stale release")
    development.mkdir(parents=True)
    private.mkdir(parents=True)
    for relative in DEVELOPMENT:
        copy_entry(materialized / relative, development / relative)
    for relative in PRIVATE:
        copy_entry(materialized / relative, private / relative)
    development_files = {path.relative_to(development).as_posix()
                         for path in development.rglob("*") if path.is_file()}
    if any(path.startswith("public/test/") or path.startswith("trusted/test/")
           for path in development_files):
        raise AssertionError("Development runtime contains test data")
    if any(Path(path).name in FORBIDDEN_DEVELOPMENT_NAMES for path in development_files):
        raise AssertionError("Development runtime contains controller-only data")
    write_inventory(development, "development")
    write_inventory(private, "private-test")
    return development, private


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialized", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    dev, private = stage(args.materialized, args.destination)
    print(json.dumps({"development": str(dev), "private": str(private)}, indent=2))
