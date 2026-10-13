#!/usr/bin/env python3
"""Build and verify manifests for already-materialized, label-separated releases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "massspecgym-materialized-release/v1"
PUBLIC_FILES = {
    "development/manifest.json", "development/train.jsonl", "development/validation.jsonl",
    "development/validation_candidate_pools.jsonl", "verifier/manifest.json",
    "verifier/queries.jsonl", "verifier/candidate_pools.jsonl",
    "development/glacier/manifest.json", "development/glacier/preparation.json",
    "development/glacier/targets.sqlite", "development/glacier/train.jsonl",
    "development/glacier/validation.jsonl", "development/synthetic_augmentation/manifest.json",
    "development/synthetic_augmentation/labels.tsv", "development/synthetic_augmentation/subformulae.sqlite",
    "development/synthetic_augmentation/sanitation_report.json",
    "development/synthetic_augmentation/training_data_audit.json",
}
PRIVATE_FILES = {"manifest.json", "labels.jsonl"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    part = PurePosixPath(relative)
    if not relative or part.is_absolute() or ".." in part.parts or str(part) != relative or "\\" in relative:
        raise ValueError(f"unsafe release path: {relative!r}")
    path = root.joinpath(*part.parts)
    for candidate in (path, *path.parents):
        if candidate == root.parent:
            break
        if candidate.is_symlink():
            raise ValueError(f"release paths cannot contain symlinks: {relative}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"release path escapes root: {relative}")
    return path


def validate_manifest(manifest: dict[str, Any], component: str) -> None:
    expected = PUBLIC_FILES if component == "public" else PRIVATE_FILES
    if manifest.get("schema") != SCHEMA or manifest.get("component") != component:
        raise ValueError("wrong release manifest schema/component")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != expected:
        raise ValueError("release manifest must contain the complete, exact component file set")
    total = 0
    for name, entry in files.items():
        safe_path(Path("/tmp/release-path-validation"), name)
        if not isinstance(entry, dict) or set(entry) != {"bytes", "sha256"}:
            raise ValueError("invalid release entry")
        size, digest = entry["bytes"], entry["sha256"]
        if type(size) is not int or not 0 < size <= 50 * 1024**3:
            raise ValueError("release file size outside allowed bounds")
        if not isinstance(digest, str) or len(digest) != 64 or any(x not in "0123456789abcdef" for x in digest):
            raise ValueError("invalid release SHA-256")
        total += size
    if manifest.get("total_bytes") != total or total > 60 * 1024**3:
        raise ValueError("invalid release total")


def verify(root: Path, manifest: dict[str, Any], component: str) -> dict[str, Any]:
    validate_manifest(manifest, component)
    for name, entry in manifest["files"].items():
        path = safe_path(root, name)
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"release file missing, modified or corrupt: {name}")
    # Existing task manifests must agree with the outer release, not merely exist.
    manifest_names = [name for name in manifest["files"] if name.endswith("manifest.json")]
    for name in manifest_names:
        inner = json.loads(safe_path(root, name).read_text())
        for relative, entry in inner["files"].items():
            path = str(PurePosixPath(name).parent / relative)
            expected = entry["sha256"] if isinstance(entry, dict) else entry
            if path not in manifest["files"] or manifest["files"][path]["sha256"] != expected:
                raise ValueError(f"nested manifest mismatch: {name}: {relative}")
    if component == "public":
        core = json.loads((root / "development/manifest.json").read_text())
        counts = core["counts"]
        if [counts[k] for k in ("train_queries", "validation_queries", "test_queries")] != [194119, 19429, 17556]:
            raise ValueError("incorrect experimental split counts")
        augmentation = json.loads((root / "development/synthetic_augmentation/manifest.json").read_text())
        if augmentation.get("rows") != 2333485:
            raise ValueError("incorrect clean augmentation count")
        glacier = json.loads((root / "development/glacier/manifest.json").read_text())
        if [glacier["counts"][k]["annotated"] for k in ("train", "validation")] != [99119, 9720]:
            raise ValueError("incorrect prepared MAGMa counts")
        if glacier["config"] != json.loads((root / "development/glacier/preparation.json").read_text()):
            raise ValueError("MAGMa preparation metadata mismatch")
    elif json.loads((root / "manifest.json").read_text()).get("queries") != 17556:
        raise ValueError("incorrect grader-label count")
    return {"component": component, "files": len(manifest["files"]), "bytes": manifest["total_bytes"], "verified": True}


def build(root: Path, component: str) -> dict[str, Any]:
    expected = PUBLIC_FILES if component == "public" else PRIVATE_FILES
    files = {}
    for name in sorted(expected):
        path = safe_path(root, name)
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    result = {"schema": SCHEMA, "component": component,
              "source_revision": "d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a",
              "license": "mit", "total_bytes": sum(entry["bytes"] for entry in files.values()), "files": files}
    verify(root, result, component)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--component", choices=("public", "private"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "build":
        manifest = build(args.root, args.component)
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"manifest": str(args.manifest), "sha256": sha256(args.manifest), "bytes": manifest["total_bytes"]}))
    else:
        print(json.dumps(verify(args.root, json.loads(args.manifest.read_text()), args.component)))


if __name__ == "__main__":
    main()
