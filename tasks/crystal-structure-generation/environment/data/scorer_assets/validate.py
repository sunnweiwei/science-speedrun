#!/usr/bin/env python3
"""Validate the prepared scorer asset boundary."""

import hashlib
import json
from pathlib import Path
import sys


EXPECTED = {
    "lemat_bulk": "0dc17eea904b860ad7288141e9870f67f8e6bb2c",
    "mlip_hull": "70d505bb294c16658a9551e2b81ace69a28d9790",
    "uma": "f611b917d9c68566bbbeccbb0aa0f7cad1696cb2",
}


def repository_cache(root: Path, name: str) -> Path:
    kind = "models" if name == "uma" else "datasets"
    repo_id = "facebook--UMA" if name == "uma" else (
        "LeMaterial--LeMat-Bulk" if name == "lemat_bulk"
        else "LeMaterial--LeMat-Bulk-MLIP-Hull"
    )
    return root / "huggingface" / "hub" / f"{kind}--{repo_id}"


def tree_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    count = 0
    size = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "READY.json"):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        item_digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                item_digest.update(block)
        digest.update(item_digest.hexdigest().encode() + b"\n")
        count += 1
        size += path.stat().st_size
    return digest.hexdigest(), count, size


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate.py ASSET_DIR")
    root = Path(sys.argv[1]).resolve()
    document = json.loads((root / "READY.json").read_text())
    if document.get("schema") != "crystalgen-scorer-assets/v0":
        raise ValueError("invalid scorer asset schema")
    if not document.get("hull_license_review_acknowledged"):
        raise ValueError("hull license review was not acknowledged")
    if not document.get("mlips_warmed"):
        raise ValueError("ORB, MACE and UMA caches were not warmed")
    home_cache = root / "home" / ".cache"
    fairchem_cache = home_cache / "fairchem"
    if (
        not fairchem_cache.is_symlink()
        or fairchem_cache.resolve() != (root / "huggingface" / "hub").resolve()
    ):
        raise ValueError("FairChem cache is not bound to the sealed hub cache")
    if not (home_cache / "mace" / "macempa0mediummodel").is_file():
        raise ValueError("warmed MACE checkpoint is missing")
    orb_cache = home_cache / "cached_path"
    if not any(path.is_file() and path.stat().st_size > 50_000_000 for path in orb_cache.glob("*")):
        raise ValueError("warmed ORB checkpoint is missing")
    for name, revision in EXPECTED.items():
        if document.get("assets", {}).get(name, {}).get("revision") != revision:
            raise ValueError(f"wrong {name} revision")
        if (repository_cache(root, name) / "refs/main").read_text().strip() != revision:
            raise ValueError(f"{name} default branch is not pinned to the audited revision")
    digest, count, size = tree_digest(root)
    if (document.get("tree_sha256"), document.get("file_count"), document.get("byte_count")) != (digest, count, size):
        raise ValueError("scorer asset tree changed after preparation")
    print(json.dumps({"schema": document["schema"], "files": count, "bytes": size}, sort_keys=True))


if __name__ == "__main__":
    main()
