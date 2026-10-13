#!/usr/bin/env python3
"""Check a downloaded release against the sizes/SHA-256 recorded in its manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--dev-only", action="store_true")
    ap.add_argument("--hash", action="store_true", help="also verify SHA-256 (slow)")
    a = ap.parse_args()
    manifest = json.loads((a.root / "manifest.json").read_text())
    bad = 0
    for rel, info in manifest["release_files"].items():
        if a.dev_only and rel.startswith("private/"):
            continue
        if rel.startswith("dev/se600m/"):
            continue  # fetched separately from arcinstitute/SE-600M
        p = a.root / rel
        if not p.exists() or p.stat().st_size != info["size"]:
            print(f"BAD size/missing: {rel}"); bad += 1; continue
        if a.hash and info.get("sha256") and sha256(p) != info["sha256"]:
            print(f"BAD sha256: {rel}"); bad += 1
    print(f"verified {len(manifest['release_files'])} entries, {bad} problems")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
