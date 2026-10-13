#!/usr/bin/env python3
"""Record file sizes and SHA-256 of the built release into DATA_ROOT/manifest.json (and the dev copy)."""
import argparse
import hashlib
import json
from pathlib import Path

from materialize.config import agent_manifest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    a = ap.parse_args()
    manifest = json.loads((a.root / "manifest.json").read_text())
    files = {}
    total = 0
    for p in sorted(a.root.rglob("*")):
        # stage logs (e.g. se_embeddings.log) are build records, not release content
        if p.is_file() and p.name != "manifest.json" and p.suffix != ".log":
            rel = str(p.relative_to(a.root))
            size = p.stat().st_size
            total += size
            files[rel] = {"size": size, "sha256": sha256(p) if size < 64 * 1024 ** 3 else None}
    manifest["release_files"] = files
    manifest["release_total_bytes"] = total
    (a.root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    dev_manifest = agent_manifest({k: v for k, v in manifest.items() if k != "release_files"})
    dev_manifest["release_files"] = {k: v for k, v in files.items() if k.startswith("dev/")}
    (a.root / "dev" / "manifest.json").write_text(json.dumps(dev_manifest, indent=2, sort_keys=True) + "\n")
    print(f"release: {len(files)} files, {total / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
