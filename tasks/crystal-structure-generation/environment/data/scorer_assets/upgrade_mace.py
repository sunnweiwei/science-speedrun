#!/usr/bin/env python3
"""Make a separate hull-aligned asset tree from a validated prior tree on the same filesystem."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from materialize import tree_digest

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), "/opt/crystalgen"]
from scoring.mace_assets import CHECKPOINTS, validate_mace_assets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("energy_checkpoint", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError("Choose a new destination outside the original asset tree")
    original = json.loads((source / "READY.json").read_text())
    if not original.get("hull_license_review_acknowledged") or not original.get("mlips_warmed"):
        raise ValueError("Expected an approved, warmed historical asset tree")
    if tree_digest(source) != (original["tree_sha256"], original["file_count"], original["byte_count"]):
        raise ValueError("The historical asset tree has changed")
    filename = "macemp0b3mediummodel"
    with args.energy_checkpoint.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != CHECKPOINTS[filename]:
            raise ValueError("Expected official MACE-MP-0b3-medium weights")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".crystal-assets-") as temp:
        target = Path(temp) / "assets"
        shutil.copytree(source, target, copy_function=os.link, symlinks=True)
        (target / "READY.json").unlink()
        energy = target / "home/.cache/mace" / filename
        if energy.exists():
            energy.unlink()
        shutil.copyfile(args.energy_checkpoint, energy)
        validate_mace_assets(target)
        digest, count, size = tree_digest(target)
        ready = dict(original, schema="crystalgen-scorer-assets/v1", tree_sha256=digest,
                     file_count=count, byte_count=size, mace_sha256=CHECKPOINTS,
                     upgraded_from_tree_sha256=original["tree_sha256"])
        (target / "READY.json").write_text(json.dumps(ready, indent=2, sort_keys=True) + "\n")
        target.rename(destination)


if __name__ == "__main__":
    main()
