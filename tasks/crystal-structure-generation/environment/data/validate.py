#!/usr/bin/env python3
"""Validate a prepared CrystalGen MP-20 directory without loading structures."""

import csv
import hashlib
import json
from pathlib import Path
import sys


EXPECTED_ROWS = {"train": 27136, "val": 9047, "test": 9046}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate.py PREPARED_DIR")
    root = Path(sys.argv[1]).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "crystalgen-mp20/v0":
        raise ValueError("invalid MP-20 manifest schema")
    if set(path.name for path in root.iterdir()) != {
        "train.csv", "val.csv", "test.csv", "manifest.json"
    }:
        raise ValueError("unexpected files in prepared MP-20 directory")
    for split, expected_rows in EXPECTED_ROWS.items():
        path = root / f"{split}.csv"
        record = manifest["prepared_splits"][split]
        if record != {"rows": expected_rows, "sha256": sha256(path)}:
            raise ValueError(f"prepared {split} manifest mismatch")
        with path.open(newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            if "density" not in header or sum(1 for _ in reader) != expected_rows:
                raise ValueError(f"prepared {split} CSV mismatch")
    print(json.dumps({"schema": manifest["schema"], "rows": EXPECTED_ROWS}, sort_keys=True))


if __name__ == "__main__":
    main()
