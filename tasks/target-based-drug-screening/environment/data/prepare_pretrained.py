#!/usr/bin/env python3
"""Download and checksum the only permitted generic Uni-Mol initialization assets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import urllib.request

from stage_task_data import sha256


def prepare(destination: Path) -> None:
    template = Path(__file__).resolve().parents[1] / "baseline/drugclip/pretrained_manifest.example.json"
    manifest = json.loads(template.read_text())
    destination.mkdir(parents=True, exist_ok=True)
    for key in ("molecule", "molecule_dictionary", "pocket_dictionary"):
        item = manifest[key]
        path = destination / item["path"]
        if path.is_file() and path.stat().st_size == item["size"] and sha256(path) == item["sha256"]:
            continue
        url = item.get("source_url") or (
            f"https://raw.githubusercontent.com/{item['source_repository']}/"
            f"{item['source_revision']}/{item['source_path']}")
        temporary = path.with_suffix(path.suffix + ".download")
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            if temporary.stat().st_size != item["size"] or sha256(temporary) != item["sha256"]:
                raise ValueError(f"download integrity failure: {key}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    prepare(parser.parse_args().destination)
