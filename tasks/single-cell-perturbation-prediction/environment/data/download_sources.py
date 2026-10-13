#!/usr/bin/env python3
"""Fetch every pinned source in download-source.json into the cache (resumable, size-checked)."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--cache", required=True, type=Path)
    a = ap.parse_args()
    manifest = json.loads(a.manifest.read_text())
    for key, entry in manifest["files"].items():
        if key == "prepared_release":
            continue  # the prepared release is fetched by fetch_release.sh, not rebuilt here
        dest = a.cache / entry["path"]
        if entry.get("directory"):
            if (dest / "README.md").exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, "-m", "huggingface_hub.cli.hf", "download", entry["hf_repo"],
                            "--repo-type", entry["hf_repo_type"], "--revision", entry["hf_revision"], "--local-dir", str(dest)],
                           check=True)
            continue
        if dest.exists() and (entry.get("size") is None or dest.stat().st_size == entry["size"]):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if "url" in entry:
            print(f"[download] {key} -> {dest}", file=sys.stderr)
            subprocess.run(["curl", "--fail", "--location", "--retry", "10", "--retry-all-errors", "--continue-at", "-",
                            "--output", str(dest) + ".partial", entry["url"]], check=True)
            (dest.with_name(dest.name + ".partial")).rename(dest)
        elif "hf_repo" in entry:
            print(f"[download] {key} from HF {entry['hf_repo']}@{entry['hf_revision']}", file=sys.stderr)
            subprocess.run([sys.executable, "-m", "huggingface_hub.cli.hf", "download", entry["hf_repo"],
                            Path(entry["path"]).name, "--revision", entry["hf_revision"], "--local-dir", str(dest.parent)], check=True)
        else:
            raise SystemExit(f"{key}: no download method")
        if entry.get("size") is not None and dest.stat().st_size != entry["size"]:
            raise SystemExit(f"{key}: size mismatch {dest.stat().st_size} != {entry['size']}")


if __name__ == "__main__":
    main()
