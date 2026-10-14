#!/usr/bin/env python3
"""Fetch Google's full WN2 pilot assets, verifying GCS size and MD5.

The 2024 sample is for execution checks, not held-out WN2 skill evaluation.
No existing track assets or private evaluation data are used.
"""
import argparse
import base64
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

OBJECTS = {
    "wn2.npz": "weathernext2/params/WeatherNext2_<2025_model1.npz",
    "cyclones.npz": "weathernext2/params/WeatherNextCyclones_<2024_model1.npz",
    "sample.nc": "weathernext2/dataset/source-hres_forecast_init-2024-10-07 00:00:00_res-0.25_levels-13_steps-01.nc",
}


def fetch(root, filename, name):
    metadata_url = "https://storage.googleapis.com/storage/v1/b/dm_graphcast/o/" + urllib.parse.quote(name, safe="")
    with urllib.request.urlopen(metadata_url, timeout=60) as response:
        meta = json.load(response)
    url = "https://storage.googleapis.com/dm_graphcast/" + urllib.parse.quote(name, safe="/")
    url += "?generation=" + meta["generation"]
    dest = root / filename
    if not dest.exists():
        part = dest.with_suffix(dest.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as out:
            while chunk := response.read(8 * 1024 * 1024):
                out.write(chunk)
        part.replace(dest)
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with dest.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            md5.update(chunk)
            sha256.update(chunk)
    if dest.stat().st_size != int(meta["size"]) or base64.b64encode(md5.digest()).decode() != meta["md5Hash"]:
        raise ValueError(f"Asset checksum/size mismatch: {dest}")
    result = {"file": filename, "object": name, "url": url, "generation": meta["generation"],
              "bytes": dest.stat().st_size, "md5": meta["md5Hash"], "sha256": sha256.hexdigest()}
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch, args.output, filename, name) for filename, name in OBJECTS.items()]
        assets = [future.result() for future in futures]
    manifest = {"assets": assets, "source_commit": "89c4b2a77a1c57b328b909c575550fd2e5aadc9c",
                "purpose": "Native 0.25-degree execution and adaptation-budget pilot",
                "sample_init": "2024-10-07T00:00:00", "lead_hours": 6,
                "wn2_sample_is_held_out": False,
                "cyclones_sample_after_pretraining_cutoff": True,
                "scientific_skill_evaluation": False}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
