"""Acquire one fixed manifest row; failures remain visible and are never filtered."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from pyproj import Transformer
from rasterio.transform import from_origin


def task_inputs(manifest, task_index):
    rows = [
        (site, observation)
        for site in manifest["sites"]
        for observation in site["observations"]
    ]
    if not 0 <= task_index < len(rows):
        raise ValueError("Task index outside fixed manifest")
    site, observation = rows[task_index]
    lon, lat = site["longitude"], site["latitude"]
    zone = min(60, int((lon + 180) // 6) + 1)
    crs = f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    transform = from_origin(
        (math.floor(x / 10) - 31) * 10, (math.floor(y / 10) + 32) * 10, 10, 10
    )
    child = {"sites": [{**site, "observations": [observation]}]}
    grid = {
        "site_id": site["site_id"],
        "date": observation["date"],
        "crs": crs,
        "transform": list(transform),
    }
    return child, grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--official-config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args()
    index = (
        args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    )
    manifest = json.loads(args.manifest.read_text())
    child, grid = task_inputs(manifest, index)
    out = args.root / f"{index:04d}"
    out.mkdir(parents=True, exist_ok=False)
    parent_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    child["parent_manifest_sha256"] = parent_hash
    child["parent_task_index"] = index
    child_path = out / "manifest.json"
    child_path.write_text(json.dumps(child, indent=2) + "\n")
    (out / "grid.json").write_text(json.dumps(grid, indent=2) + "\n")
    source = Path(__file__).parent
    steps = [
        [
            "materialize_external_lfmc.py",
            "--manifest",
            str(child_path),
            "--probe",
            str(out / "grid.json"),
            "--official-config",
            str(args.official_config),
            "--output",
            str(out / "dataset"),
        ],
        [
            "verify_external_lfmc_mosaic.py",
            "--dataset",
            str(out / "dataset"),
            "--output",
            str(out / "verification.json"),
        ],
        [
            "prepare_external_lfmc_input.py",
            "--dataset",
            str(out / "dataset"),
            "--verification",
            str(out / "verification.json"),
            "--manifest",
            str(child_path),
            "--output",
            str(out / "input"),
        ],
    ]
    receipt = {
        "task_index": index,
        "site_id": grid["site_id"],
        "date": grid["date"],
        "parent_manifest_sha256": parent_hash,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "target_used": False,
        "model_used": False,
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in [
                Path(__file__).name,
                "materialize_external_lfmc.py",
                "verify_external_lfmc_mosaic.py",
                "prepare_external_lfmc_input.py",
                "external_lfmc_input.py",
                "lfmc_window.py",
                "external_lfmc_availability.py",
            ]
        },
    }
    (out / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for step in steps:
        result = subprocess.run([sys.executable, str(source / step[0]), *step[1:]])
        if result.returncode:
            (out / "failure.json").write_text(
                json.dumps({"step": step[0], "exit_code": result.returncode}, indent=2)
                + "\n"
            )
            raise SystemExit(result.returncode)
    receipt["input_sha256"] = hashlib.sha256(
        (out / "input/input.npz").read_bytes()
    ).hexdigest()
    (out / "completion.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
