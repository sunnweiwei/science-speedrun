"""Prepare one fixed external tensor after successful raster verification."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from pyproj import Transformer
from external_lfmc_input import read_external


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    completion = json.loads((args.dataset / "completion.json").read_text())
    verification = json.loads(args.verification.read_text())
    if (
        not verification["structural_checks_passed"]
        or verification["target_used"]
        or verification["model_used"]
    ):
        raise ValueError("Missing clean raster verification")
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if manifest_hash != completion["manifest_sha256"]:
        raise ValueError("Changed candidate manifest")
    site = json.loads(args.manifest.read_text())["sites"][0]
    if site["site_id"] != completion["site_id"]:
        raise ValueError("Wrong fixed site")
    paths = list((args.dataset / "windows").glob("*/*/metadata.json"))
    if len(paths) != 1:
        raise ValueError("Wrong window count")
    window = paths[0].parent
    meta = json.loads(paths[0].read_text())
    x, y = Transformer.from_crs(
        "EPSG:4326", meta["projection"]["crs"], always_xy=True
    ).transform(site["longitude"], site["latitude"])
    px = math.floor(x / meta["projection"]["x_resolution"])
    py = math.floor(y / meta["projection"]["y_resolution"])
    bounds = (px - 16, py - 16, px + 16, py + 16)
    b = meta["bounds"]
    if not (
        b[0] <= bounds[0] < bounds[2] <= b[2] and b[1] <= bounds[1] < bounds[3] <= b[3]
    ):
        raise ValueError("Point-centered crop outside acquired window")
    periods = json.loads((args.dataset / "periods.json").read_text())
    # Bind verification to current raster contents, not only a success flag.
    for row in verification["periods"]:
        layer = (
            row["modality"]
            if row["index"] == 0
            else f"{row['modality']}.{row['index']}"
        )
        rasters = list((window / "layers" / layer).rglob("*.tif"))
        if (
            len(rasters) != 1
            or hashlib.sha256(rasters[0].read_bytes()).hexdigest() != row["file_sha256"]
        ):
            raise ValueError("Raster changed after verification")
    inputs = read_external(window, bounds, periods)
    arrays = {
        **{m: v for m, v in inputs["pixels"].items()},
        **{m + "_valid": v for m, v in inputs["valid"].items()},
    }
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Nonfinite normalized input")
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "input.npz", **arrays)
    report = {
        "site_id": site["site_id"],
        "date": completion["date"],
        "crop_bounds": bounds,
        "crop_rule": "32x32centered on raster pixel containing declared geographic point, index16",
        "target_used": False,
        "model_used": False,
        "manifest_sha256": manifest_hash,
        "verification_sha256": hashlib.sha256(
            args.verification.read_bytes()
        ).hexdigest(),
        "input_sha256": hashlib.sha256(
            (args.output / "input.npz").read_bytes()
        ).hexdigest(),
        "arrays": {
            k: {"shape": list(v.shape), "dtype": str(v.dtype)}
            for k, v in arrays.items()
        },
        "scope": "One unscored external input. Full cohort, spatial exclusion audit and locked evaluation remain outstanding.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
