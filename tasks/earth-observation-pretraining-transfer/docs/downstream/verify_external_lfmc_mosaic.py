"""Verify actual external rasters and temporal coverage without reading targets."""

import argparse
import hashlib
import json
from pathlib import Path
from datetime import timedelta

import numpy as np
import rasterio
from external_lfmc_availability import BANDS, parse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    completion = json.loads((args.dataset / "completion.json").read_text())
    periods = json.loads((args.dataset / "periods.json").read_text())
    if completion["target_used"] or completion["model_used"] or len(periods) != 24:
        raise ValueError("Wrong acquisition scope")
    expected = {(m, i) for m in BANDS for i in range(12)}
    if {(p["modality"], p["index"]) for p in periods} != expected:
        raise ValueError("Missing or duplicate periods")
    windows = list((args.dataset / "windows").glob("*/*/metadata.json"))
    if len(windows) != 1:
        raise ValueError("Expected one fixed window")
    root = windows[0].parent
    meta = json.loads(windows[0].read_text())
    end = parse(completion["date"])
    reports = []
    for p in periods:
        stop = end - timedelta(days=14 * p["index"])
        start = stop - timedelta(days=14)
        if parse(p["start"]) != start or parse(p["end"]) != stop:
            raise ValueError("Temporal direction/period mismatch")
        files = list((root / "layers" / p["layer_name"]).rglob("*.tif"))
        if len(files) != 1:
            raise ValueError("Missing/extra period raster")
        path = files[0]
        with rasterio.open(path) as src:
            arr = src.read()
            if arr.shape != (len(BANDS[p["modality"]]), 63, 63):
                raise ValueError("Wrong raster dimensions/bands")
            if str(src.crs) != meta["projection"]["crs"]:
                raise ValueError("CRS mismatch")
            b = meta["bounds"]
            if not np.allclose(
                tuple(src.transform)[:6], (10, 0, b[0] * 10, 0, -10, b[1] * -10)
            ):
                raise ValueError("Pixel grid mismatch")
            if not np.isfinite(arr).all():
                raise ValueError("Nonfinite pixels")
            if p["missing_catalog_group"] and np.any(arr):
                raise ValueError("Missing period unexpectedly contains pixels")
            reports.append(
                {
                    "modality": p["modality"],
                    "index": p["index"],
                    "dtype": str(arr.dtype),
                    "nodata_metadata": src.nodata,
                    "nonzero_pixels_per_band": np.count_nonzero(
                        arr, axis=(1, 2)
                    ).tolist(),
                    "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    result = {
        "structural_checks_passed": True,
        "periods": reports,
        "target_used": False,
        "model_used": False,
        "limitations": "One site/date. Nonzero counts describe raw raster coverage, not cloud-free observations or model input validity. Final crop/normalization and full cohort remain unverified.",
    }
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print("Verified24period rasters; no target or model use", flush=True)


if __name__ == "__main__":
    main()
