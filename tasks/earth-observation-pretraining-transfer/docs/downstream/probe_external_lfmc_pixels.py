"""Read one predetermined external site's S1/S2 assets; no labels or models."""

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import planetary_computer
from pyproj import Transformer
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
import requests

from external_lfmc_availability import BANDS, COLLECTIONS, parse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--availability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    site = manifest["sites"][0]
    date = site["observations"][0]["date"]
    end = parse(date)
    start = end - timedelta(days=14)
    availability = json.loads(
        (
            args.availability
            / (hashlib.sha256(site["site_id"].encode()).hexdigest() + ".json")
        ).read_text()
    )
    lon, lat = site["longitude"], site["latitude"]
    zone = min(60, int((lon + 180) // 6) + 1)
    crs = f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    # Probe grid only; final release-aligned crop geometry remains separately audited.
    transform = from_origin(
        (math.floor(x / 10) - 31) * 10, (math.floor(y / 10) + 32) * 10, 10, 10
    )
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "site_id": site["site_id"],
        "date": date,
        "selection": "First site/date in fixed metadata manifest; most recent14-day period",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "crs": crs,
        "transform": list(transform),
        "shape": [63, 63],
        "target_used": False,
        "model_used": False,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "scope": "Asset access and raw nearest-neighbor raster probe only. Not harmonized, not a mosaic, not final model input.",
        "modalities": {},
    }
    session = requests.Session()
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_TIMEOUT="60",
        GDAL_HTTP_MAX_RETRY="2",
    ):
        for modality, collection in COLLECTIONS.items():
            items = [
                i
                for i in availability["modalities"][modality]["items"]
                if i["assets_complete"] and start <= parse(i["datetime"]) < end
            ]
            if not items:
                raise ValueError("No catalog candidates in predetermined period")
            if modality == "sentinel2":
                items.sort(
                    key=lambda i: (
                        i["cloud_cover"]
                        if i["cloud_cover"] is not None
                        else float("inf"),
                        i["id"],
                    )
                )
            else:
                items.sort(key=lambda i: i["id"])
            item = items[0]
            response = session.get(
                f"https://planetarycomputer.microsoft.com/api/stac/v1/collections/{collection}/items/{item['id']}",
                timeout=60,
            )
            response.raise_for_status()
            stac = response.json()
            arrays, masks, bands = [], [], []
            for band in BANDS[modality]:
                url = planetary_computer.sign(stac["assets"][band]["href"])
                with rasterio.open(url) as source:
                    with WarpedVRT(
                        source,
                        crs=crs,
                        transform=transform,
                        width=63,
                        height=63,
                        resampling=Resampling.nearest,
                    ) as vrt:
                        array = vrt.read(1)
                        valid = vrt.read_masks(1) > 0
                if not np.isfinite(array).all():
                    raise ValueError("Nonfinite raw raster")
                arrays.append(array)
                masks.append(valid)
                bands.append(
                    {
                        "band": band,
                        "dtype": str(array.dtype),
                        "valid_pixels": int(valid.sum()),
                        "nonzero_pixels": int(np.count_nonzero(array)),
                        "raw_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                    }
                )
                print(json.dumps({"modality": modality, **bands[-1]}), flush=True)
            path = args.output / (modality + ".npz")
            np.savez_compressed(path, pixels=np.stack(arrays), valid=np.stack(masks))
            report["modalities"][modality] = {
                "item_id": item["id"],
                "bands": bands,
                "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    report["completed_utc"] = datetime.now().astimezone().isoformat()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
