#!/usr/bin/env python3
"""Add contemporaneous ERA5 100m winds to the official HRES execution fixture.

This mixed-source fixture is only for compute validation, never a skill score.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import xarray as xr

SOURCE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
FIELDS = ["100m_u_component_of_wind", "100m_v_component_of_wind"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path)
    args = parser.parse_args()
    metadata = urllib.request.urlopen(SOURCE.replace("gs://", "https://storage.googleapis.com/") + "/.zmetadata", timeout=60).read()
    (args.assets / "winds-source.zmetadata.json").write_bytes(metadata)
    source = xr.open_zarr(SOURCE, consolidated=True, chunks=None, storage_options={"token": "anon"})
    with xr.open_dataset(args.assets / "sample.nc") as sample:
        stamps = sample.datetime.values[0]
        frames = []
        for stamp in stamps:
            print(f"Reading ERA5 100m winds at {stamp}", flush=True)
            frame = source[FIELDS].sel(time=stamp).load().rename({"latitude": "lat", "longitude": "lon"}).sortby("lat")
            np.testing.assert_array_equal(frame.lat.values, sample.lat.values)
            np.testing.assert_array_equal(frame.lon.values, sample.lon.values)
            frames.append(frame)
        winds = xr.concat(frames, dim="time").assign_coords(time=sample.time.values).expand_dims(batch=[0])
        winds = winds.assign_coords(datetime=(("batch", "time"), sample.datetime.values))
        for name in FIELDS:
            if not np.isfinite(winds[name].values).all():
                raise ValueError(f"Nonfinite source data: {name}")
        winds = winds.astype(np.float32)
        winds.to_netcdf(args.assets / "winds.nc")
    info = {"source": SOURCE, "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
            "timestamps": [str(t) for t in stamps], "variables": FIELDS,
            "sha256": hashlib.sha256((args.assets / "winds.nc").read_bytes()).hexdigest(),
            "purpose": "Compute-only mixed HRES/ERA5 fixture; no scientific skill claim"}
    (args.assets / "winds.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
