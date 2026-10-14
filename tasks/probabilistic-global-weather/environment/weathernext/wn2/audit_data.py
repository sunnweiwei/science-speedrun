#!/usr/bin/env python3
"""Audit public archive coordinates/schema without downloading weather fields."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import xarray as xr

SOURCES = {
    "hres_fc0": "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr",
    "era5": "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
    "arco_era5": "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    report = dict(protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                  status="started", fields_materialized=False, sources={})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        info = dict(url=url)
        report["sources"][name] = info
        try:
            metadata = urllib.request.urlopen(url.replace("gs://", "https://storage.googleapis.com/") + "/.zmetadata", timeout=60).read()
            info["metadata_sha256"] = hashlib.sha256(metadata).hexdigest()
            with xr.open_zarr(url, consolidated=True, chunks=None, storage_options={"token": "anon"}) as ds:
                times = np.asarray(ds.time.values).astype("datetime64[ns]")
                info.update(first_time=str(times[0]), last_time=str(times[-1]), sizes=dict(ds.sizes),
                            variables={k:dict(dims=list(v.dims), units=v.attrs.get("units")) for k,v in ds.data_vars.items()})
                if name == "hres_fc0":
                    required = np.asarray(protocol["evaluation_initializations"], dtype="datetime64[ns]")[:,None] + np.asarray([-6,0]+protocol["lead_hours"], dtype="timedelta64[h]")[None,:]
                    missing = required[~np.isin(required, times)]
                    info["missing_required_timestamps"] = [str(v) for v in missing]
                    info["missing_scored_variables"] = sorted(set(protocol["variables"])-set(ds.data_vars))
                    info["missing_pressure_levels"] = sorted(set(protocol["pressure_levels_hpa"])-set(ds.level.values.tolist()))
                    info["evaluation_coordinates_available"] = not (len(missing) or info["missing_scored_variables"] or info["missing_pressure_levels"])
            info["status"] = "passed"
        except Exception as exc:
            info.update(status="failed", error=repr(exc))
        args.output.write_text(json.dumps(report,indent=2)+"\n")
    passed = all(v["status"]=="passed" for v in report["sources"].values()) and report["sources"]["hres_fc0"].get("evaluation_coordinates_available",False)
    report.update(status="passed" if passed else "failed",
                  scope="Public coordinate/schema availability only; no frozen data release or training statistics",
                  remaining_gates=["Materialize and hash complete training/development/evaluation windows",
                                   "Reproduce all released cyclone training targets from causally eligible observations",
                                   "Compute per-stage normalization statistics only from training data",
                                   "Measure full rollout and freeze final case count under the evaluation budget"])
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    if not passed: raise SystemExit(1)


if __name__ == "__main__": main()
