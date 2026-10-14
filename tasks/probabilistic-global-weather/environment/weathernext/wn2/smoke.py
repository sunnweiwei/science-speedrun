#!/usr/bin/env python3
"""Run the evaluator CLI on the real public WN2 sample at native resolution."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import xarray as xr

from evaluate import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    expected = next(x["sha256"] for x in json.loads((args.assets / "manifest.json").read_text())["assets"] if x["file"] == "sample.nc")
    if sha256(args.assets / "sample.nc") != expected: raise ValueError("Wrong sample hash")
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    protocol.update(status="public_execution_fixture_only", evaluation_year=2024,
                    evaluation_initializations=["2024-10-07T00:00:00"], lead_hours=[6])
    protocol["variables"] = {"2m_temperature": protocol["variables"]["2m_temperature"]}
    protocol_path = args.output / "smoke-protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2)+"\n")
    with xr.open_dataset(args.assets / "sample.nc") as ds:
        frames = ds["2m_temperature"].isel(batch=0).load()
        init = np.datetime64("2024-10-07T00:00:00", "ns")
        assert ds.datetime.values[0, 1] == init
        assert ds.datetime.values[0, 2] == init + np.timedelta64(6, "h")
        current, future = [xr.DataArray(frames.isel(time=i).values, dims=("lat", "lon"),
                                       coords={"lat": ds.lat, "lon": ds.lon}, attrs={"units": "K"}) for i in (1,2)]
    def package(field):
        data = field.expand_dims(case=[init], lead=[np.timedelta64(6,"h")]).to_dataset(name="2m_temperature")
        return data.assign_coords(level=protocol["pressure_levels_hpa"], valid_time=(("case","lead"), [[init+np.timedelta64(6,"h")]]))
    truth = package(future)
    truth.attrs["truth_source"] = "HRES-fc0"
    prediction = package(current).expand_dims(member=np.arange(8))
    truth.to_netcdf(args.output / "truth.nc")
    prediction.to_netcdf(args.output / "prediction.nc")
    result_path = args.output / "scores.json"
    subprocess.run([sys.executable, str(Path(__file__).with_name("evaluate.py")),
                    "--protocol", str(protocol_path), "--prediction", str(args.output / "prediction.nc"),
                    "--truth", str(args.output / "truth.nc"), "--output", str(result_path)], check=True)
    report = json.loads(result_path.read_text())
    assert len(report["aggregate"]) == 3
    # Identical persistence members reduce empirical CRPS to absolute error.
    assert all(row["spread"] < 1e-10 and row["crps"] > 0 for row in report["aggregate"])
    report.update(scientific_skill_evaluation=False, source_sample_sha256=expected,
                  smoke="Native 721x1440 grid, eight identical persistence members, 2m temperature at +6h, real public 2024 sample")
    result_path.write_text(json.dumps(report, indent=2)+"\n")
    print("WN2_NATIVE_GRID_REAL_DATA_EVALUATOR_OK")


if __name__ == "__main__": main()
