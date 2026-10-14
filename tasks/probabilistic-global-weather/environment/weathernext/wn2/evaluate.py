#!/usr/bin/env python3
"""Stream strict WN2 forecast panels through pinned WeatherBench-X metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import xarray as xr
from weatherbenchX.metrics import probabilistic
import weatherbenchX

WBX_REVISION = "88af2fde7a78f13b77496a2abb8ca88905fa5e95"
SPATIAL = ("lat", "lon")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_metric_source():
    root = Path(weatherbenchX.__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True)
    if revision != WBX_REVISION or dirty:
        raise ValueError("Evaluator requires a clean, pinned WeatherBench-X source checkout")
    return revision


def cell_weights(lat):
    """Spherical cell areas, including half-height polar cells."""
    values = np.asarray(lat, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.diff(values) > 0):
        raise ValueError("Latitude must increase strictly")
    edges = np.concatenate(([-90.], (values[:-1] + values[1:]) / 2, [90.]))
    return xr.DataArray(np.diff(np.sin(np.deg2rad(edges))), dims="lat", coords={"lat": lat})


def pool(field, weights, factor):
    """Non-overlapping area means; retain partial polar blocks explicitly."""
    area = weights.broadcast_like(field.isel(member=0, drop=True) if "member" in field.dims else field)
    blocks = {"lat": factor, "lon": factor}
    total = area.coarsen(blocks, boundary="pad").sum(skipna=True)
    pooled = (field * area).coarsen(blocks, boundary="pad").sum(skipna=True) / total
    return pooled, total


def panel(prediction, target, factor=1, minimum_valid_area_fraction=1.0):
    """One field/level/case/lead, returning additive sufficient statistics."""
    if set(prediction.dims) != {"member", "lat", "lon"} or set(target.dims) != set(SPATIAL):
        raise ValueError("Unexpected panel dimensions")
    xr.align(prediction, target, join="exact", copy=False)
    if prediction.sizes["member"] < 2:
        raise ValueError("At least two ensemble members required")
    if not np.isfinite(prediction.values).all() or np.isinf(target.values).any():
        raise ValueError("Nonfinite prediction or infinite truth")
    prediction, target = prediction.astype("float64"), target.astype("float64")
    weights = cell_weights(target.lat).broadcast_like(target)
    valid = np.isfinite(target)
    coverage = float(weights.where(valid, 0).sum() / weights.sum())
    if coverage + 1e-12 < minimum_valid_area_fraction:
        raise ValueError("Insufficient valid truth area")
    # Only a trusted truth mask can remove a cell. Predictions are checked for
    # finiteness everywhere above, including cells without observations.
    weights = weights.where(valid, 0)
    prediction = prediction.where(valid)
    if factor != 1:
        prediction, _ = pool(prediction, weights, factor)
        target, weights = pool(target, weights, factor)
    metric = probabilistic.CRPSEnsemble(ensemble_dim="member", fair=False, use_sort=True,
                                      skipna_ensemble=False)
    stats = {key: stat.compute({"field": prediction}, {"field": target})
             for key, stat in metric.statistics.items()}
    mean_stats = {key: {"field": value["field"].weighted(weights).mean(SPATIAL, skipna=True)}
                  for key, value in stats.items()}
    crps = float(metric.values_from_mean_statistics(mean_stats)["field"])
    mean_error = prediction.mean("member", skipna=False) - target
    mse = float((mean_error ** 2).weighted(weights).mean(SPATIAL, skipna=True))
    variance = float(prediction.var("member", ddof=1, skipna=False).weighted(weights).mean(SPATIAL, skipna=True))
    return {"crps": crps, "mse": mse, "ensemble_variance": variance, "valid_area_fraction": coverage}


def validate_contract(prediction, truth, protocol):
    coverage = protocol.get("minimum_valid_area_fraction", 1.0)
    if not np.isfinite(coverage) or not 0 < coverage <= 1:
        raise ValueError("Invalid truth coverage threshold")
    variables = protocol["variables"]
    if set(prediction.data_vars) != set(variables) or set(truth.data_vars) != set(variables):
        raise ValueError("Missing or extra scored variables")
    expected = {
        "case": np.asarray(protocol["evaluation_initializations"], dtype="datetime64[ns]"),
        "lead": np.asarray(protocol["lead_hours"], dtype="timedelta64[h]").astype("timedelta64[ns]"),
        "lat": np.linspace(-90, 90, protocol["grid"][0]),
        "lon": np.arange(protocol["grid"][1]) * 360 / protocol["grid"][1],
        "level": np.asarray(protocol["pressure_levels_hpa"]),
    }
    for ds in (prediction, truth):
        for coord, values in expected.items():
            if coord not in ds.coords or ds[coord].dims != (coord,) or not np.array_equal(ds[coord].values, values):
                raise ValueError(f"Incorrect or reordered coordinate: {coord}")
        valid = expected["case"][:, None] + expected["lead"][None, :]
        if "valid_time" not in ds.coords or ds.valid_time.dims != ("case", "lead") or not np.array_equal(ds.valid_time.values, valid):
            raise ValueError("valid_time must equal initialization + lead exactly")
    if "member" not in prediction.coords or prediction.member.dims != ("member",) or not np.array_equal(prediction.member.values, np.arange(protocol["members"])):
        raise ValueError("Incorrect ensemble membership")
    for name, spec in variables.items():
        dims = {"case", "lead", "lat", "lon"} | ({"level"} if spec["pressure_levels"] else set())
        if set(truth[name].dims) != dims or set(prediction[name].dims) != dims | {"member"}:
            raise ValueError(f"Wrong dimensions: {name}")
        if truth[name].attrs.get("units") != spec["units"] or prediction[name].attrs.get("units") != spec["units"]:
            raise ValueError(f"Units must be explicit and canonical: {name}")
    if truth.attrs.get("truth_source") != protocol["truth_source"]:
        raise ValueError("Truth provenance differs from the protocol")
    xr.align(prediction, truth, join="exact", copy=False)


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["variable"], row["level_hpa"], row["lead_hours"], row["pool_cells"])
        groups.setdefault(key, []).append(row)
    result = []
    for (name, level, lead, factor), cases in groups.items():
        mse = float(np.mean([r["mse"] for r in cases]))
        variance = float(np.mean([r["ensemble_variance"] for r in cases]))
        result.append(dict(variable=name, level_hpa=level, lead_hours=lead, pool_cells=factor,
                           cases=len(cases), crps=float(np.mean([r["crps"] for r in cases])),
                           rmse=float(np.sqrt(mse)), spread=float(np.sqrt(variance)),
                           spread_skill=float(np.sqrt(variance / mse)) if mse > 0 else None))
    return result


def open_dataset(path):
    return xr.open_zarr(path, consolidated=True, chunks=None) if Path(path).is_dir() else xr.open_dataset(path)


def evaluate(prediction, truth, protocol, input_digest=None, truth_digest=None):
    validate_contract(prediction, truth, protocol)
    for case in protocol["evaluation_initializations"]:
        for lead in protocol["lead_hours"]:
            select = dict(case=np.datetime64(case, "ns"), lead=np.timedelta64(lead, "h"))
            for name, spec in protocol["variables"].items():
                levels = protocol["pressure_levels_hpa"] if spec["pressure_levels"] else [None]
                for level in levels:
                    index = select | ({"level": level} if level is not None else {})
                    # Bound RAM to eight members of one two-dimensional field.
                    p, t = prediction[name].sel(index).load(), truth[name].sel(index).load()
                    if input_digest is not None:
                        input_digest.update(json.dumps([case, lead, name, level]).encode())
                        for value, dims in ((p, ("member", "lat", "lon")), (t, SPATIAL)):
                            input_digest.update(np.asarray(value.transpose(*dims).values, dtype="<f8").tobytes())
                    if truth_digest is not None:
                        truth_digest.update(json.dumps([case, lead, name, level]).encode())
                        truth_digest.update(np.asarray(t.transpose(*SPATIAL).values, dtype="<f8").tobytes())
                    for factor in protocol["pool_cells"]:
                        yield dict(case=case, variable=name, level_hpa=level, lead_hours=lead,
                                   pool_cells=factor, **panel(p, t, factor, protocol.get("minimum_valid_area_fraction", 1.0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    revision = verify_metric_source()
    protocol = json.loads(args.protocol.read_text())
    input_digest = hashlib.sha256()
    truth_digest = hashlib.sha256()
    with open_dataset(args.prediction) as prediction, open_dataset(args.truth) as truth:
        rows = list(evaluate(prediction, truth, protocol, input_digest, truth_digest))
    report = dict(status="passed", protocol_sha256=sha256(args.protocol), weatherbenchx_revision=revision,
                  evaluator_sha256=sha256(__file__), estimator="empirical CRPS; fair=False",
                  spread_skill_definition="sqrt(mean unbiased ensemble variance / mean ensemble-mean squared error)",
                  scope="Reduced WN2-style variable panel; see the protocol for truth sources and excluded endpoints",
                  protocol_status=protocol["status"],
                  scored_input_values_sha256=input_digest.hexdigest(),
                  truth_values_sha256=truth_digest.hexdigest(),
                  training_provenance_verified=False,
                  per_case=rows, aggregate=summarize(rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
