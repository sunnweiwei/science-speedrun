#!/usr/bin/env python3
"""Derive bounded NDMI statistics from the immutable alignment cache, on CPU."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


SCHEMA_VERSION = 1
CHANNELS = ("ndmi_temporal_mean", "ndmi_temporal_population_std")
S2_BANDS = ("B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09")
WOODY_CLASSES = (10, 20, 95)
REQUIRED_ARRAYS = ("sentinel2_l2a", "sentinel2_l2a_valid", "worldcover_fractions", "worldcover_weight")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def inverse_band(values, spec, multiplier=2.0):
    """Invert upstream COMPUTED normalization; there is no clipping in it."""
    mean, std = float(spec["mean"]), float(spec["std"])
    if not np.isfinite([mean, std, multiplier]).all() or std <= 0 or multiplier <= 0:
        raise ValueError("Invalid normalization coefficients")
    return values.astype(np.float64) * (2 * multiplier * std) + mean - multiplier * std


def moisture_targets(pixels, validity, normalization, min_sum, patch_size=4):
    """Spatially average valid pixel ratios, then compute observed-time moments."""
    n, height, width, times, bands = pixels.shape
    if height % patch_size or width % patch_size or bands != len(S2_BANDS):
        raise ValueError("Unexpected S2 geometry")
    if validity.shape != (n, times, 3) or min_sum <= 0 or not np.isfinite(min_sum):
        raise ValueError("Invalid validity geometry or denominator threshold")
    nir = inverse_band(pixels[..., S2_BANDS.index("B8A")], normalization["B8A"])
    swir = inverse_band(pixels[..., S2_BANDS.index("B11")], normalization["B11"])
    observed = np.broadcast_to(validity[:, None, None, :, 1], nir.shape)
    finite = np.isfinite(nir) & np.isfinite(swir)
    nonnegative = (nir >= 0) & (swir >= 0)
    denominator = nir + swir
    usable = observed & finite & nonnegative & (denominator > min_sum)
    ratio = np.zeros_like(nir)
    np.divide(nir - swir, denominator, out=ratio, where=usable)
    if not np.isfinite(ratio).all() or np.any(np.abs(ratio[usable]) > 1 + 1e-12):
        raise ValueError("NDMI is not finite or bounded")
    grid_h, grid_w = height // patch_size, width // patch_size
    layout = (n, grid_h, patch_size, grid_w, patch_size, times)
    pixel_count = usable.reshape(layout).sum(axis=(2, 4))
    monthly = ratio.reshape(layout).sum(axis=(2, 4)) / np.maximum(pixel_count, 1)
    monthly_valid = pixel_count > 0
    count = monthly_valid.sum(axis=-1)
    mean = (monthly * monthly_valid).sum(axis=-1) / np.maximum(count, 1)
    variance = (((monthly - mean[..., None]) ** 2) * monthly_valid).sum(axis=-1) / np.maximum(count, 1)
    std = np.sqrt(np.maximum(variance, 0))
    valid = np.stack((count >= 1, count >= 2), axis=-1)
    targets = np.stack((mean, std), axis=-1)
    targets[~valid] = 0
    audit = {
        "observed_pixels": int(observed.sum()),
        "usable_pixels": int(usable.sum()),
        "rejected_nonfinite_pixels": int((observed & ~finite).sum()),
        "rejected_negative_pixels": int((observed & finite & ~nonnegative).sum()),
        "rejected_small_denominator_pixels": int((observed & finite & nonnegative & (denominator <= min_sum)).sum()),
        "valid_slots_histogram": np.bincount(count.ravel(), minlength=times + 1).tolist(),
        "partial_pixel_patch_slots": int(((pixel_count > 0) & (pixel_count < patch_size ** 2)).sum()),
    }
    return targets.astype(np.float32), valid, audit


def woody_weights(fractions, coverage, worldcover_classes, vegetation_classes):
    if not np.isfinite(fractions).all() or not np.isfinite(coverage).all():
        raise ValueError("Nonfinite WorldCover fractions/coverage")
    if np.any(fractions < 0) or np.any(fractions > 1) or np.any(coverage < 0) or np.any(coverage > 1):
        raise ValueError("WorldCover fractions/coverage outside [0,1]")
    if not vegetation_classes or len(set(vegetation_classes)) != len(vegetation_classes):
        raise ValueError("Vegetation class list must be unique and nonempty")
    indices = [worldcover_classes.index(code) for code in vegetation_classes]
    weights = fractions[..., indices].sum(axis=-1) * coverage
    if np.any(weights > 1 + 1e-6):
        raise ValueError("Vegetation weight exceeds one")
    return np.minimum(weights, 1).astype(np.float32)


def target_statistics(targets, valid, weights, std_floor):
    if not np.isfinite(std_floor) or std_floor <= 0:
        raise ValueError("Target standard-deviation floor must be positive")
    effective = valid.astype(np.float64) * weights[..., None]
    denominator = effective.sum(axis=(0, 1, 2))
    if np.any(denominator <= 0):
        raise ValueError("A moisture channel has no positively weighted observations")
    values = targets.astype(np.float64)
    mean = (values * effective).sum(axis=(0, 1, 2)) / denominator
    variance = (((values - mean) ** 2) * effective).sum(axis=(0, 1, 2)) / denominator
    observed_std = np.sqrt(np.maximum(variance, 0))
    if np.any(observed_std <= 1e-8):
        raise ValueError("A moisture channel is degenerate")
    distributions = []
    for channel in range(len(CHANNELS)):
        selected = values[..., channel][effective[..., channel] > 0]
        distributions.append({
            "channel": CHANNELS[channel],
            "positively_weighted_patch_count": int(len(selected)),
            "weighted_denominator": float(denominator[channel]),
            "weighted_mean": float(mean[channel]),
            "weighted_std": float(observed_std[channel]),
            "unweighted_quantiles_on_positive_weight": dict(zip(
                ("min", "p01", "p05", "p50", "p95", "p99", "max"),
                np.quantile(selected, [0, .01, .05, .5, .95, .99, 1]).tolist(),
            )),
        })
    return mean.tolist(), np.maximum(observed_std, std_floor).tolist(), distributions


def prepare(cache_manifest, output, *, chunk_rows=128, vegetation_classes=WOODY_CLASSES,
            min_sum_fraction=.001, target_std_floor=.01, command=None):
    started = time.monotonic()
    cache_manifest, output = Path(cache_manifest).resolve(), Path(output).resolve()
    base = json.loads(cache_manifest.read_text())
    if not base.get("complete") or base.get("patch_size") != 4 or base.get("sequence_length") != 12:
        raise ValueError("Expected a complete patch4/12-slot alignment cache")
    if chunk_rows <= 0 or not np.isfinite(min_sum_fraction) or min_sum_fraction <= 0:
        raise ValueError("Invalid preparation controls")
    normal = base["normalization"]
    if normal.get("strategy") != "COMPUTED" or normal.get("std_multiplier") != 2:
        raise ValueError("Expected invertible COMPUTED normalization with multiplier2")
    normalization_path = cache_manifest.parent / normal["config_path"]
    if sha256(normalization_path) != normal["config_sha256"]:
        raise ValueError("Normalization hash mismatch")
    normalization = json.loads(normalization_path.read_text())["sentinel2_l2a"]
    min_sum = min_sum_fraction * (abs(normalization["B8A"]["mean"]) + abs(normalization["B11"]["mean"]))
    if not np.isfinite(min_sum) or min_sum <= 0:
        raise ValueError("Invalid source-unit denominator threshold")
    inputs = {}
    for name in REQUIRED_ARRAYS:
        entry = base["arrays"][name]
        path = cache_manifest.parent / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Input hash mismatch: {name}")
        inputs[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(inputs[name].shape) != entry["shape"] or str(inputs[name].dtype) != entry["dtype"]:
            raise ValueError(f"Input geometry mismatch: {name}")
    n, crop, grid = int(base["rows"]), int(base["crop_size"]), int(base["crop_size"]) // 4
    shapes = {
        "sentinel2_l2a": (n, crop, crop, 12, 12),
        "sentinel2_l2a_valid": (n, 12, 3),
        "worldcover_fractions": (n, grid, grid, len(base["worldcover_classes"])),
        "worldcover_weight": (n, grid, grid),
    }
    for name, shape in shapes.items():
        if inputs[name].shape != shape:
            raise ValueError(f"Unexpected parent geometry: {name}")
    # Existing or incomplete outputs are never silently reused.
    output.mkdir(parents=True, exist_ok=False)
    arrays = {
        "moisture_targets": np.lib.format.open_memmap(output / "moisture_targets.npy", mode="w+", dtype="float32", shape=(n, grid, grid, 2)),
        "moisture_valid": np.lib.format.open_memmap(output / "moisture_valid.npy", mode="w+", dtype="bool", shape=(n, grid, grid, 2)),
        "moisture_weight": np.lib.format.open_memmap(output / "moisture_weight.npy", mode="w+", dtype="float32", shape=(n, grid, grid)),
    }
    audit = {}
    for first in range(0, n, chunk_rows):
        last = min(first + chunk_rows, n)
        targets, valid, current = moisture_targets(
            inputs["sentinel2_l2a"][first:last], inputs["sentinel2_l2a_valid"][first:last],
            normalization, min_sum,
        )
        weights = woody_weights(inputs["worldcover_fractions"][first:last],
                                inputs["worldcover_weight"][first:last],
                                base["worldcover_classes"], list(vegetation_classes))
        arrays["moisture_targets"][first:last] = targets
        arrays["moisture_valid"][first:last] = valid
        arrays["moisture_weight"][first:last] = weights
        for key, value in current.items():
            if isinstance(value, list):
                audit[key] = (np.asarray(audit.get(key, np.zeros(len(value), dtype=int))) + value).tolist()
            else:
                audit[key] = audit.get(key, 0) + value
    for array in arrays.values():
        array.flush()
    mean, std, distributions = target_statistics(arrays["moisture_targets"], arrays["moisture_valid"], arrays["moisture_weight"], target_std_floor)
    positive = arrays["moisture_valid"] & (arrays["moisture_weight"][..., None] > 0)
    audit.update({
        "rows_with_positive_weight_mean": int(positive[..., 0].reshape(n, -1).any(1).sum()),
        "rows_with_positive_weight_std": int(positive[..., 1].reshape(n, -1).any(1).sum()),
        "rows_without_moisture_supervision": int((~positive.reshape(n, -1).any(1)).sum()),
        "positive_weight_spatial_patches": int((arrays["moisture_weight"] > 0).sum()),
        "total_spatial_patches": n * grid * grid,
        "distributions": distributions,
    })
    manifest = {
        "schema_version": SCHEMA_VERSION, "kind": "olmoearth_moisture_targets", "complete": True,
        "rows": n, "crop_size": crop, "patch_size": 4, "channels": list(CHANNELS),
        "base_cache_manifest": str(cache_manifest), "base_cache_manifest_sha256": sha256(cache_manifest),
        "base_array_sha256": {key: base["arrays"][key]["sha256"] for key in REQUIRED_ARRAYS},
        "normalization_sha256": sha256(normalization_path),
        "recipe": {
            "formula": "(raw_B8A-raw_B11)/(raw_B8A+raw_B11)",
            "inverse_normalization": "raw=normalized*(4*band_std)+band_mean-2*band_std; float64 computation",
            "source_bands": ["B8A", "B11"], "source_bandset": 1,
            "pixel_validity": "observed source bandset; finite nonnegative recovered bands; sum > min_sum_source_units",
            "min_sum_fraction_of_normalization_band_mean_sum": float(min_sum_fraction),
            "min_sum_source_units": float(min_sum),
            "spatial_aggregation": "mean of usable per-pixel ratios in each patch4 and slot; at least1 usable pixel",
            "temporal_aggregation": "equal observed-slot temporal mean and population std of patch means; no calendar ordering",
            "minimum_slots": [1, 2], "invalid_target_fill": 0,
            "vegetation_classes": list(vegetation_classes),
            "weight": "WorldCover known coverage times summed selected class fractions; same weight per target channel",
            "normalization_statistics": "per-channel weighted mean/population std over valid pretraining targets",
            "target_std_floor": float(target_std_floor),
            "loss": "standardized squared error weighted by moisture_weight and channel validity; global weighted denominator",
            "limits": ["No direct LFMC labels", "No explicit pixel cloud mask", "Patch vegetation weighting does not isolate vegetation pixels", "Order-invariant summaries do not teach measurement-date moisture"],
        },
        "target_mean": mean, "target_std": std, "audit": audit,
        "arrays": {key: {"path": key + ".npy", "shape": list(value.shape), "dtype": str(value.dtype),
                          "bytes": (output / (key + ".npy")).stat().st_size,
                          "sha256": sha256(output / (key + ".npy"))} for key, value in arrays.items()},
        "preparation": {"command": list(command or []), "script_sha256": sha256(__file__),
                        "finished_utc": datetime.now(timezone.utc).isoformat(),
                        "wall_seconds": time.monotonic() - started, "chunk_rows": chunk_rows,
                        "gpu_hours": 0, "benchmark_data_used": False},
    }
    write_json(output / "manifest.json", manifest)
    return manifest


class MoistureCache:
    """Read and verify derived arrays bound to one already-verified AlignmentCache."""

    def __init__(self, manifest_path, base_cache):
        self.path = self.manifest_path = Path(manifest_path).resolve()
        self.manifest_hash = self.manifest_sha256 = sha256(self.path)
        self.manifest = json.loads(self.path.read_text())
        m = self.manifest
        if m.get("schema_version") != SCHEMA_VERSION or m.get("kind") != "olmoearth_moisture_targets" or not m.get("complete"):
            raise ValueError("Invalid or incomplete moisture cache")
        if m.get("base_cache_manifest_sha256") != base_cache.manifest_hash:
            raise ValueError("Moisture cache parent hash mismatch")
        if m.get("base_array_sha256") != {key: base_cache.manifest["arrays"][key]["sha256"] for key in REQUIRED_ARRAYS}:
            raise ValueError("Moisture cache source array identity mismatch")
        if m.get("normalization_sha256") != base_cache.manifest["normalization"]["config_sha256"]:
            raise ValueError("Moisture cache normalization identity mismatch")
        n, crop = base_cache.size, base_cache.manifest["crop_size"]
        if m.get("rows") != n or m.get("crop_size") != crop or m.get("patch_size") != 4 or m.get("channels") != list(CHANNELS):
            raise ValueError("Moisture target geometry or channels mismatch")
        self.target_width = len(CHANNELS)
        self.mean, self.std = np.asarray(m["target_mean"], dtype=np.float32), np.asarray(m["target_std"], dtype=np.float32)
        if self.mean.shape != (2,) or self.std.shape != (2,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("Invalid moisture target normalization")
        self.arrays = {}
        for key, shape, dtype in (
            ("moisture_targets", (n, crop // 4, crop // 4, 2), "float32"),
            ("moisture_valid", (n, crop // 4, crop // 4, 2), "bool"),
            ("moisture_weight", (n, crop // 4, crop // 4), "float32"),
        ):
            entry = m["arrays"][key]
            path = self.path.parent / entry["path"]
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"Moisture array hash mismatch: {key}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or str(array.dtype) != dtype or list(shape) != entry["shape"] or entry["dtype"] != dtype:
                raise ValueError(f"Moisture array geometry mismatch: {key}")
            self.arrays[key] = array
        values, valid, weights = (self.arrays[k] for k in ("moisture_targets", "moisture_valid", "moisture_weight"))
        if not np.isfinite(values).all() or np.any(np.abs(values[..., 0]) > 1 + 1e-6) or np.any(values[..., 1] < 0) or np.any(values[..., 1] > 1 + 1e-6) or np.any(values[~valid] != 0):
            raise ValueError("Invalid moisture target values")
        if not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights > 1):
            raise ValueError("Invalid moisture weights")
        mean, std, _ = target_statistics(values, valid, weights, m["recipe"]["target_std_floor"])
        if not np.allclose(mean, self.mean, rtol=1e-6, atol=1e-7) or not np.allclose(std, self.std, rtol=1e-6, atol=1e-7):
            raise ValueError("Moisture statistics do not match arrays")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=128)
    parser.add_argument("--vegetation-classes", type=int, nargs="+", default=list(WOODY_CLASSES))
    parser.add_argument("--min-sum-fraction", type=float, default=.001)
    parser.add_argument("--target-std-floor", type=float, default=.01)
    args = parser.parse_args()
    manifest = prepare(args.cache_manifest, args.output, chunk_rows=args.chunk_rows,
                       vegetation_classes=args.vegetation_classes, min_sum_fraction=args.min_sum_fraction,
                       target_std_floor=args.target_std_floor, command=sys.argv)
    print(json.dumps({"manifest": str(args.output / "manifest.json"), "audit": manifest["audit"],
                      "target_mean": manifest["target_mean"], "target_std": manifest["target_std"],
                      "wall_seconds": manifest["preparation"]["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
