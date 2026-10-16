"""Immutable pretraining-only S1/S2 crops, WorldCover fractions and moments."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import resource
import struct
import sys
import time
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from olmoearth_pretrain.data.constants import (
    MISSING_VALUE,
    Modality,
    WORLDCOVER_CLASSES,
)
from olmoearth_pretrain.data.normalize import Normalizer, Strategy, load_computed_config

MODALITIES = ("sentinel2_l2a", "sentinel1")
BANDSETS = {
    "sentinel2_l2a": (slice(0, 4), slice(4, 10), slice(10, 12)),
    "sentinel1": (slice(0, 2),),
}
EXCLUSION = (
    "AWF bbox lon32..43 lat-6..6; LFMC Euclidean lonlat radius0.1deg, all labels"
)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def spectral_targets(pixels, validity, crop):
    grid = crop // 4
    values_out, masks_out = [], []
    for name in MODALITIES:
        data = pixels[name].astype(np.float32)
        channels = data.shape[-1]
        monthly = data.reshape(grid, 4, grid, 4, 12, channels).mean(axis=(1, 3))
        valid = np.zeros((12, channels), dtype=bool)
        for bandset, channel_slice in enumerate(BANDSETS[name]):
            valid[:, channel_slice] = validity[name][:, bandset, None]
        weight = valid[None, None].astype(np.float32)
        count = weight.sum(axis=2)
        mean = (monthly * weight).sum(axis=2) / np.maximum(count, 1)
        var = ((monthly - mean[:, :, None]) ** 2 * weight).sum(axis=2) / np.maximum(
            count, 1
        )
        std = np.sqrt(np.maximum(var, 0))
        present = np.broadcast_to(count > 0, mean.shape)
        values_out.extend((mean, std))
        masks_out.extend((present, present))
    return np.concatenate(values_out, axis=-1).astype(np.float32), np.concatenate(
        masks_out, axis=-1
    )


def worldcover_targets(raw, crop):
    grid = crop // 4
    known = np.isin(raw, WORLDCOVER_CLASSES)
    weight = known.reshape(grid, 4, grid, 4).mean(axis=(1, 3))
    counts = np.stack(
        [
            (raw == code).reshape(grid, 4, grid, 4).sum(axis=(1, 3))
            for code in WORLDCOVER_CLASSES
        ],
        axis=-1,
    )
    fractions = counts / np.maximum(counts.sum(axis=-1, keepdims=True), 1)
    return fractions.astype(np.float32), weight.astype(np.float32)


def prepare_sample(job):
    root, source, crop, n_crops, seed = job
    index = source["sample_index"]
    path = Path(root) / f"sample_{index}.h5"
    payload = path.read_bytes()
    source_hash = hashlib.sha256(payload).hexdigest()
    normalizer = Normalizer(Strategy.COMPUTED)
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    result, skipped = [], []
    with h5py.File(io.BytesIO(payload), "r") as h:
        timestamps = h["timestamps"][:]
        if timestamps.ndim != 2 or timestamps.shape[1] != 3 or not len(timestamps):
            raise ValueError(f"Invalid timestamps in {path}")
        if "sentinel2_l2a" not in h:
            raise ValueError(f"Expected S2 in {path}")
        height, width = h["sentinel2_l2a"].shape[:2]
        if min(height, width) < crop:
            raise ValueError("Crop exceeds source")
        raw = {name: h[name][:] for name in MODALITIES if name in h}
        wc = (
            h["worldcover"][:, :, 0, 0]
            if "worldcover" in h
            else np.zeros((height, width), dtype=np.uint8)
        )
        masks = {}
        for name in MODALITIES:
            if name not in raw:
                masks[name] = np.zeros(len(timestamps), dtype=bool)
                continue
            if raw[name].shape[:2] != (height, width) or raw[name].shape[-1] != len(
                Modality.get(name).band_order
            ):
                raise ValueError(f"Unexpected geometry: {name} in {path}")
            mask_path = f"missing_timesteps_masks/{name}"
            mask = (
                h[mask_path][:].astype(bool)
                if mask_path in h
                else np.ones(len(timestamps), dtype=bool)
            )
            if mask.shape != (len(timestamps),) or mask.sum() != raw[name].shape[2]:
                raise ValueError(f"Observation-count mismatch: {name} in {path}")
            masks[name] = mask
        seen = set()
        for crop_id in range(n_crops):
            for _ in range(1000):
                top = int(rng.integers(height - crop + 1))
                left = int(rng.integers(width - crop + 1))
                start = int(rng.integers(max(0, len(timestamps) - 12) + 1))
                if (top, left, start) not in seen:
                    seen.add((top, left, start))
                    break
            else:
                raise ValueError("Cannot select distinct crops")
            pixels, valid, invalidated = {}, {}, {}
            for name in MODALITIES:
                channels = len(Modality.get(name).band_order)
                data = np.zeros((crop, crop, 12, channels), dtype=np.float32)
                valid[name] = np.zeros((12, len(BANDSETS[name])), dtype=bool)
                invalidated[name] = 0
                if name in raw:
                    for payload_t, source_t in enumerate(np.flatnonzero(masks[name])):
                        target_t = int(source_t) - start
                        if not 0 <= target_t < 12:
                            continue
                        patch = raw[name][
                            top : top + crop, left : left + crop, payload_t
                        ].astype(np.float32)
                        # Pretraining S1 is already dB. Do not apply log10 again.
                        normalized = normalizer.normalize(
                            Modality.get(name), patch
                        ).astype(np.float32)
                        for bandset, cs in enumerate(BANDSETS[name]):
                            x, y = patch[..., cs], normalized[..., cs]
                            okay = (
                                np.isfinite(x).all()
                                and (x != MISSING_VALUE).all()
                                and np.isfinite(y).all()
                                and (np.abs(y) <= np.finfo(np.float16).max).all()
                            )
                            if okay:
                                data[:, :, target_t, cs] = y
                                valid[name][target_t, bandset] = True
                            else:
                                invalidated[name] += 1
                pixels[name] = data.astype(np.float16)
            detail = {
                "sample_index": index,
                "source": source["source"],
                "source_path": str(path.resolve()),
                "source_sha256": source_hash,
                "source_bytes": len(payload),
                "latlon": source["latlon"],
                "crop_id": crop_id,
                "top": top,
                "left": left,
                "crop_size": crop,
                "source_start_t": start,
                "source_timestamps": timestamps[start : start + 12].tolist(),
                "invalidated_bandset_observations": invalidated,
            }
            if not any(v.any() for v in valid.values()):
                skipped.append(dict(detail, reason="no_observed_s1_or_s2_bandsets"))
                continue
            fractions, weight = worldcover_targets(
                wc[top : top + crop, left : left + crop], crop
            )
            moments, moments_valid = spectral_targets(pixels, valid, crop)
            arrays = dict(
                pixels,
                worldcover_fractions=fractions,
                worldcover_weight=weight,
                spectral_moments=moments,
                spectral_moments_valid=moments_valid,
            )
            arrays.update({name + "_valid": v for name, v in valid.items()})
            result.append((detail, arrays))
    return result, skipped


def array_specs(rows, crop):
    g = crop // 4
    return {
        "sentinel2_l2a": ((rows, crop, crop, 12, 12), "float16"),
        "sentinel1": ((rows, crop, crop, 12, 2), "float16"),
        "sentinel2_l2a_valid": ((rows, 12, 3), "bool"),
        "sentinel1_valid": ((rows, 12, 1), "bool"),
        "worldcover_fractions": ((rows, g, g, 11), "float32"),
        "worldcover_weight": ((rows, g, g), "float32"),
        "spectral_moments": ((rows, g, g, 28), "float32"),
        "spectral_moments_valid": ((rows, g, g, 28), "bool"),
    }


def shrink_npy(path, rows):
    """Trim optional all-missing rows without copying the multi-GB arrays."""
    with path.open("r+b") as f:
        if np.lib.format.read_magic(f) != (1, 0):
            raise ValueError("Expected NPY v1")
        length = struct.unpack("<H", f.read(2))[0]
        start = f.tell()
        f.seek(0)
        np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        offset = f.tell()
        shape = (rows, *shape[1:])
        header = repr(
            {
                "descr": np.lib.format.dtype_to_descr(dtype),
                "fortran_order": fortran,
                "shape": shape,
            }
        ).encode("latin1")
        if len(header) + 1 > length:
            raise ValueError("Updated header too long")
        f.seek(start)
        f.write(header + b" " * (length - len(header) - 1) + b"\n")
        f.truncate(offset + int(np.prod(shape)) * dtype.itemsize)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--crop-size", type=int, choices=(16, 32), default=16)
    p.add_argument("--crops-per-sample", type=int, default=2)
    p.add_argument("--seed", type=int, default=42017)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument(
        "--max-samples",
        type=int,
        help="Deterministic source prefix for a separate health-check cache",
    )
    p.add_argument("--plan-only", action="store_true")
    a = p.parse_args()
    started, started_utc = time.monotonic(), utc()
    if not 1 <= a.workers <= 16 or not 1 <= a.crops_per_sample <= 16 or a.seed < 0:
        p.error("workers/crops must be1..16 and seed nonnegative")
    source = a.source.resolve()
    provenance_path = source / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("exclusion") != EXCLUSION or not provenance.get(
        "lfmc_annotations_sha256"
    ):
        raise ValueError("Source lacks established geographic exclusions")
    samples = provenance["samples"]
    if len(samples) != provenance["kept"] or [
        s["sample_index"] for s in samples
    ] != list(range(len(samples))):
        raise ValueError("Invalid source mapping")
    if a.max_samples is not None:
        if a.max_samples <= 0:
            p.error("max-samples must be positive")
        samples = samples[: a.max_samples]
    rows_max = len(samples) * a.crops_per_sample
    specs = array_specs(rows_max, a.crop_size)
    plan = {
        "source_samples": len(samples),
        "max_rows": rows_max,
        "array_payload_bytes": sum(
            int(np.prod(shape)) * np.dtype(dtype).itemsize
            for shape, dtype in specs.values()
        ),
        "workers": a.workers,
        "crop_size": a.crop_size,
        "crops_per_sample": a.crops_per_sample,
    }
    if a.plan_only:
        print(json.dumps(plan, indent=2))
        return
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "preparation-started.json").write_text(
        json.dumps(
            {"started_utc": started_utc, "command": sys.argv, "plan": plan}, indent=2
        )
        + "\n"
    )
    (out / "source-provenance.json").write_bytes(provenance_path.read_bytes())
    arrays = {
        name: np.lib.format.open_memmap(
            out / f"{name}.npy", mode="w+", dtype=dtype, shape=shape, version=(1, 0)
        )
        for name, (shape, dtype) in specs.items()
    }
    sums, squares, counts = (np.zeros(28, dtype=np.float64) for _ in range(3))
    class_counts = np.zeros(11, dtype=np.float64)
    rows = sources_read = skipped_total = source_bytes = 0
    # At most 2*workers submitted sources: no eager full-dataset result queue.
    with ProcessPoolExecutor(max_workers=a.workers) as pool, (
        out / "records.jsonl"
    ).open("w") as records, (out / "skipped.jsonl").open("w") as skips:
        jobs = iter(
            (str(source), s, a.crop_size, a.crops_per_sample, a.seed) for s in samples
        )
        pending = []
        for _ in range(2 * a.workers):
            try:
                pending.append(pool.submit(prepare_sample, next(jobs)))
            except StopIteration:
                break
        while pending:
            result, skipped = pending.pop(0).result()
            sources_read += 1
            if result or skipped:
                source_bytes += (result[0][0] if result else skipped[0])["source_bytes"]
            for detail in skipped:
                skips.write(json.dumps(detail, sort_keys=True) + "\n")
            skipped_total += len(skipped)
            for detail, values in result:
                for name, data in values.items():
                    arrays[name][rows] = data
                detail["record_index"] = rows
                records.write(json.dumps(detail, sort_keys=True) + "\n")
                x = values["spectral_moments"].astype(np.float64)
                valid = values["spectral_moments_valid"]
                sums += (x * valid).sum(axis=(0, 1))
                squares += (x * x * valid).sum(axis=(0, 1))
                counts += valid.sum(axis=(0, 1))
                class_counts += (
                    values["worldcover_fractions"]
                    * values["worldcover_weight"][..., None]
                    * 16
                ).sum(axis=(0, 1))
                rows += 1
            try:
                pending.append(pool.submit(prepare_sample, next(jobs)))
            except StopIteration:
                pass
            if sources_read % 500 == 0:
                print(
                    json.dumps(
                        {
                            "sources_read": sources_read,
                            "rows": rows,
                            "wall_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
    if rows == 0:
        raise ValueError("No observed cache rows")
    for name in list(arrays):
        arrays[name].flush()
        del arrays[name]
    array_manifest = {}
    for name, (shape, dtype) in specs.items():
        path = out / f"{name}.npy"
        if rows < rows_max:
            shrink_npy(path, rows)
        array_manifest[name] = {
            "path": path.name,
            "shape": [rows, *shape[1:]],
            "dtype": dtype,
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        }
    means = sums / np.maximum(counts, 1)
    stds = np.sqrt(np.maximum(squares / np.maximum(counts, 1) - means**2, 0))
    channels = [
        f"{short}_{moment}_{band}"
        for name, short in zip(MODALITIES, ("s2", "s1"))
        for moment in ("mean", "std")
        for band in Modality.get(name).band_order
    ]
    config = load_computed_config()
    normalization = {name: config[name] for name in MODALITIES}
    (out / "normalization.json").write_text(
        json.dumps(normalization, sort_keys=True, indent=2) + "\n"
    )
    elapsed = time.monotonic() - started
    manifest = {
        "schema_version": 1,
        "complete": True,
        "rows": rows,
        "arrays": array_manifest,
        "patch_size": 4,
        "crop_size": a.crop_size,
        "sequence_length": 12,
        "worldcover_classes": list(WORLDCOVER_CLASSES),
        "worldcover_unknown_policy": "Unknown codes including0 receive zero weight; fractions normalize recognized pixels only",
        "worldcover_pixel_counts": class_counts.tolist(),
        "moment_channels": channels,
        "moment_mean": means.tolist(),
        "moment_std": stds.tolist(),
        "moment_valid_counts": counts.tolist(),
        "moment_definition": "For each patch4, spatial mean of normalized cached pixels per observed month, then temporal population mean/std. S2 mean12/std12 followed by S1 mean2/std2. One observation has std0.",
        "normalization": {
            "strategy": "COMPUTED",
            "std_multiplier": 2,
            "source_sentinel1_units": "dB; no additional log10",
            "cache_dtype": "float16",
            "missing_fill": 0,
            "config_path": "normalization.json",
            "config_sha256": digest(out / "normalization.json"),
        },
        "encoder_timestamps": [[1, month, 2024] for month in range(12)],
        "timestamps_policy": "Fixed downstream synthetic timestamps; source order preserved without calendar reordering; source timestamps retained per record; padded tail marked missing",
        "selection": {
            "seed": a.seed,
            "crops_per_sample": a.crops_per_sample,
            "source_samples_read": sources_read,
            "source_collection_samples": provenance["kept"],
            "source_prefix_limit": a.max_samples,
            "skipped_no_observation_rows": skipped_total,
            "selection_rationale": "Uniform deterministic spatial crops of every selected geographically excluded source; no benchmark labels, class rebalancing or score-based filtering",
        },
        "provenance": {
            "source": str(source),
            "source_provenance_sha256": digest(provenance_path),
            "copied_source_provenance": "source-provenance.json",
            "exclusion": provenance["exclusion"],
            "lfmc_annotations_sha256": provenance["lfmc_annotations_sha256"],
            "source_release": provenance.get("source_release"),
            "records_path": "records.jsonl",
            "records_sha256": digest(out / "records.jsonl"),
            "skipped_path": "skipped.jsonl",
            "skipped_sha256": digest(out / "skipped.jsonl"),
            "per_source_file_sha256": True,
            "source_bytes_read": source_bytes,
        },
        "preparation": {
            "command": sys.argv,
            "script_sha256": digest(Path(__file__)),
            "started_utc": started_utc,
            "finished_utc": utc(),
            "wall_seconds": elapsed,
            "gpu_hours": 0,
            "workers": a.workers,
            "parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "max_child_peak_rss_kib": resource.getrusage(
                resource.RUSAGE_CHILDREN
            ).ru_maxrss,
            "rss_note": "Parent peak and largest reaped child peak, not summed concurrent RSS",
            "plan": plan,
        },
    }
    temporary = out / "manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(out / "manifest.json")
    print(
        json.dumps(
            {
                "manifest": str(out / "manifest.json"),
                "rows": rows,
                "wall_seconds": elapsed,
                "gpu_hours": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
