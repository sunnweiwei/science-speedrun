"""Pack complete public rslearn tar releases, retaining spatial split tags.

No label filtering by prediction quality. Missing timesteps and complete band groups are masked explicitly. Broken groups fail
preparation; no label-bearing window is silently dropped.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, deque
import json
from pathlib import Path
import tarfile
import h5py
import numpy as np
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

BANDS = [
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
    "B01",
    "B09",
]


def convert(job):
    name, files, task, metadata_only = job
    meta = json.loads(files["metadata.json"])
    label_key = (
        "layers/label/category/geotiff.tif"
        if task == "awf"
        else "layers/labels/value/geotiff.tif"
    )
    if label_key not in files:
        raise ValueError(f"{name}: no label")
    with MemoryFile(files[label_key]) as mem, mem.open() as ds:
        labels = ds.read(1)
        valid = labels != (9 if task == "awf" else -1)
        yy, xx = np.where(valid)
        if not len(yy):
            raise ValueError(f"{name}: no valid label pixels")
        values = np.unique(labels[valid])
        if len(values) != 1:
            raise ValueError(f"{name}: expected one point label, got {values}")
        cy, cx = int(np.median(yy)), int(np.median(xx))
        x, y = ds.xy(cy, cx)
        lon, lat = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True).transform(
            x, y
        )
        label = float(values[0])
        if not np.isfinite(label):
            raise ValueError(f"{name}: nonfinite label")
    if metadata_only:
        return name, meta["options"]["split"], label, [lat, lon], meta["time_range"], {}
    size = 16 if task == "awf" else 32
    bounds = (x - size * 5, y - size * 5, x + size * 5, y + size * 5)
    # Published windows are centered on point labels. Align crop edges to 10m
    # pixel boundaries with the label at index size//2.
    bounds = (bounds[0] - 5, bounds[1] + 5, bounds[2] - 5, bounds[3] + 5)
    arrays = {}
    modalities = ["sentinel2"] if task == "awf" else ["sentinel2", "sentinel1"]
    for mod in modalities:
        bands = BANDS if mod == "sentinel2" else ["vv", "vh"]
        arr = np.zeros((size, size, 12, len(bands)), np.float32)
        bandsets = (
            [BANDS[:4], BANDS[4:10], BANDS[10:]]
            if mod == "sentinel2"
            else [["vv", "vh"]]
        )
        validity = np.zeros((12, len(bandsets)), np.uint8)
        for t in range(12):
            layer = mod + (f".{t}" if t else "")
            found = set()
            prefix = f"layers/{layer}/"
            for key, content in files.items():
                if not key.startswith(prefix) or not key.endswith("/geotiff.tif"):
                    continue
                band_names = key.split("/")[-2].split("_")
                if not all(b in bands for b in band_names):
                    continue
                with MemoryFile(content) as mem, mem.open() as ds:
                    window = from_bounds(*bounds, transform=ds.transform)
                    pixels = ds.read(
                        window=window,
                        out_shape=(len(band_names), size, size),
                        resampling=Resampling.nearest,
                        boundless=True,
                        fill_value=0,
                    )
                for j, b in enumerate(band_names):
                    arr[:, :, t, bands.index(b)] = pixels[j]
                    found.add(b)
            for group_index, bandset in enumerate(bandsets):
                present = set(bandset) & found
                if present and present != set(bandset):
                    raise ValueError(f"{name}: incomplete band group {bandset}")
                validity[t, group_index] = bool(present)
        if not np.isfinite(arr).all():
            raise ValueError(f"{name}: nonfinite pixels")
        key = "sentinel2_l2a" if mod == "sentinel2" else mod
        arrays[key] = arr
        arrays[key + "_valid"] = validity
    return name, meta["options"]["split"], label, [lat, lon], meta["time_range"], arrays


def groups(tar_path, task, skip_images=0):
    seen = set()
    current, files = None, {}
    with tarfile.open(tar_path, "r:") as tar:
        for member in tar:
            parts = member.name.split("/")
            if "windows" not in parts:
                tar.members.clear()
                continue
            idx = parts.index("windows")
            if len(parts) < idx + 4:
                tar.members.clear()
                continue
            name = parts[idx + 2]
            if name != current:
                if current is not None:
                    yield current, files, task, len(seen) < skip_images
                    seen.add(current)
                if name in seen:
                    raise ValueError("Tar window files are not contiguous")
                current, files = name, {}
            rel = "/".join(parts[idx + 3 :])
            if member.isfile() and (
                rel == "metadata.json" or rel.endswith("/geotiff.tif")
            ):
                if (
                    len(seen) < skip_images
                    and rel != "metadata.json"
                    and "/label/" not in rel
                    and "/labels/" not in rel
                ):
                    continue
                if task != "awf" or "/sentinel1" not in rel:
                    files[rel] = tar.extractfile(member).read()
            tar.members.clear()
        if current is not None:
            yield current, files, task, len(seen) < skip_images


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tar", type=Path, required=True)
    p.add_argument("--task", choices=["awf", "lfmc"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = []
    out = args.output / "inputs.h5"
    with (
        h5py.File(out.with_suffix(".partial"), "a") as h,
        ProcessPoolExecutor(args.workers) as pool,
    ):
        keys = ["sentinel2_l2a", "sentinel2_l2a_valid"] + (
            ["sentinel1", "sentinel1_valid"] if args.task == "lfmc" else []
        )
        resume = min((h[key].shape[0] if key in h else 0) for key in keys)
        for key in keys:
            if key not in h:
                continue
            if h[key].shape[0] != resume:
                h[key].resize(resume, axis=0)
            if key.endswith("_valid") and h[key].ndim == 2:
                nsets = 3 if key.startswith("sentinel2") else 1
                old = h[key][:]
                del h[key]
                h.create_dataset(
                    key,
                    data=np.repeat(old[:, :, None], nsets, axis=2),
                    maxshape=(None, 12, nsets),
                    chunks=(1, 12, nsets),
                    compression="lzf",
                )
        h.attrs["source_tar"] = str(args.tar.resolve())
        print("Resuming", resume, "packed windows", flush=True)
        pending = deque()

        def write(future):
            name, split, label, latlon, time_range, arrays = future.result()
            index = len(report)
            for key, arr in arrays.items():
                if key not in h:
                    h.create_dataset(
                        key,
                        shape=(0, *arr.shape),
                        maxshape=(None, *arr.shape),
                        chunks=(1, *arr.shape),
                        dtype=(
                            "u1"
                            if key.endswith("_valid")
                            else "u2"
                            if key == "sentinel2_l2a"
                            else "f4"
                        ),
                        compression="lzf",
                    )
                ds = h[key]
                ds.resize(index + 1, axis=0)
                ds[index] = arr
            report.append(
                {
                    "index": index,
                    "id": name,
                    "split": split,
                    "label": label,
                    "latlon": latlon,
                    "time_range": time_range,
                }
            )
            if len(report) % 200 == 0:
                print(args.task, len(report), flush=True)

        for job in groups(args.tar, args.task, skip_images=resume):
            pending.append(pool.submit(convert, job))
            if len(pending) >= args.workers * 2:
                write(pending.popleft())
        while pending:
            write(pending.popleft())
        h.attrs["task"] = args.task
        h.create_dataset(
            "ids", data=np.array([r["id"] for r in report], dtype=h5py.string_dtype())
        )
    out.with_suffix(".partial").rename(out)
    (args.output / "labels.json").write_text(json.dumps(report))
    summary = {
        "task": args.task,
        "rows": len(report),
        "splits": dict(Counter(r["split"] for r in report)),
        "source": str(args.tar),
        "spatial_window": 16 if args.task == "awf" else 32,
        "point_probe": True,
        "timestamps": "published project legacy month-index encoding; original window time ranges retained",
        "resampling": "nearest to 10m grid",
        "label_pixel": "center of valid label region",
    }
    (args.output / "release.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
