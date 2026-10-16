"""Read an original LFMC crop with explicit per-period/per-band availability.

Uses the same rslearn GeoTIFF decoder and nearest-neighbor resampling as the
project DataInput, while retaining samples with unavailable layers.
"""

import json
from pathlib import Path
import numpy as np
from rasterio.crs import CRS
from rasterio.enums import Resampling
from upath import UPath
from rslearn.utils.geometry import Projection
from rslearn.utils.raster_format import GeotiffRasterFormat
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy

BAND_GROUPS = {
    "sentinel2_l2a": [
        ["B02", "B03", "B04", "B08"],
        ["B05", "B06", "B07", "B8A", "B11", "B12"],
        ["B01", "B09"],
    ],
    "sentinel1": [["vv", "vh"]],
}


def crop_bounds(metadata, rng, size=32):
    x1, y1, x2, y2 = metadata["bounds"]

    def offset(width):
        return rng.randint(min(0, width - size), max(0, width - size))

    x = x1 + offset(x2 - x1)
    y = y1 + offset(y2 - y1)
    return (x, y, x + size, y + size)


def read_window(path, *, rng=None, bounds=None, center_on_label=False):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata["options"]["split"] not in ["train", "val"]:
        raise ValueError("Development reader does not open heldout window payloads")
    projection = Projection(
        CRS.from_string(metadata["projection"]["crs"]),
        metadata["projection"]["x_resolution"],
        metadata["projection"]["y_resolution"],
    )
    reader = GeotiffRasterFormat()
    if center_on_label:
        if bounds is not None or rng is not None:
            raise ValueError(
                "Label-centered evaluation cannot also specify crop RNG/bounds"
            )
        full_labels = reader.decode_raster(
            UPath(path / "layers/labels/value"),
            projection,
            tuple(metadata["bounds"]),
            resampling=Resampling.nearest,
        ).array[0, 0]
        yy, xx = np.where(full_labels != -1)
        if not len(yy):
            raise ValueError("Released window has no valid label")
        x = metadata["bounds"][0] + int(np.median(xx)) - 16
        y = metadata["bounds"][1] + int(np.median(yy)) - 16
        bounds = (x, y, x + 32, y + 32)
    if bounds is None:
        if rng is None:
            raise ValueError("Provide an explicit crop RNG or bounds")
        bounds = crop_bounds(metadata, rng)
    if bounds[2] - bounds[0] != 32 or bounds[3] - bounds[1] != 32:
        raise ValueError("Official LFMC crop must be 32x32")
    normalizer = Normalizer(Strategy.COMPUTED)
    pixels = {}
    valid = {}
    for name, groups in BAND_GROUPS.items():
        channels = sum(map(len, groups))
        x = np.zeros((32, 32, 12, channels), dtype=np.float32)
        mask = np.zeros((12, len(groups)), dtype=bool)
        for t in range(12):
            layer = "sentinel2" if name == "sentinel2_l2a" else "sentinel1"
            if t:
                layer += f".{t}"
            channel = 0
            for g, bands in enumerate(groups):
                directory = path / "layers" / layer / "_".join(bands)
                if (directory / "geotiff.tif").exists():
                    arr = reader.decode_raster(
                        UPath(directory),
                        projection,
                        bounds,
                        resampling=Resampling.nearest,
                    ).array
                    if (
                        arr.shape != (len(bands), 1, 32, 32)
                        or not np.isfinite(arr).all()
                    ):
                        raise ValueError(f"Invalid raster group: {directory}")
                    x[:, :, t, channel : channel + len(bands)] = arr[:, 0].transpose(
                        1, 2, 0
                    )
                    mask[t, g] = True
                channel += len(bands)
        if name == "sentinel1":
            x = 10 * np.log10(np.maximum(x, 1e-6))
        pixels[name] = normalizer.normalize(Modality.get(name), x).astype(np.float32)
        valid[name] = mask
    labels = reader.decode_raster(
        UPath(path / "layers/labels/value"),
        projection,
        bounds,
        resampling=Resampling.nearest,
    ).array
    if labels.shape != (1, 1, 32, 32) or not np.isfinite(labels).all():
        raise ValueError("Invalid label raster")
    target = labels[0, 0].astype(np.float32)
    target_valid = target != -1
    # Official random training crops can miss the point. Preserve that sample
    # and its empty mask; the official regression head returns zero masked loss.
    if center_on_label and not target_valid.any():
        raise ValueError("Label-centered evaluation lost its label")
    return {
        "id": metadata["name"],
        "split": metadata["options"]["split"],
        "bounds": bounds,
        "pixels": pixels,
        "valid": valid,
        "target": target,
        "target_valid": target_valid,
        "center_on_label": center_on_label,
    }
