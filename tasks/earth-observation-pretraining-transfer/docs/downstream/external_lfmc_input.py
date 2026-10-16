"""Read label-free, official-config external rasters into LFMC encoder inputs."""

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

from external_lfmc_availability import BANDS
from lfmc_window import BAND_GROUPS


def read_external(path, bounds, periods):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata["options"]["split"] != "external_unscored":
        raise ValueError("External reader requires explicit unscored external split")
    if bounds[2] - bounds[0] != 32 or bounds[3] - bounds[1] != 32:
        raise ValueError("Require32x32crop")
    expected = {(m, i) for m in BANDS for i in range(12)}
    if len(periods) != 24 or {(p["modality"], p["index"]) for p in periods} != expected:
        raise ValueError("Require complete explicit period manifest")
    records = {(p["modality"], p["index"]): p for p in periods}
    projection = Projection(
        CRS.from_string(metadata["projection"]["crs"]),
        metadata["projection"]["x_resolution"],
        metadata["projection"]["y_resolution"],
    )
    reader = GeotiffRasterFormat()
    normalizer = Normalizer(Strategy.COMPUTED)
    pixels, valid = {}, {}
    for model_name, groups in BAND_GROUPS.items():
        name = "sentinel2" if model_name == "sentinel2_l2a" else "sentinel1"
        channels = [b for g in groups for b in g]
        reorder = [BANDS[name].index(b) for b in channels]
        values = np.zeros((32, 32, 12, len(channels)), dtype=np.float32)
        mask = np.zeros((12, len(groups)), dtype=bool)
        for index in range(12):
            record = records[name, index]
            if record["missing_catalog_group"]:
                continue
            layer = name if index == 0 else f"{name}.{index}"
            directory = path / "layers" / layer / "_".join(BANDS[name])
            arr = reader.decode_raster(
                UPath(directory), projection, bounds, resampling=Resampling.nearest
            ).array
            if arr.shape != (len(channels), 1, 32, 32) or not np.isfinite(arr).all():
                raise ValueError("Invalid external raster")
            values[:, :, index, :] = arr[reorder, 0].transpose(1, 2, 0)
            # Match development file-availability masks. Pixel-level missingness
            # remains recorded in the raster audit, not silently filtered here.
            mask[index, :] = True
        if model_name == "sentinel1":
            values = 10 * np.log10(np.maximum(values, 1e-6))
        pixels[model_name] = normalizer.normalize(
            Modality.get(model_name), values
        ).astype(np.float32)
        valid[model_name] = mask
    return {"pixels": pixels, "valid": valid, "bounds": tuple(bounds)}
