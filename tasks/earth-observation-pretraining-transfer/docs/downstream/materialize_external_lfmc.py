"""Use official rslearn matching/materialization for fixed external site/date periods."""

import argparse
from datetime import timedelta
import hashlib
import json
from pathlib import Path

import rasterio
from rasterio.crs import CRS
from rslearn.config import QueryConfig
from rslearn.data_sources.planetary_computer import Sentinel1, Sentinel2
from rslearn.dataset import Dataset, Window
from rslearn.utils import Projection, STGeometry
from shapely.geometry import box
from upath import UPath

from external_lfmc_availability import BANDS, parse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--official-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    site = manifest["sites"][0]
    observation = site["observations"][0]
    probe = json.loads(args.probe.read_text())
    if probe["site_id"] != site["site_id"] or probe["date"] != observation["date"]:
        raise ValueError("Probe differs from predetermined candidate")
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.official_config.read_text())
    config["layers"] = {m: config["layers"][m] for m in ["sentinel1", "sentinel2"]}
    for layer in config["layers"].values():
        layer.pop(
            "data_source"
        )  # Explicit per-period calls below preserve missing slots.
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    dataset = Dataset(UPath(args.output))
    transform = probe["transform"]
    left, top = round(transform[2] / 10), round(-transform[5] / 10)
    bounds = (left, top, left + 63, top + 63)
    projection = Projection(CRS.from_string(probe["crs"]), 10, -10)
    end = parse(observation["date"])
    window = Window(
        dataset.storage,
        "external",
        "first-fixed-site-date",
        projection,
        bounds,
        (end, end),
        {"split": "external_unscored", "site_id": site["site_id"]},
    )
    window.save()
    sources = {
        "sentinel2": Sentinel2(
            harmonize=True,
            assets=BANDS["sentinel2"],
            sort_by="eo:cloud_cover",
            timeout=timedelta(seconds=60),
        ),
        "sentinel1": Sentinel1(
            band_names=BANDS["sentinel1"],
            query={
                "sar:instrument_mode": {"eq": "IW"},
                "sar:polarizations": {"eq": ["VV", "VH"]},
            },
            timeout=timedelta(seconds=60),
        ),
    }
    records = []
    item_groups = {m: [] for m in sources}
    time_ranges = []
    query = QueryConfig(space_mode="MOSAIC", max_matches=1, min_matches=0)
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_TIMEOUT="60",
        GDAL_HTTP_MAX_RETRY="2",
    ):
        for index in range(12):
            stop = end - timedelta(days=14 * index)
            start = stop - timedelta(days=14)
            time_ranges.append((start, stop))
            geometry = STGeometry(projection, box(*bounds), (start, stop))
            for modality, source in sources.items():
                groups = source.get_items([geometry], query)[0]
                name = modality if index == 0 else f"{modality}.{index}"
                record = {
                    "modality": modality,
                    "index": index,
                    "start": start.isoformat(),
                    "end": stop.isoformat(),
                    "layer_name": name,
                    "missing_catalog_group": not groups,
                }
                items = []
                if groups:
                    if len(groups) != 1:
                        raise ValueError("Unexpected mosaic group count")
                    items = groups[0].items
                    record["items"] = [item.serialize() for item in items]
                item_groups[modality].append(items)
                records.append(record)
                (args.output / "periods.json").write_text(
                    json.dumps(records, indent=2) + "\n"
                )
                print(
                    json.dumps({k: v for k, v in record.items() if k != "items"}),
                    flush=True,
                )
        for modality, source in sources.items():
            source.materialize(
                window,
                item_groups[modality],
                modality,
                dataset.layers[modality],
                group_time_ranges=time_ranges,
            )
            print(
                json.dumps({"materialized_modality": modality, "groups": 12}),
                flush=True,
            )
    report = {
        "site_id": site["site_id"],
        "date": observation["date"],
        "period_records": len(records),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "official_config_sha256": hashlib.sha256(
            args.official_config.read_bytes()
        ).hexdigest(),
        "target_used": False,
        "model_used": False,
        "scope": "One fixed site/date, twelve periods, official matcher/materializer and S2 callback. Probe grid; final crop/normalization parity remains unverified.",
        "explicit_deviation": "Independent14-day queries preserve gaps instead of whole-window min_matches=12 rejection.",
    }
    (args.output / "completion.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
