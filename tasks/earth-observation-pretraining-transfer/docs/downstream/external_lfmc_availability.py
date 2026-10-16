"""Metadata-only STAC coverage audit; never filters the fixed candidate cohort."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time

import requests
from shapely.geometry import Point, shape

ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
COLLECTIONS = {"sentinel2": "sentinel-2-l2a", "sentinel1": "sentinel-1-rtc"}
BANDS = {
    "sentinel1": ["vv", "vh"],
    "sentinel2": [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    ],
}


def parse(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(
        tzinfo=timezone.utc
    )


def query(session, site, modality):
    dates = [parse(o["date"]) for o in site["observations"]]
    lon, lat = site["longitude"], site["latitude"]
    point = Point(lon, lat)
    params = {
        "collections": COLLECTIONS[modality],
        "bbox": f"{lon - 0.00001},{lat - 0.00001},{lon + 0.00001},{lat + 0.00001}",
        "datetime": f"{(min(dates) - timedelta(days=168)).isoformat()}/{max(dates).isoformat()}",
        "limit": 500,
    }
    if modality == "sentinel1":
        params["query"] = json.dumps(
            {
                "sar:instrument_mode": {"eq": "IW"},
                "sar:polarizations": {"eq": ["VV", "VH"]},
            }
        )
    url, method, body = ENDPOINT, "GET", None
    items, seen_pages = {}, set()
    while url:
        key = json.dumps([url, method, params, body], sort_keys=True)
        if key in seen_pages:
            raise ValueError("Repeated STAC page")
        seen_pages.add(key)
        for attempt in range(4):
            try:
                response = session.request(
                    method, url, params=params, json=body, timeout=60
                )
                response.raise_for_status()
                page = response.json()
                break
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        for item in page["features"]:
            if not shape(item["geometry"]).covers(point):
                continue
            prop = item["properties"]
            items[item["id"]] = {
                "id": item["id"],
                "datetime": prop["datetime"],
                "assets_complete": all(b in item["assets"] for b in BANDS[modality]),
                "cloud_cover": prop.get("eo:cloud_cover"),
            }
        links = [v for v in page.get("links", []) if v["rel"] == "next"]
        if len(links) > 1:
            raise ValueError("Ambiguous pagination")
        if not links:
            break
        link = links[0]
        url, method, params, body = (
            link["href"],
            link.get("method", "GET"),
            None,
            link.get("body"),
        )
    periods = []
    for observation in site["observations"]:
        end = parse(observation["date"])
        start = end - timedelta(days=168)
        counts = []
        for period in range(12):
            a = start + timedelta(days=14 * period)
            b = a + timedelta(days=14)
            counts.append(
                sum(
                    i["assets_complete"] and a <= parse(i["datetime"]) < b
                    for i in items.values()
                )
            )
        periods.append({"date": observation["date"], "items_per_period": counts})
    return {
        "items": sorted(items.values(), key=lambda i: (i["datetime"], i["id"])),
        "periods": periods,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    reports = []
    for site in manifest["sites"]:
        result = {"site_id": site["site_id"], "modalities": {}}
        for modality in COLLECTIONS:
            try:
                result["modalities"][modality] = query(session, site, modality)
            except Exception as exc:
                # An acquisition error must never be mistaken for genuine missingness.
                result["modalities"][modality] = {
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                }
        filename = hashlib.sha256(site["site_id"].encode()).hexdigest() + ".json"
        (args.output / filename).write_text(json.dumps(result, indent=2) + "\n")
        reports.append(
            {
                "site_id": site["site_id"],
                "report": filename,
                "errors": [m for m, r in result["modalities"].items() if "error" in r],
            }
        )
        print(json.dumps(reports[-1]), flush=True)
    summary = {
        "manifest_sha256": manifest_hash,
        "sites": reports,
        "site_count": len(reports),
        "sites_with_errors": sum(bool(r["errors"]) for r in reports),
        "target_used": False,
        "predictions_used": False,
        "cohort_filtered": False,
        "limitations": "Point-level STAC metadata only; half-open chronological 14-day bins. Not raster coverage, valid pixels, or verified rslearn mosaic alignment.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
