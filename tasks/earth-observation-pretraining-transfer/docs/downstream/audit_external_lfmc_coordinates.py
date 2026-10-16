"""Audit candidate-to-development coordinates without selecting on outcomes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pyproj import Geod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sites = json.loads(args.manifest.read_text())["sites"]
    sources = {}
    for task in ["awf", "lfmc"]:
        path = args.data_root / "eval" / task / "labels.json"
        # Only coordinates/splits are selected; never inspect the numeric label field.
        rows = json.loads(path.read_text())
        coordinates = np.asarray(
            [r["latlon"] for r in rows if r["split"] in ["train", "val"]], dtype=float
        )
        sources[task + "_development"] = (path, coordinates)
    for name in ["pretrain-two-shards-filtered", "pretrain-filtered"]:
        path = args.data_root / name / "latlon_distribution.npy"
        sources[name] = (path, np.load(path))
    geod = Geod(ellps="WGS84")
    results = {}
    for name, (path, coordinates) in sources.items():
        if (
            coordinates.ndim != 2
            or coordinates.shape[1] != 2
            or not np.isfinite(coordinates).all()
        ):
            raise ValueError("Invalid coordinate source")
        unique = np.unique(coordinates, axis=0)
        records = []
        for site in sites:
            lat, lon = site["latitude"], site["longitude"]
            _, _, distance = geod.inv(
                np.full(len(unique), lon),
                np.full(len(unique), lat),
                unique[:, 1],
                unique[:, 0],
            )
            index = int(np.argmin(distance))
            records.append(
                {
                    "site_id": site["site_id"],
                    "nearest_distance_m": float(distance[index]),
                    "nearest_latlon": unique[index].tolist(),
                    "minimum_euclidean_degrees": float(
                        np.linalg.norm(unique - [lat, lon], axis=1).min()
                    ),
                }
            )
        results[name] = {
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_rows": len(coordinates),
            "unique_coordinates": len(unique),
            "minimum_geodesic_distance_m": min(
                r["nearest_distance_m"] for r in records
            ),
            "exact_coordinate_matches": sum(
                r["nearest_distance_m"] < 1e-6 for r in records
            ),
            "sites": records,
        }
    report = {
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "site_count": len(sites),
        "sources": results,
        "numeric_targets_inspected": False,
        "cohort_filtered": False,
        "full_footprint_nonoverlap_proven": False,
        "scope": "Geodesic distances to development coordinates and known pretraining centers. Full pretraining raster bounds are not available in these coordinate files.",
    }
    with args.output.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(
        json.dumps(
            {
                k: {n: v for n, v in r.items() if n != "sites"}
                for k, r in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
