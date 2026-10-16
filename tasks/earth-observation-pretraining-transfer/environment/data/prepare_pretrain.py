"""Spatial exclusion and reproducible indexing of the official HDF5 subset."""

import argparse
import hashlib
import json
from pathlib import Path
import h5py
import hdf5plugin  # noqa
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--lfmc-annotations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-source-samples", type=int, default=3996)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    points = np.array(
        [
            f["geometry"]["coordinates"]
            for f in json.loads(args.lfmc_annotations.read_text())["features"]
        ]
    )
    # Conservative exclusion: whole southern Kenya bounding region for AWF,
    # 0.1 degree coordinate distance (~10 km) around every published LFMC point.
    tree = cKDTree(points)
    rows, coords, excluded, provenance = [], [], [], []
    files = sorted(
        args.source.glob("sample_*.h5"), key=lambda x: int(x.stem.split("_")[1])
    )
    if len(files) != args.expected_source_samples:
        raise ValueError(
            f"Expected {args.expected_source_samples} source subtiles, found {len(files)}"
        )
    for f in files:
        with h5py.File(f) as h:
            lat, lon = np.asarray(h["latlon"]).reshape(-1)
            if (32 <= lon <= 43 and -6 <= lat <= 6) or tree.query([lon, lat])[0] < 0.1:
                excluded.append(f.name)
                continue
            i = len(rows)
            row = {"sample_index": i}
            row.update(
                {
                    k: 1
                    for k in h
                    if k not in ("latlon", "timestamps", "missing_timesteps_masks")
                }
            )
            dst = args.output / f"sample_{i}.h5"
            if dst.exists() and dst.resolve() != f.resolve():
                raise ValueError(f"Stale sample mapping: {dst}")
            if not dst.exists():
                dst.symlink_to(f.resolve())
            rows.append(row)
            coords.append([float(lat), float(lon)])
            provenance.append(
                {"sample_index": i, "source": f.name, "latlon": coords[-1]}
            )
    pd.DataFrame(rows).fillna(0).astype(int).to_csv(
        args.output / "sample_metadata.csv", index=False
    )
    np.save(args.output / "latlon_distribution.npy", np.array(coords))
    report = {
        "source": str(args.source),
        "source_release": json.loads((args.source / "source.json").read_text())
        if (args.source / "source.json").exists()
        else None,
        "source_samples": len(files),
        "kept": len(rows),
        "excluded": excluded,
        "samples": provenance,
        "lfmc_annotations_sha256": hashlib.sha256(
            args.lfmc_annotations.read_bytes()
        ).hexdigest(),
        "exclusion": "AWF bbox lon32..43 lat-6..6; LFMC Euclidean lonlat radius0.1deg, all labels",
    }
    (args.output / "provenance.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("samples", "excluded")}
        )
    )


if __name__ == "__main__":
    main()
