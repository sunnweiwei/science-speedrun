"""Extract the official RFD4 arms and check measured points against separate publication tables."""
import argparse
import csv
import hashlib
import io
import math
from pathlib import Path
import tempfile
from urllib.request import urlopen

from benchmark_curve import ARMS, FIELDS, HERE, TRACKS, operating_points, write_points

REVISION = "6306a210dd17dcae8e4e0f381b5b74a719d9b30e"
BASE = f"https://raw.githubusercontent.com/RosettaCommons/RFD4-Proteina/{REVISION}/website/benchmarks/"
HASHES = {
    "source_data.csv": "2652e4ac1703898975e0eb5f4c73a55a9248e85106279dec962adb3df59ddbc5",
    "scatter.csv": "28523b5b133034a40cbdcae0f159061cd263a80f3f514e90e057361e6d42ddb4",
}
# V1 supplement, Table S29, in the printed precision: pass %, generation-only and AF3-inclusive yield.
S29 = {
    "Protein binding": {"High": (10.6, 120.7, 66.5), "Search": (14.9, 8.8, 8.4)},
    "Small-molecule binding": {"High": (14.3, 787.4, 237.1), "Search": (20.9, 23.0, 22.0)},
    "Enzyme scaffolding": {"High": (12.1, 453.4, 155.2), "Search": (16.6, 14.0, 13.5)},
    "DNA binding": {"High": (3.5, 140.6, 52.3), "Search": (4.8, 5.4, 5.1)},
}


def checked_table(filename, source_dir):
    if source_dir:
        data = (source_dir / filename).read_bytes()
    else:
        with urlopen(BASE + filename, timeout=60) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != HASHES[filename]:
        raise ValueError(f"Unexpected upstream digest for {filename}")
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def upstream_key(track, method):
    if method == "Search":
        model = "rfd4-search-multimer" if track == "Protein binding" else "rfd4_proteina_high_search"
        return model, "native_pipeline"
    return "rfd4_proteina_" + method.lower(), "one_shot"


def verify(source_dir, refresh):
    source = checked_table("source_data.csv", source_dir)
    scatter = checked_table("scatter.csv", source_dir)
    with tempfile.TemporaryDirectory() as tmp:
        targets = Path(tmp) / "targets.csv"
        with targets.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            for track, (dataset, criterion, _) in TRACKS.items():
                for method in ARMS:
                    model, protocol = upstream_key(track, method)
                    selected = [r for r in source if (r["dataset"], r["criterion"], r["model"], r["protocol"])
                                == (dataset, criterion, model, protocol)]
                    for raw in sorted(selected, key=lambda r: r["target"]):
                        row = dict(raw, method=method)
                        writer.writerow({key: row[key] for key in FIELDS})
        points = operating_points(targets)
        for p in points:
            track, method = p["track"], p["method"]
            model, protocol = upstream_key(track, method)
            golden, = (r for r in scatter if (r["model"], r["protocol"], r["benchmark"])
                       == (model, protocol, TRACKS[track][0]))
            for key, col in (("targets", "targets"), ("attempts", "attempts"), ("scored", "resolved"),
                             ("passes", "passes"), ("clusters", "clusters"), ("covered_targets", "solved"),
                             ("pass_pct", "pass_percent"), ("clustered_pct", "clustered_percent"),
                             ("generation_attempts_h", "designs_per_gpu_hour"),
                             ("generation_clusters_h", "mean_unique_passers_per_gpu_hour"),
                             ("total_clusters_h", "mean_unique_passers_per_gpu_hour_including_af3")):
                if not math.isclose(p[key], float(golden[col]), rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(f"{track}/{method} disagrees with scatter.csv on {key}")
            if method in S29[track]:
                values = [p[k] for k in ("clustered_pct", "generation_clusters_h", "total_clusters_h")]
                if tuple(round(v, 1) for v in values) != S29[track][method]:
                    raise ValueError(f"{track}/{method} disagrees with supplementary Table S29")
        summary = Path(tmp) / "operating_points.csv"
        with summary.open("w", newline="") as f:
            write_points(points, f)
        for path in (targets, summary):
            bundled = HERE / path.name
            if refresh:
                bundled.write_bytes(path.read_bytes())
            elif bundled.read_bytes() != path.read_bytes():
                raise ValueError(f"{path.name} differs from the checked upstream reconstruction")
    print("Verified 564 target records, 16 RFD4 operating points and all 24 numerical cells of Table S29.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, help="Directory with pinned source_data.csv and scatter.csv")
    p.add_argument("--refresh", action="store_true", help="Replace bundled CSVs after verification")
    args = p.parse_args()
    verify(args.source_dir, args.refresh)
