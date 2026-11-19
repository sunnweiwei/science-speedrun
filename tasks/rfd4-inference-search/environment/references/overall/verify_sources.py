"""Extract the official RFD4 arms and check measured points against separate publication tables."""
import argparse
import csv
import hashlib
import io
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path
import tempfile
from urllib.request import urlopen

from benchmark_curve import ARMS, FIELDS, HERE, TRACKS, Measurement, frontier, operating_points, summarize, write_points

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


def checked_table(filename, source_dir, archive=None):
    if source_dir:
        data = (source_dir / filename).read_bytes()
    else:
        with urlopen(BASE + filename, timeout=60) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != HASHES[filename]:
        raise ValueError(f"Unexpected upstream digest for {filename}")
    if archive is not None:
        (archive / filename).write_bytes(data)
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def upstream_key(track, method):
    if method == "Search":
        model = "rfd4-search-multimer" if track == "Protein binding" else "rfd4_proteina_high_search"
        return model, "native_pipeline"
    return "rfd4_proteina_" + method.lower(), "one_shot"


METRICS = {
    "targets": "targets", "attempts": "attempts", "scored": "resolved", "passes": "passes",
    "clusters": "clusters", "covered_targets": "solved", "pass_pct": "pass_percent",
    "clustered_pct": "clustered_percent", "generation_attempts_h": "designs_per_gpu_hour",
    "generation_clusters_h": "mean_unique_passers_per_gpu_hour",
    "total_clusters_h": "mean_unique_passers_per_gpu_hour_including_af3",
}
PANELS = {"ppi": "Protein binding", "small_molecule": "Small-molecule binding",
          "mcsa_fixed": "Enzyme scaffolding", "dna_unfixed": "DNA binding"}


def check_point(point, golden, allow_missing_total=False):
    for key, col in METRICS.items():
        if allow_missing_total and not golden[col] and key == "total_clusters_h":
            continue  # Some comparator rates can only be checked in the target-level export.
        if not math.isclose(point[key], float(golden[col]), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{point['track']}/{point['method']} disagrees with scatter.csv on {key}")


def comparison_field(source, scatter):
    points = []
    for golden in scatter:
        track = PANELS[golden["dataset"]]
        reference_dataset, criterion, n = TRACKS[track]
        model = golden["model"].split("__")[0]  # Plot labels can carry conditioning suffixes.
        rows = [r for r in source if (r["model"], r["protocol"], r["dataset"], r["criterion"]) ==
                (model, golden["protocol"], golden["benchmark"], criterion)]
        roster = {r["target"] for r in source if (r["model"], r["dataset"], r["criterion"]) ==
                  ("rfd4_proteina_low", reference_dataset, criterion)}
        if len(rows) != n or len(roster) != n or {r["target"] for r in rows} != roster:
            raise ValueError(f"Incomplete comparator roster: {track}/{model}")
        measurements = {}
        for r in rows:
            a, s, p, c = (int(r[k]) for k in FIELDS[4:8])
            g, f = float(r["generation_gpu_s"]), float(r["af3_gpu_s"])
            if not (0 <= c <= p <= s <= a == 200 and math.isfinite(g) and g > 0 and math.isfinite(f)
                    and f >= 0 and (not p or f > 0)):
                raise ValueError(f"Invalid comparator record: {track}/{model}/{r['target']}")
            measurements[r["target"]] = Measurement(a, s, p, c, g, f)
        method = " ".join(golden["display_label"].split()).replace("Proteina- Complexa", "Proteina-Complexa")
        point = summarize(track, method, measurements, "author archive")
        check_point(point, golden, allow_missing_total=not model.startswith("rfd4"))
        point.update(model_id=golden["model"], protocol=golden["protocol"], input_condition=golden["benchmark"],
                     figure2_displayed=golden["displayed"] == "True", is_rfd4=model.startswith("rfd4"),
                     total_yield_in_plot_export=bool(golden[METRICS["total_clusters_h"]]),
                     generation_proxy_attempts_reported=(int(float(golden["generation_proxy_attempts"]))
                                                         if golden["generation_proxy_attempts"] else None),
                     timing_basis=" | ".join(sorted({r["timing_basis"] for r in rows})))
        points.append(point)
    for track in TRACKS:
        group = [p for p in points if p["track"] == track]
        best = frontier(group, "generation_attempts_h")
        for p in group:
            p["field_generation_pareto"] = p in best
    return points


def reconstruct(source, scatter, directory):
    targets = directory / "targets.csv"
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
        check_point(p, golden)
        if method in S29[track]:
            values = [p[k] for k in ("clustered_pct", "generation_clusters_h", "total_clusters_h")]
            if tuple(round(v, 1) for v in values) != S29[track][method]:
                raise ValueError(f"{track}/{method} disagrees with supplementary Table S29")
    summary = directory / "operating_points.csv"
    with summary.open("w", newline="") as f:
        write_points(points, f)
    return points


def export_replay(directory, points, field):
    from plot_curve import plot
    peers = [p for p in field if p["figure2_displayed"] and not p["is_rfd4"]]
    plot(points, directory / "quality-throughput.png")
    plot(points, directory / "quality-throughput-af3.png", include_af3=True)
    plot(points, directory / "quality-throughput-with-comparators.png", peers=peers)
    metrics = {p["track"]: p["method"] for p in points if p["highest_total_yield"]}
    receipt = {
        "evidence": "Recalculated from published target records. No fresh model or AF3 runs.",
        "upstream_revision": REVISION, "source_sha256": HASHES,
        "build_software": {"python": platform.python_version(), "matplotlib": version("matplotlib")},
        "code_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                        for name in ("verify_sources.py", "benchmark_curve.py", "plot_curve.py")},
        "figure2_displayed_points": sum(p["figure2_displayed"] for p in field),
        "s29_search_overlays": sum(not p["figure2_displayed"] for p in field),
        "rfd4_points": len(points), "rfd4_target_records": sum(p["targets"] for p in points),
        "original_rfd4_attempts_represented": sum(p["attempts"] for p in points),
        "best_rfd4_end_to_end_yield": metrics,
        "generation_pareto_methods": {
            track: [p["method"] for p in field if p["track"] == track and p["field_generation_pareto"]]
            for track in TRACKS},
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(directory.iterdir()) if p.is_file()},
    }
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def verify(source_dir, refresh=False, rebuild=None):
    if rebuild is not None:
        if refresh:
            raise ValueError("Choose either a clean rebuild or a reference refresh")
        rebuild = Path(rebuild).resolve()
        if rebuild.exists():
            raise FileExistsError(f"Choose a new output directory: {rebuild}")
        rebuild.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=rebuild.parent if rebuild else None) as tmp:
        directory = Path(tmp) / "figure"
        directory.mkdir()
        archive = directory / "inputs" if rebuild else None
        if archive is not None:
            archive.mkdir()
        source = checked_table("source_data.csv", source_dir, archive)
        scatter = checked_table("scatter.csv", source_dir, archive)
        points = reconstruct(source, scatter, directory)
        field = comparison_field(source, scatter)
        with (directory / "comparison_field.csv").open("w", newline="") as f:
            write_points(field, f)
        if rebuild:
            export_replay(directory, points, field)
            directory.rename(rebuild)
            print(f"Rebuilt the RFD4 curves and checked all {len(scatter)} published comparison points in {rebuild}")
        else:
            for path in directory.iterdir():
                bundled = HERE / path.name
                if refresh:
                    bundled.write_bytes(path.read_bytes())
                elif bundled.read_bytes() != path.read_bytes():
                    raise ValueError(f"{path.name} differs from the checked upstream reconstruction")
    print("Verified 564 target records, 16 RFD4 operating points and all 24 numerical cells of Table S29.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, help="Directory with pinned source_data.csv and scatter.csv")
    output = p.add_mutually_exclusive_group()
    output.add_argument("--refresh", action="store_true", help="Replace bundled CSVs after verification")
    output.add_argument("--rebuild", type=Path, help="Build all figures and a hash receipt in a fresh directory")
    args = p.parse_args()
    verify(args.source_dir, args.refresh, args.rebuild)
