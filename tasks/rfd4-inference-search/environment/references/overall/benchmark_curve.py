"""Figure 2 / S29 scoring for frozen, controller-owned target measurements."""
import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import fmean
import sys

HERE = Path(__file__).resolve().parent
# Exact Figure 2 RFD4 conditioning arms, including the newer small-molecule pass definition.
TRACKS = {
    "Protein binding": ("ppi_backbone", "ppi", 24),
    "Small-molecule binding": ("studio_figure_2c_unfixed", "sm_binder", 77),
    "Enzyme scaffolding": ("mcsa_fixed", "ame", 37),
    "DNA binding": ("dna_unfixed", "dna", 3),
}
ARMS = ("Low", "Medium", "High", "Search")
FIELDS = ("dataset", "criterion", "method", "target", "attempts", "scored", "passes",
          "clusters", "generation_gpu_s", "af3_gpu_s")


@dataclass(frozen=True)
class Measurement:
    attempts: int
    scored: int
    passes: int
    clusters: int
    generation: float
    af3: float


TRIAL_FIELDS = ("dataset", "criterion", "method", "target", "attempt_id", "scored", "passed",
                "cluster_id", "generation_gpu_s", "af3_gpu_s")


NATIVE_STATUS_FIELDS = ("pre_oracle_pass", "all_metrics_present")


def validate_native_status(row, scored, passed):
    pre, complete = (float(row[k]) for k in NATIVE_STATUS_FIELDS)
    if pre not in (0, 1) or complete not in (0, 1):
        raise ValueError("Invalid native scorer status")
    if pre and not complete:
        raise ValueError("Indeterminate final metrics after passing the native filter")
    if (passed and not pre) or (pre and complete and not scored):
        raise ValueError("Inconsistent native pass decision")


def read_trials(path):
    """Reduce an external evaluator's sealed pass decisions, cluster IDs and cost ledger."""
    tracks = {(d, c): key for key, (d, c, _) in TRACKS.items()}
    trials = {}
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        header = tuple(reader.fieldnames or ())
        if header not in (TRIAL_FIELDS, TRIAL_FIELDS + NATIVE_STATUS_FIELDS):
            raise ValueError(f"{path}: invalid attempt columns")
        native_status = header == TRIAL_FIELDS + NATIVE_STATUS_FIELDS
        for row in reader:
            track = tracks.get((row["dataset"], row["criterion"]))
            if track is None or None in row or None in row.values():
                raise ValueError(f"{path}:{reader.line_num}: incomplete record or unexpected cohort")
            method, target, attempt = (row[k] for k in ("method", "target", "attempt_id"))
            if any(not x.strip() for x in (method, target, attempt)) or method.strip() != method:
                raise ValueError(f"{path}:{reader.line_num}: missing method, target or attempt ID")
            group = trials.setdefault((track, method, target), {})
            if attempt in group:
                raise ValueError(f"{path}:{reader.line_num}: duplicate original attempt")
            scored, passed = int(row["scored"]), int(row["passed"])
            cluster = row["cluster_id"].strip()
            if scored not in (0, 1) or passed not in (0, 1) or passed > scored or (passed and not cluster):
                raise ValueError(f"{path}:{reader.line_num}: invalid adjudication or missing cluster ID")
            if native_status:
                validate_native_status(row, scored, passed)
            gen, af3 = float(row["generation_gpu_s"]), float(row["af3_gpu_s"])
            if not all(math.isfinite(v) and v >= 0 for v in (gen, af3)):
                raise ValueError(f"{path}:{reader.line_num}: missing or invalid GPU time")
            group[attempt] = (scored, passed, cluster, gen, af3)
    groups = {}
    for (track, method, target), attempts in trials.items():
        if len(attempts) != 200:
            raise ValueError(f"{track}/{method}/{target}: expected 200 original attempts")
        scores, passes, clusters, g, f = zip(*attempts.values())
        distinct = {c for ok, c in zip(passes, clusters) if ok}
        m = Measurement(200, sum(scores), sum(passes), len(distinct), math.fsum(g), math.fsum(f))
        if m.generation <= 0 or (m.passes and m.af3 <= 0):
            raise ValueError(f"{track}/{method}/{target}: incomplete cohort timing")
        groups.setdefault((track, method), {})[target] = m
    return groups


def read_targets(path):
    tracks = {(d, c): key for key, (d, c, _) in TRACKS.items()}
    groups = {}
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"{path}: expected columns {', '.join(FIELDS)}")
        for row in reader:
            if None in row or None in row.values():
                raise ValueError(f"{path}:{reader.line_num}: incomplete measurement")
            track = tracks.get((row["dataset"], row["criterion"]))
            method, target = row["method"], row["target"]
            if track is None or not method or method.strip() != method or not target.strip():
                raise ValueError(f"{path}:{reader.line_num}: invalid cohort, method or target")
            group = groups.setdefault((track, method), {})
            if target in group:
                raise ValueError(f"{path}:{reader.line_num}: duplicate target {target}")
            a, s, p, c = (int(row[k]) for k in FIELDS[4:8])
            g, v = (float(row[k]) for k in FIELDS[8:])
            if not (0 <= c <= p <= s <= a == 200):
                raise ValueError(f"{path}:{reader.line_num}: invalid counts")
            if not (math.isfinite(g) and g > 0 and math.isfinite(v) and v >= 0 and (p == 0 or v > 0)):
                raise ValueError(f"{path}:{reader.line_num}: incomplete GPU timing")
            group[target] = Measurement(a, s, p, c, g, v)
    return groups


def validate_rosters(groups, methods, reference, tracks=TRACKS):
    if not methods or set(groups) != {(track, method) for track in tracks for method in methods}:
        raise ValueError("Each method needs a complete cohort on every selected benchmark")
    for (track, method), targets in groups.items():
        if len(targets) != TRACKS[track][2] or targets.keys() != reference[track, "Low"].keys():
            raise ValueError(f"{track} / {method}: incomplete or different target roster")


def summarize(track, method, by_target, source):
    rows = list(by_target.values())
    attempts = sum(r.attempts for r in rows)
    generation = math.fsum(r.generation for r in rows)
    total = generation + math.fsum(r.af3 for r in rows)
    return dict(
        track=track, method=method, source=source, targets=len(rows), attempts=attempts,
        scored=sum(r.scored for r in rows), passes=sum(r.passes for r in rows),
        clusters=sum(r.clusters for r in rows), covered_targets=sum(r.passes > 0 for r in rows),
        pass_pct=fmean(100 * r.passes / r.attempts for r in rows),
        clustered_pct=fmean(100 * r.clusters / r.attempts for r in rows),
        generation_attempts_h=3600 * attempts / generation, total_attempts_h=3600 * attempts / total,
        generation_clusters_h=fmean(3600 * r.clusters / r.generation for r in rows),
        total_clusters_h=fmean(3600 * r.clusters / (r.generation + r.af3) for r in rows),
    )


def frontier(points, speed):
    """Measured nondominated points for one track; identical observations share a rank."""
    return [p for p in points if not any(
        q[speed] >= p[speed] and q["clustered_pct"] >= p["clustered_pct"]
        and (q[speed] > p[speed] or q["clustered_pct"] > p["clustered_pct"])
        for q in points)]


def operating_points(reference_file, observations=None, trials=None, only_track=None):
    groups = read_targets(reference_file)
    validate_rosters(groups, ARMS, groups)
    if only_track is not None and only_track not in TRACKS:
        raise ValueError(f"Unknown benchmark: {only_track}")
    tracks = (only_track,) if only_track is not None else tuple(TRACKS)
    methods = list(ARMS)
    if observations is not None and trials is not None:
        raise ValueError("Choose one input measurement level")
    if observations is not None or trials is not None:
        more = read_trials(trials) if trials is not None else read_targets(observations)
        more = {key: cohort for key, cohort in more.items() if key[0] in tracks}
        names = sorted({method for _, method in more})
        if set(names).intersection(ARMS):
            raise ValueError("Observation names must be distinct from the four official arms")
        validate_rosters(more, names, groups, tracks)
        groups.update(more)
        methods.extend(names)
    points = []
    for track in tracks:
        track_points = [summarize(track, m, groups[track, m], "author archive" if m in ARMS else "supplied observations")
                        for m in methods]
        reference_points = [p for p in track_points if p["method"] in ARMS]
        for name, axis in (("generation_pareto", "generation_attempts_h"), ("total_pareto", "total_attempts_h")):
            keep = {p["method"] for p in frontier(track_points, axis)}
            for p in track_points:
                p[name] = p["method"] in keep
                comparable = [q["clustered_pct"] for q in reference_points if q[axis] >= p[axis]]
                margin = p["clustered_pct"] - max(comparable) if comparable else None
                p[f"quality_margin_pp_at_{axis}"] = margin
        reference_yield = max(p["total_clusters_h"] for p in reference_points)
        best_yield = max(p["total_clusters_h"] for p in track_points)
        best_rate = max(p["clustered_pct"] for p in track_points)
        for p in track_points:
            p["yield_gain_pct_vs_best_reference"] = 100 * (p["total_clusters_h"] / reference_yield - 1)
            p["highest_total_yield"] = p["total_clusters_h"] == best_yield
            p["highest_clustered_rate"] = p["clustered_pct"] == best_rate
        points.extend(track_points)
    return points


def write_points(points, f):
    writer = csv.DictWriter(f, fieldnames=list(points[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--observations", type=Path, help="Optional trusted per-target results for additional methods")
    inputs.add_argument("--trials", type=Path, help="Trusted attempt-level scores, cluster assignments and timings")
    parser.add_argument("--track", choices=tuple(TRACKS), help="Score a single complete suite")
    parser.add_argument("--output", type=Path, help="Output CSV, defaults to stdout")
    parser.add_argument("--plot", type=Path, help="Save a plot of the selected suites, e.g. as PNG or PDF")
    parser.add_argument("--include-af3", action="store_true", help="Include common AF3 in the plot's throughput axis")
    args = parser.parse_args()
    points = operating_points(HERE / "targets.csv", args.observations, args.trials, only_track=args.track)
    if args.output:
        with args.output.open("w", newline="") as f:
            write_points(points, f)
    else:
        write_points(points, sys.stdout)
    if args.plot:
        from plot_curve import plot
        plot(points, args.plot, args.include_af3)


if __name__ == "__main__":
    main()
