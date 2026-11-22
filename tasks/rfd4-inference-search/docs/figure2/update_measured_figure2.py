"""Incrementally aggregate completed Figure 2 target cohorts into reproduced_targets.csv and render 4-panel measured plots."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

TASK = Path(__file__).resolve().parents[2]
OVERALL = TASK / "environment/references/overall"
sys.path.insert(0, str(OVERALL))
from benchmark_curve import ARMS, FIELDS, TRACKS, operating_points, read_targets
from measured_report import render


DATASET_TO_TRACK = {
    "ppi_backbone": ("Protein binding", "ppi"),
    "studio_figure_2c_unfixed": ("Small-molecule binding", "sm_binder"),
    "mcsa_fixed": ("Enzyme scaffolding", "ame"),
    "dna_unfixed": ("DNA binding", "dna"),
}


def load_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def sync_completed_suites(cohorts_dir: Path, mirror_measured_dir: Path | None = None) -> dict:
    # 1. Official target roster order from targets.csv
    author_rows = load_csv_rows(OVERALL / "targets.csv")
    roster_by_ds: dict[str, list[str]] = {}
    for r in author_rows:
        if r["method"] == "Low":
            roster_by_ds.setdefault(r["dataset"], []).append(r["target"])

    # 2. Existing pinned rows in reproduced_targets.csv
    reproduced_path = OVERALL / "reproduced_targets.csv"
    existing_rows = load_csv_rows(reproduced_path)
    by_suite_method: dict[tuple[str, str], list[dict]] = {}
    for r in existing_rows:
        by_suite_method.setdefault((r["dataset"], r["method"]), []).append(r)

    newly_completed = []
    for ds in ("ppi_backbone", "studio_figure_2c_unfixed", "mcsa_fixed", "dna_unfixed"):
        _, crit = DATASET_TO_TRACK[ds]
        targets = roster_by_ds[ds]
        for method in ARMS:
            if (ds, method) in by_suite_method and len(by_suite_method[(ds, method)]) == len(targets):
                continue
            done_files = [cohorts_dir / ds / method / tgt / "done.json" for tgt in targets]
            if not all(p.is_file() for p in done_files):
                continue
            suite_rows = []
            for tgt, dp in zip(targets, done_files, strict=True):
                d = json.loads(dp.read_text())
                if (d["dataset"], d["method"], d["target"], int(d["attempts"])) != (ds, method, tgt, 200):
                    raise ValueError(f"Cohort identity mismatch in {dp}")
                suite_rows.append({
                    "dataset": ds,
                    "criterion": crit,
                    "method": method,
                    "target": tgt,
                    "attempts": 200,
                    "scored": int(d["scored"]),
                    "passes": int(d["passes"]),
                    "clusters": int(d["clusters"]),
                    "generation_gpu_s": float(d["generation_gpu_s"]),
                    "af3_gpu_s": float(d["af3_gpu_s"]),
                })
            by_suite_method[(ds, method)] = suite_rows
            newly_completed.append(f"{ds}/{method}")

    # 3. Canonical ordering matching TRACKS x ARMS x official target order
    ordered_rows = []
    for ds in ("ppi_backbone", "studio_figure_2c_unfixed", "mcsa_fixed", "dna_unfixed"):
        for method in ARMS:
            if (ds, method) in by_suite_method:
                row_map = {r["target"]: r for r in by_suite_method[(ds, method)]}
                for tgt in roster_by_ds[ds]:
                    ordered_rows.append(row_map[tgt])

    write_csv_rows(reproduced_path, list(FIELDS), ordered_rows)

    # 4. Regenerate environment/references/overall/measured/{points.csv,coverage.json,generation.png,including-af3.png}
    measured_dir = OVERALL / "measured"
    stage_out = OVERALL / "measured_stage_tmp"
    if stage_out.exists():
        shutil.rmtree(stage_out)
    status = render(stage_out)
    for fn in ("points.csv", "coverage.json", "generation.png", "including-af3.png"):
        shutil.copy2(stage_out / fn, measured_dir / fn)
    shutil.rmtree(stage_out)

    # 5. Build direct side-by-side comparison CSV against author operating_points.csv
    author_pts = {(p["track"], p["method"]): p for p in operating_points(OVERALL / "targets.csv")}
    local_pts = load_csv_rows(measured_dir / "points.csv")
    comparison_rows = []
    for lp in local_pts:
        ap = author_pts[(lp["track"], lp["method"])]
        l_pct = float(lp["clustered_pct"])
        a_pct = float(ap["clustered_pct"])
        l_gen_h = float(lp["generation_attempts_h"])
        a_gen_h = float(ap["generation_attempts_h"])
        l_tot_h = float(lp["total_attempts_h"])
        a_tot_h = float(ap["total_attempts_h"])
        comparison_rows.append({
            "track": lp["track"],
            "method": lp["method"],
            "targets": int(lp["targets"]),
            "attempts": int(lp["attempts"]),
            "paper_passes": int(ap["passes"]),
            "paper_clusters": int(ap["clusters"]),
            "paper_clustered_pct": round(a_pct, 4),
            "local_passes": int(lp["passes"]),
            "local_clusters": int(lp["clusters"]),
            "local_clustered_pct": round(l_pct, 4),
            "rate_diff_pp": round(l_pct - a_pct, 4),
            "paper_gen_attempts_h": round(a_gen_h, 2),
            "local_warm_gen_attempts_h": round(l_gen_h, 2),
            "gen_speed_ratio_vs_h200": round(l_gen_h / a_gen_h, 4),
            "paper_total_attempts_h": round(a_tot_h, 2),
            "local_warm_total_attempts_h": round(l_tot_h, 2),
            "paper_total_clusters_h": round(float(ap["total_clusters_h"]), 3),
            "local_warm_total_clusters_h": round(float(lp["total_clusters_h"]), 3),
        })
    if comparison_rows:
        write_csv_rows(measured_dir / "comparison_vs_paper.csv", list(comparison_rows[0]), comparison_rows)

    # 6. Build complete 16-point (4 tracks x Low/Medium/High/Search) reference/baseline frontier table
    local_by_key = {(lp["track"], lp["method"]): lp for lp in local_pts}
    frontier_rows = []
    for track in TRACKS:
        for method in ARMS:
            ap = author_pts[(track, method)]
            lp = local_by_key.get((track, method))
            frontier_rows.append({
                "track": track,
                "method": method,
                "targets": int(ap["targets"]),
                "attempts": int(ap["attempts"]),
                "author_clustered_pct": round(float(ap["clustered_pct"]), 4),
                "author_gen_attempts_h": round(float(ap["generation_attempts_h"]), 2),
                "author_total_attempts_h": round(float(ap["total_attempts_h"]), 2),
                "author_total_clusters_h": round(float(ap["total_clusters_h"]), 3),
                "local_status": "reproduced" if lp is not None else "author_reference_only",
                "local_clustered_pct": round(float(lp["clustered_pct"]), 4) if lp is not None else "",
                "local_gen_attempts_h": round(float(lp["generation_attempts_h"]), 2) if lp is not None else "",
                "local_total_attempts_h": round(float(lp["total_attempts_h"]), 2) if lp is not None else "",
                "local_total_clusters_h": round(float(lp["total_clusters_h"]), 3) if lp is not None else "",
            })
    write_csv_rows(measured_dir / "baseline_frontier_reference.csv", list(frontier_rows[0]), frontier_rows)

    if mirror_measured_dir is not None:
        m_out = mirror_measured_dir.resolve()
        m_out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reproduced_path, m_out / "reproduced_targets.csv")
        for fn in (
            "points.csv", "coverage.json", "generation.png", "including-af3.png",
            "comparison_vs_paper.csv", "baseline_frontier_reference.csv",
        ):
            if (measured_dir / fn).is_file():
                shutil.copy2(measured_dir / fn, m_out / fn)

    return {"newly_completed": newly_completed, "coverage": status}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohorts-dir", type=Path, required=True)
    p.add_argument("--mirror-measured-dir", type=Path, default=None)
    args = p.parse_args()
    res = sync_completed_suites(args.cohorts_dir.resolve(), args.mirror_measured_dir)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
