"""Score and plot independent DNA-binding benchmark cohorts against the official Figure 2 archive."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "environment/references/overall"))
from benchmark_curve import operating_points, write_points


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def collect_from_campaign(campaign_dir: Path, targets_csv: Path, out_dir: Path):
    warm_rows, cold_rows, target_rows = [], [], []
    with targets_csv.open() as f:
        author_targets = {
            (r["method"], r["target"]): r
            for r in csv.DictReader(f)
            if r["dataset"] == "dna_unfixed"
        }

    for method in ("Low", "Medium", "High"):
        for target in ("7m5w", "7n5u", "7rte"):
            w_dir = campaign_dir / method / target
            with (w_dir / "trials_warm.csv").open() as f:
                warm_rows.extend(csv.DictReader(f))
            with (w_dir / "trials_cold.csv").open() as f:
                cold_rows.extend(csv.DictReader(f))
            d = json.loads((w_dir / "done.json").read_text())
            ar = author_targets[(method, target)]
            target_rows.append({
                "method": method,
                "target": target,
                "attempts": d["attempts"],
                "author_passes": int(ar["passes"]),
                "author_clusters": int(ar["clusters"]),
                "author_clustered_pct": 100.0 * int(ar["clusters"]) / 200,
                "local_scored": d["scored"],
                "local_passes": d["passes"],
                "local_clusters": d["passing_clusters"],
                "local_clustered_pct": d["clustered_pass_pct"],
                "difference_pp": d["clustered_pass_pct"] - (100.0 * int(ar["clusters"]) / 200),
                "author_generation_gpu_s": float(ar["generation_gpu_s"]),
                "author_af3_gpu_s": float(ar["af3_gpu_s"]),
                "local_warm_generation_gpu_s": d["warm_generation_gpu_s"],
                "local_warm_af3_gpu_s": d["warm_af3_gpu_s"],
                "local_cold_generation_wall_s": d["cold_generation_wall_s"],
                "local_cold_eval_wall_s": d["cold_eval_wall_s"],
                "af3_warmup_compile_s": d["af3_warmup_compile_s"],
            })

    write_csv(out_dir / "trials_warm.csv", warm_rows)
    write_csv(out_dir / "trials_cold.csv", cold_rows)
    write_csv(out_dir / "trials_all.csv", warm_rows + cold_rows)
    write_csv(out_dir / "target_breakdown.csv", target_rows)


def render_dna_reports_and_plots(targets_csv: Path, trials_warm_csv: Path, trials_all_csv: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    pts_all = operating_points(targets_csv, trials=trials_all_csv, only_track="DNA binding")
    with (out_dir / "operating_points_comparison.csv").open("w", newline="") as f:
        write_points(pts_all, f)

    by_method = {p["method"]: p for p in pts_all}
    side_by_side = []
    for m in ("Low", "Medium", "High"):
        auth = by_method[m]
        lw = by_method[f"Local {m} (warm)"]
        lc = by_method[f"Local {m} (cold)"]
        side_by_side.append({
            "setting": f"DNA-{m}",
            "attempts": 600,
            "author_clusters": auth["clusters"],
            "author_clustered_pct": auth["clustered_pct"],
            "local_passes": lw["passes"],
            "local_clusters": lw["clusters"],
            "local_clustered_pct": lw["clustered_pct"],
            "rate_diff_pp": lw["clustered_pct"] - auth["clustered_pct"],
            "author_gen_attempts_h": auth["generation_attempts_h"],
            "local_warm_gen_attempts_h": lw["generation_attempts_h"],
            "local_cold_gen_attempts_h": lc["generation_attempts_h"],
            "warm_gen_speed_ratio_vs_author": lw["generation_attempts_h"] / auth["generation_attempts_h"],
            "author_total_attempts_h": auth["total_attempts_h"],
            "local_warm_total_attempts_h": lw["total_attempts_h"],
            "local_cold_total_attempts_h": lc["total_attempts_h"],
            "warm_total_speed_ratio_vs_author": lw["total_attempts_h"] / auth["total_attempts_h"],
            "author_total_clusters_h": auth["total_clusters_h"],
            "local_warm_total_clusters_h": lw["total_clusters_h"],
            "local_cold_total_clusters_h": lc["total_clusters_h"],
        })
    write_csv(out_dir / "side_by_side_summary.csv", side_by_side)

    pts_warm = operating_points(targets_csv, trials=trials_warm_csv, only_track="DNA binding")
    for inc_af3, fname in ((False, "quality-throughput-dna.png"), (True, "quality-throughput-dna-af3.png")):
        speed_key = "total_attempts_h" if inc_af3 else "generation_attempts_h"
        fig, ax = plt.subplots(1, 1, figsize=(10.2, 5.8))
        fig.subplots_adjust(left=0.10, right=0.97, bottom=0.16, top=0.73)
        fig.text(0.06, 0.945, "RFD4-Proteina   /   DNA binding quality and throughput",
                 fontsize=19, weight="bold", color="#162b36")
        scope = "Generation + common AF3 evaluation" if inc_af3 else "Generation including internal search"
        fig.text(0.062, 0.898, f"{scope}    ·    3 targets × 200 attempts per setting",
                 fontsize=10.8, color="#445361")

        auth_pts = [p for p in pts_warm if p["source"] == "author archive"]
        loc_pts = [p for p in pts_warm if p["source"] != "author archive"]
        ax.plot([p[speed_key] for p in auth_pts], [p["clustered_pct"] for p in auth_pts],
                ":", lw=1.4, color="#939ea4", zorder=1, label="Author H200 frontier")
        ax.plot([p[speed_key] for p in loc_pts], [p["clustered_pct"] for p in loc_pts],
                "--", lw=1.3, color="#b5bfc4", zorder=1, label="Local RTX A6000 curve")

        auth_styles = {"Low": ("#198572", "o"), "Medium": ("#2674aa", "s"), "High": ("#b06b16", "^"), "Search": ("#803f87", "D")}
        loc_styles = {"Local Low (warm)": ("#198572", "P"), "Local Medium (warm)": ("#2674aa", "X"), "Local High (warm)": ("#b06b16", "v")}
        for p in auth_pts:
            c, mk = auth_styles[p["method"]]
            ax.scatter(p[speed_key], p["clustered_pct"], color=c, marker=mk, s=78, facecolors="white", linewidths=1.8, zorder=3,
                       label=f"Author {p['method']} ({p['clustered_pct']:.2f}%)")
        for p in loc_pts:
            c, mk = loc_styles[p["method"]]
            short = p["method"].replace(" (warm)", "")
            ax.scatter(p[speed_key], p["clustered_pct"], color=c, marker=mk, s=82, zorder=4,
                       label=f"{short} ({p['clustered_pct']:.2f}%)")

        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 3)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}" if x >= 1 else f"{x:g}"))
        ax.tick_params(which="minor", bottom=False)
        ax.set_ylim(0, 5.6)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="both", color="#dde3e6", linewidth=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#a6b1b7")
        ax.tick_params(labelsize=9.5, length=0, pad=5, colors="#445361")
        ax.set_axisbelow(True)
        ax.set_ylabel("Clustered pass rate (%)", fontsize=11)
        ax.set_xlabel("Attempts per GPU-hour (log scale)", fontsize=11)
        fig.legend(loc="upper left", bbox_to_anchor=(0.055, 0.875), frameon=False, ncol=4, fontsize=8.7, columnspacing=1.2)
        fig.text(0.062, 0.022,
                 "Open markers: author H200 archive. Filled markers: independent local RTX A6000 runs (warm batch-amortized GPU time).",
                 fontsize=8.5, color="#54636b")
        fig.savefig(out_dir / fname, dpi=185, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", type=Path, default=TASK / "environment/references/overall/targets.csv")
    p.add_argument("--dna-dir", type=Path, default=TASK / "environment/references/dna_binding")
    p.add_argument("--out-dir", type=Path, default=None, help="Destination directory (defaults to --dna-dir)")
    p.add_argument("--campaign-dir", type=Path, default=None, help="Optional raw dna_suite directory to rebuild trial CSVs")
    args = p.parse_args()

    dna_dir = args.dna_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else dna_dir
    targets_csv = args.targets.resolve()
    if args.campaign_dir is not None:
        collect_from_campaign(args.campaign_dir.resolve(), targets_csv, out_dir)
        warm_csv, all_csv = out_dir / "trials_warm.csv", out_dir / "trials_all.csv"
    else:
        warm_csv, all_csv = dna_dir / "trials_warm.csv", dna_dir / "trials_all.csv"
    render_dna_reports_and_plots(targets_csv, warm_csv, all_csv, out_dir)


if __name__ == "__main__":
    main()
