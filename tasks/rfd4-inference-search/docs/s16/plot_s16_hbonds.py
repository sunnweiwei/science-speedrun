"""Aggregate (optional) and plot Supplementary Table S16 SAM hydrogen-bond search results."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

TASK = Path(__file__).resolve().parents[2]
PAPER_S16 = {
    "stochastic-tsr": ("Stochastic TSR (default)", 60, 14.24),
    "prior-tsr": ("Stochastic TSR (prior schedule)", 60, 5.18),
    "full-sde": ("Full-length SDE", 400, 12.72),
    "short-sde": ("Shortened SDE", 60, 3.31),
}


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def collect_from_campaign(campaign_dir: Path, out_dir: Path):
    summary_rows, design_rows = [], []
    for mode in ("base_1pass", "midpeak"):
        arms = ("stochastic-tsr", "prior-tsr", "full-sde", "short-sde") if mode == "base_1pass" else ("stochastic-tsr", "prior-tsr", "short-sde")
        for arm in arms:
            w_dir = campaign_dir / mode / arm
            d = json.loads((w_dir / "done.json").read_text())
            pname, steps, p_hb = PAPER_S16[arm]
            summary_rows.append({
                "schedule_mode": mode,
                "sampler": arm,
                "paper_method": pname,
                "steps": steps,
                "search_runs": d["search_runs"],
                "final_designs": d["final_designs"],
                "local_mean_hbonds": d["mean_hbond_count"],
                "local_stdev_hbonds": d["stdev_hbond_count"],
                "local_median_hbonds": d["median_hbond_count"],
                "paper_mean_hbonds": p_hb,
                "difference_hbonds": d["mean_hbond_count"] - p_hb,
                "warm_s_per_search_run": d["warm_s_per_search_run"],
                "wall_seconds": d["wall_seconds"],
            })
            with (w_dir / "outputs/inference_metrics/per_example_metrics.csv").open() as f:
                for r in csv.DictReader(f):
                    design_rows.append({
                        "schedule_mode": mode,
                        "sampler": arm,
                        "example_id": r["example_id"],
                        "hbond_atom_pair_count_coi": float(r["interchain_interactions.hbond_atom_pair_count_coi"]),
                    })
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "designs.csv", design_rows)


def render_s16_plot(summary_csv: Path, output_png: Path):
    with summary_csv.open() as f:
        summary_rows = list(csv.DictReader(f))

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 4.8))
    fig.subplots_adjust(left=0.25, right=0.96, bottom=0.22, top=0.78)
    fig.text(0.04, 0.93, "Small-molecule beam search   /   Table S16 SAM hydrogen-bond ablation",
             fontsize=16.5, weight="bold", color="#142c35")
    fig.text(0.041, 0.87,
             "50 independent search runs per setting (beam width 2, branch 2, checkpoints at 25% & 50% steps, 100 final designs)",
             fontsize=9.5, color="#46575f")

    order = ["stochastic-tsr", "full-sde", "prior-tsr", "short-sde"]
    ylabels = ["Stochastic TSR (60)", "Full-length SDE (400)", "TSR, prior sched. (60)", "Shortened SDE (60)"]
    base_map = {r["sampler"]: r for r in summary_rows if r["schedule_mode"] == "base_1pass"}
    mid_map = {r["sampler"]: r for r in summary_rows if r["schedule_mode"] == "midpeak"}

    for i, arm in enumerate(order):
        p_hb = PAPER_S16[arm][2]
        b_hb = float(base_map[arm]["local_mean_hbonds"])
        vals = [p_hb, b_hb]
        if arm in mid_map:
            vals.append(float(mid_map[arm]["local_mean_hbonds"]))
        ax.plot([min(vals), max(vals)], [i, i], color="#cbd5d8", lw=1.4, zorder=1)
        ax.scatter(p_hb, i, marker="o", facecolors="white", edgecolors="#3c4853", s=75, linewidths=1.5, zorder=3)
        ax.scatter(b_hb, i, marker="D", color="#156c64", s=55, zorder=4)
        if arm in mid_map:
            ax.scatter(float(mid_map[arm]["local_mean_hbonds"]), i, marker="s", color="#b06b16", s=52, zorder=4)

    ax.set_yticks(range(4), ylabels)
    ax.set_ylim(3.6, -0.6)
    ax.set_xlim(-0.8, 16.5)
    ax.set_xlabel("Mean protein–SAM hydrogen-bond atom-pair count")
    ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", color="#3c4853", label="Paper Table S16"),
            Line2D([], [], marker="D", linestyle="none", color="#156c64", label="Local n_recycle=0 (1 pass/step)"),
            Line2D([], [], marker="s", linestyle="none", color="#b06b16", label="Local Midpeak Medium (150p)"),
        ],
        loc="lower center", bbox_to_anchor=(0.45, -0.28), ncol=3, frameon=False, fontsize=8.8,
    )
    ax.grid(axis="x", alpha=0.45, color="#cfd8dc")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#aab7bc")
    ax.tick_params(length=0, pad=6)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    default_dir = TASK / "environment/references/s16"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s16-dir", type=Path, default=default_dir)
    p.add_argument("--campaign-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    s16_dir = args.s16_dir.resolve()
    if args.campaign_dir is not None:
        collect_from_campaign(args.campaign_dir.resolve(), s16_dir)
    out_png = args.output.resolve() if args.output else (s16_dir / "s16-hbonds.png")
    render_s16_plot(s16_dir / "summary.csv", out_png)


if __name__ == "__main__":
    main()
