"""Render the 2-panel unconditional monomer scaling & native AF3-guided beam comparison plot."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

TASK = Path(__file__).resolve().parents[2]


def render_monomer_beam_plot(matched_csv: Path, scaling_csv: Path, output_png: Path):
    with matched_csv.open() as f:
        matched_summary = list(csv.DictReader(f))
    with scaling_csv.open() as f:
        scaling_150 = list(csv.DictReader(f))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 5.8))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.29, top=0.79, wspace=0.26)
    fig.text(0.05, 0.945, "Unconditional monomers   /   One-shot refinement & AF3-guided beam",
             fontsize=17.5, weight="bold", color="#142c35")
    fig.text(0.051, 0.892,
             "Left: 150-slot paper-size policies (L=100, 150, 250)   ·   Right: 30-slot paired control with official AF3 Beam (held-out AF3 seed 314159)",
             fontsize=9.6, color="#46575f")

    oneshot = scaling_150[:4]
    ax1.plot([float(r["end_to_end_attempts_per_gpu_h"]) for r in oneshot],
             [float(r["clustered_pct"]) for r in oneshot],
             "-", color="#939ea4", lw=1.4, zorder=1)
    colors_150 = ["#62707c", "#198572", "#2674aa", "#b06b16", "#955b35", "#8a4b76"]
    markers_150 = ["v", "o", "s", "^", "P", "X"]
    for r, c, mk in zip(scaling_150, colors_150, markers_150):
        ax1.scatter(float(r["end_to_end_attempts_per_gpu_h"]), float(r["clustered_pct"]),
                    color=c, marker=mk, s=72, zorder=3, label=r["label"])
    ax1.set_title("150-slot monomer policies (including AF3)", loc="left", fontsize=11.5, weight="bold")
    ax1.set_xlabel("Attempts per GPU-hour (including final AF3)")
    ax1.set_ylabel("Clustered pass rate (%)")
    ax1.set_ylim(25, 62)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=8.0, columnspacing=1.0)

    ax2.plot([float(r["end_to_end_attempts_per_gpu_h"]) for r in matched_summary],
             [float(r["clustered_pct_seed_314159"]) for r in matched_summary],
             ":", color="#939ea4", lw=1.4, zorder=1)
    c_beam = ["#198572", "#2674aa", "#b06b16", "#803f87"]
    m_beam = ["o", "s", "^", "D"]
    for r, c, mk in zip(matched_summary, c_beam, m_beam):
        pct = float(r["clustered_pct_seed_314159"])
        yld = float(r["end_to_end_clusters_per_gpu_h_seed_314159"])
        ax2.scatter(float(r["end_to_end_attempts_per_gpu_h"]), pct,
                    color=c, marker=mk, s=80, zorder=3,
                    label=f"{r['label']}: {pct:.1f}% ({yld:.1f}/h)")
    ax2.set_xscale("log")
    ax2.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
    ax2.tick_params(which="minor", bottom=False)
    ax2.set_ylim(55, 95)
    ax2.set_title("30-slot paired control: One-shot vs Native AF3 Beam", loc="left", fontsize=11.5, weight="bold")
    ax2.set_xlabel("Attempts per GPU-hour, including internal + final AF3 (log)")
    ax2.set_ylabel("Clustered pass rate (%)")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=8.0, columnspacing=0.8)

    for ax in (ax1, ax2):
        ax.grid(axis="both", alpha=0.45, color="#cfd8dc")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#aab7bc")
        ax.tick_params(length=0, pad=5)

    fig.text(0.05, 0.018, "Local supplementary monomer study on RTX A6000 (not counted toward the 16 official four-suite points).",
             fontsize=8.5, color="#53636c")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    default_dir = TASK / "environment/references/monomer_beam"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mb-dir", type=Path, default=default_dir)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    mb_dir = args.mb_dir.resolve()
    out_png = args.output.resolve() if args.output else (mb_dir / "monomer-quality-throughput.png")
    render_monomer_beam_plot(mb_dir / "matched_30slot_summary.csv", mb_dir / "scaling_150slot_summary.csv", out_png)


if __name__ == "__main__":
    main()
