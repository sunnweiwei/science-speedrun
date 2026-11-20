"""Render the five unconditional sampler observations from the audited report."""
import argparse
import csv
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


LABELS = {"stochastic-tsr": "Stochastic TSR", "prior-tsr": "TSR, prior schedule", "ode-tsr": "ODE TSR",
          "full-sde": "Full-length SDE", "short-sde": "Shortened SDE"}
COLORS = ("#156c64", "#955b35", "#356db0", "#80518c", "#62707c")


def plot(source, destination):
    with source.open(newline="") as f:
        values = list(csv.DictReader(f))
    if [row["sampler"] for row in values] != list(LABELS):
        raise ValueError("Expected the five S15 methods in report order")
    fig, (rates, speed) = plt.subplots(1, 2, figsize=(11.3, 5.6))
    fig.subplots_adjust(left=.175, right=.975, bottom=.19, top=.76, wspace=.29)
    fig.text(.055, .935, "Unconditional monomers   /   S15 sampler comparison", fontsize=19, weight="bold", color="#142c35")
    fig.text(.056, .88, "150 trials per method  ·  AF3 self-consistency and structural clustering",
             fontsize=10.5, color="#46575f")
    rates.set_title("Clustered pass rate (%)", loc="left", fontsize=12, weight="bold")
    for i, (row, color) in enumerate(zip(values, COLORS, strict=True)):
        local, paper = float(row["local_clustered_pct"]), float(row["paper_clustered_pct"])
        rates.plot([local, paper], [i, i], color="#b6c2c7", lw=1.2)
        rates.scatter(paper, i, marker="o", facecolors="white", edgecolors="#3c4853", s=65, zorder=2)
        rates.scatter(local, i, marker="D", color=color, s=31, zorder=3)
        speed.barh(i, float(row["generation_seconds"]) / int(row["slots"]), height=.43, color=color)
    rates.set_yticks(range(5), [LABELS[r["sampler"]] for r in values])
    rates.legend(handles=[Line2D([], [], marker="D", linestyle="none", color="#38545c", label="Local GPU"),
                          Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                                 color="#3c4853", label="Paper S15")], loc="lower center",
                 bbox_to_anchor=(.5, -0.21), ncol=2, frameon=False)
    speed.set_title("Local generation seconds per design", loc="left", fontsize=12, weight="bold")
    speed.set_yticks(range(5), [])
    for ax in (rates, speed):
        ax.set_ylim(4.6, -.6)
        ax.grid(axis="x", alpha=.5, color="#c3cdd0")
        ax.set_axisbelow(True)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.spines["bottom"].set_color("#aab7bc")
        ax.tick_params(length=0, pad=6)
    rates.set_xlim(left=-2)
    fig.text(.055, .018, "RTX A6000, native eager sampling. Local timing includes generation startup and output work. AF3 is timed separately.",
             fontsize=8.5, color="#53636c")
    fig.savefig(destination, dpi=175, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.summary, args.output)
