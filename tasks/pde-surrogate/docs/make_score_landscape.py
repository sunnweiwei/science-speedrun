"""Where every measured system sits between each track's floor and ceiling."""
import argparse
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
_ap.add_argument("--out", default=str(HERE / "score_landscape.png"),
                 help="where to write the figure")
_args = _ap.parse_args()

INK, MUT, RULE = "#15191E", "#69737E", "#C6CCD2"
C = {"control": "#9AA4AE", "ours_scratch": "#0E7C86", "ours_ft": "#4A4FBF",
     "external": "#B3690B", "ceiling": "#B42D3F"}
LAB = {"control": "trivial control", "ours_scratch": "trained from scratch by us",
       "ours_ft": "fine-tuned by us", "external": "released checkpoint, replayed",
       "ceiling": "bound"}

PDE_LO, PDE_HI = 0.2827, 0.4203   # docs/validation/score_bounds.json (pde-surrogate), 2026-09-22
TS_LO, TS_HI = 0.008, 0.4948
n = lambda r, lo, hi: (r - lo) / (hi - lo)

PDE = [("zero prediction", 0.2202, "control"), ("DPOT-M", 0.2172, "external"),
       ("DPOT-Ti", 0.2216, "external"),
       ("sc-s fine-tune, all three transfer pools", 0.2711, "ours_ft"),
       ("CNextU-net, active matter", 0.2750, "ours_scratch"),
       ("sc-s fine-tune, active matter", 0.2794, "ours_ft"),
       ("persistence  — floor", 0.2827, "ceiling"),
       ("sc-s fine-tune, turbulent layer", 0.2881, "ours_ft"),
       ("CNextU-net, turbulent layer", 0.3080, "ours_scratch"),
       ("spatial mean of last frame", 0.3105, "control"),
       ("sc-s fine-tune, PDEArena", 0.3109, "ours_ft"),
       ("task starter, 473k, 40 min", 0.3139, "ours_scratch"),
       ("PDE-Transformer mc-l, 518M", 0.3317, "external"),
       ("PDE-Transformer mc-s, 33M", 0.3349, "external"),
       ("PDE-Transformer mc-b, 131M", 0.3363, "external"),
       ("CNextU-net, PDEArena", 0.3446, "ours_scratch"),
       ("per-group oracle  — ceiling", 0.4203, "ceiling")]
TS = [("context-quantile control", 0.0000, "control"), ("Seasonal Naive  — floor", 0.0080, "ceiling"),
      ("task starter, 6.4M, 16 min", 0.1530, "ours_scratch"), ("scaled starter, 41.8M, 28 GPU-h", 0.2071, "ours_scratch"),
      ("Chronos-Bolt-base, 205M", 0.3254, "external"),
      ("TimesFM-2.5, 231M", 0.4121, "external"), ("Chronos-2, 119M", 0.4180, "external"),
      ("TimesFM-3, 331M", 0.4465, "external"),
      ("best non-agentic on the board", 0.4723, "external"),
      ("best on the board  — ceiling", 0.4948, "ceiling")]

fig, axes = plt.subplots(1, 2, figsize=(12.6, 7.3), gridspec_kw=dict(wspace=0.60))
fig.subplots_adjust(top=0.760, bottom=0.125, left=0.225, right=0.975)

for ax, rows, lo, hi, title, sub in [
        (axes[0], PDE, PDE_LO, PDE_HI, "PDE surrogate",
         "floor = persistence · ceiling = per-group oracle\n1,085 held-out cases, four groups"),
        (axes[1], TS, TS_LO, TS_HI, "Time series",
         "floor = Seasonal Naive · ceiling = best of 130 published systems\n97 configs, 326,490 instances")]:
    rows = sorted(rows, key=lambda r: r[1])
    ys = range(len(rows))
    ax.axvspan(-0.65, 0, color="#F2F4F5", zorder=0)
    for xv in (0, 1):
        ax.axvline(xv, color=RULE, lw=1.0, ls="--", zorder=1)
    for y, (name, raw, kind) in zip(ys, rows):
        x = n(raw, lo, hi)
        ax.plot([min(0, x), x], [y, y], color=RULE, lw=0.8, zorder=2)
        ax.scatter([x], [y], s=64 if kind == "ceiling" else 52, color=C[kind], zorder=3,
                   marker="D" if kind == "ceiling" else "o",
                   edgecolor="white", linewidth=0.9)
        ax.text(x + 0.035, y, f"{x:+.2f}", va="center", fontsize=7.4, color=MUT,
                fontfamily="monospace")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.2, color=INK)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", labelsize=7.6, colors=MUT, length=3)
    ax.set_xlim(-0.65, 1.30); ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xticks([-0.5, 0, 0.5, 1.0])
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE); ax.spines["bottom"].set_linewidth(0.8)
    ax.text(0, 1.115, title, transform=ax.transAxes, fontsize=12.6, color=INK, va="bottom", weight="bold")
    ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=7.8, color=MUT, va="bottom", linespacing=1.55)
    ax.set_xlabel("normalised score", fontsize=8.2, color=MUT, labelpad=6)

h = [plt.Line2D([], [], marker=("D" if k == "ceiling" else "o"), ls="", color=C[k],
                markersize=6.4, markeredgecolor="white", markeredgewidth=0.9) for k in LAB]
fig.legend(h, list(LAB.values()), loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=5,
           frameon=False, fontsize=8.4, labelcolor=MUT, handletextpad=0.5, columnspacing=1.8)
fig.text(0.5, 0.968, "Two tracks on one scale", ha="center", va="top", fontsize=16, color=INK)
fig.text(0.5, 0.930,
         "Each track's raw reward rescaled so its trivial floor reads 0 and its ceiling reads 1. "
         "The time-series ceiling is a real system\nanyone can submit against; the PDE ceiling is an oracle that "
         "routes each group to a different model, which the contract forbids —\nno legal single package has passed "
         "+0.45, and that gap is the headroom the task offers.",
         ha="center", va="top", fontsize=8.6, color=MUT, linespacing=1.6)
fig.savefig(_args.out, dpi=200, facecolor="white", bbox_inches="tight")
print(f"wrote {_args.out}")
