"""Science introduction figure for timeseries-forecasting."""
import argparse
from pathlib import Path
import h5py, numpy as np

HERE = Path(__file__).resolve().parent
_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
_ap.add_argument("--work", default="/tmp/timeseries-forecasting-verify",
                 help="a verifier work directory holding input/ and truth/ for the configs shown")
_ap.add_argument("--predictions", required=True,
                 help="a package's output directory for those same configs, e.g. produced with\n"
                      "  bash environment/baseline/starter/infer.sh WORK/input /tmp/preds")
_ap.add_argument("--out", default=str(HERE / "science_intro.png"),
                 help="where to write the figure")
_args = _ap.parse_args()
# after --help: the verifier image has no matplotlib, and a reader may ask for the inputs without it
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

IN = Path(_args.work) / "input" / "cases"
TRU = Path(_args.work) / "truth" / "cases"
OUT = Path(_args.predictions) / "cases"
for _d in (IN, TRU, OUT):
    if not _d.exists():
        raise SystemExit(f"{_d} is missing. --work needs a completed verifier run and --predictions a "
                         f"package's output for the same configs.")
INK, MUT, ACC, WARN, RULE, SOFT = "#15191E", "#69737E", "#0E7C86", "#B3690B", "#C6CCD2", "#F1F4F5"

fig = plt.figure(figsize=(13.2, 9.6)); fig.patch.set_facecolor("white")
def band(y0, y1):
    fig.add_artist(FancyBboxPatch((0.035, y0), 0.93, y1 - y0, transform=fig.transFigure,
                                  boxstyle="round,pad=0.006,rounding_size=0.012",
                                  fc=SOFT, ec="none", zorder=-5))

fig.text(0.5, 0.982, "Can one model forecast every domain?", ha="center", va="top", fontsize=21, color=INK)
fig.text(0.5, 0.944,
         "Forecasting is still done one dataset at a time: a bespoke pipeline for electricity load, another for "
         "hospital admissions, another for cloud traffic.\nTime-series foundation models claim a single pretrained "
         "model replaces all of them. This task asks an agent to build one from scratch and prove it.",
         ha="center", va="top", fontsize=9.8, color=MUT, linespacing=1.65)

# ---- 1. the problem -------------------------------------------------------
band(0.632, 0.882)
fig.text(0.055, 0.868, "1 — THE PROBLEM", fontsize=8.6, color=WARN, weight="bold")
fig.text(0.055, 0.845, "Seven domains, ten frequencies, three horizons — one model for all of it",
         fontsize=12.4, color=INK, va="top", weight="bold")
DOM = [("Energy", 32), ("Web/CloudOps", 20), ("Nature", 15), ("Transport", 15),
       ("Econ/Fin", 6), ("Healthcare", 5), ("Sales", 4)]
ax = fig.add_axes([0.065, 0.672, 0.40, 0.118])
ys = np.arange(len(DOM))[::-1]
ax.barh(ys, [d[1] for d in DOM], height=0.62, color=ACC)
for y, (n, c) in zip(ys, DOM):
    ax.text(c + 0.7, y, str(c), va="center", fontsize=7.4, color=MUT, fontfamily="monospace")
ax.set_yticks(ys); ax.set_yticklabels([d[0] for d in DOM], fontsize=8, color=INK)
ax.set_xlim(0, 37); ax.set_xticks([])
ax.tick_params(axis="y", length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("configurations per domain  (97 total)", fontsize=8, color=MUT, loc="left", pad=6)
fig.text(0.510, 0.790,
         "23 datasets  ·  97 configurations  ·  326,490 test instances\n"
         "frequencies from 10 seconds to yearly  ·  1 to 862 variates\n"
         "horizons of 6 to 900 steps, split into short, medium and long",
         fontsize=9.4, color=INK, va="top", linespacing=1.9)
fig.text(0.510, 0.703,
         "The agent may pretrain only on a fixed, pinned 41 GB corpus that shares no\n"
         "series with any of these — a constraint this task verified by content rather\n"
         "than by name, and had to enforce by excluding six leaking directories.",
         fontsize=8.6, color=MUT, va="top", linespacing=1.8)

# ---- 2. input and output --------------------------------------------------
band(0.300, 0.606)
fig.text(0.055, 0.592, "2 — INPUT AND OUTPUT", fontsize=8.6, color=WARN, weight="bold")
fig.text(0.055, 0.569, "History in, a calibrated distribution out — not just a line",
         fontsize=12.4, color=INK, va="top", weight="bold")
cid, w, v = "us_births__D__short", 0, 0
with h5py.File(IN / f"{cid}.h5") as h:
    vals, offs, at = h["values"][()], h["offsets"][()], dict(h.attrs)
    nv, H = int(at["n_variates"]), int(at["prediction_length"])
    ctx = vals[offs[w]:offs[w + 1]].reshape(nv, -1)[v]
with h5py.File(TRU / f"{cid}.h5") as h: truth = h["labels"][()][w, v]
with h5py.File(OUT / f"{cid}.h5") as h: mean, q = h["mean"][()][w, v], h["quantiles"][()][w, v]

axc = fig.add_axes([0.068, 0.392, 0.295, 0.128])
c = ctx[-150:]
axc.plot(np.arange(-len(c), 0), c, lw=1.1, color=INK)
axc.set_xticks([]); axc.set_yticks([])
for s in ("top", "right", "left"): axc.spines[s].set_visible(False)
axc.spines["bottom"].set_color(ACC); axc.spines["bottom"].set_linewidth(1.6)
fig.text(0.2155, 0.372, "INPUT  ·  history, up to 16,384 points", ha="center", va="top",
         fontsize=8.4, color=ACC, weight="bold")
fig.text(0.2155, 0.352, "one series, its frequency and its forecast start.\n"
                        "No covariates, no dataset name, no fitting on the test window.",
         ha="center", va="top", fontsize=7.6, color=MUT, linespacing=1.55)

fig.add_artist(FancyArrowPatch((0.383, 0.455), (0.436, 0.455), transform=fig.transFigure,
                               arrowstyle="-|>", mutation_scale=16, lw=1.6, color=INK, zorder=6))
fig.text(0.409, 0.466, "forecaster", ha="center", fontsize=8.4, color=INK, weight="bold")

axf = fig.add_axes([0.455, 0.392, 0.295, 0.128])
tf = np.arange(H)
axf.fill_between(tf, q[0], q[8], color=ACC, alpha=0.16, lw=0)
axf.fill_between(tf, q[2], q[6], color=ACC, alpha=0.22, lw=0)
axf.plot(tf, mean, lw=1.6, color=ACC)
axf.plot(tf, truth, lw=1.1, color=INK, ls=(0, (3.2, 2.2)))
axf.set_xticks([]); axf.set_yticks([])
for s in ("top", "right", "left"): axf.spines[s].set_visible(False)
axf.spines["bottom"].set_color(WARN); axf.spines["bottom"].set_linewidth(1.6)
fig.text(0.6025, 0.372, "OUTPUT  ·  mean plus nine quantiles, every step", ha="center", va="top",
         fontsize=8.4, color=WARN, weight="bold")
fig.text(0.6025, 0.352, "scored on point accuracy (MASE) and on how well the whole\n"
                        "interval is calibrated (weighted quantile loss), against Seasonal Naive.",
         ha="center", va="top", fontsize=7.6, color=MUT, linespacing=1.55)
fig.text(0.790, 0.455, "dashed line:\nwhat actually\nhappened", fontsize=7.6, color=MUT,
         va="center", linespacing=1.7)
fig.text(0.963, 0.305, "shown: US births, daily, 30-step horizon  ·  forecast by this task's from-scratch starter",
         ha="right", fontsize=7.4, color=MUT)

# ---- 3. why it matters ----------------------------------------------------
band(0.045, 0.274)
fig.text(0.055, 0.260, "3 — WHY IT MATTERS", fontsize=8.6, color=WARN, weight="bold")
fig.text(0.055, 0.237, "A measurable target, and a long way still to climb",
         fontsize=12.4, color=INK, va="top", weight="bold")
ROWS = [("context-quantile control", 0.000, MUT), ("Seasonal Naive", 0.008, MUT),
        ("this task's starter, 6.4M, 16 min", 0.153, ACC),
        ("Chronos-Bolt-base, 205M", 0.325, WARN), ("TimesFM-2.5, 231M", 0.412, WARN),
        ("Chronos-2, 119M", 0.418, WARN), ("TimesFM-3, 331M", 0.447, WARN),
        ("best of 130 published systems", 0.495, "#B42D3F")]
ax = fig.add_axes([0.305, 0.082, 0.34, 0.108])
ys = np.arange(len(ROWS))[::-1]
ax.barh(ys, [r[1] for r in ROWS], height=0.62, color=[r[2] for r in ROWS])
for y, (n, s, _) in zip(ys, ROWS):
    ax.text(s + 0.012, y, f"{s:.3f}", va="center", fontsize=7.2, color=MUT, fontfamily="monospace")
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in ROWS], fontsize=7.6, color=INK)
ax.set_xlim(0, 0.60); ax.set_xticks([])
ax.tick_params(axis="y", length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("benchmark score  (0 = Seasonal Naive, higher is better)", fontsize=7.8, color=MUT,
             loc="left", pad=5)
fig.text(0.680, 0.196,
         "Every released model on that list was replayed here under this task's own\n"
         "contract, and TimesFM-2.5 reproduces its published per-configuration numbers\n"
         "to 4.3e-6 — so the ladder above is measured, not quoted.",
         fontsize=9.0, color=INK, va="top", linespacing=1.8)
fig.text(0.680, 0.116,
         "A 6.4M model trained from scratch for sixteen minutes already reaches a third\n"
         "of the way to the best published system. What closes the rest of that gap,\n"
         "on a fixed corpus and a fixed GPU budget, is the open question.",
         fontsize=9.0, color=MUT, va="top", linespacing=1.8)
fig.savefig(_args.out, dpi=170, facecolor="white", bbox_inches="tight")
print(f"wrote {_args.out}")
