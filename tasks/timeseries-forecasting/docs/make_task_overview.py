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
_ap.add_argument("--out", default=str(HERE / "task_overview.png"),
                 help="where to write the figure")
_args = _ap.parse_args()
# after --help: the verifier image has no matplotlib, and a reader may ask for the inputs without it
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN = Path(_args.work) / "input" / "cases"
TRU = Path(_args.work) / "truth" / "cases"
OUT = Path(_args.predictions) / "cases"
for _d in (IN, TRU, OUT):
    if not _d.exists():
        raise SystemExit(f"{_d} is missing. --work needs a completed verifier run and --predictions a "
                         f"package's output for the same configs.")

PANELS = [("us_births__D__short", "US births", "Healthcare  ·  daily", 0, 0),
          ("bizitobs_l2c__H__short", "Cloud service load", "Web/CloudOps  ·  hourly  ·  7 variates", 0, 0),
          ("hospital__M__short", "Hospital admissions", "Healthcare  ·  monthly", 0, 0),
          ("solar__W__short", "Solar generation", "Energy  ·  weekly", 0, 0)]

INK, MUT, ACC, BAND = "#15191E", "#69737E", "#0E7C86", "#0E7C86"
fig, axes = plt.subplots(2, 2, figsize=(11.4, 6.4))
fig.subplots_adjust(top=0.800, bottom=0.115, left=0.055, right=0.985, hspace=0.46, wspace=0.16)

for ax, (cid, title, sub, w, v) in zip(axes.ravel(), PANELS):
    with h5py.File(IN / f"{cid}.h5") as h:
        vals, offs = h["values"][()], h["offsets"][()]
        attrs = dict(h.attrs); nv = int(attrs["n_variates"]); H = int(attrs["prediction_length"])
        ctx = vals[offs[w]:offs[w + 1]].reshape(nv, -1)[v]
    with h5py.File(TRU / f"{cid}.h5") as h:
        truth = h["labels"][()][w, v]
    with h5py.File(OUT / f"{cid}.h5") as h:
        mean, q = h["mean"][()][w, v], h["quantiles"][()][w, v]

    show = min(len(ctx), max(4 * H, 60))
    c = ctx[-show:]
    tc = np.arange(-len(c), 0); tf = np.arange(0, H)
    ax.plot(tc, c, lw=1.15, color=INK, solid_joinstyle="round")
    ax.fill_between(tf, q[0], q[8], color=BAND, alpha=0.16, lw=0, label="10–90% interval")
    ax.fill_between(tf, q[2], q[6], color=BAND, alpha=0.20, lw=0, label="30–70% interval")
    ax.plot(tf, mean, lw=1.5, color=ACC, label="forecast median")
    ax.plot(tf, truth, lw=1.15, color=INK, ls=(0, (3.2, 2.2)), label="what happened")
    ax.axvline(0, color=MUT, lw=0.7, ls=":")
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=13, weight="bold")
    ax.text(0, 1.045, sub, transform=ax.transAxes, fontsize=7.2, color=MUT, va="bottom")
    ax.text(0.0, -0.155, f"← context, up to 16,384 points      forecast, {H} steps →",
            transform=ax.transAxes, fontsize=6.9, color=MUT)
    ax.tick_params(labelsize=7, colors=MUT, length=2.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#C6CCD2"); ax.spines[s].set_linewidth(0.7)
    ax.set_xticks([])

h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4, frameon=False,
           fontsize=8, labelcolor=MUT, handlelength=1.9, columnspacing=1.9)
fig.text(0.5, 0.965, "One forecaster, 97 configurations it has never seen", ha="center", fontsize=15, color=INK)
fig.text(0.5, 0.900,
         "Trained once on a fixed corpus, then asked for a probabilistic forecast on 23 datasets across seven domains, "
         "ten frequencies and three horizons.\nScored against Seasonal Naive by point accuracy (MASE) and calibration "
         "of the whole interval (weighted quantile loss).",
         ha="center", fontsize=8.4, color=MUT, linespacing=1.5)
fig.savefig(_args.out, dpi=200, facecolor="white", bbox_inches="tight")
print(f"wrote {_args.out}")
