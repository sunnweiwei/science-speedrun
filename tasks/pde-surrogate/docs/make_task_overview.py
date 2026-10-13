"""Science introduction figure for pde-surrogate.

Four sources, one contract: four context frames in, a free-running rollout out, scored against the
simulation the frames came from. Every prediction here is the task's own from-scratch starter, so the
figure shows one model meeting all four families - which is what the task asks for.
"""
import argparse
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

HERE = Path(__file__).resolve().parent
_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
_ap.add_argument("--work", default="/tmp/pde-verify-canon",
                 help="a completed verifier work directory: input/, truth/ and output/ for the same cases")
_ap.add_argument("--out", default=str(HERE / "task_overview.png"), help="where to write the figure")
_args = _ap.parse_args()

W = Path(_args.work)
for _needed in ("input/index.json", "truth", "output/cases"):
    if not (W / _needed).exists():
        raise SystemExit(f"{W / _needed} is missing: --work must be a completed verifier run")
PRED = W / "output/cases"          # the task starter: the one package that covers every group

SHOW = [
    dict(cid="ape2d-full-kolm_flow-test-0000", label="Kolmogorov flow", ch=0, chlab="velocity x",
         sub="corpus  ·  256²  ·  2 channels  ·  26 steps", K=26, cmap="RdBu_r", scale="sym", pct=99.5),
    dict(cid="well-turbulent_radiative_layer_2D-turbulent_radiative_layer_tcool_0.03-0",
         label="Turbulent radiative layer", ch=2, chlab="velocity x",
         sub="The Well  ·  128×384  ·  4 channels  ·  30 steps", K=30, cmap="RdBu_r", scale="sym", pct=99),
    dict(cid="well-active_matter-active_matter_L_10.0_zeta_1.0_alpha_-1.0-0",
         label="Active matter", ch=0, chlab="concentration",
         sub="The Well  ·  256²  ·  11 channels  ·  30 steps", K=30, cmap="RdBu_r", scale="centered", pct=99),
    dict(cid="pdearena-126488-00", label="PDEArena Navier–Stokes", ch=0, chlab="smoke density",
         sub="PDEArena  ·  128²  ·  3 channels  ·  10 steps", K=10, cmap="magma", scale="lin", pct=99.5),
]

def load(cid):
    with h5py.File(W / f"input/cases/{cid}.h5", "r") as f:
        x = f["input"][()].astype(np.float32)
    with h5py.File(W / f"truth/cases/{cid}.h5", "r") as f:
        k = "target" if "target" in f else list(f.keys())[0]
        y = f[k][()].astype(np.float32)
    with h5py.File(PRED / f"{cid}.h5", "r") as f:
        p = f["prediction"][()].astype(np.float32)
    return x, y, p

NCOL = 6
INK, MUT, ACC = "#15191E", "#69737E", "#0E7C86"
fig = plt.figure(figsize=(11.6, 10.6))
outer = fig.add_gridspec(len(SHOW), 1, hspace=0.46, top=0.900, bottom=0.030, left=0.175, right=0.985)

for r, S in enumerate(SHOW):
    x, y, p = load(S["cid"]); ch, K = S["ch"], S["K"]
    steps = sorted({0, max(1, K // 6), K // 3, K // 2, (2 * K) // 3, K - 1})[:NCOL - 1]
    field = np.concatenate([x[:, ch], y[:, ch]])
    if S["scale"] == "sym":
        m = float(np.nanpercentile(np.abs(field), S["pct"])) or 1.0
        norm = Normalize(-m, m)
    elif S["scale"] == "centered":
        mid = float(np.nanmedian(field))
        m = float(np.nanpercentile(np.abs(field - mid), S["pct"])) or 1.0
        norm = Normalize(mid - m, mid + m)
    else:
        norm = Normalize(float(np.nanpercentile(field, 1)), float(np.nanpercentile(field, S["pct"])))
    kw = dict(cmap=S["cmap"], norm=norm, interpolation="nearest", aspect="auto", origin="lower")

    inner = outer[r].subgridspec(2, NCOL, wspace=0.045, hspace=0.085)
    for c in range(NCOL):
        for row in (0, 1):
            ax = fig.add_subplot(inner[row, c])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5); sp.set_color("#C6CCD2")
            if c == 0:
                ax.imshow(x[-1, ch].T, **kw)
                if row == 0:
                    ax.set_title("given: last of 4 context frames", fontsize=6.9, color=INK, pad=3)
                else:
                    ax.set_xlabel("the rollout starts here", fontsize=6.4, color=MUT, labelpad=2)
            else:
                s = steps[c - 1]
                ax.imshow((y if row == 0 else p)[s, ch].T, **kw)
                if row == 0:
                    ax.set_title(f"step +{s + 1}", fontsize=6.9, color=INK, pad=3)
            if c == 0:
                ax.text(0.035, 0.94, "simulation" if row == 0 else "model",
                        transform=ax.transAxes, fontsize=6.6, va="top", ha="left",
                        color=INK if row == 0 else ACC,
                        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.82))

    pos = outer[r].get_position(fig)
    fig.text(0.163, pos.y1 - 0.004, S["label"], ha="right", va="top", fontsize=10.2, color=INK, weight="bold")
    fig.text(0.163, pos.y1 - 0.030, S["sub"], ha="right", va="top", fontsize=7.0, color=MUT)
    fig.text(0.163, pos.y1 - 0.047, f"shown: {S['chlab']}", ha="right", va="top",
             fontsize=7.0, color=MUT, style="italic")

fig.text(0.5, 0.977, "One surrogate, four families of 2D PDEs", ha="center", fontsize=15, color=INK)
fig.text(0.5, 0.948,
         "Four context frames are given. The model then rolls its own output forward for the whole horizon with no "
         "further input,\nand is scored per channel and per step against the simulation. Predictions here are the "
         "task's from-scratch starter.",
         ha="center", fontsize=8.5, color=MUT, linespacing=1.5)
fig.savefig(_args.out, dpi=200, facecolor="white", bbox_inches="tight")
print(f"wrote {_args.out}")
