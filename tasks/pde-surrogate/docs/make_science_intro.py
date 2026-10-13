"""Science introduction figure for pde-surrogate: the problem, the contract, and why it matters."""
import argparse
from pathlib import Path
import h5py, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
_ap.add_argument("--work", default="/tmp/pde-verify-canon",
                 help="a completed verifier work directory: needs input/, truth/ and output/ for the same "
                      "cases. Produce one with tests/main_verifier.py --work THIS --submission <the starter>.")
_ap.add_argument("--out", default=str(HERE / "science_intro.png"), help="where to write the figure")
_args = _ap.parse_args()

W = Path(_args.work)
for _needed in ("input/index.json", "truth", "output/cases"):
    if not (W / _needed).exists():
        raise SystemExit(f"{W / _needed} is missing: --work must be a completed verifier run, e.g.\n"
                         f"  python3 tests/main_verifier.py --submission <starter package> --data DATA \\\n"
                         f"      --manifest environment/data/split.json --work {W} --output /tmp/out")
INK, MUT, ACC, WARN, RULE = "#15191E", "#69737E", "#0E7C86", "#B3690B", "#C6CCD2"
SOFT = "#F1F4F5"

FAM = [("ape2d-full-kolm_flow-test-0000", 0, "Kolmogorov flow", "corpus, 16 datasets", "RdBu_r", "sym", 99.5),
       ("well-turbulent_radiative_layer_2D-turbulent_radiative_layer_tcool_0.03-0", 2,
        "Turbulent radiative layer", "radiative cooling", "RdBu_r", "sym", 99),
       ("well-active_matter-active_matter_L_10.0_zeta_1.0_alpha_-1.0-0", 0,
        "Active matter", "nematic continuum", "RdBu_r", "centered", 99),
       ("pdearena-126488-00", 0, "Navier–Stokes", "buoyant smoke", "magma", "lin", 99.5)]

def frames(cid):
    with h5py.File(W / f"input/cases/{cid}.h5") as f:
        x = f["input"][()].astype(np.float32); a = dict(f.attrs)
    with h5py.File(W / f"truth/cases/{cid}.h5") as f:
        k = "target" if "target" in f else list(f.keys())[0]
        y = f[k][()].astype(np.float32)
    with h5py.File(W / f"output/cases/{cid}.h5") as f:
        p = f["prediction"][()].astype(np.float32)
    return x, y, p, a

def norm_of(field, mode, pct):
    if mode == "sym":
        m = float(np.nanpercentile(np.abs(field), pct)) or 1.0; return Normalize(-m, m)
    if mode == "centered":
        mid = float(np.nanmedian(field)); m = float(np.nanpercentile(np.abs(field - mid), pct)) or 1.0
        return Normalize(mid - m, mid + m)
    return Normalize(float(np.nanpercentile(field, 1)), float(np.nanpercentile(field, pct)))

fig = plt.figure(figsize=(13.2, 9.6)); fig.patch.set_facecolor("white")
def band(y0, y1, c=SOFT):
    # figure-level artists and Axes share one z-order space and Axes default to 0,
    # so a band at zorder 0 added first silently paints over every panel in its section
    fig.add_artist(FancyBboxPatch((0.035, y0), 0.93, y1 - y0, transform=fig.transFigure,
                                  boxstyle="round,pad=0.006,rounding_size=0.012",
                                  fc=c, ec="none", zorder=-5))

# ---- header ---------------------------------------------------------------
fig.text(0.5, 0.982, "Can one surrogate learn many physics?", ha="center", va="top",
         fontsize=21, color=INK)
fig.text(0.5, 0.944,
         "Direct simulation of 2D PDEs is the bottleneck in fluid dynamics, astrophysics and active-matter research: "
         "one parameter sweep is thousands of CPU-hours.\nA neural surrogate is thousands of times cheaper — if it "
         "generalises. This task asks whether one model, built from scratch, can roll forward four unrelated "
         "families of physics.",
         ha="center", va="top", fontsize=9.8, color=MUT, linespacing=1.65)

# ---- 1. the problem -------------------------------------------------------
band(0.638, 0.882)
fig.text(0.055, 0.868, "1 — THE PROBLEM", fontsize=8.6, color=WARN, weight="bold")
fig.text(0.055, 0.845, "Four families that share no equations, no channels and no grid",
         fontsize=12.4, color=INK, va="top", weight="bold")
for i, (cid, ch, name, note, cmap, mode, pct) in enumerate(FAM):
    x, y, p, a = frames(cid)
    ax = fig.add_axes([0.065 + i * 0.225, 0.663, 0.185, 0.128])
    ax.imshow(y[len(y) // 2, ch].T, cmap=cmap, norm=norm_of(np.concatenate([x[:, ch], y[:, ch]]), mode, pct),
              aspect="auto", origin="lower", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(RULE); s.set_linewidth(0.6)
    ax.set_title(name, fontsize=9.6, color=INK, pad=4)
    shape = "×".join(str(v) for v in y.shape[2:])
    ax.set_xlabel(f"{note}\n{y.shape[1]} channels · {shape}", fontsize=7.4, color=MUT, labelpad=4, linespacing=1.5)

# ---- 2. input and output --------------------------------------------------
band(0.298, 0.612)
fig.text(0.055, 0.598, "2 — INPUT AND OUTPUT", fontsize=8.6, color=WARN, weight="bold")
fig.text(0.055, 0.576, "Four frames in, a free-running rollout out — the model never sees the truth again",
         fontsize=12.4, color=INK, va="top", weight="bold")
cid, ch, cmap, mode, pct = FAM[0][0], FAM[0][1], FAM[0][4], FAM[0][5], FAM[0][6]
x, y, p, a = frames(cid)
nrm = norm_of(np.concatenate([x[:, ch], y[:, ch]]), mode, pct)
kw = dict(cmap=cmap, norm=nrm, aspect="auto", origin="lower", interpolation="nearest")
for i in range(4):                                   # context
    ax = fig.add_axes([0.068 + i * 0.079, 0.372, 0.070, 0.095]); ax.imshow(x[i, ch].T, **kw)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(ACC); s.set_linewidth(1.0)
fig.text(0.068 + 1.58 * 0.079, 0.356, "INPUT  ·  4 context frames  ·  (4, C, X, Y)",
         ha="center", va="top", fontsize=8.4, color=ACC, weight="bold")
fig.text(0.068 + 1.58 * 0.079, 0.337, "C = 1 to 11 channels, grids 128² to 256², plus boundary conditions,\n"
         "domain extent and the physical constants of that simulation",
         ha="center", va="top", fontsize=7.6, color=MUT, linespacing=1.5)
fig.add_artist(FancyArrowPatch((0.395, 0.420), (0.448, 0.420), transform=fig.transFigure,
                               arrowstyle="-|>", mutation_scale=16, lw=1.6, color=INK, zorder=6))
fig.text(0.421, 0.431, "surrogate", ha="center", fontsize=8.4, color=INK, weight="bold")
for i, s in enumerate([0, 8, 16, 25]):               # rollout
    ax = fig.add_axes([0.465 + i * 0.079, 0.425, 0.070, 0.095]); ax.imshow(p[s, ch].T, **kw)
    ax.set_xticks([]); ax.set_yticks([])
    for s2 in ax.spines.values(): s2.set_color(WARN); s2.set_linewidth(1.0)
    ax.set_title(f"+{s+1}", fontsize=7.2, color=MUT, pad=2)
    ax2 = fig.add_axes([0.465 + i * 0.079, 0.318, 0.070, 0.095]); ax2.imshow(y[s, ch].T, **kw)
    ax2.set_xticks([]); ax2.set_yticks([])
    for s2 in ax2.spines.values(): s2.set_color(RULE); s2.set_linewidth(0.8)
fig.text(0.462, 0.487, "OUTPUT", ha="right", va="center", fontsize=8.4, color=WARN, weight="bold")
fig.text(0.462, 0.468, "prediction", ha="right", va="center", fontsize=7.6, color=MUT)
fig.text(0.462, 0.366, "simulation", ha="right", va="center", fontsize=7.6, color=MUT)
fig.text(0.783, 0.305, "(K, C, X, Y),  K = 26 / 30 / 10 steps  ·  scored by variance-scaled RMSE, per channel and step",
         ha="right", fontsize=7.6, color=MUT)

# ---- 3. why it matters ----------------------------------------------------
band(0.045, 0.292)
fig.text(0.055, 0.278, "3 — WHY IT MATTERS", fontsize=8.6, color=WARN, weight="bold")
fig.text(0.055, 0.255, "The transfer claim is still open — and that is the point",
         fontsize=12.4, color=INK, va="top", weight="bold")
GROUPS = ["corpus\n850 cases", "turbulent layer\n9 cases", "active matter\n26 cases", "PDEArena\n200 cases"]
SCRATCH = [1.1455, 0.6743, 1.8857, 0.2978]    # task starter, CNextU-net x3 (docs/validation/)
PRETRAIN = [0.3415, 1.3332, 1.3182, 0.7455]   # mc-l, mc-b, mc-s, DPOT-Ti (docs/validation/)
ax = fig.add_axes([0.075, 0.098, 0.44, 0.118])
ix = np.arange(4); w = 0.36
ax.bar(ix - w / 2, SCRATCH, w, color=ACC, label="best trained from scratch here")
ax.bar(ix + w / 2, PRETRAIN, w, color=WARN, label="best released checkpoint, zero-shot")
ax.axhline(1.0, color=MUT, lw=0.9, ls="--")
ax.text(3.45, 1.05, "no skill", fontsize=7, color=MUT, va="bottom", ha="right")
ax.set_xticks(ix); ax.set_xticklabels(GROUPS, fontsize=7.6, color=INK, linespacing=1.5)
ax.set_ylabel("VRMSE  (lower is better)", fontsize=7.8, color=MUT)
ax.tick_params(axis="y", labelsize=7, colors=MUT, length=2.5); ax.tick_params(axis="x", length=0)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(RULE)
ax.legend(fontsize=7.6, frameon=False, labelcolor=MUT, loc="upper center",
          bbox_to_anchor=(0.5, -0.30), ncol=2, handlelength=1.3)
fig.text(0.560, 0.214,
         "A 518M-parameter pretrained transformer owns the corpus it was trained on.\n"
         "An 18.6M model trained from scratch on this task's own data beats every\n"
         "released checkpoint on two of the four groups, by 2× and 2.5×.",
         fontsize=9.2, color=INK, va="top", linespacing=1.75)
fig.text(0.560, 0.138,
         "No published model leads everywhere, so the question the task poses —\n"
         "does pretraining across many equations actually transfer to new physics? —\n"
         "has no settled answer. That unclaimed ground is the benchmark's headroom.",
         fontsize=9.2, color=MUT, va="top", linespacing=1.75)
fig.text(0.965, 0.055, "1,085 held-out trajectories  ·  4 sources  ·  sealed replay", ha="right",
         fontsize=7.4, color=MUT)
fig.savefig(_args.out, dpi=170, facecolor="white", bbox_inches="tight")
print(f"wrote {_args.out}")
