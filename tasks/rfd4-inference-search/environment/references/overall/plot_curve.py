"""Publication-style plots of measured operating points, using matplotlib only for rendering."""
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator

from benchmark_curve import ARMS, TRACKS, frontier


def plot(points, path, include_af3=False, peers=(), *, panel_tracks=None, source_note=None):
    speed = "total_attempts_h" if include_af3 else "generation_attempts_h"
    colors = ["#198572", "#2674aa", "#b06b16", "#803f87"]
    markers = ["o", "s", "^", "D"]
    methods = list(dict.fromkeys(p["method"] for p in points))
    styles = dict(zip(ARMS, zip(colors, markers)))
    extras = [m for m in methods if m not in ARMS]
    for i, method in enumerate(extras):
        styles[method] = (plt.get_cmap("tab10")(i % 10), ["P", "X", "v", "h"][i % 4])

    tracks = list(panel_tracks) if panel_tracks is not None else [
        track for track in TRACKS if any(p["track"] == track for p in points)]
    if len(tracks) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10.2, 5.8), squeeze=False)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.3), sharex=True)
    fig.subplots_adjust(left=.102, right=.98, bottom=.15, top=.79, wspace=.19, hspace=.38)
    fig.text(.064, .956, "RFD4-Proteina   /   Quality and throughput", fontsize=20, weight="bold", color="#162b36")
    scope = "Generation + common AF3 evaluation" if include_af3 else "Generation including internal search"
    fig.text(.066, .913, f"{scope}    ·    200 attempts per target", fontsize=11, color="#445361")
    handles = [Line2D([], [], linestyle="none", marker=styles[m][1], color=styles[m][0], label=m)
               for m in methods]
    if peers:
        handles.append(Line2D([], [], linestyle="none", marker="o", markerfacecolor="none",
                              color="#71808d", label="Other Fig. 2 methods"))
    handles.append(Line2D([], [], linestyle=":", color="#687681", label="Measured Pareto frontier"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.058, .895), frameon=False,
               ncol=min(6, len(handles)), fontsize=9.5, columnspacing=1.7, handletextpad=.5)
    for unused in list(axes.flat)[len(tracks):]:
        unused.set_visible(False)
    for ax, track in zip(axes.flat, tracks):
        n = TRACKS[track][2]
        cohort = [p for p in points if p["track"] == track]
        background = [p for p in peers if p["track"] == track]
        if not cohort and not background:
            ax.set_title(f"{track}  ·  {n} targets", loc="left", fontsize=12, weight="bold", pad=8)
            ax.text(.5, .5, "No independent measurement", ha="center", va="center",
                    transform=ax.transAxes, color="#62737c")
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue
        if background:
            ax.scatter([p[speed] for p in background], [p["clustered_pct"] for p in background],
                       facecolors="none", edgecolors="#71808d", s=35, linewidths=1, zorder=2)
        edge = sorted(frontier(cohort + background, speed), key=lambda p: p[speed])
        ax.plot([p[speed] for p in edge], [p["clustered_pct"] for p in edge], ":",
                linewidth=1.3, color="#939ea4", zorder=1)
        for p in cohort:
            color, marker = styles[p["method"]]
            ax.scatter(p[speed], p["clustered_pct"], c=[color], marker=marker, s=75,
                       edgecolors="white", linewidths=.9, zorder=3)
        ax.set_title(f"{track}  ·  {n} targets", loc="left", fontsize=12, weight="bold", pad=8)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 3)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}" if x >= 1 else f"{x:g}"))
        ax.tick_params(which="minor", bottom=False)
        ax.set_ylim(0, max(1, max(p["clustered_pct"] for p in cohort + background) * 1.15))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="both", color="#dde3e6", linewidth=.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#a6b1b7")
        ax.tick_params(labelsize=9, length=0, pad=5, colors="#445361")
        ax.set_axisbelow(True)
    fig.supylabel("Clustered pass rate (%)", x=.025, y=.48, fontsize=11)
    fig.supxlabel("Attempts per GPU-hour (log scale)", y=.068, fontsize=11)
    provenance = source_note or ("Author archive and supplied observations" if extras else "Author archive for Figure 2 and Table S29")
    fig.text(.066, .021, f"{provenance}. Dotted segments join measured points.",
             fontsize=8.5, color="#54636b")
    fig.savefig(path, dpi=185, bbox_inches="tight", facecolor="white")
    plt.close(fig)
