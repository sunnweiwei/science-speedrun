"""Aggregate (optional) and plot S15 n_recycle=0 vs thinking_medium and 5-seed paired repeatability."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

TASK = Path(__file__).resolve().parents[2]
SAMPLERS = ("stochastic-tsr", "prior-tsr", "ode-tsr", "full-sde", "short-sde")
PAPER_S15 = {
    "stochastic-tsr": ("Stochastic TSR", 60, 34.7, 2.99),
    "prior-tsr": ("Stochastic TSR (prior schedule)", 60, 0.0, 2.97),
    "ode-tsr": ("ODE TSR", 60, 33.3, 3.04),
    "full-sde": ("Full-length SDE", 400, 37.3, 17.68),
    "short-sde": ("Shortened SDE", 60, 0.0, 2.95),
}


def write_csv(path: Path, rows: list[dict], delimiter: str = ","):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=delimiter, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def collect_from_campaign(campaign: Path, s15_dir: Path, seeds: tuple[int, ...] = (727000, 728000, 729000, 730000, 731000)):
    with (s15_dir / "summary.csv").open() as f:
        med_summary = {r["sampler"]: r for r in csv.DictReader(f)}

    base_points, base_lengths, base_slots = [], [], []
    for arm in SAMPLERS:
        _, steps, p_pct, p_sec = PAPER_S15[arm]
        passes_per_slot = 400 if arm == "full-sde" else 60
        af3_tot = 0.0
        for n in (100, 150, 250):
            w_dir = campaign / f"s15_base1pass/{arm}/{n}"
            gen = json.loads((w_dir / "generation.json").read_text())
            res = json.loads((w_dir / "result.json").read_text())
            meas = json.loads((w_dir / f"evaluation/{n}/measurements.json").read_text())
            counts = res["by_length"][str(n)]
            af3_tot += res["af3_seconds"]
            for i in range(50):
                slot = f"l{n}d{i:03d}"
                v = meas.get(slot, {})
                base_slots.append({
                    "sampler": arm, "length": n, "slot": slot,
                    "scRMSD_A": v.get("scRMSD_A", ""),
                    "foldseek_cluster": v.get("cluster", ""),
                })
            base_lengths.append({
                "sampler": arm, "length": n, "slots": 50,
                "emitted": counts["emitted"], "refolded": counts["refolded"],
                "passes": counts["passes"], "passing_clusters": counts["passing_clusters"],
                "trunk_passes_per_slot": passes_per_slot,
                "generation_seconds": gen["generation_seconds"],
                "af3_seconds": res["af3_seconds"],
                "local_warm_predict_s_per_design": gen["warm_predict_s_per_design"],
            })
        arm_rows = [r for r in base_lengths if r["sampler"] == arm]
        uniq = sum(r["passing_clusters"] for r in arm_rows)
        pct = 100.0 * uniq / 150
        gen_s = sum(r["generation_seconds"] for r in arm_rows)
        hours = (gen_s + af3_tot) / 3600.0
        warm_s = statistics.fmean(r["local_warm_predict_s_per_design"] for r in arm_rows)
        med_pct = float(med_summary[arm]["local_clustered_pct"])
        base_points.append({
            "sampler": arm,
            "steps": steps,
            "trunk_passes_per_slot": passes_per_slot,
            "slots": 150,
            "refolded": sum(r["refolded"] for r in arm_rows),
            "passes": sum(r["passes"] for r in arm_rows),
            "passing_clusters": uniq,
            "base_clustered_pct": pct,
            "medium_recycle_clustered_pct": med_pct,
            "paper_clustered_pct": p_pct,
            "base_minus_paper_pp": pct - p_pct,
            "medium_minus_paper_pp": med_pct - p_pct,
            "total_trunk_passes": 150 * passes_per_slot,
            "generation_seconds": gen_s,
            "af3_seconds": af3_tot,
            "local_warm_predict_s_per_design": warm_s,
            "warm_generation_attempts_per_gpu_h": 3600.0 / warm_s,
            "generation_and_af3_attempts_per_gpu_h": 150.0 / hours,
            "generation_and_af3_passing_clusters_per_gpu_h": uniq / hours,
            "paper_reported_seconds": p_sec,
        })

    write_csv(s15_dir / "base_schedule_summary.csv", base_points)
    write_csv(s15_dir / "base_schedule_by_length.csv", base_lengths)
    write_csv(s15_dir / "base_schedule_slots.tsv", base_slots, delimiter="\t")

    with (s15_dir / "by_length.csv").open() as f:
        s727_len = {(r["sampler"], int(r["length"])): r for r in csv.DictReader(f)}
    with (s15_dir / "slots.tsv").open() as f:
        s727_slots = [r for r in csv.DictReader(f, delimiter="\t") if r["sampler"] in ("stochastic-tsr", "ode-tsr")]

    repeat_rows = []
    repeat_slots = [{"base_seed": 727000, **r} for r in s727_slots]
    for seed in seeds:
        for arm in ("stochastic-tsr", "ode-tsr"):
            by_l = {}
            passes_tot = 0
            for n in (100, 150, 250):
                if seed == 727000 and not (campaign / f"multi_seed_repeats/seed_{seed}/{arm}/{n}").is_dir():
                    rec = s727_len[(arm, n)]
                    p_cnt, c_cnt = int(rec["passes"]), int(rec["passing_clusters"])
                else:
                    w_dir = campaign / f"multi_seed_repeats/seed_{seed}/{arm}/{n}"
                    res = json.loads((w_dir / "result.json").read_text())["by_length"][str(n)]
                    meas = json.loads((w_dir / f"evaluation/{n}/measurements.json").read_text())
                    p_cnt, c_cnt = res["passes"], res["passing_clusters"]
                    for i in range(50):
                        slot = f"l{n}d{i:03d}"
                        v = meas.get(slot, {})
                        repeat_slots.append({
                            "base_seed": seed, "sampler": arm, "length": n, "slot": slot,
                            "scRMSD_A": v.get("scRMSD_A", ""), "foldseek_cluster": v.get("cluster", ""),
                        })
                by_l[n] = c_cnt
                passes_tot += p_cnt
            tot_c = sum(by_l.values())
            repeat_rows.append({
                "base_seed": seed,
                "sampler": arm,
                "slots": 150,
                "passes": passes_tot,
                "passing_clusters": tot_c,
                "clustered_pass_pct": 100.0 * tot_c / 150,
                "clusters_L100": by_l[100],
                "clusters_L150": by_l[150],
                "clusters_L250": by_l[250],
            })

    write_csv(s15_dir / "multi_seed_repeats.csv", repeat_rows)
    write_csv(s15_dir / "multi_seed_slots.tsv", repeat_slots, delimiter="\t")


def render_plot(base_csv: Path, repeats_csv: Path, output_png: Path):
    with base_csv.open() as f:
        base_points = list(csv.DictReader(f))
    with repeats_csv.open() as f:
        repeat_rows = list(csv.DictReader(f))

    seeds = sorted({int(r["base_seed"]) for r in repeat_rows})
    st_vals = [int(r["passing_clusters"]) for s in seeds for r in repeat_rows
               if int(r["base_seed"]) == s and r["sampler"] == "stochastic-tsr"]
    od_vals = [int(r["passing_clusters"]) for s in seeds for r in repeat_rows
               if int(r["base_seed"]) == s and r["sampler"] == "ode-tsr"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 5.4))
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.20, top=0.77, wspace=0.34)
    fig.text(0.04, 0.94, "Unconditional monomers   /   S15 schedule resolution & multi-seed stability",
             fontsize=17.5, weight="bold", color="#142c35")
    fig.text(0.041, 0.885,
             "Left: 1-pass base schedule (60/400p) vs Medium recycling (180/1199p) and Paper S15   ·   Right: 5 independent 150-slot seeds",
             fontsize=9.8, color="#46575f")

    labels = ["Stochastic TSR", "TSR, prior sched.", "ODE TSR", "Full-length SDE", "Shortened SDE"]
    for i, bp in enumerate(base_points):
        p_val = float(bp["paper_clustered_pct"])
        b_val = float(bp["base_clustered_pct"])
        m_val = float(bp["medium_recycle_clustered_pct"])
        ax1.plot([min(p_val, b_val, m_val), max(p_val, b_val, m_val)], [i, i], color="#cbd5d8", lw=1.4, zorder=1)
        ax1.scatter(p_val, i, marker="o", facecolors="white", edgecolors="#3c4853", s=70, linewidths=1.5, zorder=3)
        ax1.scatter(b_val, i, marker="D", color="#156c64", s=52, zorder=4)
        ax1.scatter(m_val, i, marker="s", color="#b06b16", s=50, zorder=3)
    ax1.set_yticks(range(5), labels)
    ax1.set_ylim(4.6, -0.6)
    ax1.set_xlim(-2.5, 56)
    ax1.set_xlabel("Clustered pass rate (%)")
    ax1.set_title("Effect of recycling on S15 samplers", loc="left", fontsize=11.5, weight="bold")
    ax1.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", color="#3c4853", label="Paper S15"),
            Line2D([], [], marker="D", linestyle="none", color="#156c64", label="Local n_recycle=0 (60/400p)"),
            Line2D([], [], marker="s", linestyle="none", color="#b06b16", label="Local Medium (180/1199p)"),
        ],
        loc="lower center", bbox_to_anchor=(0.48, -0.27), ncol=3, frameon=False, fontsize=8.3, columnspacing=1.0,
    )

    x_seeds = np.arange(len(seeds))
    st_pct = [100 * v / 150 for v in st_vals]
    od_pct = [100 * v / 150 for v in od_vals]
    ax2.plot(x_seeds, st_pct, "-o", color="#156c64", lw=1.6, markersize=6.5, label=f"Stochastic TSR (mean {statistics.fmean(st_pct):.1f}%)")
    ax2.plot(x_seeds, od_pct, "--s", color="#356db0", lw=1.6, markersize=6.0, label=f"ODE TSR (mean {statistics.fmean(od_pct):.1f}%)")
    ax2.scatter([0], [100 * 82 / 150], marker="^", color="#955b35", s=65, zorder=5, label="Prior campaign seed 727k (54.7%)")
    ax2.set_xticks(x_seeds, [f"{s // 1000}k" for s in seeds])
    ax2.set_xlabel("Generator base seed")
    ax2.set_ylabel("Clustered pass rate (%)")
    ax2.set_ylim(35, 60)
    ax2.set_title("Paired 5-seed repeatability (Medium, 150 slots)", loc="left", fontsize=11.5, weight="bold")
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.27), ncol=2, frameon=False, fontsize=8.3)

    for ax in (ax1, ax2):
        ax.grid(axis="both", alpha=0.45, color="#cfd8dc")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#aab7bc")
        ax.tick_params(length=0, pad=5)

    fig.text(0.04, 0.018, "RTX A6000, RFD4 0.0.3, AF3 3.0.2 (5 recycles, seed 727000), Foldseek 10-941cd33.",
             fontsize=8.5, color="#53636c")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    default_dir = TASK / "environment/references/s15"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s15-dir", type=Path, default=default_dir)
    p.add_argument("--campaign-dir", type=Path, default=None, help="Optional raw campaign root to rebuild CSVs before plotting")
    p.add_argument("--output", type=Path, default=None, help="Output PNG path (defaults to <s15-dir>/s15-schedule-and-repeats.png)")
    args = p.parse_args()

    s15_dir = args.s15_dir.resolve()
    if args.campaign_dir is not None:
        collect_from_campaign(args.campaign_dir.resolve(), s15_dir)
    out_png = args.output.resolve() if args.output else (s15_dir / "s15-schedule-and-repeats.png")
    render_plot(s15_dir / "base_schedule_summary.csv", s15_dir / "multi_seed_repeats.csv", out_png)


if __name__ == "__main__":
    main()
