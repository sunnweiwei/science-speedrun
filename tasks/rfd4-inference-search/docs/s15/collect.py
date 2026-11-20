"""Build a complete Table S15 local report from finished, length-only cohorts."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean
import tempfile

from run_cohort import SAMPLERS, SEED, SLOTS, TASK, LENGTHS, config

PAPER_NAMES = {"stochastic-tsr": "Stochastic TSR", "prior-tsr": "Stochastic TSR (prior schedule)",
               "ode-tsr": "ODE TSR", "full-sde": "Full-length SDE", "short-sde": "Shortened SDE"}


def read_rows(path, delimiter=","):
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def read_json(path):
    return json.loads(path.read_bytes())


def write_rows(path, rows, delimiter=","):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=delimiter, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def collect(root, output):
    source = TASK / "environment/references/overall/ablation_results.csv"
    paper = {r["method"]: r for r in read_rows(source) if r["table"] == "S15"}
    old = TASK / "environment/baseline/stochastic-tsr/reproduction"
    baseline = read_json(old / "paper-size.json")
    if baseline["af3_recycles"] != 5 or baseline["af3_samples"] != 1 or baseline["af3_model_seed"] != SEED:
        raise ValueError("Unexpected baseline refolding protocol")
    prior_records = read_rows(old / "paper-size-slots.tsv", delimiter="\t")
    points, lengths, slots = [], [], []
    for arm in SAMPLERS:
        reused = arm == "stochastic-tsr" and not (root / arm).exists()
        seconds_af3 = baseline["af3_seconds"] if reused else 0
        for n in LENGTHS:
            nsteps = config(arm, n, Path("/assets"), Path("/run"))["generation"]
            passes_per_slot = sum(width*depth for width, depth in nsteps["n_recycle_schedule"])
            if reused:
                counts = baseline["by_length"][str(n)]
                measurements = {r["slot"]: r for r in prior_records if int(r["length"]) == n and r["scRMSD_A"]}
                gen_seconds, warm_seconds = counts["generation_seconds"], None
            else:
                path = root / arm / str(n)
                gen = read_json(path / "generation.json")
                if (gen["sampler"], gen["length"], gen["seed"], gen["slots"]) != (arm, n, SEED+n, SLOTS):
                    raise ValueError(f"Wrong generation cohort {path}")
                scored = read_json(path / "result.json")
                counts = scored["by_length"][str(n)]
                measurements = {key: {"scRMSD_A": v["scRMSD_A"], "foldseek_cluster": v["cluster"]}
                                for key, v in read_json(path / "evaluation" / str(n) / "measurements.json").items()}
                gen_seconds = gen["generation_seconds"]
                warm = gen["batches"][1:]
                warm_seconds = math.fsum(b["seconds"] for b in warm) / sum(b["size"] for b in warm)
                seconds_af3 += scored["af3_seconds"]
            if any(not math.isfinite(float(v["scRMSD_A"])) or float(v["scRMSD_A"]) < 0 or not v["foldseek_cluster"]
                   for v in measurements.values()):
                raise ValueError(f"Invalid structure measurement for {arm}, L={n}")
            hits = [v["foldseek_cluster"] for v in measurements.values() if float(v["scRMSD_A"]) <= 1.5]
            expected = {f"l{n}d{i:03d}" for i in range(SLOTS)}
            if (not measurements.keys() <= expected or counts["slots"] != SLOTS or
                    (len(measurements), len(hits), len(set(hits))) !=
                    (counts["refolded"], counts["passes"], counts["passing_clusters"])):
                raise ValueError(f"Slot counts disagree for {arm}, L={n}")
            for key in sorted(expected):
                v = measurements.get(key, {})
                slots.append(dict(sampler=arm, length=n, slot=key, scRMSD_A=v.get("scRMSD_A", ""),
                                  foldseek_cluster=v.get("foldseek_cluster", "")))
            lengths.append(dict(sampler=arm, length=n, slots=SLOTS, emitted=counts["emitted"],
                                refolded=len(measurements), passes=len(hits), passing_clusters=len(set(hits)),
                                trunk_passes_per_slot=passes_per_slot, generation_seconds=gen_seconds,
                                local_warm_predict_s_per_design=warm_seconds))
        rows = [r for r in lengths if r["sampler"] == arm]
        originals = sum(r["slots"] for r in rows)
        unique = sum(r["passing_clusters"] for r in rows)
        paper_row = paper[PAPER_NAMES[arm]]
        pct = 100 * unique / originals
        points.append(dict(sampler=arm, steps=int(paper_row["steps"]), slots=originals,
                           refolded=sum(r["refolded"] for r in rows), passes=sum(r["passes"] for r in rows),
                           passing_clusters=unique, local_clustered_pct=pct, paper_clustered_pct=float(paper_row["clustered_pass_pct"]),
                           difference_pp=pct-float(paper_row["clustered_pass_pct"]),
                           total_trunk_passes=sum(r["slots"]*r["trunk_passes_per_slot"] for r in rows),
                           generation_seconds=sum(r["generation_seconds"] for r in rows), af3_seconds=seconds_af3,
                           local_warm_predict_s_per_design=None if reused else fmean(r["local_warm_predict_s_per_design"] for r in rows),
                           paper_reported_seconds=float(paper_row["reported_seconds"]),
                           source="previous local release run" if reused else "new local release run"))
    if output.exists():
        raise FileExistsError("Choose a new report directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "result"
        stage.mkdir()
        write_rows(stage / "summary.csv", points)
        write_rows(stage / "by_length.csv", lengths)
        write_rows(stage / "slots.tsv", slots, delimiter="\t")
        stage.rename(output)
    print(f"Checked {len(points)} samplers, {len(lengths)} cohorts and {len(slots)} slots.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    collect(args.root.resolve(), args.output.resolve())
