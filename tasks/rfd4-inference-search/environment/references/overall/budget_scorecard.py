"""Three-budget quality/cost report from frozen, paired numerical measurements."""
import argparse
from collections import defaultdict
import csv
from decimal import Decimal
import hashlib
import io
import json
import math
from pathlib import Path
import re
from statistics import fmean, stdev
import tempfile

from benchmark_curve import HERE, TRACKS, TRIAL_FIELDS

PROTOCOL = "rfd4-budget-curve-v1"
PROFILE = "rfd4-paired-program-v1"
TIERS = (("tenth", 10), ("quarter", 4), ("full", 1))
TRACK_NAMES = {(v[0], v[1]): k for k, v in TRACKS.items()}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("Expected a SHA-256 digest")
    return value


def positive_integer(value):
    if type(value) is not int or value < 1:
        raise ValueError("Expected a positive integer")
    return value


def read_tier(folder, tier, cap):
    summary_bytes = (folder / "summary.json").read_bytes()
    report = json.loads(summary_bytes)
    raw = (folder / "trials.csv").read_bytes()
    if sha(raw) != digest(report["trials_sha256"]):
        raise ValueError(f"{tier}: trial ledger checksum mismatch")
    if (report["timing_profile"] != PROFILE
            or report["generation_timing_boundary"] != "per_request_infer_subprocess_wall_clock"
            or report["af3_timing_boundary"] != "shared_persistent_af3_server_warm_fold"):
        raise ValueError(f"{tier}: incompatible cost accounting")
    if Decimal(str(report["seconds_per_slot"])) != cap:
        raise ValueError(f"{tier}: expected a declared budget of {cap} seconds per slot")
    attempts = positive_integer(report["attempts_per_target"])
    repeats = positive_integer(report["repeats"])
    panel = report["panel"]
    if panel == "full" and attempts != 200:
        raise ValueError("Full-panel publication requires 200 attempts per target")

    # A public report may omit its seed identity. A controller can bind an existing
    # numeric report to the matching draw record without disclosing private seeds.
    review_file = folder / "budget_review.json"
    review_bytes = review_file.read_bytes() if review_file.is_file() else None
    review = json.loads(review_bytes) if review_bytes is not None else {}
    if review and digest(review["summary_sha256"]) != sha(summary_bytes):
        raise ValueError(f"{tier}: controller review refers to a different report")
    draws = digest(review.get("draws_sha256") or report.get("suite_sha256"))
    if "all_arms_complete" in review and type(review["all_arms_complete"]) is not bool:
        raise ValueError(f"{tier}: expected a boolean completion attestation")

    method = report["method"]
    ref_hashes = report["reference_sha256"]
    if not ref_hashes or set(ref_hashes) != set(report["paired_references"]):
        raise ValueError(f"{tier}: incomplete reference package identities")
    by_track = report["by_track"]
    if set(by_track) != set(TRACKS):
        raise ValueError(f"{tier}: all four tracks are required")
    labels = {}
    for track in TRACKS:
        refs = by_track[track]["baselines"]
        refs_here = {r["method"]: r["reference_package"] for r in refs}
        if (len(refs_here) != len(refs) or len(refs) != len(ref_hashes)
                or set(refs_here.values()) != set(ref_hashes) or method in refs_here
                or (labels and labels != refs_here)):
            raise ValueError(f"{tier}: ambiguous reference labels")
        labels = refs_here
    packages = {method: digest(report["submission_sha256"])}
    packages.update({label: digest(ref_hashes[key]) for label, key in labels.items()})
    device = report["device"]
    if not device.get("uuid") or not device.get("driver"):
        raise ValueError(f"{tier}: missing device identity")
    identity = dict(panel=panel, attempts_per_target=attempts, repeats=repeats, candidate=method,
                    packages=packages, device_uuid=device["uuid"], driver=device["driver"],
                    suite_sha256=report.get("suite_sha256"), draws_sha256=draws)

    ledger = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    if tuple(ledger.fieldnames or ()) != ("repeat",) + TRIAL_FIELDS:
        raise ValueError(f"{tier}: unrecognized trial ledger")
    groups = defaultdict(dict)
    for r in ledger:
        if None in r or None in r.values():
            raise ValueError(f"{tier}: malformed trial row")
        track = TRACK_NAMES[(r["dataset"], r["criterion"])]
        rep, label = int(r["repeat"]), r["method"]
        if label not in packages or not 0 <= rep < repeats or not r["target"] or not r["attempt_id"]:
            raise ValueError(f"{tier}: invalid trial identity")
        slots = groups[rep, label, track, r["target"]]
        scored, passed = int(r["scored"]), int(r["passed"])
        cluster = r["cluster_id"].strip()
        g, a = float(r["generation_gpu_s"]), float(r["af3_gpu_s"])
        if (r["attempt_id"] in slots or not 0 <= passed <= scored <= 1 or (passed and not cluster)
                or any(not math.isfinite(x) or x < 0 for x in (g, a))):
            raise ValueError(f"{tier}: duplicate or invalid trial")
        slots[r["attempt_id"]] = (cluster if passed else "", g, a)

    with (HERE / "targets.csv").open() as f:
        official = defaultdict(set)
        for row in csv.DictReader(f):
            official[TRACK_NAMES[row["dataset"], row["criterion"]]].add(row["target"])

    roster = {}
    stats = defaultdict(list)
    within_cap = dict.fromkeys(packages, True)
    for (rep, label, track, target), slots in groups.items():
        slot_ids = frozenset(slots)
        if len(slot_ids) != attempts or roster.setdefault((track, target), slot_ids) != slot_ids:
            raise ValueError(f"{tier}: mismatched attempt identities or denominator")
        clusters, gen, af3 = zip(*slots.values())
        seconds = math.fsum(gen)
        within_cap[label] &= seconds <= float(cap) * attempts
        count = len(set(clusters) - {""})
        stats[rep, label, track].append((target, count, seconds, math.fsum(af3)))
    for track in TRACKS:
        targets = {t for k, t in roster if k == track}
        if not targets or not targets <= official[track] or (panel == "full" and targets != official[track]):
            raise ValueError(f"{tier}: missing or different {track} roster")
        for rep in range(repeats):
            for label in packages:
                if {z[0] for z in stats[rep, label, track]} != targets:
                    raise ValueError(f"{tier}: unmatched arms or repeats")

    evidence = report["per_repeat"] if repeats > 1 else [report]
    if len(evidence) != repeats:
        raise ValueError(f"{tier}: missing repeat summaries")
    candidate_complete = within_cap[method] and all(
        r["by_track"][t]["candidate"]["complete"] is True for r in evidence for t in TRACKS)
    ref_status = [b.get("complete") for r in evidence for t in TRACKS for b in r["by_track"][t]["baselines"]]
    paired_complete = (candidate_complete and all(within_cap.values())
                       and all(x is None or x is True for x in ref_status)
                       and review.get("all_arms_complete", all(x is True for x in ref_status)))
    rows = []
    for label in packages:
        for track in TRACKS:
            qualities, gen_cost, total_cost, yields = [], [], [], []
            for rep in range(repeats):
                records = stats[rep, label, track]
                n = len(records) * attempts
                quality = 100 * sum(z[1] for z in records) / n
                item = evidence[rep]["by_track"][track]
                reported = item["candidate"] if label == method else next(b for b in item["baselines"] if b["method"] == label)
                g = math.fsum(z[2] for z in records)
                af3 = math.fsum(z[3] for z in records)
                if g <= 0 or (quality and af3 <= 0):
                    raise ValueError(f"{tier}: missing measured time for {track}/{label}")
                expected = {"clustered_pct": quality, "generation_attempts_h": 3600*n/g,
                            "total_attempts_h": 3600*n/(g + af3)}
                if (reported["method"] != label or reported["attempts"] != n or reported["targets"] != len(records)
                        or any(not math.isclose(reported[k], v, rel_tol=1e-8, abs_tol=1e-8) for k, v in expected.items())):
                    raise ValueError(f"{tier}: trial outcomes disagree with summary for {track}/{label}")
                qualities.append(quality)
                gen_cost.append(g / n)
                total_cost.append((g + af3) / n)
                if any(z[2] <= 0 for z in records):
                    raise ValueError(f"{tier}: missing target generation timing")
                yields.append(fmean(3600 * z[1] / (z[2] + z[3]) for z in records))
            rows.append(dict(tier=tier, budget_s_per_slot=float(cap), track=track, method=label,
                             package_sha256=packages[label], clustered_pct=fmean(qualities),
                             repeat_sd_pp=stdev(qualities) if repeats > 1 else None,
                             generation_s_per_attempt=fmean(gen_cost), total_s_per_attempt=fmean(total_cost),
                             generation_attempts_per_gpu_hour=3600 / fmean(gen_cost),
                             total_attempts_per_gpu_hour=3600 / fmean(total_cost),
                             distinct_clusters_per_gpu_hour=fmean(yields)))
    hashes = {"summary_sha256": sha(summary_bytes), "trials_sha256": sha(raw),
              "budget_review_sha256": sha(review_bytes) if review_bytes is not None else None}
    return identity, roster, rows, candidate_complete, paired_complete, hashes


def build(folders, max_budget):
    if not max_budget.is_finite() or max_budget <= 0:
        raise ValueError("The maximum budget must be finite and positive")
    rows, hashes, candidate_ok, paired_ok = [], {}, {}, {}
    first = None
    for tier, divisor in TIERS:
        identity, roster, observed, candidate_ok[tier], paired_ok[tier], hashes[tier] = read_tier(
            folders[tier], tier, max_budget / divisor)
        if first is None:
            first = (identity, roster)
        elif (identity, roster) != first:
            raise ValueError(f"{tier}: submission, paired references, workload, draw identity or device changed")
        rows.extend(observed)

    identity, _ = first
    candidate, packages = identity["candidate"], identity["packages"]
    cells = {(p["tier"], p["track"], p["method"]): p for p in rows}
    refs = [m for m in packages if m != candidate]
    by_track = {}
    for track in TRACKS:
        q = {t: cells[t, track, candidate]["clustered_pct"] for t, _ in TIERS}
        delta = {t: q[t] - max(cells[t, track, ref]["clustered_pct"] for ref in refs) if paired_ok[t] else None
                 for t, _ in TIERS}
        by_track[track] = {"quality_pct": q, "delta_pp_vs_best_paired_at_tier": delta,
                           "retention_vs_full": {t: q[t] / q["full"] if q["full"] > 0 and candidate_ok[t] and candidate_ok["full"] else None
                                                 for t in ("tenth", "quarter")}}
    macro = {t: fmean(v["quality_pct"][t] for v in by_track.values()) for t, _ in TIERS}
    qualified = all(paired_ok.values())
    return rows, {**identity, "protocol": PROTOCOL, "timing_profile": PROFILE,
                  "submission_sha256": packages[candidate], "input_hashes": hashes,
                  "max_budget_s_per_slot": float(max_budget),
                  "candidate_complete_at_tier": candidate_ok, "paired_complete_at_tier": paired_ok,
                  "status": "complete" if qualified else "descriptive_only",
                  "primary_full_budget_quality": macro["full"]/100 if candidate_ok["full"] else None,
                  "macro_quality_pct": macro,
                  "macro_low_budget_quality_pct": ((macro["tenth"] + macro["quarter"])/2
                                                   if candidate_ok["tenth"] and candidate_ok["quarter"] else None),
                  "macro_delta_pp_vs_best_paired_at_tier": {
                      t: fmean(v["delta_pp_vs_best_paired_at_tier"][t] for v in by_track.values()) if paired_ok[t] else None
                      for t, _ in TIERS},
                  "by_track": by_track}


def publish(output, rows, report):
    from matplotlib import pyplot as plt

    if output.exists():
        raise FileExistsError("Choose a fresh report directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "report"
        stage.mkdir()
        (stage / "scorecard.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        with (stage / "points.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        for ax, track in zip(axes.flat, TRACKS):
            track_rows = [r for r in rows if r["track"] == track]
            for method in dict.fromkeys(r["method"] for r in track_rows):
                pts = [r for r in track_rows if r["method"] == method]
                xs = [r["total_s_per_attempt"] for r in pts]
                ys = [r["clustered_pct"] for r in pts]
                line, = ax.plot(xs, ys, 'o-', label=method)
                if method == report["candidate"]:
                    for r, x, y in zip(pts, xs, ys):
                        ax.annotate(r["tier"], (x, y), xytext=(4, 5), textcoords="offset points",
                                    fontsize=8, color=line.get_color())
            ax.set(title=track, xlabel="Measured GPU-seconds per attempt, including final AF3",
                   ylabel="Clustered pass rate (%)")
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        fig.suptitle(f"Three-budget quality and cost | {report['panel']} | {report['status'].replace('_',' ')}")
        fig.savefig(stage / "cost-quality.png", dpi=160)
        plt.close(fig)
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name, _ in TIERS:
        parser.add_argument("--" + name, required=True, type=Path, help="Frozen report directory")
    parser.add_argument("--max-budget", type=Decimal, default=Decimal("60"), help="Predeclared generation seconds per slot")
    parser.add_argument("--output", required=True, type=Path, help="New report directory")
    args = parser.parse_args()
    rows, report = build({t: getattr(args,t) for t, _ in TIERS}, args.max_budget)
    publish(args.output, rows, report)
    print(json.dumps({"status": report["status"], "macro_quality_pct": report["macro_quality_pct"]}))
