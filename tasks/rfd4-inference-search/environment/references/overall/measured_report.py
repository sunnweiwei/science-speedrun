"""Plot the official suites using independently measured, controller-owned records."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile

from benchmark_curve import ARMS, HERE, TRACKS, read_targets, read_trials, summarize, write_points
from plot_curve import plot

PROFILE = "rfd4-a6000-warm-v1"


def points(baselines, candidate):
    # Only target identities come from the official roster.
    roster = read_targets(HERE / "targets.csv")
    results = []
    for groups, source in ((baselines, "local baseline"), (candidate, "independent submission measurement")):
        for (track, method), targets in groups.items():
            if targets.keys() != roster[track, "Low"].keys():
                raise ValueError(f"Incomplete or mismatched official roster: {track} / {method}")
            if (method in ARMS) != (source == "local baseline"):
                raise ValueError(f"Reserve Low, Medium, High and Search for reproduced baselines: {method}")
            result = summarize(track, method, targets, source)
            speed = result["generation_attempts_h"]
            comparators = [p for p in results if p["track"] == track and p["source"] == "local baseline"
                           and p["generation_attempts_h"] >= speed]
            result["gain_pp_at_matched_or_faster_baseline"] = (
                result["clustered_pct"] - max(p["clustered_pct"] for p in comparators)
                if source != "local baseline" and comparators else None)
            results.append(result)
    return results


def submission_measurements(receipts, digest):
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Expected a submission SHA-256")
    receipt = receipts / (digest + ".json")
    ledger = receipts / (digest + ".csv")
    if not receipt.exists():
        if ledger.exists():
            raise ValueError("Independent trial ledger has no controller receipt")
        return {}
    meta = json.loads(receipt.read_text())
    if meta["submission_sha256"] != digest or meta["profile"] != PROFILE:
        raise ValueError("Package identity or measurement profile mismatch")
    if hashlib.sha256(ledger.read_bytes()).hexdigest() != meta["trials_sha256"]:
        raise ValueError("Independent trial ledger differs from its controller receipt")
    groups = read_trials(ledger)
    if not groups:
        raise ValueError("An independent trial ledger must have at least one complete cohort")
    return groups


def render(output, *, receipts=None, digest=None, baseline_file=None):
    groups = read_targets(HERE / "reproduced_targets.csv")
    if baseline_file:
        for key, targets in read_targets(baseline_file).items():
            if key in groups and groups[key] != targets:
                raise ValueError(f"Additional evidence conflicts with the pinned baseline: {key}")
            groups[key] = targets
    candidate = submission_measurements(receipts, digest) if receipts is not None else {}
    rows = points(groups, candidate)
    status = {"profile": PROFILE, "submission_sha256": digest,
              "baseline_points": len(groups), "official_baseline_points": len(TRACKS) * len(ARMS),
              "submission_points": len(candidate), "tracks": {}}
    for track in TRACKS:
        status["tracks"][track] = {"reproduced": [m for m in ARMS if (track, m) in groups],
                                   "missing": [m for m in ARMS if (track, m) not in groups],
                                   "submission": sorted(m for t, m in candidate if t == track)}
    if output.exists():
        raise FileExistsError(f"Choose a fresh report directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "report"
        stage.mkdir()
        with (stage / "points.csv").open("w", newline="") as f:
            write_points(rows, f)
        (stage / "coverage.json").write_text(json.dumps(status, indent=2) + "\n")
        for name, af3 in (("generation.png", False), ("including-af3.png", True)):
            plot(rows, stage / name, include_af3=af3, panel_tracks=TRACKS,
                 source_note="Independent A6000 measurements, warm GPU timing")
        stage.rename(output)
    return status


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--receipts", type=Path, help="Read-only controller directory for sealed submission measurements")
    p.add_argument("--submission-sha256")
    p.add_argument("--baseline-targets", type=Path, help="Additional trusted independent baseline targets in the same profile")
    args = p.parse_args()
    if (args.receipts is None) != (args.submission_sha256 is None):
        p.error("Provide both the receipts directory and the submission digest, or neither")
    print(json.dumps(render(args.output, receipts=args.receipts, digest=args.submission_sha256,
                            baseline_file=args.baseline_targets)))
