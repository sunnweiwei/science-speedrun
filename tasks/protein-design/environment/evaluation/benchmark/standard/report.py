#!/usr/bin/env python
"""Stage 4: the 14 per-target score files -> one benchmark table.

    python report.py --scores <dir> [--out <json>] [--label <name>]

Per-target pass rates are the unit of this benchmark, and a track is their MEAN over
its targets - not pooled over designs - so a target cannot dominate by contributing
more designs. That matches `grader.py`, which also averages per-target success.

Designability is the only metric: the fraction of designs satisfying the track's
published criterion, as `success.txt` defines it. The task's own geometry and conditioning
gates were deleted on 2026-09-26 and are no longer recorded - see
../../RESULTS.md for what they had been costing.
"""

import argparse
import json
import pathlib
import statistics

TRACKS = ("unconditional", "sm_binder", "ppi_binder", "enzyme")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--label", default="")
    arguments = parser.parse_args()

    by_track, missing = {track: [] for track in TRACKS}, []
    for path in sorted(arguments.scores.glob("*.json")):
        summary = json.loads(path.read_text())["summary"]
        by_track.setdefault(summary["track"], []).append(summary)

    print(f"# standard benchmark{' - ' + arguments.label if arguments.label else ''}\n")
    header = (f"{'track':<14} {'targets':>7} {'designs':>8} "
              f"{'PASS':>7} {'pass rate':>10}   per-target")
    print(header)
    print("-" * len(header))

    table, modes = {}, set()
    for track in TRACKS:
        rows = by_track.get(track) or []
        if not rows:
            missing.append(track)
            continue
        modes.update(row.get("criterion", "?") for row in rows)
        rates = [row["pass_rate"] for row in rows]
        designs = sum(row["designs"] for row in rows)
        passed = sum(row["passed"] for row in rows)
        mean = statistics.fmean(rates)
        detail = " ".join(f"{row['target'].replace('rfd3_', '')}={row['pass_rate']:.2f}"
                          for row in sorted(rows, key=lambda r: r["target"]))
        table[track] = {
            "targets": len(rows), "designs": designs,
            "passed": passed, "track_pass_rate": mean,
            "per_target": {row["target"]: row["pass_rate"] for row in rows},
        }
        print(f"{track:<14} {len(rows):>7} {designs:>8} {passed:>7} "
              f"{mean:>10.3f}   {detail}")

    combined = None
    if set(table) >= {"unconditional", "sm_binder", "ppi_binder", "enzyme"}:
        combined = (table["unconditional"]["track_pass_rate"] / 51.0
                    + (50.0 / 153.0) * (table["sm_binder"]["track_pass_rate"]
                                        + table["ppi_binder"]["track_pass_rate"]
                                        + table["enzyme"]["track_pass_rate"]))
        print(f"\nscore (V3 target-weighted): {combined:.6f}")
    if table:
        print(f"\ncriterion: {', '.join(sorted(modes))}")
    if missing:
        print(f"\nMISSING (not scored yet): {', '.join(missing)}")

    if arguments.out:
        payload = {"label": arguments.label, "criterion": sorted(modes),
                   "tracks": table, "missing": missing}
        if combined is not None:
            payload["score"] = combined
            payload["reward"] = combined
        arguments.out.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"\nwrote {arguments.out}")


if __name__ == "__main__":
    main()
