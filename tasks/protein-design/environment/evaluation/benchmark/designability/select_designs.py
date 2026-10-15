#!/usr/bin/env python
"""Stage 0: choose which designs go through the designability pipeline.

Subsetting is a compute decision, not a scientific one. Scoring all 360 evaluation
designs is `SEQUENCES_PER_BACKBONE x MODELS_PER_SEQUENCE` RoseTTAFold3 seed-inferences
per design - order 8,000 - which is far past a single session. Designs are taken evenly
across targets so that no one target dominates and every chosen target contributes.

    python select_designs.py \
        --source unconditional=$RUNS/ep60-uncond/submission/predictions:$DATA/development/tune \
        --source enzyme=$RUNS/ep60-evaluation/submission/predictions:$DATA/evaluation \
        --targets unconditional=20,enzyme=12 \
        --per-target unconditional=1,enzyme=1 \
        --out $DESIG_WORK/selection.json

`--source TRACK=<submission predictions dir>:<view dir>`. The predictions directory is
what `benchmark/collect.py` or `starter/rfd3_collect.py` wrote; the view directory is
the one the requests came from, because the conditioning context is read from there.
"""

import argparse
import json
import pathlib


def parse_map(text, cast=int):
    """`a=1,b=2` -> {"a": 1, "b": 2}."""
    out = {}
    for item in (text or "").split(","):
        if not item.strip():
            continue
        key, _, value = item.partition("=")
        out[key.strip()] = cast(value)
    return out


def parse_source(text):
    track, _, rest = text.partition("=")
    predictions, _, view = rest.rpartition(":")
    if not track or not predictions or not view:
        raise argparse.ArgumentTypeError(
            f"--source wants TRACK=<predictions dir>:<view dir>, got {text!r}")
    return track.strip(), pathlib.Path(predictions), pathlib.Path(view)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=parse_source, required=True,
                        help="TRACK=<predictions dir>:<view dir>, repeatable")
    parser.add_argument("--targets", default="",
                        help="per-track cap on targets, e.g. 'enzyme=12,sm_binder=4'")
    parser.add_argument("--per-target", default="",
                        help="per-track designs per target, default 1")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    arguments = parser.parse_args()

    caps = parse_map(arguments.targets)
    per_target = parse_map(arguments.per_target)

    selection = {}
    for track, predictions, view in arguments.source:
        requests = {}
        for line in (view / f"{track}.jsonl").read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                requests[row["example_id"]] = row
        rows = [json.loads(line)
                for line in (predictions / f"{track}.jsonl").read_text().splitlines()
                if line.strip()]
        rows.sort(key=lambda row: row["example_id"])

        cap = caps.get(track, len(rows))
        step = max(1, len(rows) // cap) if cap else 1
        chosen = rows[::step][:cap]

        selection[track] = {}
        for row in chosen:
            request = requests[row["example_id"]]
            for design in row["designs"][:per_target.get(track, 1)]:
                key = f"{row['example_id']}#{design['design_index']}"
                selection[track][key] = {
                    "example_id": row["example_id"],
                    "structure": str(predictions / "structures"
                                     / design["structure_path"]),
                    "context": str(view / request["context_path"]),
                    "motif_map": design.get("motif_map"),
                }
        print(f"{track:15s} {len(chosen):3d} of {len(rows):3d} targets "
              f"-> {len(selection[track]):3d} designs")

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(selection, indent=1) + "\n")
    print(f"\n{sum(len(v) for v in selection.values())} designs -> {arguments.out}")


if __name__ == "__main__":
    main()
