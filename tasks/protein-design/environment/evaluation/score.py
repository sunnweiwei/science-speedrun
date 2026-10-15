#!/usr/bin/env python3
"""Self-service development scoring.

Runs exactly the criterion the verifier scores, against a labelled development split.
There are no hidden labels in this task - designability is intrinsic - so the agent may
score itself as often as it likes.

    python score.py --input /data/tune --designs /tmp/designs

There is no geometry-only shortcut. One existed while AlphaFold 3's parameters kept the
real scorer out of this image; RoseTTAFold3's are public and ship here, so the real
metric is what runs. It was removed on 2026-09-26 along with the pre-fold gate it
measured - without that gate it reported every parseable design as designable, which is
a wrong number rather than a cheap one.
"""

import argparse
import os
import json
import sys
from pathlib import Path

# In the image the grader modules sit in `lib/` beside this file, put there by the
# Dockerfile from the task's own `tests/`. Outside the image - running this script
# straight out of a checkout - that directory does not exist, so fall back to the
# source it would have been built from. Same files either way; the Dockerfile is what
# makes that true (`COPY tests/common.py ... /workspace/evaluation/lib/`).
_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE / "lib", _HERE.parent.parent / "tests"):
    if _candidate.is_dir():
        sys.path.insert(0, str(_candidate))
        break

import grader                                        # noqa: E402
from common import TRACKS, validate_input            # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--designs", required=True, type=Path)
    parser.add_argument("--assets", default=os.environ.get(
        "PROTEIN_DESIGN_ASSETS", "/assets"),
        help="LigandMPNN + RoseTTAFold3 parameters; see "
             "environment/fetch_designability_weights.sh")
    parser.add_argument("--ligandmpnn", default=os.environ.get(
        "LIGANDMPNN_ROOT", "/opt/ligandmpnn"))
    parser.add_argument("--tracks", nargs="*", default=list(TRACKS))
    parser.add_argument("--reference", default=os.environ.get(
        "PROTEIN_DESIGN_NOVELTY_REFERENCE", "/workspace/data/fit/structures"),
        help="fit-corpus structures for the novelty gate, as the verifier applies it; "
             "--reference none scores designability alone")
    arguments = parser.parse_args()

    # fit and tune are labelled `development`; the verifier's own view is `verifier`.
    _, requests = validate_input(arguments.input, views=("verifier", "development"))
    from pipeline import DesignabilityScorer
    scorer = DesignabilityScorer(assets_root=arguments.assets,
                                 ligandmpnn_root=arguments.ligandmpnn)

    novelty = None
    if arguments.reference != "none":
        from novelty import FitNovelty
        novelty = FitNovelty(arguments.reference)

    predictions = arguments.designs / "predictions"
    # Per-track pass rates only, exactly as the trusted grader reports them. There is no
    # composite here either: see tests/grader.py's docstring.
    report = {}
    for track in arguments.tracks:
        report[track] = grader.score_track(track, predictions / f"{track}.jsonl",
                                           requests[track], arguments.input, scorer, novelty)
    print(json.dumps(report, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
