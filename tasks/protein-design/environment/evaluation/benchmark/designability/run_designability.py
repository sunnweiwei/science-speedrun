#!/usr/bin/env python
"""Score a batch of designs on the real designability criterion.

This is a *batch driver*, not a second implementation: every criterion, threshold and
geometry call comes from `tests/pipeline.py::DesignabilityScorer`, which is the grader.
The only thing added here is looping over many designs, keeping per-design records and
summarising per track - so this harness and the grader cannot drift apart.

    python run_designability.py --selection selection.json --out designability.json \\
        [--assets /assets] [--ligandmpnn /opt/ligandmpnn]

Both models are on the machine that runs this: LigandMPNN from the pinned clone, and
RoseTTAFold3 from the source vendored with the starter plus a public checkpoint. Fetch
the parameters anywhere with:

    bash environment/fetch_designability_weights.sh /assets

Cost: one RF3 call per design (all its sequences batched into that call), each loading a
~2.8 GB checkpoint once. Budget roughly a minute per design on one modern GPU, so a
360-design evaluation view is hours, not minutes - subset with `select_designs.py` when
you only need a read on where a checkpoint stands.
"""

import argparse
import collections
import json
import os
import pathlib
import sys
import time

import numpy as np

def _task_root(start):
    """The directory holding task.toml. Located by MARKER, not by a fixed number of
    `.parents[]` hops, because this file has already been relocated once and a depth
    count breaks silently when it moves."""
    for parent in [start, *start.parents]:
        if (parent / "task.toml").is_file():
            return parent
    raise RuntimeError(f"cannot locate the task root (no task.toml at or above {start})")


TASK = pathlib.Path(os.environ.get(
    "PROTEIN_DESIGN_TASK", _task_root(pathlib.Path(__file__).resolve())))
sys.path.insert(0, str(TASK / "tests"))

from geometry import parse_allatom_pdb                              # noqa: E402
from pipeline import DesignabilityScorer, ScorerUnavailable         # noqa: E402

TRACKS = ("unconditional", "sm_binder", "ppi_binder", "enzyme")


def design_length_of(path):
    return len({line[22:27] for line in path.read_text().splitlines()
                if line.startswith("ATOM")})


def load_context(path):
    """npz -> float64 arrays, which is what the criteria expect."""
    raw = np.load(path, allow_pickle=False)
    return {name: (value.astype(np.float64) if value.dtype.kind in "fiu" else value)
            for name, value in raw.items()}


def summarise(records):
    summary = {}
    for track in TRACKS:
        rows = {key: value for (name, key), value in records.items() if name == track}
        if not rows:
            continue
        passed = sum(1 for row in rows.values() if row.get("passed"))
        finite = [row["sc_rmsd"] for row in rows.values()
                  if isinstance(row.get("sc_rmsd"), float) and np.isfinite(row["sc_rmsd"])]
        entry = {"designs": len(rows), "passed": passed,
                 "pass_rate": passed / len(rows),
                 "median_sc_rmsd": float(np.median(finite)) if finite else None,
                 "min_sc_rmsd": float(np.min(finite)) if finite else None,
                 "errors": sum(1 for row in rows.values() if row.get("error"))}

        def collect(field):
            return [row[field] for row in rows.values()
                    if isinstance(row.get(field), (int, float))]

        for field in ("iptm", "interface_pae", "motif_rmsd", "ligand_rmsd"):
            values = collect(field)
            if values:
                entry[f"median_{field}"] = float(np.median(values))
                entry[f"best_{field}"] = float(
                    np.max(values) if field == "iptm" else np.min(values))
        summary[track] = entry

        print(f"### {track}")
        for name, value in entry.items():
            print(f"      {name:24s} {value}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True, type=pathlib.Path,
                        help="JSON from select_designs.py")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--assets", default=os.environ.get(
        "PROTEIN_DESIGN_ASSETS", "/assets"))
    parser.add_argument("--ligandmpnn", default=os.environ.get(
        "LIGANDMPNN_ROOT", "/opt/ligandmpnn"))
    parser.add_argument("--timeout", type=int, default=3600)
    arguments = parser.parse_args()

    try:
        scorer = DesignabilityScorer(assets_root=arguments.assets,
                                     ligandmpnn_root=arguments.ligandmpnn,
                                     timeout=arguments.timeout)
    except ScorerUnavailable as error:
        raise SystemExit(f"cannot score: {error}")

    selection = json.loads(arguments.selection.read_text())
    records = collections.OrderedDict()

    for track in TRACKS:
        for key, info in selection.get(track, {}).items():
            pdb = pathlib.Path(info["structure"])
            structure = parse_allatom_pdb(pdb, design_length_of(pdb))
            context = load_context(info["context"])
            started = time.time()
            try:
                verdict = scorer.evaluate(track, structure, context,
                                          info.get("motif_map"))
            except ScorerUnavailable as error:
                # One design failing to fold must not lose the other 67.
                verdict = {"passed": False, "error": str(error)[:300]}
            verdict["seconds"] = round(time.time() - started, 1)
            records[(track, key)] = verdict
            flag = "PASS" if verdict.get("passed") else (
                "ERR " if verdict.get("error") else "fail")
            print(f"{flag} {track:15s} {key:28s} {verdict['seconds']:6.1f}s "
                  f"sc_rmsd={verdict.get('sc_rmsd')}", flush=True)

    print()
    summary = summarise(records)
    arguments.out.write_text(json.dumps(
        {"summary": summary,
         "per_design": {f"{track}|{key}": value
                        for (track, key), value in records.items()}}, indent=1) + "\n")
    print(f"\nwrote {arguments.out}")


if __name__ == "__main__":
    main()
