#!/usr/bin/env python
"""Turn a protein-design input view into RFdiffusion3 inference specs.

The task hands out requests as `<track>.jsonl` rows plus an `.npz` conditioning context.
Upstream RFD3 wants an InputSpecification JSON: an input structure, a contig, and
selections over it. For the three conditioned tracks the bridge is almost free, because
the task vendors upstream's OWN benchmark specs beside the target structures
(`environment/data/targets/{enzyme,sm,ppi}/input.yaml`) - the same `unindex` /
`select_fixed_atoms` / `select_hotspots` strings upstream used to publish its numbers.
We reuse those verbatim and change exactly one thing: the design length is pinned to the
length the request asks for, because the grader rejects a design whose residue count is
not `design_length` (`tests/geometry.py::parse_allatom_pdb`).

unconditional has no structure at all - a length is the whole request.
"""

import argparse
import json
from pathlib import Path

import yaml


def _task_root(start):
    """The directory holding task.toml. Located by MARKER, not by a fixed number of
    `.parents[]` hops, because this file has already been relocated once and a depth
    count breaks silently when it moves."""
    for parent in [start, *start.parents]:
        if (parent / "task.toml").is_file():
            return parent
    raise RuntimeError(f"cannot locate the task root (no task.toml at or above {start})")


def load_requests(view, track):
    path = view / f"{track}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pin_length(contig, length):
    """Replace the leading designed range of a contig with an exact length.

    Upstream samples `40-120` uniformly; the request asks for a specific number, so the
    first comma-separated field - always the designed segment in these benchmarks - is
    replaced by that number. Fixed segments carry a chain letter and are left alone.
    """
    fields = contig.split(",")
    if fields[0][:1].isalpha():
        raise ValueError(f"contig does not start with a designed segment: {contig}")
    # `150` on its own is parsed as residue index 150, not a 150-residue design; a
    # range is what makes it a designed segment. So pin it as `150-150`.
    fields[0] = f"{length}-{length}"
    return ",".join(fields)


def build(view, targets, out, tracks):
    out.mkdir(parents=True, exist_ok=True)
    specs, counts = {}, {}

    # --- unconditional: length only, no input structure -----------------------------
    rows = load_requests(view, "unconditional") if "unconditional" in tracks else []
    if rows:
        specs["unconditional"] = {
            row["example_id"]: {"length": f"{row['design_length']}-{row['design_length']}"}
            for row in rows}
        counts["unconditional"] = len(rows)

    # --- the three conditioned tracks: upstream's own specs, lengths pinned ---------
    for track, folder, key in (("enzyme", "enzyme", "length"),
                               ("sm_binder", "sm", "contig"),
                               ("ppi_binder", "ppi", "contig")):
        rows = load_requests(view, track) if track in tracks else []
        if not rows:
            continue
        published = yaml.safe_load((targets / folder / "input.yaml").read_text())
        track_specs = {}
        for row in rows:
            example, length = row["example_id"], row["design_length"]
            if example not in published:
                # Only the vendored evaluation targets ship upstream's published spec.
                # A materialized view (fit/tune) has no equivalent, so say so plainly
                # rather than inventing a spec that would not be comparable.
                raise SystemExit(
                    f"{track}: no vendored RFD3 spec for {example}. Only the evaluation "
                    f"view has them; for fit/tune run --tracks unconditional.")
            spec = dict(published[example])
            spec["input"] = str((targets / folder / spec["input"]).resolve())
            if key == "length":
                spec["length"] = f"{length}-{length}"
            else:
                spec["contig"] = pin_length(spec["contig"], length)
                spec.pop("length", None)
            spec["dialect"] = spec.get("dialect", 2)
            track_specs[example] = spec
        specs[track] = track_specs
        counts[track] = len(rows)

    for track, track_specs in specs.items():
        (out / f"{track}.json").write_text(json.dumps(track_specs, indent=2) + "\n")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", required=True, type=Path,
                        help="an input view: evaluation, tune, or fit")
    parser.add_argument("--targets", type=Path,
                        default=_task_root(Path(__file__).resolve())
                                / "environment" / "data" / "targets")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tracks", nargs="*",
                        default=["unconditional", "sm_binder", "ppi_binder", "enzyme"])
    arguments = parser.parse_args()
    counts = build(arguments.view, arguments.targets, arguments.out, set(arguments.tracks))
    for track, number in counts.items():
        print(f"{track:15s} {number:3d} targets -> {arguments.out / (track + '.json')}")


if __name__ == "__main__":
    main()
