#!/usr/bin/env python
"""Stage 1 of the standard benchmark: frozen target list -> RFD3 input specs.

    python make_specs.py --out <dir> [--view <evaluation view>] [--track <name>]

One JSON per track, in RFD3's `InputSpecification` dialect. The three conditioned
tracks reuse upstream's PUBLISHED specs verbatim from
`environment/data/targets/*/input.yaml` - the same `unindex`, `select_fixed_atoms` and
`select_hotspots` strings - with only the design length pinned, exactly as
`benchmark/make_inputs.py` does. That is what makes a number here comparable with
upstream's own benchmark.

`unconditional` is different by construction: the standard benchmark asks for a fixed
LENGTH (100) rather than a materialized target, so its spec is synthesised here and
needs no view, no context and no input structure.
"""

import argparse
import json
import pathlib
import sys

import yaml

def _task_root(start):
    """The directory holding task.toml. Located by MARKER, not by a fixed number of
    `.parents[]` hops, because this file has already been relocated once and a depth
    count breaks silently when it moves."""
    for parent in [start, *start.parents]:
        if (parent / "task.toml").is_file():
            return parent
    raise RuntimeError(f"cannot locate the task root (no task.toml at or above {start})")


HERE = pathlib.Path(__file__).resolve().parent
TASK = _task_root(HERE)
sys.path.insert(0, str(HERE.parent))   # benchmark/, for make_inputs.py

from make_inputs import pin_length                                   # noqa: E402

FOLDER = {"sm_binder": "sm", "ppi_binder": "ppi", "enzyme": "enzyme"}
KEY = {"sm_binder": "contig", "ppi_binder": "contig", "enzyme": "length"}


def build(definition, view, targets_root, out, only=None):
    out.mkdir(parents=True, exist_ok=True)
    counts = {}

    for track, entry in definition["tracks"].items():
        if only and track != only:
            continue

        if track == "unconditional":
            specs = {name: {"length": f"{info['design_length']}-{info['design_length']}"}
                     for name, info in entry["targets"].items()}
            (out / "unconditional.json").write_text(json.dumps(specs, indent=2) + "\n")
            counts[track] = len(specs)
            continue

        published = yaml.safe_load(
            (targets_root / FOLDER[track] / "input.yaml").read_text())
        lengths = {}
        for line in (view / f"{track}.jsonl").read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                lengths[row["example_id"]] = int(row["design_length"])

        specs = {}
        for example in entry["targets"]:
            if example not in published:
                raise SystemExit(f"{track}: {example} has no vendored RFD3 spec")
            spec = dict(published[example])
            spec["input"] = str((targets_root / FOLDER[track] / spec["input"]).resolve())
            if KEY[track] == "length":
                spec["length"] = f"{lengths[example]}-{lengths[example]}"
            else:
                spec["contig"] = pin_length(spec["contig"], lengths[example])
                spec.pop("length", None)
            spec["dialect"] = spec.get("dialect", 2)
            if track == "ppi_binder":
                # Upstream's published specs carry no origin strategy, and with both
                # `ori_token` and `infer_ori_strategy` unset, `set_com`
                # (`rfd3/utils/inference.py:460`) falls through to "COM of fixed motif".
                # For a ppi spec the whole target chain is fixed motif, so the origin
                # becomes the TARGET's centre of mass and the diffused binder is
                # initialised inside it. Measured on the 2026-09-20 run: binder-target
                # minimum atom distance 0.07-0.66 A. `protein_binder_design.md` says
                # the "hotspots" strategy "works best"; it puts the origin 10 A along
                # the outward normal from the hotspot centroid.
                spec.setdefault("infer_ori_strategy", "hotspots")
            specs[example] = spec
        (out / f"{track}.json").write_text(json.dumps(specs, indent=2) + "\n")
        counts[track] = len(specs)

    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--definition", type=pathlib.Path,
                        default=HERE / "targets.json")
    parser.add_argument("--view", type=pathlib.Path,
                        default=pathlib.Path("/net/tukwila/yanjing/protein-design-run"
                                             "/data/evaluation"))
    parser.add_argument("--targets-root", type=pathlib.Path,
                        default=TASK / "environment" / "data" / "targets")
    parser.add_argument("--track", default=None)
    arguments = parser.parse_args()

    definition = json.loads(arguments.definition.read_text())
    counts = build(definition, arguments.view, arguments.targets_root,
                   arguments.out, arguments.track)
    for track, number in sorted(counts.items()):
        print(f"{track:15s} {number:3d} targets -> {arguments.out / (track + '.json')}")
    print(f"\n{sum(counts.values())} targets x "
          f"{definition['designs_per_target']} designs")


if __name__ == "__main__":
    main()
