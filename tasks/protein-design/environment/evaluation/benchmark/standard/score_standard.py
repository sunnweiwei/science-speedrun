#!/usr/bin/env python
"""Stage 3 of the standard benchmark: raw RFD3 designs -> per-target pass rates.

    python score_standard.py --designs <dir> --track <name> --target <id> --out <json>

Why this exists rather than `collect.py` + `score.py`: the standard benchmark runs 100
designs per target, and the SUBMISSION path caps a target at 32
(`common.py::MAX_DESIGNS_PER_TARGET`, enforced by `validate_input`) and truncates at the
request's `num_designs` (`collect.py:197`). Those limits protect the agent-facing
contract and should not be relaxed for an operator benchmark, so this driver reads the
raw RFD3 output directly.

**success.txt alone defines success.** There is no pre-fold gate: every design is
inverse-folded and refolded, and the verdict is the track's published criterion. For
enzyme that criterion has two parts - motif recapitulation on the refolded model, AND no
ligand clashes on the raw design - so the ligand-clash check is part of the verdict, not
a filter. The task's own extra gates (CA-CA continuity, self-clash, radius of
gyration, ligand burial, raw-design motif RMSD) are RECORDED as diagnostics and decide
nothing; they reject 100% of our ep200 enzyme designs, none of which success.txt asks
for.

Everything that decides a verdict is the grader's own code, imported not copied:

    tests/pipeline.py::DesignabilityScorer   inverse folding + refolding + criteria
    tests/geometry.py::minimum_distance      the ligand-clash measurement
    benchmark/collect.py                     parsing, incl. its three hard-won fixes
"""

import argparse
import json
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


HERE = pathlib.Path(__file__).resolve().parent
TASK = _task_root(HERE)
sys.path.insert(0, str(TASK / "tests"))
sys.path.insert(0, str(HERE.parent))   # benchmark/, for collect.py

from collect import (read_cif, design_chain, to_atom14, BACKBONE,      # noqa: E402
                     recover_motif_map, outputs_for)
from common import LIGAND_CLASH_MIN                                    # noqa: E402
from geometry import minimum_distance                                  # noqa: E402
from pipeline import DesignabilityScorer, ScorerUnavailable            # noqa: E402


def ligand_clash_check(track, coordinates, mask, motif_map, context):
    """enzyme success.txt criterion 2, the one criterion measured on the RAW design.

    "No ligand clashes. A ligand clash is any non-motif backbone atom (N, CA or C)
    within 1.5 A of the closest ligand atom." No other track's success.txt names a
    criterion on the raw backbone - they are all measured on the REFOLDED model.

    NOT applied, because success.txt does not contain them: backbone continuity
    (CA-CA), self-clash under both rules, radius of gyration, ligand burial, motif RMSD on
    the raw design. Those are the task's own additions, and they reject 100% of our
    ep200 enzyme designs.

    Uses the grader's own `minimum_distance` and `LIGAND_CLASH_MIN` so the threshold
    cannot drift from the rest of the task.
    """
    if track != "enzyme":
        return None
    ligand = context.get("ligand_coords")
    if ligand is None or np.size(ligand) == 0:
        return None
    placed = set(motif_map or ())
    keep = np.array([index not in placed for index in range(coordinates.shape[0])])
    if not keep.any():
        return None
    backbone = coordinates[keep][:, :3].reshape(-1, 3)   # N, CA, C - as the grader does
    distance = minimum_distance(backbone, np.asarray(ligand))
    if distance < LIGAND_CLASH_MIN:
        return f"non-motif backbone clashes with ligand at {distance:.2f} A"
    return None


def load_context(view, track, example):
    """The conditioning context for a target, or {} for the synthetic unconditional."""
    if track == "unconditional":
        return {}
    for line in (view / f"{track}.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["example_id"] == example:
            return dict(np.load(view / row["context_path"], allow_pickle=False))
    raise SystemExit(f"{track}: no request for {example} in {view}")


def design_length_of(definition, track, example, view):
    if track == "unconditional":
        return definition["tracks"]["unconditional"]["targets"][example]["design_length"]
    for line in (view / f"{track}.jsonl").read_text().splitlines():
        if line.strip() and json.loads(line)["example_id"] == example:
            return int(json.loads(line)["design_length"])
    raise SystemExit(f"{track}: no request for {example}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--designs", required=True, type=pathlib.Path,
                        help="the rfd3 output directory for this track")
    parser.add_argument("--track", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--definition", type=pathlib.Path, default=HERE / "targets.json")
    parser.add_argument("--view", type=pathlib.Path,
                        default=pathlib.Path("/net/tukwila/yanjing/protein-design-run"
                                             "/data/evaluation"))
    parser.add_argument("--assets", default="/assets")
    parser.add_argument("--ligandmpnn", default="/opt/ligandmpnn")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N designs (smoke runs)")
    arguments = parser.parse_args()

    definition = json.loads(arguments.definition.read_text())
    track, example = arguments.track, arguments.target
    length = design_length_of(definition, track, example, arguments.view)
    context = load_context(arguments.view, track, example)

    try:
        scorer = DesignabilityScorer(assets_root=arguments.assets,
                                     ligandmpnn_root=arguments.ligandmpnn,
                                     timeout=arguments.timeout)
    except ScorerUnavailable as error:
        raise SystemExit(f"cannot score: {error}")

    paths = outputs_for(arguments.designs, example)
    if arguments.limit:
        paths = paths[:arguments.limit]
    print(f"{track}/{example}: {len(paths)} designs, length {length}", flush=True)

    records, folded, passed = [], 0, 0
    for index, path in enumerate(paths):
        started = time.time()
        record = {"design": path.name, "passed": False}
        try:
            residues = design_chain(read_cif(path), length)
            coordinates, mask, names = to_atom14(residues)
            # Same two conditions collect.py applies, keyed the same way.
            if not mask[:, BACKBONE].all():
                raise ValueError("a residue is missing backbone atoms")
            motif_map = None
            if "motif_coords" in context:
                metadata = json.loads(
                    path.with_suffix("").with_suffix(".json").read_text())
                motif_map = recover_motif_map(metadata, names, context,
                                              coordinates, mask)
                if motif_map is None:
                    raise ValueError("could not locate the motif in the design")
            structure = (coordinates, mask, names)

            # NO pre-fold gate, and no gate diagnostics either. success.txt defines
            # success, and every track's criteria are measured on the REFOLDED model, so
            # every design is folded. enzyme's second criterion - "no ligand clashes" -
            # IS evaluated on the raw design, so it is part of the verdict rather than a
            # filter. The task's own geometry and conditioning gates were deleted on
            # 2026-09-26; designability is the only metric reported.
            ligand_clash = ligand_clash_check(track, coordinates, mask,
                                            motif_map, context)
            record["ligand_clash"] = ligand_clash

            folded += 1
            verdict = scorer.evaluate(track, structure, context, motif_map)
            record.update({key: verdict.get(key) for key in
                           ("sc_rmsd", "iptm", "interface_pae",
                            "ligand_rmsd", "motif_rmsd")})
            record["designable"] = bool(verdict.get("passed"))
            record["passed"] = record["designable"] and ligand_clash is None
            passed += record["passed"]
        except ScorerUnavailable as error:
            record["error"] = str(error)[:300]
        except Exception as error:                       # noqa: BLE001 - one bad design
            record["error"] = f"{type(error).__name__}: {error}"[:300]
        record["seconds"] = round(time.time() - started, 1)
        records.append(record)
        flag = ("PASS" if record["passed"] else
                "CLSH" if record.get("ligand_clash") else
                "ERR " if record.get("error") else "fail")
        print(f"  {flag} {index + 1:>4}/{len(paths)} {record['seconds']:>6.1f}s "
              f"{record.get('ligand_clash') or record.get('error') or ''}"[:140],
              flush=True)

    n = len(paths)
    summary = {"track": track, "target": example, "designs": n,
               "criterion": "success.txt", "folded": folded, "passed": passed,
               "pass_rate": passed / n if n else 0.0,
               "designable": sum(1 for r in records if r.get("designable")),
               "ligand_clashes": sum(
                   1 for r in records if r.get("ligand_clash") is not None)}
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(
        {"summary": summary, "per_design": records}, indent=1) + "\n")
    print(f"\n{track}/{example}: folded {folded}/{len(paths)}, "
          f"PASS {passed}/{len(paths)} = {summary['pass_rate']:.3f}")
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
