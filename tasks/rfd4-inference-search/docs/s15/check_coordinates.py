"""Recheck finished monomer observations from raw CIFs using Gemmi's independent fit."""
import argparse
import csv
import json
import math
from pathlib import Path

import gemmi

from run_cohort import LENGTHS, SAMPLERS, SEED, SLOTS


def read_backbone(path, length):
    model, = gemmi.read_structure(str(path))
    chain, = model
    residues = sorted(chain, key=lambda r: r.label_seq)
    if [r.label_seq for r in residues] != list(range(1, length + 1)):
        raise ValueError(f"Incomplete monomer in {path}")
    sequence = "".join(gemmi.find_tabulated_residue(r.name).one_letter_code for r in residues)
    atoms = [gemmi.Position(*r[key][0].pos.tolist()) for r in residues for key in ("N", "CA", "C", "O")]
    return sequence, atoms


def check_cohort(root, arm, length):
    work = root / arm / str(length)
    refolds = work / "evaluation"
    folder = refolds / str(length)
    observations = json.loads((folder / "measurements.json").read_text())
    paths = json.loads((work / "generation.json").read_text())["paths"]
    if len(paths) != SLOTS or len(set(paths)) != SLOTS:
        raise ValueError(f"Incomplete generation manifest at {work}")
    assignments = {slot: slot for slot in observations}
    if len(observations) > 1:
        with (folder / "clusters_cluster.tsv").open() as f:
            assignments = {Path(member).stem: Path(rep).stem for rep, member in csv.reader(f, delimiter="\t")}
    queries = {p.stem for p in (refolds / "queries").glob("*.json")}
    if set(assignments) != set(observations) or queries != set(observations):
        raise ValueError(f"Inconsistent AF3 query, cluster or observation set at {work}")
    hits = []
    differences = []
    for index, path in enumerate(paths):
        slot = f"l{length}d{index:03d}"
        if slot not in queries:
            continue
        q = json.loads((refolds / "queries" / f"{slot}.json").read_text())
        seq, coords = read_backbone(work / path, length)
        seq_refold, reference = read_backbone(refolds / "refolds" / slot / f"{slot}_model.cif", length)
        protein, = q["sequences"]
        expected = {"id": "A", "sequence": seq, "unpairedMsa": "", "pairedMsa": "", "templates": []}
        if seq != seq_refold or q["name"] != slot or q["modelSeeds"] != [SEED] or protein["protein"] != expected:
            raise ValueError(f"AF3 query does not match monomer slot at {work}/{slot}")
        rmsd = gemmi.superpose_positions(coords, reference).rmsd
        record = observations[slot]
        if not math.isclose(rmsd, record["scRMSD_A"], abs_tol=1e-6) or assignments[slot] != record["cluster"]:
            raise ValueError(f"RMSD or cluster differs at {work}/{slot}")
        differences.append(abs(rmsd - record["scRMSD_A"]))
        if rmsd <= 1.5:
            hits.append(assignments[slot])
    counts = json.loads((work / "result.json").read_text())["by_length"][str(length)]
    if (len(differences), len(hits), len(set(hits))) != (counts["refolded"], counts["passes"], counts["passing_clusters"]):
        raise ValueError(f"Raw and recorded outcomes differ at {work}")
    return {"refolds_checked": len(differences), "passes": len(hits), "passing_clusters": len(set(hits)),
            "max_abs_scRMSD_difference_A": max(differences, default=0)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results = {arm: {n: check_cohort(args.root, arm, n) for n in LENGTHS} for arm in SAMPLERS}
    with args.output.open("x") as f:
        json.dump({"gemmi_version": gemmi.__version__, "by_sampler_and_length": results}, f, indent=2)
        f.write("\n")
