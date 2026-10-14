"""The verifier's nine track scores from raw FoldBench scorer CSVs (same rules as tests/grade.py).

    python3 track_scores.py SCORING_TARGETS_DIR RAW_DIR

SCORING_TARGETS_DIR holds one <track>.csv of target rows; RAW_DIR the scorer's <track>_ost.csv or
<track>_dockqv2.csv. Every target row stays in the denominator; the highest ranking_score row per
target is used.
- monomers: mean lDDT;
- interfaces: DockQ >= 0.23 and CAPRI acceptable geometry (ligand RMSD <= 10 A or interface
  RMSD <= 4 A); collapsing chains onto each other passes DockQ alone but not this;
- protein-ligand: RMSD < 2 A and lDDT-PLI > 0.8.
"""
import csv
import json
import math
import sys
from pathlib import Path

NA_TRACKS = {"interface_protein_rna", "interface_protein_dna"}
DOCKQ_SUCCESS, CAPRI_LRMSD, CAPRI_IRMSD = 0.23, 10.0, 4.0


def number(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def target_key(track, row):
    if track.startswith("monomer_"):
        return (row["pdb_id"].lower(), row["chain_id"])
    return (row["pdb_id"].lower(),
            row.get("interface_chain_id_1", row.get("native_chain_id_1")),
            row.get("interface_chain_id_2", row.get("native_chain_id_2")))


def capri_acceptable_geometry(row):
    lrmsd, irmsd = number(row.get("lrmsd")), number(row.get("irmsd"))
    return ((lrmsd is not None and 0 <= lrmsd <= CAPRI_LRMSD)
            or (irmsd is not None and 0 <= irmsd <= CAPRI_IRMSD))


def success(track, row):
    if track.startswith("monomer_"):
        value = number(row.get("lddt"))
        return value if value is not None and 0 <= value <= 1 else 0.0
    if track == "interface_protein_ligand":
        rmsd, pli = number(row.get("rmsd")), number(row.get("lddt-pli"))
        return float(rmsd is not None and 0 <= rmsd < 2.0 and pli is not None and 0.8 < pli <= 1)
    value = number(row.get("dockq_score", row.get("dockq")))
    return float(value is not None and DOCKQ_SUCCESS <= value <= 1 and capri_acceptable_geometry(row))


def track_score(track, targets, raw):
    expected = {target_key(track, row) for row in targets}
    selected = {}
    for row in raw:
        key, ranking = target_key(track, row), number(row.get("ranking_score"))
        if key in expected and ranking is not None and (
                key not in selected or ranking > number(selected[key]["ranking_score"])):
            selected[key] = row
    return sum(success(track, selected[key]) for key in expected if key in selected) / max(1, len(targets))


def main():
    targets_dir, raw_dir = Path(sys.argv[1]), Path(sys.argv[2])
    scores = {}
    for path in sorted(targets_dir.glob("*.csv")):
        track = path.stem
        raw_path = raw_dir / f"{track}_{'dockqv2' if track in NA_TRACKS else 'ost'}.csv"
        raw = list(csv.DictReader(raw_path.open())) if raw_path.exists() else []
        scores[track] = track_score(track, list(csv.DictReader(path.open())), raw)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
