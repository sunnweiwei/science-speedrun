#!/usr/bin/env python3
"""Write only GLACIER candidate rows absent from existing PredSpecDB files.

This recovery helper is intentionally answer-free. It compares the public
candidate table with prediction keys already committed to one or more HDF5
files. Invalid structures are left for the scorer's deterministic worst-score
path rather than retried forever.
"""

import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path

from rdkit import Chem

import ms_pred.common as common
from ms_pred.common import chem_utils


@lru_cache(maxsize=262144)
def candidate_prediction_key(smiles: str) -> tuple[str, str]:
    """Return GLACIER's lookup key and the upstream support classification."""
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return "", "rdkit_invalid"
        if any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()):
            return "", "wildcard_atom"
        if any(
            atom.GetAtomicNum() not in chem_utils.VALID_ATOM_NUM
            for atom in molecule.GetAtoms()
        ):
            return Chem.MolToInchiKey(molecule), "unsupported_atom"
        molecule = Chem.RemoveHs(molecule)
        if molecule.GetNumAtoms() > 100:
            return Chem.MolToInchiKey(molecule), "more_than_100_heavy_atoms"
        return Chem.MolToInchiKey(molecule), "supported"
    except Exception:
        return "", "structure_exception"


def existing_pairs(databases, prediction_name: str) -> set[tuple[str, str]]:
    pairs = set()
    for database in databases:
        spectra, present = database.read_from_name(prediction_name)
        if not present:
            continue
        for remark, collision_spectra in spectra.items():
            normalized = str(remark).removeprefix("ikey ").strip()
            pairs.update(
                (normalized, common.get_collision_energy(collision_energy))
                for collision_energy in collision_spectra
            )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--pred-file", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    databases = [common.PredSpecDB(path.resolve(), mode="r") for path in args.pred_file]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    input_rows = 0
    missing_rows = 0
    excluded_structures = 0
    unsupported_structures = {}
    duplicate_missing_pairs = 0
    query_count = 0
    queries_with_missing = set()
    seen_missing = set()
    active_spec = None
    active_pairs = set()
    with args.candidates.open(encoding="utf-8", newline="") as source, args.output.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        rows = csv.DictReader(source, delimiter="\t")
        if rows.fieldnames is None:
            raise ValueError("candidate table has no header")
        writer = csv.DictWriter(target, fieldnames=rows.fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            input_rows += 1
            spec = row["spec"]
            if spec != active_spec:
                active_spec = spec
                active_pairs = existing_pairs(databases, f"pred_{spec}")
                query_count += 1
            inchikey, support = candidate_prediction_key(row["smiles"])
            if support != "supported":
                excluded_structures += 1
                unsupported_structures[support] = (
                    unsupported_structures.get(support, 0) + 1
                )
                continue
            collision_energy = common.get_collision_energy(
                json.loads(row["collision_energies"])[0]
            )
            pair = (inchikey, collision_energy)
            if pair in active_pairs:
                continue
            missing_key = (spec, *pair)
            if missing_key in seen_missing:
                duplicate_missing_pairs += 1
                continue
            seen_missing.add(missing_key)
            writer.writerow(row)
            missing_rows += 1
            queries_with_missing.add(spec)

    for database in databases:
        database.close()
    report = {
        "schema": "glacier-resume-candidates/v2",
        "queries": query_count,
        "input_rows": input_rows,
        "missing_rows": missing_rows,
        "queries_with_missing": len(queries_with_missing),
        "excluded_structure_rows_not_retried": excluded_structures,
        "invalid_structure_rows_not_retried": sum(
            unsupported_structures.get(reason, 0)
            for reason in ("rdkit_invalid", "structure_exception")
        ),
        "model_unsupported_rows_not_retried": dict(sorted(unsupported_structures.items())),
        "duplicate_missing_pairs_not_retried": duplicate_missing_pairs,
        "answer_fields_read": [],
        "prediction_files": len(databases),
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
