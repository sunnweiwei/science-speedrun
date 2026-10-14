#!/usr/bin/env python3
"""Project the frozen 100K split into an official Protenix native index.

One native index row is selected deterministically for every supported logical
training example.  The source split is never supplemented.  Records that
cannot be represented by the available official bioassembly cache are retained
in the audit as named rejections.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Iterable

from agent_pipeline.contract import (
    NATIVE_VIEW_AUDIT_SCHEMA,
    load_json,
    sha256_file,
    validate_dataset_package,
)


TASK_EVAL_TYPES = {
    "monomer_protein": {"intra_prot"},
    "monomer_rna": {"intra_rna"},
    "monomer_dna": {"intra_dna"},
    "interface_protein_protein": {"prot_prot"},
    "interface_antibody_antigen": {"prot_prot"},
    "interface_protein_peptide": {"prot_prot"},
    "interface_protein_rna": {"rna_prot"},
    "interface_protein_dna": {"dna_prot"},
    "interface_protein_ligand": {"ligand_prot"},
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-package", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--upstream-index", type=Path, required=True)
    parser.add_argument("--bioassembly-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid"), default="train")
    parser.add_argument(
        "--esmc-manifest",
        type=Path,
        help="when set, require a ready mixed-256 feature for every protein sequence",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="reject native rows above this declared model inference limit",
    )
    parser.add_argument("--model-id", default="protenix_pairformer_50m")
    return parser.parse_args()


def open_csv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def stable_row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(key, "")) for key in (
        "eval_type",
        "type",
        "chain_1_id",
        "chain_2_id",
        "entity_1_id",
        "entity_2_id",
    ))


def choose_row(record: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, str] | None:
    desired = TASK_EVAL_TYPES.get(record.get("task"))
    candidates = rows
    if desired is not None:
        candidates = [row for row in rows if row.get("eval_type") in desired]
    if not candidates:
        return None
    return min(candidates, key=stable_row_key)


def passes_official_exclusions(row: dict[str, str]) -> bool:
    """Mirror the released weighted-PDB ion exclusion before auditing rows."""

    return row.get("mol_1_type") != "ions" and row.get("mol_2_type") != "ions"


def ready_sequences(path: Path) -> set[str]:
    ready: set[str] = set()
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("local_status") == "ready":
                ready.add(record["sequence_sha256"])
    return ready


def protein_sequence_digests_from_bioassembly(
    bioassembly: dict[str, Any],
) -> set[str]:
    """Return every protein entity sequence consumed by a stochastic crop.

    The unified record may describe only the benchmark-relevant chain, while
    Protenix crops from the complete official bioassembly.  ESM features must
    therefore cover every protein entity in that native object, not merely the
    sequence copied into the common DatasetPackage record.
    """

    sequences = bioassembly.get("sequences", {})
    entity_poly_type = bioassembly.get("entity_poly_type", {})
    if not isinstance(sequences, dict) or not isinstance(entity_poly_type, dict):
        raise ValueError("official bioassembly is missing sequence metadata")
    protein_digests = set()
    for entity_id, polymer_type in entity_poly_type.items():
        if not str(polymer_type).lower().startswith("polypeptide"):
            continue
        try:
            sequence = sequences[str(entity_id)]
        except KeyError as error:
            raise ValueError(
                f"protein entity {entity_id} has no official sequence"
            ) from error
        protein_digests.add(hashlib.sha256(sequence.encode()).hexdigest())
    return protein_digests


def bioassembly_protein_sequence_digests(path: Path) -> set[str]:
    with gzip.open(path, "rb") as stream:
        bioassembly = pickle.load(stream)
    if not isinstance(bioassembly, dict):
        raise ValueError(f"official bioassembly is not a mapping: {path}")
    return protein_sequence_digests_from_bioassembly(bioassembly)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else Path.open
    if path.suffix == ".gz":
        stream = opener(path, "wt", newline="", encoding="utf-8")
    else:
        stream = opener(path, "w", newline="", encoding="utf-8")
    with stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = arguments()
    package = load_json(args.dataset_package)
    paths = validate_dataset_package(package, args.data_root)
    package_split = "validation" if args.split == "valid" else "train"
    split_path = paths[package_split]

    records: list[dict[str, Any]] = []
    wanted_pdb_ids: set[str] = set()
    with split_path.open() as stream:
        for line in stream:
            record = json.loads(line)
            records.append(record)
            if record.get("source") == "openfold3_pdb":
                pdb_id = str(record.get("pdb_id", record["source_id"])).lower()
                wanted_pdb_ids.add(pdb_id)

    native_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    with open_csv(args.upstream_index) as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "pdb_id" not in reader.fieldnames:
            raise ValueError(f"upstream index has no pdb_id column: {args.upstream_index}")
        fieldnames = list(reader.fieldnames)
        for row in reader:
            pdb_id = row["pdb_id"].lower()
            if pdb_id in wanted_pdb_ids:
                native_rows[pdb_id].append(row)

    cached_sequences = ready_sequences(args.esmc_manifest) if args.esmc_manifest else None
    accepted: list[dict[str, str]] = []
    rejected: Counter[str] = Counter()
    task_accepts: Counter[str] = Counter()
    bioassembly_sequence_cache: dict[str, set[str]] = {}
    for record in records:
        if record.get("source") != "openfold3_pdb":
            rejected["native_source_unavailable"] += 1
            continue
        pdb_id = str(record.get("pdb_id", record["source_id"])).lower()
        bioassembly_path = args.bioassembly_dir / f"{pdb_id}.pkl.gz"
        if not bioassembly_path.is_file():
            rejected["missing_official_bioassembly"] += 1
            continue
        pdb_rows = native_rows.get(pdb_id, [])
        admitted_rows = [row for row in pdb_rows if passes_official_exclusions(row)]
        row = choose_row(record, admitted_rows)
        if row is None:
            if pdb_id not in native_rows:
                reason = "missing_upstream_pdb"
            elif pdb_rows and not admitted_rows:
                reason = "upstream_ion_exclusion"
            else:
                reason = "no_task_compatible_native_row"
            rejected[reason] += 1
            continue
        if args.max_tokens is not None:
            if args.max_tokens < 1:
                raise ValueError("--max-tokens must be positive")
            native_tokens = int(row["num_tokens"])
            if native_tokens > args.max_tokens:
                rejected["exceeds_model_max_tokens"] += 1
                continue
        if cached_sequences is not None:
            required_sequences = bioassembly_sequence_cache.get(pdb_id)
            if required_sequences is None:
                required_sequences = bioassembly_protein_sequence_digests(
                    bioassembly_path
                )
                bioassembly_sequence_cache[pdb_id] = required_sequences
            if required_sequences - cached_sequences:
                rejected["missing_esmc_feature"] += 1
                continue
        projected = dict(row)
        projected["unified_example_id"] = record["example_id"]
        projected["unified_task"] = record.get("task", "other")
        accepted.append(projected)
        task_accepts[record.get("task", "other")] += 1

    accepted.sort(key=lambda row: row["unified_example_id"])
    output_index = args.output_dir / f"{args.split}.csv.gz"
    output_fields = [*fieldnames, "unified_example_id", "unified_task"]
    write_csv(output_index, output_fields, accepted)
    pdb_list = args.output_dir / "pdb_ids.txt"
    pdb_ids = sorted({row["pdb_id"].lower() for row in accepted})
    pdb_list.write_text("".join(f"{pdb_id}\n" for pdb_id in pdb_ids))
    audit = {
        "schema_version": (
            NATIVE_VIEW_AUDIT_SCHEMA
            if args.split == "train"
            else "autofold-native-validation-view/v1"
        ),
        "model_id": args.model_id,
        "dataset_id": package["dataset_id"],
        "source_split_sha256": package["splits"][package_split]["sha256"],
        "source_examples": package["splits"][package_split]["examples"],
        "accepted_examples": len(accepted),
        "rejected_examples": sum(rejected.values()),
        "rejection_reasons": dict(sorted(rejected.items())),
        "native_manifest": str(output_index),
        "native_manifest_sha256": sha256_file(output_index),
        "selection_policy": "capability_only",
    }
    audit["details"] = {
        "source_split": package_split,
        "upstream_index": str(args.upstream_index),
        "upstream_index_sha256": sha256_file(args.upstream_index),
        "bioassembly_dir": str(args.bioassembly_dir),
        "accepted_task_counts": dict(sorted(task_accepts.items())),
        "unique_pdbs_requested": len(wanted_pdb_ids),
        "unique_pdbs_in_native_index": len(native_rows),
        "native_pdbs": len(pdb_ids),
        "pdb_list": str(pdb_list),
        "esmc_manifest": str(args.esmc_manifest) if args.esmc_manifest else None,
        "esmc_coverage_basis": (
            "all_official_bioassembly_protein_entities"
            if args.esmc_manifest
            else None
        ),
        "max_tokens": args.max_tokens,
    }
    # The strict public audit is kept free of backend-specific details.  Write
    # the additional provenance next to it rather than weakening the schema.
    details = audit.pop("details")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "native_view_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "native_view_details.json").write_text(
        json.dumps(details, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({**audit, "details": details}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
