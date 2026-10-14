#!/usr/bin/env python3
"""Project FoldingTrain into native OpenDDE JSON features and atom targets."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from agent_pipeline.contract import (
    NATIVE_VIEW_AUDIT_SCHEMA,
    load_json,
    sha256_file,
    validate_dataset_package,
)


POLYMER_TYPES = {
    "protein": "proteinChain",
    "dna": "dnaSequence",
    "rna": "rnaSequence",
}
SUPPORTED_TASKS = {
    "monomer_protein",
    "monomer_rna",
    "monomer_dna",
    "interface_protein_protein",
    "interface_antibody_antigen",
    "interface_protein_peptide",
    "interface_protein_rna",
    "interface_protein_dna",
    "interface_protein_ligand",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-package", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid"), default="train")
    parser.add_argument("--max-train-tokens", type=int, default=384)
    parser.add_argument("--max-eval-tokens", type=int, default=2048)
    parser.add_argument("--model-id", default="opendde_structural_50m")
    return parser.parse_args()


def data_path(value: str, data_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        try:
            relative = path.relative_to("/data")
        except ValueError as error:
            raise ValueError(f"structure path outside /data: {value}") from error
        return data_root / relative
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe structure path: {value}")
    return data_root.joinpath(*pure.parts)


def portable_path(path: Path, data_root: Path) -> str:
    return "data://" + path.resolve().relative_to(data_root.resolve()).as_posix()


def source_token_count(record: dict[str, Any]) -> int:
    declared = record.get("token_count")
    if declared is not None:
        return int(declared)
    polymer = sum(int(sequence.get("length", 0)) for sequence in record.get("sequences", []))
    ligand = int(record.get("ligand_heavy_atoms", 0) or 0)
    return polymer + ligand


def _residue_groups(npz) -> list[tuple[str, int, str, str, bool]]:
    import numpy as np

    chains = np.asarray(npz["chain_id"]).astype(str)
    residue_ids = np.asarray(npz["res_id"])
    insertion = (
        np.asarray(npz["ins_code"]).astype(str)
        if "ins_code" in npz
        else np.full(chains.shape, "")
    )
    names = np.asarray(npz["res_name"]).astype(str)
    hetero = np.asarray(npz["hetero"]).astype(bool)
    groups: list[tuple[str, int, str, str, bool]] = []
    seen: set[tuple[str, int, str]] = set()
    for index in range(len(chains)):
        key = (chains[index], int(residue_ids[index]), insertion[index])
        if key in seen:
            continue
        seen.add(key)
        groups.append((*key, names[index], bool(hetero[index])))
    return groups


def native_input(record: dict[str, Any], path: Path) -> tuple[dict[str, Any] | None, str | None]:
    import numpy as np

    if record.get("task") not in SUPPORTED_TASKS:
        return None, "unsupported_task"
    sequences: list[dict[str, Any]] = []
    polymer_chains: set[str] = set()
    for sequence in record.get("sequences", []):
        molecule_type = str(sequence.get("molecule_type"))
        native_type = POLYMER_TYPES.get(molecule_type)
        chain_id = str(sequence.get("chain_id", ""))
        value = str(sequence.get("sequence", ""))
        if native_type is None or not chain_id or not value:
            return None, "unsupported_polymer"
        if chain_id in polymer_chains:
            return None, "duplicate_chain_id"
        polymer_chains.add(chain_id)
        sequences.append(
            {
                native_type: {
                    "sequence": value,
                    "count": 1,
                    "id": [chain_id],
                }
            }
        )
    if not sequences:
        return None, "missing_polymer"
    if not path.is_file():
        return None, "missing_structure"
    try:
        with np.load(path, allow_pickle=False) as npz:
            required = {
                "coord",
                "chain_id",
                "res_id",
                "res_name",
                "atom_name",
                "hetero",
            }
            if required - set(npz.files):
                return None, "unsupported_structure_schema"
            coords = np.asarray(npz["coord"])
            if coords.ndim != 2 or coords.shape[-1] != 3:
                return None, "unsupported_structure_schema"
            groups = _residue_groups(npz)
            nonpolymer = [group for group in groups if group[0] not in polymer_chains]
            by_chain: dict[str, list[tuple[str, int, str, str, bool]]] = {}
            for group in nonpolymer:
                by_chain.setdefault(group[0], []).append(group)
            for chain_id in sorted(by_chain):
                residues = by_chain[chain_id]
                if len(residues) != 1:
                    return None, "multi_residue_nonpolymer_chain"
                ccd = residues[0][3]
                if not ccd or ccd in {"HOH", "DOD"}:
                    return None, "unsupported_nonpolymer"
                sequences.append(
                    {
                        "ligand": {
                            "ligand": f"CCD_{ccd}",
                            "count": 1,
                            "id": [chain_id],
                        }
                    }
                )
            finite_fraction = float(np.isfinite(coords).all(axis=-1).mean())
            if finite_fraction < 0.5:
                return None, "insufficient_finite_coordinates"
    except (OSError, ValueError, KeyError):
        return None, "unreadable_structure"
    return {
        "name": record["example_id"],
        "modelSeeds": [],
        "sequences": sequences,
        "covalent_bonds": [],
    }, None


def prepare_view(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_train_tokens < 1 or args.max_eval_tokens < 1:
        raise ValueError("token limits must be positive")
    package = load_json(args.dataset_package)
    paths = validate_dataset_package(package, args.data_root)
    package_split = "validation" if args.split == "valid" else "train"
    limit = args.max_eval_tokens if args.split == "valid" else args.max_train_tokens
    accepted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    with paths[package_split].open() as stream:
        records = [json.loads(line) for line in stream]
    by_example_id = {record["example_id"]: record for record in records}
    for logical_record in records:
        record = logical_record
        base_id = logical_record.get("base_example_id")
        if not logical_record.get("sequences") and base_id is not None:
            base = by_example_id.get(str(base_id))
            if base is None:
                rejected["missing_base_example"] += 1
                continue
            # Interface-focused logical examples deliberately carry only
            # their task identity.  Their coordinate-free input and target
            # structure are inherited from the frozen base row.
            record = {**base, **logical_record}
        token_count = source_token_count(record)
        if token_count < 1:
            rejected["missing_token_count"] += 1
            continue
        if token_count > limit:
            rejected[
                "exceeds_model_max_tokens"
                if args.split == "valid"
                else "exceeds_training_crop_without_native_cropper"
            ] += 1
            continue
        structure = record.get("structure_path")
        if not structure:
            rejected["missing_structure"] += 1
            continue
        path = data_path(str(structure), args.data_root)
        input_value, reason = native_input(record, path)
        if reason is not None or input_value is None:
            rejected[reason or "unsupported_record"] += 1
            continue
        accepted.append(
            {
                "example_id": record["example_id"],
                "source": record.get("source"),
                "source_id": record.get("source_id"),
                "task": record["task"],
                "token_count": token_count,
                "structure": portable_path(path, args.data_root),
                "input": input_value,
            }
        )
        task_counts[record["task"]] += 1
    accepted.sort(key=lambda item: item["example_id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / f"{args.split}.jsonl"
    manifest.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in accepted))
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
        "native_manifest": str(manifest),
        "native_manifest_sha256": sha256_file(manifest),
        "selection_policy": "capability_only",
    }
    (args.output_dir / "native_view_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    details = {
        "source_split": package_split,
        "token_limit": limit,
        "accepted_task_counts": dict(sorted(task_counts.items())),
        "input_featurizer": "released OpenDDE SampleDictToFeatures",
        "target_alignment": "chain/residue/atom identity against immutable NPZ",
    }
    (args.output_dir / "native_view_details.json").write_text(
        json.dumps(details, indent=2, sort_keys=True) + "\n"
    )
    return {**audit, "details": details}


def iter_manifest(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def resolve_structure(uri: str, data_root: Path) -> Path:
    if not uri.startswith("data://"):
        raise ValueError(f"expected data:// URI, got {uri!r}")
    relative = PurePosixPath(uri[len("data://") :])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe data URI: {uri!r}")
    return data_root.joinpath(*relative.parts)


def load_example(record: dict[str, Any], data_root: Path):
    """Return released no-MSA/no-template features plus aligned coordinates."""

    import numpy as np
    import torch
    from opendde.data.inference.json_to_feature import SampleDictToFeatures
    from opendde.data.utils import data_type_transform, make_dummy_feature

    converter = SampleDictToFeatures(record["input"])
    features, atom_array, _ = converter.get_feature_dict()
    features["distogram_rep_atom_mask"] = torch.as_tensor(
        atom_array.distogram_rep_atom_mask
    ).long()
    features = make_dummy_feature(features_dict=features, dummy_feats=["msa", "template"])
    features = data_type_transform(features)
    path = resolve_structure(record["structure"], data_root)
    with np.load(path, allow_pickle=False) as npz:
        chains = np.asarray(npz["chain_id"]).astype(str)
        residue_ids = np.asarray(npz["res_id"])
        residue_names = np.asarray(npz["res_name"]).astype(str)
        atom_names = np.asarray(npz["atom_name"]).astype(str)
        coords = np.asarray(npz["coord"], dtype=np.float32)
        occupancy = (
            np.asarray(npz["occupancy"], dtype=np.float32)
            if "occupancy" in npz
            else np.ones(chains.shape, dtype=np.float32)
        )
        target = torch.zeros((len(atom_array), 3), dtype=torch.float32)
        target_mask = torch.zeros(len(atom_array), dtype=torch.bool)
        for atom_index in range(len(atom_array)):
            chain = str(atom_array.chain_id[atom_index])
            name = str(atom_array.atom_name[atom_index])
            residue = str(atom_array.res_name[atom_index])
            mask = (chains == chain) & (atom_names == name) & (residue_names == residue)
            if atom_array.mol_type[atom_index] in {"protein", "dna", "rna"}:
                mask &= residue_ids == int(atom_array.res_id[atom_index])
            indices = np.flatnonzero(mask)
            if len(indices):
                chosen = int(indices[np.argmax(occupancy[indices])])
                if np.isfinite(coords[chosen]).all():
                    target[atom_index] = torch.from_numpy(coords[chosen])
                    target_mask[atom_index] = True
    if int(target_mask.sum()) < 3:
        raise ValueError(f"fewer than three target atoms mapped for {record['example_id']}")
    return features, target, target_mask


def main() -> None:
    print(json.dumps(prepare_view(arguments()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
