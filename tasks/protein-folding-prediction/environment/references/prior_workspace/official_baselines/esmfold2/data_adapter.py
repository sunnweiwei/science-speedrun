#!/usr/bin/env python3
"""Build the all-track, all-atom native view for ESMFold2-50M.

The adapter preserves every FoldBench-supported biomolecule in each immutable
FoldingTrain example.  It uses Biohub's released ``StructurePredictionInput``
and ``ESMFold2InputBuilder`` for featurization, while aligning the resulting
official atom order to the frozen OpenFold3 coordinate target by
chain/residue/atom identity.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from agent_pipeline.contract import (
    FOLDBENCH_TRACKS,
    NATIVE_VIEW_AUDIT_SCHEMA,
    load_json,
    sha256_file,
    validate_dataset_package,
)


POLYMER_TYPES = frozenset({"protein", "rna", "dna"})
SUPPORTED_TASKS = frozenset(FOLDBENCH_TRACKS)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-package", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid"), default="train")
    parser.add_argument("--model-id", default="esmfold2_fast_50m")
    parser.add_argument("--max-train-tokens", type=int, default=384)
    parser.add_argument("--max-eval-tokens", type=int, default=2048)
    parser.add_argument("--log-interval", type=int, default=1000)
    return parser.parse_args()


def _data_path(path: str, data_root: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        try:
            relative = value.relative_to("/data")
        except ValueError as error:
            raise ValueError(f"structure path is outside /data: {path}") from error
        return data_root / relative
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"structure path escapes data root: {path}")
    return data_root.joinpath(*pure.parts)


def _portable_path(path: Path, data_root: Path) -> str:
    relative = path.resolve().relative_to(data_root.resolve())
    return f"data://{relative.as_posix()}"


def resolve_structure_uri(uri: str, data_root: Path) -> Path:
    prefix = "data://"
    if not uri.startswith(prefix):
        raise ValueError(f"expected data:// structure URI, got {uri!r}")
    relative = PurePosixPath(uri[len(prefix) :])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"structure URI escapes data root: {uri!r}")
    return data_root.joinpath(*relative.parts)


def source_token_count(record: dict[str, Any]) -> int:
    declared = record.get("token_count")
    if declared is not None:
        return int(declared)
    polymer = sum(
        int(sequence.get("length", len(str(sequence.get("sequence", "")))))
        for sequence in record.get("sequences", [])
    )
    return polymer + int(record.get("ligand_heavy_atoms", 0) or 0)


def _residue_groups(npz, chain_id: str | None = None):
    """Return ordered NPZ residue groups keyed by chain/residue/insertion."""

    import numpy as np

    chains = np.asarray(npz["chain_id"]).astype(str)
    residue_ids = np.asarray(npz["res_id"])
    insertion = (
        np.asarray(npz["ins_code"]).astype(str)
        if "ins_code" in npz
        else np.full(chains.shape, "")
    )
    names = np.asarray(npz["res_name"]).astype(str)
    hetero = (
        np.asarray(npz["hetero"]).astype(bool)
        if "hetero" in npz
        else np.zeros(chains.shape, dtype=bool)
    )
    ordered: list[dict[str, Any]] = []
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for atom_index, chain in enumerate(chains):
        if chain_id is not None and chain != chain_id:
            continue
        key = (chain, int(residue_ids[atom_index]), str(insertion[atom_index]))
        group = by_key.get(key)
        if group is None:
            group = {
                "chain_id": chain,
                "res_id": key[1],
                "ins_code": key[2],
                "res_name": str(names[atom_index]),
                "hetero": bool(hetero[atom_index]),
                "indices": [],
            }
            by_key[key] = group
            ordered.append(group)
        group["indices"].append(int(atom_index))
    return ordered


def native_input(
    record: dict[str, Any], path: Path
) -> tuple[dict[str, Any] | None, str | None]:
    """Create a JSON-safe Biohub input while retaining the full assembly."""

    import numpy as np

    if record.get("task") not in SUPPORTED_TASKS:
        return None, "unsupported_task"
    sequences: list[dict[str, Any]] = []
    polymer_chains: set[str] = set()
    for value in record.get("sequences", []):
        kind = str(value.get("molecule_type", ""))
        chain_id = str(value.get("chain_id", ""))
        sequence = str(value.get("sequence", ""))
        if kind not in POLYMER_TYPES or not chain_id or not sequence:
            return None, "unsupported_polymer"
        if chain_id in polymer_chains:
            return None, "duplicate_chain_id"
        polymer_chains.add(chain_id)
        sequences.append(
            {
                "type": kind,
                "id": chain_id,
                "sequence": sequence,
                "modifications": [],
            }
        )
    if not sequences:
        return None, "missing_polymer"
    if not path.is_file():
        return None, "missing_structure"
    try:
        with np.load(path, allow_pickle=False) as npz:
            required = {
                "coord", "chain_id", "res_id", "res_name", "atom_name",
            }
            if required - set(npz.files):
                return None, "unsupported_structure_schema"
            coords = np.asarray(npz["coord"])
            if coords.ndim != 2 or coords.shape[-1] != 3:
                return None, "unsupported_structure_schema"
            by_chain: dict[str, list[dict[str, Any]]] = {}
            for group in _residue_groups(npz):
                if group["chain_id"] not in polymer_chains:
                    by_chain.setdefault(group["chain_id"], []).append(group)
            for chain_id in sorted(by_chain):
                residues = by_chain[chain_id]
                if len(residues) != 1:
                    return None, "multi_residue_nonpolymer_chain"
                ccd = residues[0]["res_name"]
                if not ccd or ccd in {"HOH", "DOD"}:
                    return None, "unsupported_nonpolymer"
                sequences.append(
                    {"type": "ligand", "id": chain_id, "ccd": [ccd]}
                )
            finite_fraction = float(np.isfinite(coords).all(axis=-1).mean())
            if finite_fraction < 0.5:
                return None, "insufficient_finite_coordinates"
    except (OSError, ValueError, KeyError):
        return None, "unreadable_structure"
    return {"sequences": sequences}, None


def prepare_view(args: argparse.Namespace) -> dict[str, Any]:
    if min(args.max_train_tokens, args.max_eval_tokens, args.log_interval) < 1:
        raise ValueError("token and log limits must be positive")
    package = load_json(args.dataset_package)
    package_paths = validate_dataset_package(package, args.data_root)
    package_split = "validation" if args.split == "valid" else "train"
    token_limit = (
        args.max_eval_tokens if args.split == "valid" else args.max_train_tokens
    )
    with package_paths[package_split].open() as stream:
        source_records = [json.loads(line) for line in stream if line.strip()]
    by_example_id = {record["example_id"]: record for record in source_records}
    accepted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    for processed, logical_record in enumerate(source_records, 1):
        if processed > 1 and (processed - 1) % args.log_interval == 0:
            print(json.dumps({
                "processed": processed - 1,
                "accepted": len(accepted),
                "rejected": sum(rejected.values()),
            }), flush=True)
        record = logical_record
        base_id = logical_record.get("base_example_id")
        if not logical_record.get("sequences") and base_id is not None:
            base = by_example_id.get(str(base_id))
            if base is None:
                rejected["missing_base_example"] += 1
                continue
            record = {**base, **logical_record}
        token_count = source_token_count(record)
        if token_count < 1:
            rejected["missing_token_count"] += 1
            continue
        if token_count > token_limit:
            rejected[
                "exceeds_model_max_tokens"
                if args.split == "valid"
                else "exceeds_training_crop_without_native_cropper"
            ] += 1
            continue
        structure_value = record.get("structure_path")
        if not structure_value:
            rejected["missing_structure"] += 1
            continue
        structure_path = _data_path(str(structure_value), args.data_root)
        input_value, reason = native_input(record, structure_path)
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
                # The reconstructed trainer does not crop complexes.  Keeping
                # this alias preserves deterministic one-pass runner metadata.
                "length": token_count,
                "structure": _portable_path(structure_path, args.data_root),
                "input": input_value,
            }
        )
        task_counts[record["task"]] += 1

    accepted.sort(key=lambda item: item["example_id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"{args.split}.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in accepted)
    )
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
        "native_manifest": str(manifest_path),
        "native_manifest_sha256": sha256_file(manifest_path),
        "selection_policy": "capability_only",
    }
    (args.output_dir / "native_view_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    details = {
        "source_split": package_split,
        "token_limit": token_limit,
        "accepted_task_counts": dict(sorted(task_counts.items())),
        "input_featurizer": "released Biohub ESMFold2InputBuilder",
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


def build_structure_prediction_input(value: dict[str, Any]):
    from esm.models.esmfold2 import (
        DNAInput,
        LigandInput,
        Modification,
        ProteinInput,
        RNAInput,
        StructurePredictionInput,
    )

    classes = {
        "protein": ProteinInput,
        "rna": RNAInput,
        "dna": DNAInput,
        "ligand": LigandInput,
    }
    sequences = []
    for item in value["sequences"]:
        kind = item["type"]
        kwargs: dict[str, Any] = {"id": item["id"]}
        if kind == "ligand":
            kwargs["ccd"] = item.get("ccd")
            kwargs["smiles"] = item.get("smiles")
        else:
            kwargs["sequence"] = item["sequence"]
            kwargs["modifications"] = [
                Modification(**modification)
                for modification in item.get("modifications", [])
            ]
        sequences.append(classes[kind](**kwargs))
    return StructurePredictionInput(sequences=sequences)


def _load_feature(uri: str, data_root: Path, expected_length: int):
    import torch
    from safetensors.torch import load_file

    path = resolve_structure_uri(uri, data_root)
    tensors = load_file(str(path), device="cpu")
    if set(tensors) != {"single"}:
        raise ValueError(f"unexpected ESMC feature keys: {sorted(tensors)}")
    value = tensors["single"]
    expected = (expected_length, 256)
    if tuple(value.shape) != expected or value.dtype != torch.bfloat16:
        raise ValueError(
            f"invalid ESMC feature {tuple(value.shape)} {value.dtype} != {expected} bf16"
        )
    if not torch.isfinite(value).all():
        raise ValueError(f"non-finite ESMC feature: {path}")
    return value


def _load_mixed256(
    record: dict[str, Any], data_root: Path, start: int, end: int
):
    """Backward-compatible loader retained for cached monomer-view audits."""

    uri = record.get("esmc_mixed256")
    if not isinstance(uri, str):
        raise ValueError("native record is missing one frozen ESMC mixed-256 feature")
    value = _load_feature(uri, data_root, len(record["sequence"]))
    return value[start:end].unsqueeze(0).contiguous()


def _complex_mixed256(record, data_root: Path, chain_infos, token_count: int):
    import torch

    output = torch.zeros((1, token_count, 256), dtype=torch.bfloat16)
    cache_records = record.get("esmc_mixed256")
    if not isinstance(cache_records, dict):
        raise ValueError("all-atom native record has no per-chain ESMC cache map")
    input_by_id = {item["id"]: item for item in record["input"]["sequences"]}
    for chain in chain_infos:
        if chain.mol_type != 0:  # Biohub MOL_TYPE_PROTEIN
            continue
        source = input_by_id[chain.chain_id]
        sequence = source["sequence"]
        entry = cache_records.get(chain.chain_id)
        if not isinstance(entry, dict):
            raise ValueError(f"missing ESMC cache for protein chain {chain.chain_id}")
        digest = hashlib.sha256(sequence.encode()).hexdigest()
        if entry.get("sequence_sha256") != digest:
            raise ValueError(f"ESMC cache digest mismatch for chain {chain.chain_id}")
        value = _load_feature(entry["uri"], data_root, len(sequence))
        for token in chain.tokens:
            output[0, token.token_index] = value[token.residue_index]
    return output


def _decode_atom_name(encoded) -> str:
    return "".join(chr(int(value) + 32) if int(value) else " " for value in encoded).strip()


def _aligned_target(path: Path, features, chain_infos):
    import numpy as np
    import torch

    target = torch.zeros_like(features["ref_pos"], dtype=torch.float32)
    target_mask = torch.zeros_like(features["atom_attention_mask"], dtype=torch.bool)
    with np.load(path, allow_pickle=False) as npz:
        atom_names = np.asarray(npz["atom_name"]).astype(str)
        coords = np.asarray(npz["coord"], dtype=np.float32)
        occupancy = (
            np.asarray(npz["occupancy"], dtype=np.float32)
            if "occupancy" in npz
            else np.ones(coords.shape[0], dtype=np.float32)
        )
        for chain in chain_infos:
            groups = _residue_groups(npz, chain.chain_id)
            if not groups:
                # The immutable sequence may include an unresolved chain.  It
                # remains valid model input but contributes no coordinate loss.
                continue
            by_label_position = {
                int(group["res_id"]): group for group in groups
            }
            for token in chain.tokens:
                if chain.mol_type == 3:  # ligand atoms share residue_index=0
                    group = groups[0]
                else:
                    # OpenFold3 ``res_id`` is label_seq_id (1-based), so this
                    # keeps missing/unresolved residues masked instead of
                    # shifting every later coordinate to the wrong token.
                    group = by_label_position.get(token.residue_index + 1)
                    if group is None:
                        continue
                best: dict[str, int] = {}
                for index in group["indices"]:
                    name = str(atom_names[index])
                    if name not in best or occupancy[index] > occupancy[best[name]]:
                        best[name] = index
                for atom_index in range(
                    token.atom_start, token.atom_start + token.atom_count
                ):
                    name = _decode_atom_name(
                        features["ref_atom_name_chars"][0, atom_index]
                    )
                    source_index = best.get(name)
                    if source_index is not None and np.isfinite(coords[source_index]).all():
                        target[0, atom_index] = torch.from_numpy(coords[source_index])
                        target_mask[0, atom_index] = True
    if not target_mask.any():
        raise ValueError("official ESMFold2 atoms have no aligned target coordinates")
    return target, target_mask


def load_cropped_example(
    record: dict[str, Any],
    data_root: Path,
    *,
    crop_start: int,
    crop_size: int,
    input_builder=None,
):
    """Return official all-atom features and aligned experimental coordinates."""

    if crop_start != 0 or int(record["token_count"]) > crop_size:
        raise ValueError("all-atom ESMFold2 view requires an uncropped admitted complex")
    if input_builder is None:
        raise ValueError("released ESMFold2InputBuilder is required")
    native = build_structure_prediction_input(record["input"])
    features, chain_infos = input_builder.prepare_input(native, seed=0)
    for key in ("msa", "msa_attention_mask", "has_deletion", "deletion_value"):
        features.pop(key, None)
    features["lm_mixed256"] = _complex_mixed256(
        record, data_root, chain_infos, features["token_index"].shape[1]
    )
    target, target_mask = _aligned_target(
        resolve_structure_uri(record["structure"], data_root), features, chain_infos
    )
    return features, target, target_mask


def main() -> None:
    print(json.dumps(prepare_view(arguments()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
