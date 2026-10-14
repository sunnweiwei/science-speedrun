"""Current task data/request adapters for the Biohub ESMFold2 graph."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any, Callable

from .features import load_esmc_final


POLYMER_TYPES = frozenset({"protein", "rna", "dna"})


def _residue_groups(npz, chain_id: str | None = None):
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


def _biohub_input(sequence_values: list[dict[str, Any]]):
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
    for item in sequence_values:
        kind = item["type"]
        kwargs: dict[str, Any] = {"id": item["id"]}
        if kind == "ligand":
            kwargs["ccd"] = item.get("ccd")
            kwargs["smiles"] = item.get("smiles")
        else:
            kwargs["sequence"] = item["sequence"]
            kwargs["modifications"] = [
                Modification(**value) for value in item.get("modifications", [])
            ]
        sequences.append(classes[kind](**kwargs))
    return StructurePredictionInput(sequences=sequences)


def request_input(request: dict[str, Any]):
    """Expand public entity multiplicities into one Biohub assembly."""

    sequence_values = []
    for entity in request["target"]["entities"]:
        copies = int(entity.get("copies", 1))
        for copy_index in range(copies):
            item: dict[str, Any] = {
                "id": entity["id"] if copies == 1 else f"{entity['id']}_{copy_index + 1}",
                "type": entity["type"],
            }
            if entity["type"] == "ligand":
                if "ccd_codes" in entity:
                    item["ccd"] = list(entity["ccd_codes"])
                else:
                    item["smiles"] = entity["smiles"]
            else:
                item["sequence"] = entity["sequence"]
                item["modifications"] = [
                    {**deepcopy(value), "position": int(value["position"]) - 1}
                    for value in entity.get("modifications", [])
                ]
            sequence_values.append(item)
    return _biohub_input(sequence_values)


def _expanded_chain_id(request: dict[str, Any], endpoint: dict[str, Any]) -> str:
    entity_id = endpoint["entity_id"]
    entities = {
        entity["id"]: entity for entity in request["target"]["entities"]
    }
    if entity_id not in entities:
        raise ValueError(f"unknown covalent-bond entity: {entity_id}")
    copies = int(entities[entity_id].get("copies", 1))
    copy_index = int(endpoint.get("copy", 1))
    if not 1 <= copy_index <= copies:
        raise ValueError(f"invalid copy {copy_index} for entity {entity_id}")
    return entity_id if copies == 1 else f"{entity_id}_{copy_index}"


def _bond_atom_index(features, chain_infos, chain_id, residue_index, atom_name):
    chain = next((value for value in chain_infos if value.chain_id == chain_id), None)
    if chain is None:
        raise ValueError(f"covalent-bond chain not constructed: {chain_id}")
    names = []
    for token in chain.tokens:
        if int(token.residue_index) != residue_index:
            continue
        for atom_index in range(token.atom_start, token.atom_start + token.atom_count):
            names.append(_decode_atom_name(features["ref_atom_name_chars"][0, atom_index]))
    matches = [index for index, name in enumerate(names) if name == atom_name]
    if len(matches) != 1:
        raise ValueError(
            f"expected one atom {atom_name!r} at {chain_id}:{residue_index + 1}; "
            f"available={names}"
        )
    return matches[0]


def prepare_request(request, builder, *, seed: int, device):
    """Prepare a request and translate the public name-based covalent bonds."""

    from esm.models.esmfold2 import CovalentBond

    native = request_input(request)
    bonds = request["target"].get("covalent_bonds", [])
    if not bonds:
        return builder.prepare_input(native, seed=seed, device=device)

    endpoints = []
    dummy = []
    for bond in bonds:
        values = []
        for side in ("left", "right"):
            endpoint = bond[side]
            chain_id = _expanded_chain_id(request, endpoint)
            residue_index = int(endpoint["position"]) - 1
            if residue_index < 0:
                raise ValueError("covalent-bond positions are one-based")
            values.append((chain_id, residue_index, endpoint["atom_name"]))
        endpoints.append(values)
        dummy.append(CovalentBond(values[0][0], values[0][1], 0, values[1][0], values[1][1], 0))

    # The first pass marks bonded chains so CCD leaving atoms are removed. The
    # second pass supplies atom ordinals in that final per-residue atom order.
    native.covalent_bonds = dummy
    provisional, chain_infos = builder.prepare_input(native, seed=seed, device=device)
    native.covalent_bonds = [
        CovalentBond(
            left[0],
            left[1],
            _bond_atom_index(provisional, chain_infos, *left),
            right[0],
            right[1],
            _bond_atom_index(provisional, chain_infos, *right),
        )
        for left, right in endpoints
    ]
    return builder.prepare_input(native, seed=seed, device=device)


def training_input(record: dict[str, Any]):
    """Create a full polymer/ligand assembly from one compact index record."""

    import numpy as np

    polymers = []
    polymer_chains = set()
    for value in record["sequences"]:
        kind = value["molecule_type"]
        chain_id = str(value["chain_id"])
        if kind not in POLYMER_TYPES or not value.get("sequence"):
            raise ValueError("unsupported polymer metadata")
        polymers.append(
            {"type": kind, "id": chain_id, "sequence": value["sequence"], "modifications": []}
        )
        polymer_chains.add(chain_id)

    with np.load(record["structure_path"], allow_pickle=False) as npz:
        nonpolymer: dict[str, list[dict[str, Any]]] = {}
        for group in _residue_groups(npz):
            if group["chain_id"] not in polymer_chains:
                nonpolymer.setdefault(group["chain_id"], []).append(group)
        for chain_id in sorted(nonpolymer):
            residues = nonpolymer[chain_id]
            if len(residues) != 1:
                raise ValueError("multi-residue nonpolymer chain")
            ccd = residues[0]["res_name"]
            if ccd in {"", "HOH", "DOD"}:
                raise ValueError("unsupported solvent/nonpolymer")
            polymers.append({"type": "ligand", "id": chain_id, "ccd": [ccd]})
    return _biohub_input(polymers)


def _feature_lookup_from_record(record: dict[str, Any]) -> dict[str, tuple[str, Path]]:
    return {
        str(value["chain_id"]): (value["sequence"], Path(value["esmc_path"]))
        for value in record["sequences"]
        if value["molecule_type"] == "protein"
    }


def final_hidden_for_chains(
    features: dict[str, Any],
    chain_infos,
    lookup: dict[str, tuple[str, Path]],
) -> Any:
    """Place one decoded final-hidden tensor in the official token ordering."""

    import torch

    token_count = int(features["token_index"].shape[1])
    output = torch.zeros((1, token_count, 2560), dtype=torch.float32)
    for chain in chain_infos:
        if int(chain.mol_type) != 0:  # Biohub protein molecule type
            continue
        if chain.chain_id not in lookup:
            raise ValueError(f"missing fixed feature for protein chain {chain.chain_id}")
        sequence, path = lookup[chain.chain_id]
        expected = hashlib.sha256(sequence.encode()).hexdigest()
        if path.stem != expected:
            raise ValueError(f"feature digest mismatch for chain {chain.chain_id}")
        value = load_esmc_final(path)
        if len(value) != len(sequence):
            raise ValueError(f"feature length mismatch for chain {chain.chain_id}")
        for token in chain.tokens:
            output[0, token.token_index] = value[token.residue_index]
    return output


def _decode_atom_name(encoded) -> str:
    return "".join(
        chr(int(value) + 32) if int(value) else " " for value in encoded
    ).strip()


def aligned_target(path: Path, features, chain_infos):
    """Align immutable OpenFold3 atoms to Biohub's official atom ordering."""

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
                continue
            by_position = {int(group["res_id"]): group for group in groups}
            for token in chain.tokens:
                group = groups[0] if int(chain.mol_type) == 3 else by_position.get(token.residue_index + 1)
                if group is None:
                    continue
                best: dict[str, int] = {}
                for index in group["indices"]:
                    name = str(atom_names[index])
                    if name not in best or occupancy[index] > occupancy[best[name]]:
                        best[name] = index
                for atom_index in range(token.atom_start, token.atom_start + token.atom_count):
                    source = best.get(_decode_atom_name(features["ref_atom_name_chars"][0, atom_index]))
                    if source is not None and np.isfinite(coords[source]).all():
                        target[0, atom_index] = torch.from_numpy(coords[source])
                        target_mask[0, atom_index] = True
    if not target_mask.any():
        raise ValueError("no aligned all-atom target coordinates")
    return target, target_mask


def prepare_training_example(record, builder):
    native = training_input(record)
    features, chain_infos = builder.prepare_input(native, seed=0)
    for key in ("msa", "msa_attention_mask", "has_deletion", "deletion_value"):
        features.pop(key, None)
    features["lm_final_hidden"] = final_hidden_for_chains(
        features, chain_infos, _feature_lookup_from_record(record)
    )
    target, target_mask = aligned_target(Path(record["structure_path"]), features, chain_infos)
    return features, target, target_mask


def request_feature_lookup(request: dict[str, Any], feature_paths: dict[str, Path]):
    """Map expanded Biohub chain IDs to exact evaluator-supplied tensors."""

    lookup: dict[str, tuple[str, Path]] = {}
    for entity in request["target"]["entities"]:
        if entity["type"] != "protein":
            continue
        sequence = entity["sequence"]
        digest = hashlib.sha256(sequence.encode()).hexdigest()
        path = Path(feature_paths[digest])
        copies = int(entity.get("copies", 1))
        for copy_index in range(copies):
            chain_id = entity["id"] if copies == 1 else f"{entity['id']}_{copy_index + 1}"
            lookup[chain_id] = (sequence, path)
    return lookup
