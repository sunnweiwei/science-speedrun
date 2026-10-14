"""Dataset and crop logic for the compact FoldingTrain reference indexes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from reference.features import load_esmc_final


TASKS = (
    "monomer_protein",
    "monomer_rna",
    "monomer_dna",
    "interface_protein_protein",
    "interface_antibody_antigen",
    "interface_protein_peptide",
    "interface_protein_ligand",
    "interface_protein_rna",
    "interface_protein_dna",
    "specialist_disorder",
    "other",
)
TASK_TO_ID = {task: index for index, task in enumerate(TASKS)}


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)


def _stable_token_id(value: str, size: int) -> int:
    return _stable_int(value) % size


class FoldingReferenceDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: Path,
        crop_size: int = 128,
        training: bool = True,
        seed: int = 0,
    ) -> None:
        self.records = [
            json.loads(line) for line in manifest.read_text().splitlines() if line
        ]
        if not self.records:
            raise ValueError(f"empty dataset: {manifest}")
        self.manifest = manifest
        self.crop_size = crop_size
        self.training = training
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def _rng(self, record: dict[str, Any]) -> np.random.Generator:
        key = f"{self.seed}\0{self.epoch}\0{record['example_id']}\0{self.training}"
        return np.random.default_rng(_stable_int(key))

    def _crop_indices(
        self,
        record: dict[str, Any],
        coords: np.ndarray,
        coord_mask: np.ndarray,
        chain_ids: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        length = len(coords)
        if length <= self.crop_size:
            return np.arange(length)
        interface = record["task"].startswith("interface_")
        valid_chains = [
            chain
            for chain in np.unique(chain_ids)
            if np.count_nonzero((chain_ids == chain) & coord_mask)
        ]
        if interface and len(valid_chains) >= 2:
            chain_a, chain_b = rng.choice(valid_chains, size=2, replace=False)
            positions_a = np.flatnonzero((chain_ids == chain_a) & coord_mask)
            positions_b = np.flatnonzero((chain_ids == chain_b) & coord_mask)
            probes = rng.choice(positions_a, size=min(16, len(positions_a)), replace=False)
            best = None
            for anchor_a in probes:
                delta = coords[positions_b] - coords[anchor_a]
                candidate = positions_b[np.argmin(np.sum(delta * delta, axis=-1))]
                distance = float(np.sum((coords[candidate] - coords[anchor_a]) ** 2))
                if best is None or distance < best[0]:
                    best = (distance, int(anchor_a), int(candidate))
            assert best is not None
            _, anchor_a, anchor_b = best
            distance_a = np.sum((coords - coords[anchor_a]) ** 2, axis=-1)
            distance_b = np.sum((coords - coords[anchor_b]) ** 2, axis=-1)
            score = np.minimum(distance_a, distance_b)
            score[~coord_mask] = np.inf
            chosen = np.argsort(score)[: self.crop_size]
            return np.sort(chosen)

        start = int(rng.integers(0, length - self.crop_size + 1))
        return np.arange(start, start + self.crop_size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        rng = self._rng(record)
        with np.load(record["structure_path"], allow_pickle=False) as structure:
            if "token_center_atom" in structure.files:
                center_indices = np.flatnonzero(structure["token_center_atom"])
                token_positions = structure["token_position"][center_indices].astype(np.int64)
                order = np.argsort(token_positions)
                center_indices = center_indices[order]
                token_positions = token_positions[order]
                if not np.array_equal(token_positions, np.arange(len(token_positions))):
                    raise ValueError(
                        f"{record['example_id']}: token positions are not contiguous/unique"
                    )
            else:
                # Older OpenFold3 distillation caches are atom arrays without
                # token metadata. Reconstruct one representative per residue.
                preferred_atom = {0: "CA", 1: "C4'", 2: "C4'"}
                groups: dict[tuple[str, int, str], list[int]] = {}
                for atom_index, key in enumerate(
                    zip(
                        structure["chain_id"].astype(str),
                        structure["res_id"].astype(np.int64),
                        structure["ins_code"].astype(str),
                        strict=True,
                    )
                ):
                    groups.setdefault(key, []).append(atom_index)
                selected = []
                for atom_indices in groups.values():
                    molecule = int(structure["molecule_type_id"][atom_indices[0]])
                    preferred = preferred_atom.get(molecule, "CA")
                    chosen = next(
                        (
                            atom_index
                            for atom_index in atom_indices
                            if str(structure["atom_name"][atom_index]) == preferred
                        ),
                        atom_indices[0],
                    )
                    selected.append(chosen)
                center_indices = np.asarray(selected, dtype=np.int64)
            coords = structure["coord"][center_indices].astype(np.float32)
            molecule_type = structure["molecule_type_id"][center_indices].astype(np.int64)
            chain_text = structure["chain_id"][center_indices].astype(str)
            residue_index = structure["res_id"][center_indices].astype(np.int64)
            residue_name = structure["res_name"][center_indices].astype(str)
            atom_name = structure["atom_name"][center_indices].astype(str)

        coord_mask = np.isfinite(coords).all(axis=-1)
        chain_vocabulary = {chain: idx for idx, chain in enumerate(dict.fromkeys(chain_text))}
        chain_ids = np.asarray([chain_vocabulary[value] for value in chain_text], dtype=np.int64)
        single = np.zeros((len(coords), 2560), dtype=np.float32)
        lm_mask = np.zeros(len(coords), dtype=np.bool_)
        for sequence in record["sequences"]:
            if sequence["molecule_type"] != "protein" or not sequence["esmc_available"]:
                continue
            positions = np.flatnonzero(chain_text == str(sequence["chain_id"]))
            feature = load_esmc_final(sequence["esmc_path"]).numpy()
            if len(positions) != len(feature):
                raise ValueError(
                    f"{record['example_id']}: chain {sequence['chain_id']} has "
                    f"{len(positions)} tokens but ESMC has {len(feature)}"
                )
            single[positions] = feature
            lm_mask[positions] = True

        indices = self._crop_indices(
            record, coords, coord_mask, chain_ids, rng
        )
        coords = coords[indices]
        coord_mask = coord_mask[indices]
        if not np.any(coord_mask):
            raise ValueError(f"{record['example_id']}: crop has no finite coordinates")
        coords = np.nan_to_num(coords)
        center = coords[coord_mask].mean(axis=0, keepdims=True)
        coords = (coords - center) / 10.0
        coords[~coord_mask] = 0.0
        residue_type = np.asarray(
            [_stable_token_id(value, 512) for value in residue_name[indices]],
            dtype=np.int64,
        )
        atom_type = np.asarray(
            [_stable_token_id(value, 128) for value in atom_name[indices]],
            dtype=np.int64,
        )
        return {
            "example_id": record["example_id"],
            "task": record["task"],
            "task_id": torch.tensor(TASK_TO_ID[record["task"]], dtype=torch.long),
            "single": torch.from_numpy(single[indices]),
            "lm_mask": torch.from_numpy(lm_mask[indices]),
            "coords": torch.from_numpy(coords),
            "coord_mask": torch.from_numpy(coord_mask),
            "molecule_type": torch.from_numpy(molecule_type[indices]),
            "chain_id": torch.from_numpy(chain_ids[indices]),
            "residue_index": torch.from_numpy(residue_index[indices]),
            "residue_type": torch.from_numpy(residue_type),
            "atom_type": torch.from_numpy(atom_type),
        }


def collate_reference_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty batch")
    max_length = max(len(item["coords"]) for item in items)
    batch_size = len(items)
    output: dict[str, Any] = {
        "example_id": [item["example_id"] for item in items],
        "task": [item["task"] for item in items],
        "task_id": torch.stack([item["task_id"] for item in items]),
        "single": torch.zeros(batch_size, max_length, 2560),
        "lm_mask": torch.zeros(batch_size, max_length, dtype=torch.bool),
        "coords": torch.zeros(batch_size, max_length, 3),
        "coord_mask": torch.zeros(batch_size, max_length, dtype=torch.bool),
        "token_mask": torch.zeros(batch_size, max_length, dtype=torch.bool),
        "molecule_type": torch.zeros(batch_size, max_length, dtype=torch.long),
        "chain_id": torch.zeros(batch_size, max_length, dtype=torch.long),
        "residue_index": torch.zeros(batch_size, max_length, dtype=torch.long),
        "residue_type": torch.zeros(batch_size, max_length, dtype=torch.long),
        "atom_type": torch.zeros(batch_size, max_length, dtype=torch.long),
    }
    for batch_index, item in enumerate(items):
        length = len(item["coords"])
        output["token_mask"][batch_index, :length] = True
        for key in (
            "single",
            "lm_mask",
            "coords",
            "coord_mask",
            "molecule_type",
            "chain_id",
            "residue_index",
            "residue_type",
            "atom_type",
        ):
            output[key][batch_index, :length] = item[key]
    return output
