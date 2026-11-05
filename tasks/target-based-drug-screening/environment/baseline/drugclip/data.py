"""Streaming LMDB input pipeline for the DrugCLIP reference model."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


ATOM_SYMBOLS = (
    "H", "B", "C", "N", "O", "F", "SI", "P", "S", "CL", "BR", "I",
    "NA", "MG", "AL", "K", "CA", "MN", "FE", "CO", "NI", "CU", "ZN",
    "SE", "MO", "RU", "RH", "PD", "AG", "CD", "SN", "W", "PT", "AU",
    "HG", "PB",
)
SPECIAL_TOKENS = ("<pad>", "<cls>", "<eos>", "<unk>")
VOCABULARY = SPECIAL_TOKENS + ATOM_SYMBOLS
UNIMOL_VOCABULARY = (
    "[PAD]", "[CLS]", "[SEP]", "[UNK]", "C", "N", "O", "S", "H",
    "CL", "F", "BR", "I", "SI", "P", "B", "NA", "K", "AL", "CA",
    "SN", "AS", "HG", "FE", "ZN", "CR", "SE", "GD", "AU", "LI", "[MASK]",
)
UNIMOL_POCKET_VOCABULARY = (
    "[PAD]", "[CLS]", "[SEP]", "[UNK]", "C", "N", "O", "S", "H", "[MASK]",
)
UNIMOL_ATOM_TO_INDEX = {atom: index for index, atom in enumerate(UNIMOL_VOCABULARY)}
UNIMOL_POCKET_ATOM_TO_INDEX = {
    atom: index for index, atom in enumerate(UNIMOL_POCKET_VOCABULARY)
}
ATOM_TO_INDEX = {atom: index for index, atom in enumerate(VOCABULARY)}
PAD_INDEX, CLS_INDEX, EOS_INDEX, UNK_INDEX = range(4)
POCKET_CROP_POLICY = "author_distance_weighted_v1"


def stable_seed(*parts: object) -> int:
    digest = hashlib.blake2b("\0".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


class LMDBRecords(Dataset):
    """Read numeric-key or cursor-key LMDBs without loading values into RAM."""

    def __init__(self, path: Path, max_records: int | None = None) -> None:
        import lmdb

        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        environment = lmdb.open(str(self.path), subdir=False, readonly=True,
                                lock=False, readahead=False, meminit=False,
                                max_readers=256)
        try:
            with environment.begin() as transaction:
                count = transaction.stat()["entries"]
                selected = min(count, max_records) if max_records is not None else count
                # Both the materializer contract and the official DrugCLIP
                # reader require contiguous ASCII keys 0..N-1. Generate them
                # from LMDB metadata instead of walking every B-tree leaf;
                # this keeps --max-*-records a genuinely quick smoke option.
                keys = [str(index).encode("ascii") for index in range(selected)]
                if keys and (transaction.get(keys[0]) is None
                             or transaction.get(keys[-1]) is None):
                    raise ValueError(f"LMDB does not use contiguous numeric keys: {self.path}")
        finally:
            environment.close()
        self.keys = keys
        self.environment = None

    def __len__(self) -> int:
        return len(self.keys)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["environment"] = None
        return state

    def __getitem__(self, index: int) -> dict:
        if self.environment is None:
            import lmdb
            self.environment = lmdb.open(
                str(self.path), subdir=False, readonly=True, lock=False,
                readahead=False, meminit=False, max_readers=256,
            )
        with self.environment.begin(buffers=True) as transaction:
            value = transaction.get(self.keys[index])
            if value is None:
                raise RuntimeError(f"missing LMDB key {self.keys[index]!r}")
            return pickle.loads(bytes(value))


def _coordinates(value, *, seed: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not array.size:
        raise ValueError("empty coordinate array")
    # Both a single structure and a conformer ensemble may be represented as
    # Python lists. Infer the meaning from rank: [atom, xyz] is one structure;
    # [conformer, atom, xyz] is an ensemble.
    if array.ndim == 3:
        array = array[seed % array.shape[0]]
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError(f"coordinates must be finite [atoms,3], got {array.shape}")
    return array


def pocket_crop_indices(coordinates: np.ndarray, max_atoms: int, *,
                        seed: int, epoch: int, index: int) -> np.ndarray:
    """Author CroppingPocketDataset probabilities and Uni-Core seed mixing.

    A local RandomState has the same choice sequence as numpy_seed's global
    RandomState, without changing the caller's global NumPy state.
    """
    if len(coordinates) <= max_atoms:
        return np.arange(len(coordinates))
    mixed_seed = int(hash((int(seed), int(epoch), int(index))) % 1e8)
    distance = np.linalg.norm(
        coordinates - coordinates.mean(axis=0), axis=1
    )
    logits = 1.0 / (distance + 1.0)
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    return np.random.RandomState(mixed_seed).choice(
        len(coordinates), max_atoms, replace=False, p=weights
    )


def structure_features(atoms: Sequence[str], coordinates, *, max_atoms: int,
                       seed: int, unimol: bool = False,
                       unimol_pocket: bool = False,
                       crop_policy: str = "molecule_uniform",
                       crop_seed: int | None = None, epoch: int = 0,
                       index: int = 0) -> dict[str, torch.Tensor]:
    if unimol and unimol_pocket:
        raise ValueError("choose one Uni-Mol dictionary")
    coordinates = _coordinates(coordinates, seed=seed)
    atoms = [str(atom).strip().upper() for atom in atoms]
    if unimol_pocket:
        # The author's AffinityPocketDataset reads atom *names* from the
        # benchmark LMDB (CA, CB, N.am, ...), then keeps the element prefix.
        atoms = [atom[1] if atom and atom[0].isdigit() else atom[:1]
                 for atom in atoms]
    if len(atoms) != len(coordinates):
        raise ValueError("atom and coordinate counts differ")
    keep = np.asarray([index for index, atom in enumerate(atoms) if atom != "H"])
    if not len(keep):
        raise ValueError("structure contains no heavy atoms")
    atoms = [atoms[index] for index in keep]
    coordinates = coordinates[keep]
    if len(atoms) > max_atoms:
        if crop_policy == "pocket_distance_weighted":
            if crop_seed is None:
                raise ValueError("pocket crop requires an explicit seed")
            keep = pocket_crop_indices(coordinates, max_atoms, seed=crop_seed,
                                       epoch=epoch, index=index)
        elif crop_policy == "molecule_uniform":
            generator = np.random.default_rng(seed)
            keep = np.sort(generator.choice(len(atoms), size=max_atoms,
                                            replace=False))
        else:
            raise ValueError(f"unknown crop policy: {crop_policy}")
        atoms = [atoms[index] for index in keep]
        coordinates = coordinates[keep]
    coordinates = coordinates - coordinates.mean(axis=0, keepdims=True)
    vocabulary = (UNIMOL_VOCABULARY if unimol else
                  UNIMOL_POCKET_VOCABULARY if unimol_pocket else VOCABULARY)
    atom_to_index = (UNIMOL_ATOM_TO_INDEX if unimol else
                     UNIMOL_POCKET_ATOM_TO_INDEX if unimol_pocket else ATOM_TO_INDEX)
    atom_tokens = [atom_to_index.get(atom, UNK_INDEX) for atom in atoms]
    tokens = np.asarray([CLS_INDEX, *atom_tokens, EOS_INDEX], dtype=np.int64)
    distance = np.zeros((len(tokens), len(tokens)), dtype=np.float32)
    distance[1:-1, 1:-1] = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :], axis=-1
    )
    edge_types = tokens[:, None] * len(vocabulary) + tokens[None, :]
    return {
        "tokens": torch.from_numpy(tokens),
        "distances": torch.from_numpy(distance),
        "edge_types": torch.from_numpy(edge_types),
    }


class PairedDataset(Dataset):
    def __init__(self, path: Path, *, max_molecule_atoms: int = 256,
                 max_pocket_atoms: int = 256, seed: int = 0,
                 max_records: int | None = None,
                 unimol_molecule: bool = False,
                 unimol_pocket: bool = False) -> None:
        self.records = LMDBRecords(path, max_records)
        self.max_molecule_atoms = max_molecule_atoms
        self.max_pocket_atoms = max_pocket_atoms
        self.seed = seed
        self.unimol_molecule = unimol_molecule
        self.unimol_pocket = unimol_pocket
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        item_seed = stable_seed(self.seed, self.epoch, index)
        molecule_id = str(record.get("smi", record.get("sample_id", index)))
        pocket_id = str(record.get("pocket", record.get("complex_id", index)))
        return {
            "molecule": structure_features(
                record["atoms"], record["coordinates"],
                max_atoms=self.max_molecule_atoms, seed=item_seed,
                unimol=self.unimol_molecule,
            ),
            "pocket": structure_features(
                record["pocket_atoms"], record["pocket_coordinates"],
                max_atoms=self.max_pocket_atoms, seed=item_seed + 1,
                unimol_pocket=self.unimol_pocket,
                crop_policy="pocket_distance_weighted", crop_seed=self.seed,
                epoch=self.epoch, index=index,
            ),
            "molecule_id": molecule_id, "pocket_id": pocket_id,
        }


class SingleStructureDataset(Dataset):
    def __init__(self, path: Path, kind: str, *, max_atoms: int,
                 max_records: int | None = None,
                 unimol_molecule: bool = False,
                 unimol_pocket: bool = False, seed: int = 1) -> None:
        if kind not in {"molecule", "pocket"}:
            raise ValueError("kind must be molecule or pocket")
        self.records = LMDBRecords(path, max_records)
        self.kind = kind
        self.max_atoms = max_atoms
        self.unimol_molecule = unimol_molecule
        self.unimol_pocket = unimol_pocket
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        atom_key = "atoms" if self.kind == "molecule" else "pocket_atoms"
        coordinate_key = "coordinates" if self.kind == "molecule" else "pocket_coordinates"
        return {
            "features": structure_features(
                record[atom_key], record[coordinate_key], max_atoms=self.max_atoms,
                seed=(stable_seed(self.seed, 0, index) if self.kind == "pocket"
                      # Private input permutations must not change conformer
                      # selection or atom cropping for the same molecule.
                      else stable_seed(self.seed, record["smi"], self.kind)),
                unimol=self.unimol_molecule and self.kind == "molecule",
                unimol_pocket=self.unimol_pocket and self.kind == "pocket",
                crop_policy=("pocket_distance_weighted" if self.kind == "pocket"
                             else "molecule_uniform"),
                crop_seed=self.seed if self.kind == "pocket" else None,
                epoch=0, index=index,
            ),
            "index": index,
            "name": str(record.get("smi" if self.kind == "molecule" else "pocket", index)),
            "label": int(record.get("label", -1)),
        }


def _pad_features(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    maximum = max(item["tokens"].shape[0] for item in items)
    batch = len(items)
    tokens = torch.full((batch, maximum), PAD_INDEX, dtype=torch.long)
    distances = torch.zeros((batch, maximum, maximum), dtype=torch.float32)
    edge_types = torch.zeros((batch, maximum, maximum), dtype=torch.long)
    for row, item in enumerate(items):
        length = item["tokens"].shape[0]
        tokens[row, :length] = item["tokens"]
        distances[row, :length, :length] = item["distances"]
        edge_types[row, :length, :length] = item["edge_types"]
    return {"tokens": tokens, "distances": distances, "edge_types": edge_types}


def collate_pairs(items: list[dict]) -> dict:
    return {
        "molecule": _pad_features([item["molecule"] for item in items]),
        "pocket": _pad_features([item["pocket"] for item in items]),
        "molecule_ids": [item["molecule_id"] for item in items],
        "pocket_ids": [item["pocket_id"] for item in items],
    }


def collate_structures(items: list[dict]) -> dict:
    return {
        "features": _pad_features([item["features"] for item in items]),
        "indices": [item["index"] for item in items],
        "names": [item["name"] for item in items],
        "labels": [item["label"] for item in items],
    }


def move_features(features: dict[str, torch.Tensor], device: torch.device) -> dict:
    return {name: value.to(device, non_blocking=True)
            for name, value in features.items()}

