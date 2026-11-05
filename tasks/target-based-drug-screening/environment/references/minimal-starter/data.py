"""Minimal LMDB reader and atom-count features for the starter baseline."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


ELEMENTS = (
    "H", "B", "C", "N", "O", "F", "SI", "P", "S", "CL", "BR", "I",
    "NA", "MG", "AL", "K", "CA", "MN", "FE", "CO", "NI", "CU", "ZN",
    "SE", "MO", "RU", "RH", "PD", "AG", "CD", "SN", "W", "PT", "AU",
    "HG", "PB", "OTHER",
)
ELEMENT_TO_INDEX = {element: index for index, element in enumerate(ELEMENTS)}


class LMDBRecords(Dataset):
    """Read contiguous numeric LMDB records without loading values into RAM."""

    def __init__(self, path: Path, max_records: int | None = None) -> None:
        import lmdb

        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        environment = lmdb.open(str(self.path), subdir=False, readonly=True,
                                lock=False, readahead=False, meminit=False)
        try:
            with environment.begin() as transaction:
                count = transaction.stat()["entries"]
                count = min(count, max_records) if max_records is not None else count
                self.keys = [str(index).encode("ascii") for index in range(count)]
                if self.keys and (transaction.get(self.keys[0]) is None
                                  or transaction.get(self.keys[-1]) is None):
                    raise ValueError(f"LMDB does not use contiguous numeric keys: {self.path}")
        finally:
            environment.close()
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


def atom_histogram(atoms) -> torch.Tensor:
    """Return a normalized element-count vector; coordinates are intentionally unused."""
    values = np.zeros(len(ELEMENTS), dtype=np.float32)
    for atom in atoms:
        symbol = str(atom).strip().upper()
        values[ELEMENT_TO_INDEX.get(symbol, ELEMENT_TO_INDEX["OTHER"])] += 1
    if not values.sum():
        raise ValueError("structure contains no atoms")
    values /= values.sum()
    return torch.from_numpy(values)


class PairedDataset(Dataset):
    def __init__(self, path: Path, max_records: int | None = None) -> None:
        self.records = LMDBRecords(path, max_records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        return {
            "molecule": atom_histogram(record["atoms"]),
            "pocket": atom_histogram(record["pocket_atoms"]),
        }


class SingleStructureDataset(Dataset):
    def __init__(self, path: Path, kind: str, max_records: int | None = None) -> None:
        if kind not in {"molecule", "pocket"}:
            raise ValueError("kind must be molecule or pocket")
        self.records = LMDBRecords(path, max_records)
        self.kind = kind

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        record = self.records[index]
        key = "atoms" if self.kind == "molecule" else "pocket_atoms"
        return index, atom_histogram(record[key])
