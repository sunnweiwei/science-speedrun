"""Shared data and chemistry helpers for the Fourier-DeepSets starter."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from torch.utils.data import Dataset


RDLogger.DisableLog("rdApp.*")


def compact_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_jsonl(path: Path) -> list[dict]:
    values = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            values.append(value)
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def molecule_id(smiles: str) -> str:
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()


def spectrum_tensor(
    row: dict,
    max_peaks: int = 60,
    mz_min: float = 10.0,
    mz_max: float = 1000.0,
) -> torch.Tensor:
    mzs = row.get("mzs")
    intensities = row.get("intensities")
    if (
        not isinstance(mzs, list)
        or not isinstance(intensities, list)
        or not mzs
        or len(mzs) != len(intensities)
    ):
        raise ValueError(f"invalid peak arrays for {row.get('query_id')}")
    peaks = []
    for mz, intensity in zip(mzs, intensities):
        if type(mz) not in (int, float) or type(intensity) not in (int, float):
            raise ValueError("peak values must be numeric")
        if not math.isfinite(mz) or not math.isfinite(intensity) or intensity < 0:
            raise ValueError("peak values must be finite and intensities nonnegative")
        if mz_min <= mz <= mz_max:
            peaks.append((float(mz), float(intensity)))
    if max(intensities) <= 0:
        raise ValueError("spectrum maximum intensity must be positive")
    peaks = sorted(peaks, key=lambda item: (-item[1], item[0]))[:max_peaks]
    normalizer = max((item[1] for item in peaks), default=0.0)
    # The input contract does not impose this model's m/z window. If filtering
    # removes every positive peak, retain the precursor token and zero padding.
    peaks = (sorted((mz, intensity / normalizer) for mz, intensity in peaks)
             if normalizer > 0 else [])
    precursor = row.get("precursor_mz")
    if type(precursor) not in (int, float) or not math.isfinite(precursor) or precursor <= 0:
        raise ValueError("precursor_mz must be positive and finite")
    result = np.zeros((max_peaks + 1, 2), dtype=np.float32)
    result[0] = (float(precursor), 1.1)
    if peaks:
        result[1:len(peaks) + 1] = np.asarray(peaks, dtype=np.float32)
    return torch.from_numpy(result)


def morgan_fingerprint(
    smiles: str, size: int = 4096, radius: int = 2, *, allow_unparseable: bool = False,
) -> torch.Tensor:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        # RDKit releases disagree on some candidate valences. Preserve every
        # candidate with a zero similarity feature when this encoder cannot
        # parse it; never remove it from the pool or relax training validation.
        if allow_unparseable:
            return torch.zeros(size, dtype=torch.float32)
        raise ValueError(f"invalid candidate SMILES: {smiles!r}")
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule, radius=radius, nBits=size
    )
    array = np.zeros((size,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return torch.from_numpy(array)


def select_records(records: list[dict], maximum: int | None, seed: int) -> list[dict]:
    if maximum is None or maximum >= len(records):
        return records
    if maximum <= 0:
        raise ValueError("maximum record count must be positive")
    return sorted(
        records,
        key=lambda row: hashlib.sha256(
            f"{seed}\0{row['query_id']}".encode("utf-8")
        ).digest(),
    )[:maximum]


class LabeledSpectra(Dataset):
    def __init__(
        self,
        path: Path,
        maximum: int | None = None,
        seed: int = 0,
        fingerprint_size: int = 4096,
        fingerprint_radius: int = 2,
    ):
        records = select_records(read_jsonl(path), maximum, seed)
        if not records:
            raise ValueError(f"empty dataset: {path}")
        fingerprints = {}
        self.examples = []
        for row in records:
            smiles = row.get("target_smiles")
            if not isinstance(smiles, str) or not smiles:
                raise ValueError("labeled rows require target_smiles")
            if smiles not in fingerprints:
                fingerprints[smiles] = morgan_fingerprint(
                    smiles, size=fingerprint_size, radius=fingerprint_radius
                )
            self.examples.append((spectrum_tensor(row), fingerprints[smiles]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def load_candidate_pools(path: Path) -> dict[str, list[dict]]:
    pools = {}
    for row in read_jsonl(path):
        if set(row) != {"candidate_pool_id", "candidates"}:
            raise ValueError("invalid candidate pool fields")
        pool_id = row["candidate_pool_id"]
        candidates = row["candidates"]
        if not isinstance(pool_id, str) or pool_id in pools:
            raise ValueError("missing or duplicate candidate pool ID")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("candidate pool must be nonempty")
        expected = []
        for candidate in candidates:
            if set(candidate) != {"candidate_id", "smiles"}:
                raise ValueError("invalid candidate fields")
            if candidate["candidate_id"] != molecule_id(candidate["smiles"]):
                raise ValueError("candidate ID does not match SMILES")
            expected.append(candidate["candidate_id"])
        if len(expected) != len(set(expected)):
            raise ValueError("duplicate candidate ID")
        pools[pool_id] = candidates
    return pools
