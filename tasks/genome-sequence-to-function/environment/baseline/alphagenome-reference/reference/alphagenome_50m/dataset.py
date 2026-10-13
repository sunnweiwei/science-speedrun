"""Streaming reader for the compact 1 Mb / eleven-family release."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import torch
from torch.utils.data import Dataset
import zarr


def _open_part(path: Path):
    if str(path).endswith(".zarr.zip"):
        store = zarr.ZipStore(str(path), mode="r")
        try:
            return zarr.open_group(store=store, mode="r"), store
        except BaseException:
            store.close()
            raise
    return zarr.open_group(str(path), mode="r"), None


DENSE_FAMILIES = (
    "atac",
    "dnase",
    "procap",
    "cage",
    "rna_seq",
    "chip_tf",
    "chip_histone",
)
def normalize_splice_positions(value) -> np.ndarray:
    """Normalize Arrow's nested-list representation to the public [4, 512] array."""

    outer = np.asarray(value, dtype=object).reshape(-1).tolist()
    positions = np.stack([np.asarray(item, dtype=np.int64) for item in outer])
    if positions.shape != (4, 512):
        raise ValueError(f"splice_site_positions must have shape (4, 512), got {positions.shape}")
    return positions


class AlphaGenomeDataset(Dataset):
    """Resolve frozen coordinates to reference DNA and one Zarr example group."""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        *,
        max_open_parts: int = 8,
        include_targets: bool = True,
    ) -> None:
        self.root = Path(data_root)
        self.split = split
        manifest = self.root / "public" / f"{split}.parquet"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing materialized manifest: {manifest}")
        self.rows = pd.read_parquet(manifest).reset_index(drop=True)
        junction_tracks = pd.read_parquet(self.root / "public/junction_tracks.parquet")
        junction_tracks = junction_tracks.sort_values("tensor_index")
        self.junction_track_halves = torch.from_numpy(
            junction_tracks.junction_channel_half.to_numpy(np.int64)
        )
        if include_targets:
            labels_path = self.root / "labels" / f"{split}.parquet"
            if not labels_path.is_file():
                raise FileNotFoundError(f"Missing trusted labels manifest: {labels_path}")
            labels = pd.read_parquet(labels_path).reset_index(drop=True)
            if self.rows.example_id.astype(str).tolist() != labels.example_id.astype(str).tolist():
                raise ValueError("public and labels manifests have different IDs/order")
            self.rows = self.rows.merge(labels, on="example_id", validate="one_to_one", sort=False)
        self.fasta_path = self.root / "reference" / "GRCh38.p13.genome.fa"
        self.max_open_parts = max_open_parts
        self.include_targets = include_targets
        self._fasta = None
        self._parts: OrderedDict[str, tuple[zarr.Group, object | None]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.rows)

    def _reference(self):
        if self._fasta is None:
            self._fasta = pysam.FastaFile(str(self.fasta_path))
        return self._fasta

    def _part(self, relative: str):
        if relative in self._parts:
            opened = self._parts.pop(relative)
            self._parts[relative] = opened
            return opened[0]
        group, store = _open_part(self.root / relative)
        if len(self._parts) >= self.max_open_parts:
            _, (_, old_store) = self._parts.popitem(last=False)
            if old_store is not None:
                old_store.close()
        self._parts[relative] = (group, store)
        return group

    @staticmethod
    def _one_hot(sequence: str) -> np.ndarray:
        encoded = np.frombuffer(sequence.upper().encode("ascii"), dtype=np.uint8)
        output = np.zeros((len(encoded), 4), dtype=np.float32)
        for index, letter in enumerate(b"ACGT"):
            output[:, index] = encoded == letter
        return output

    def _sequence(self, row) -> tuple[np.ndarray, np.ndarray]:
        reference = self._reference()
        chromosome_length = reference.get_reference_length(str(row.chromosome))
        start, end = int(row.input_start), int(row.input_end)
        clipped_start, clipped_end = max(0, start), min(chromosome_length, end)
        sequence = reference.fetch(str(row.chromosome), clipped_start, clipped_end)
        left, right = clipped_start - start, end - clipped_end
        if left or right:
            sequence = "N" * left + sequence + "N" * right
        valid = np.ones(end - start, dtype=bool)
        if left:
            valid[:left] = False
        if right:
            valid[-right:] = False
        return self._one_hot(sequence), valid

    @staticmethod
    def _decode(example, name: str, *, asinh: bool) -> np.ndarray:
        """Train targets may be stored as 8-bit codes with per-track ranges; they are decoded
        here, one example at a time, so the release stays compressed on disk."""
        array = example[name]
        if array.dtype != np.uint8:
            return np.asarray(array, dtype=np.float32)
        lo = np.asarray(example[f"{name}_lo"], dtype=np.float32)
        hi = np.asarray(example[f"{name}_hi"], dtype=np.float32)
        x = lo + np.asarray(array, dtype=np.float32) * ((hi - lo) / 255.0)
        return np.sinh(x) if asinh else x

    @staticmethod
    def _dense_target(example, family: str) -> dict[str, torch.Tensor]:
        values = AlphaGenomeDataset._decode(example, f"{family}/values", asinh=True)
        mask = np.asarray(example[f"{family}/mask"], dtype=bool)
        if mask.ndim == 1:
            mask = np.broadcast_to(mask[None, :], values.shape).copy()
        return {
            "values": torch.from_numpy(values),
            "mask": torch.from_numpy(mask),
        }

    def __getitem__(self, item: int):
        row = self.rows.iloc[item]
        dna, input_mask = self._sequence(row)
        result = {
            "example_id": str(row.example_id),
            "dna": torch.from_numpy(dna),
            "input_mask": torch.from_numpy(input_mask),
            "junction_track_halves": self.junction_track_halves,
        }
        if not self.include_targets:
            return result

        part = self._part(str(row.target_shard))
        example = part[str(row.target_group)]
        result["dense"] = {
            family: self._dense_target(example, family) for family in DENSE_FAMILIES
        }
        result["splice_sites"] = torch.from_numpy(
            np.asarray(example["splice_sites/values"], dtype=np.float32)
        )
        result["splice_usage"] = {
            "positions": torch.from_numpy(
                np.asarray(example["splice_site_usage/positions"], dtype=np.int64)
            ),
            "values": torch.from_numpy(
                np.asarray(example["splice_site_usage/values"], dtype=np.float32)
            ),
            "mask": torch.from_numpy(
                np.asarray(example["splice_site_usage/mask"], dtype=bool)
            ),
        }
        result["usage_positions"] = result["splice_usage"]["positions"]
        result["junctions"] = {
            name: torch.from_numpy(
                np.asarray(
                    example[f"splice_junctions/{name}"],
                    dtype=np.float32 if name == "value" else np.int64,
                )
            )
            for name in ("row", "column", "track", "value")
        }
        result["junctions"]["site_positions"] = torch.from_numpy(
            np.asarray(example["splice_junctions/site_positions"], dtype=np.int64)
        )
        site_positions = normalize_splice_positions(row.splice_site_positions)
        result["site_positions"] = torch.from_numpy(site_positions)
        if not torch.equal(result["junctions"]["site_positions"], result["site_positions"]):
            raise ValueError(f"public/private splice positions differ for {row.example_id}")
        scoring_path = self.root / "scoring" / self.split / f"part-{item:06d}.parquet"
        scoring = pd.read_parquet(scoring_path)
        if not (scoring.example_id.astype(str) == str(row.example_id)).all():
            raise ValueError(f"scoring part ID differs for {row.example_id}")
        result["junction_scoring"] = {
            "donor_position": torch.from_numpy(
                scoring.donor_position.to_numpy(np.int64)
            ),
            "acceptor_position": torch.from_numpy(
                scoring.acceptor_position.to_numpy(np.int64)
            ),
            "track": torch.from_numpy(scoring.track.to_numpy(np.int64)),
            "target_value": torch.from_numpy(
                scoring.target_value.to_numpy(np.float32)
            ),
            "valid": torch.from_numpy(scoring.valid.to_numpy(bool)),
        }
        result["contact_maps"] = torch.from_numpy(
            self._decode(example, "contact_maps/upper_triangle", asinh=False)
        )
        result["contact_map_mask"] = torch.from_numpy(
            np.asarray(example["contact_maps/mask"], dtype=bool)
        )
        return result


def single_example_collate(items):
    """Keep variable-size event tensors intact; the reference currently trains at batch one."""

    if len(items) != 1:
        raise ValueError("The compact event loader currently requires batch_size=1")
    return items[0]
