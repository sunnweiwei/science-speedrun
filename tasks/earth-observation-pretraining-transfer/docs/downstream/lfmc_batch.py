"""Deterministic original-window batches for complete LFMC development splits."""

import hashlib
import json
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
from lfmc_window import read_window


class LFMCWindows(Dataset):
    def __init__(self, index, split, seed=8123):
        if split not in ["train", "val"]:
            raise ValueError("Only development splits are allowed")
        self.rows = [r for r in index if r["split"] == split]
        if not self.rows or len({r["id"] for r in self.rows}) != len(self.rows):
            raise ValueError("Missing or duplicate dataset rows")
        self.rows.sort(key=lambda r: r["id"])
        self.split = split
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if self.split == "val":
            sample = read_window(row["path"], center_on_label=True)
        else:
            # No dependence on worker count, traversal timing, or model RNG.
            value = f"{self.seed}:{self.epoch}:{row['id']}".encode()
            seed = int.from_bytes(hashlib.sha256(value).digest()[:8], "big")
            sample = read_window(row["path"], rng=random.Random(seed))
        if sample["id"] != row["id"] or sample["split"] != self.split:
            raise ValueError("Window metadata differs from indexed split")
        return sample


def collate(samples):
    return {
        "ids": [r["id"] for r in samples],
        "bounds": [r["bounds"] for r in samples],
        "pixels": {
            k: torch.stack([torch.from_numpy(r["pixels"][k]) for r in samples])
            for k in samples[0]["pixels"]
        },
        "valid": {
            k: torch.stack([torch.from_numpy(r["valid"][k]) for r in samples])
            for k in samples[0]["valid"]
        },
        "target": torch.stack([torch.from_numpy(r["target"]) for r in samples]),
        "target_valid": torch.stack(
            [torch.from_numpy(r["target_valid"]) for r in samples]
        ),
    }


def build_index(root, output):
    root = Path(root)
    output = Path(output)
    if output.exists():
        raise ValueError("Preserve existing dataset index")
    extraction = json.loads((root / "extraction.json").read_text())
    if extraction["window_counts"] != {
        "train": 31175,
        "val": 8262,
        "test": 4585,
    } or extraction["extracted_splits"] != ["train", "val"]:
        raise ValueError("Incomplete source extraction")
    rows = []
    counts = {"train": 0, "val": 0}
    for path in sorted((root / "windows/spatial_split").iterdir()):
        meta = json.loads((path / "metadata.json").read_text())
        split = meta["options"]["split"]
        if split not in counts:
            raise ValueError("Heldout or unknown split in development dataset")
        counts[split] += 1
        rows.append({"id": meta["name"], "split": split, "path": str(path.resolve())})
    if counts != {"train": 31175, "val": 8262} or len({r["id"] for r in rows}) != len(
        rows
    ):
        raise ValueError("Dataset index has wrong population")
    output.write_text(
        json.dumps(
            {
                "rows": rows,
                "counts": counts,
                "test_used": False,
                "extraction_sha256": hashlib.sha256(
                    (root / "extraction.json").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    return counts


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(build_index(a.root, a.output))
