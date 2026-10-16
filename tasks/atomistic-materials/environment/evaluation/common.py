"""Small, explicit interchange format. No model code is imported by the scorer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from ase import Atoms

PROTOCOL = "atomistic-materials-v0.1"
PHONON_FIELDS = {
    "max_freq": "K",
    "avg_freq": "K",
    "entropy": "J/(mol primitive cell K)",
    "heat_capacity": "J/(mol primitive cell K)",
    "free_energy": "kJ/mol primitive cell",
}
ELASTIC_FIELDS = {"bulk_modulus_vrh": "GPa", "shear_modulus_vrh": "GPa"}


def digest(path: Path, algorithm="sha256") -> str:
    """File digest; a directory (submission) digests its sorted relative paths and file digests."""
    h = hashlib.new(algorithm)
    if path.is_dir():
        for item in sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            h.update(f"{item.relative_to(path).as_posix()}\0{digest(item, algorithm)}\n".encode())
        return h.hexdigest()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def with_oom_retry(run, attempts=4, wait=20.0):
    """run() again after a CUDA out-of-memory error, which other jobs on a shared GPU can cause.

    The verifier runs every suite at once, so a GPU can fill up briefly; that is not the model's
    failure. Any other error, or a fourth OOM, propagates and the case is recorded as failed."""
    import gc
    import time
    for attempt in range(attempts):
        try:
            return run()
        except Exception as error:
            if type(error).__name__ != "OutOfMemoryError" or attempt == attempts - 1:
                raise
            gc.collect()
            import torch
            torch.cuda.empty_cache()
            time.sleep(wait * (attempt + 1))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def read_rows(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def indexed(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        key = row["id"]
        if key in result:
            raise ValueError(f"Duplicate ID: {key}")
        result[key] = row
    if not result:
        raise ValueError("Empty evaluation")
    return result


def encode_atoms(atoms: Atoms) -> dict:
    # Whitelist geometry: never serialize calc, info, site properties, or arrays
    # such as DFT magnetic moments/charges/forces.
    return {
        "numbers": atoms.numbers.tolist(),
        "positions": atoms.positions.tolist(),
        "cell": atoms.cell.array.tolist(),
        "pbc": atoms.pbc.tolist(),
    }


def decode_atoms(data: dict) -> Atoms:
    required = {"numbers", "positions", "cell", "pbc"}
    if set(data) != required:
        raise ValueError(f"Unexpected atomic fields: {set(data) ^ required}")
    atoms = Atoms(**data)
    if not len(atoms) or not np.isfinite(atoms.positions).all():
        raise ValueError("Invalid atomic coordinates")
    if not np.isfinite(atoms.cell).all() or atoms.get_volume() <= 0:
        raise ValueError("Invalid unit cell")
    if not atoms.pbc.all():
        raise ValueError("Core tasks require fully periodic crystals")
    return atoms
