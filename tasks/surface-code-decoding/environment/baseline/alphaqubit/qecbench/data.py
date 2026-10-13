"""Physically separated data and strict, label-free inference input contract."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
import zipfile

import numpy as np

SCHEMA = "surface-code-decoding-input/v1"
NAME = re.compile(r"surface_code_b[XZ]_d[35]_r[0-9]{2}_center_[0-9]+_[0-9]+")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def contained(root: Path, relative: str) -> Path:
    if not isinstance(relative, str):
        raise ValueError("Path must be a string")
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or rel.as_posix() != relative:
        raise ValueError(f"Unsafe relative path: {relative}")
    path = root / rel
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path escapes root")
    cursor = path
    while cursor != root:
        if cursor.is_symlink():
            raise ValueError("Symbolic links are not accepted")
        cursor = cursor.parent
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_npz(path: Path, max_bytes: int) -> dict[str, np.ndarray]:
    """Bound decompression before NumPy allocates arrays; never allow pickle."""
    if path.stat().st_size > max_bytes + 65536:
        raise ValueError("NPZ file exceeds declared shot count")
    with zipfile.ZipFile(path) as archive:
        info = archive.infolist()
        names = [item.filename for item in info]
        if len(names) != len(set(names)) or len(names) > 4:
            raise ValueError("Duplicate/excess archive members")
        if any("/" in name or not name.endswith(".npy") for name in names):
            raise ValueError("Invalid NPZ member")
        if sum(item.file_size for item in info) > max_bytes + 4096:
            raise ValueError("NPZ expanded content exceeds declared shot count")
        for item in info:
            if item.flag_bits & 1 or item.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ValueError("Encrypted/unsupported NPZ compression is not accepted")
            with archive.open(item) as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
                else:
                    raise ValueError("Only NumPy numeric-array format v1/v2 is accepted")
                if dtype.hasobject or len(shape) > 4 or any(type(n) is not int or n < 0 for n in shape):
                    raise ValueError("Unsupported NumPy array header")
                required = math.prod(shape) * dtype.itemsize
                if required > max_bytes or required + handle.tell() != item.file_size:
                    raise ValueError("NumPy shape disagrees with bounded member size")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_manifest(root: Path, allow_labels: bool = False) -> dict[str, Any]:
    manifest = json.loads(contained(root, "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported input schema")
    labeled = manifest.get("split") in ("train", "validation")
    if labeled and not allow_labels:
        raise ValueError("Inference must receive a physically label-free input directory")
    rows = manifest.get("conditions")
    if not isinstance(rows, list) or not rows or len(rows) > 130:
        raise ValueError("Expected 1..130 conditions")
    seen = set()
    for record in rows:
        name = record.get("name", "")
        if not NAME.fullmatch(name) or name in seen:
            raise ValueError("Invalid/duplicate condition name")
        seen.add(name)
        if (record.get("distance") not in (3, 5) or record.get("basis") not in ("X", "Z")
                or record.get("rounds") not in range(1, 26, 2)):
            raise ValueError("Unsupported geometry or duration")
        if record.get("num_detectors") != record["rounds"] * (record["distance"] ** 2 - 1):
            raise ValueError("Invalid detector count")
        shots = record.get("shots")
        if type(shots) is not int or not 1 <= shots <= 100_000_000:
            raise ValueError("Invalid shot count")
        for field in ("arrays", "circuit"):
            path = contained(root, record[field])
            if sha256(path) != record[f"{field}_sha256"]:
                raise ValueError(f"Input checksum mismatch: {record[field]}")
    return manifest


def load_condition(root: Path, record: dict[str, Any], labels: bool = False) -> dict[str, np.ndarray]:
    n, width = record["shots"], (record["num_detectors"] + 7) // 8
    arrays = read_npz(contained(root, record["arrays"]), n * (8 + width + int(labels)))
    expected = {"sample_ids", "detectors", "labels"} if labels else {"sample_ids", "detectors"}
    if set(arrays) != expected:
        raise ValueError(f"Wrong array fields; expected {expected}")
    ids, packed = arrays["sample_ids"], arrays["detectors"]
    if ids.dtype != np.uint64 or ids.shape != (n,) or len(np.unique(ids)) != n:
        raise ValueError("sample_ids must be unique uint64[N]")
    if packed.dtype != np.uint8 or packed.shape != (n, width):
        raise ValueError("detectors must be packed uint8[N,ceil(D/8)]")
    extra = record["num_detectors"] % 8
    if extra and np.any(packed[:, -1] >> extra):
        raise ValueError("Unused packed detector bits must be zero")
    if labels:
        target = arrays["labels"]
        if target.dtype != np.uint8 or target.shape != (n,) or np.any(target > 1):
            raise ValueError("labels must be binary uint8[N]")
    return arrays


def condition_from_record(root: Path, record: dict[str, Any]) -> Any:
    from baselines.data import Condition
    return Condition(path=contained(root, record["circuit"]).parent, name=record["name"],
                     distance=record["distance"], basis=record["basis"], rounds=record["rounds"],
                     group_id=record["group"])


class DevelopmentDataset:
    """Warm-start loader reading physically split train/validation data only.

    A batch is sampled within one condition so its feature tensor has a fixed
    number of rounds. The 13 durations are selected uniformly. Labels in the
    returned batch are official-path observable flips, not raw data parity.
    """
    def __init__(self, root: Path, split: str, group: str | None = None) -> None:
        if split not in ("train", "validation"):
            raise ValueError("Development loader permits train/validation only")
        self.root = root / split
        manifest = load_manifest(self.root, allow_labels=True)
        if manifest["split"] != split:
            raise ValueError("Split mismatch")
        self.records = [row for row in manifest["conditions"] if group is None or row["group"] == group]
        if not self.records:
            raise ValueError("No matching conditions")
        self._arrays: dict[str, dict[str, np.ndarray]] = {}
        self._layouts: dict[str, Any] = {}

    def get(self, index: int) -> tuple[Any, dict[str, np.ndarray]]:
        from baselines.alphaqubit_data import AlphaQubitLayout
        record = self.records[index]
        name = record["name"]
        if name not in self._arrays:
            self._arrays[name] = load_condition(self.root, record, labels=True)
            self._layouts[name] = AlphaQubitLayout.from_condition(condition_from_record(self.root, record))
        return self._layouts[name], self._arrays[name]

    def batch(self, rng: np.random.Generator, size: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
        if size <= 0:
            raise ValueError("Batch size must be positive")
        layout, arrays = self.get(int(rng.integers(len(self.records))))
        indices = rng.integers(len(arrays["sample_ids"]), size=size)
        return layout.decode_packed_features(arrays["detectors"][indices]), arrays["labels"][indices].astype(np.float32)
