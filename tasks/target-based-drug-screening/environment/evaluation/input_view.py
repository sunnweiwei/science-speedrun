"""Trusted controller helpers for a label-free, test-shaped inference view."""

from __future__ import annotations

import hashlib
import hmac
import json
import pickle
import secrets
from pathlib import Path

import lmdb
import numpy as np

BENCHMARK_TARGETS = {"dude": 102, "lit_pcba": 15}
FIELDS = {"molecule": ("atoms", "coordinates", "smi"),
          "pocket": ("pocket_atoms", "pocket_coordinates")}


def discover_targets(root: Path, *, full_suite: bool = False) -> dict:
    targets = {}
    for benchmark, expected_count in BENCHMARK_TARGETS.items():
        directory = root / "test" / benchmark
        selected = sorted(path for path in directory.iterdir() if path.is_dir()) \
            if directory.is_dir() else []
        if full_suite and len(selected) != expected_count:
            raise ValueError(f"{benchmark}: expected {expected_count} targets, got {len(selected)}")
        for target in selected:
            molecules = target / "mols.lmdb"
            pocket = target / ("pocket.lmdb" if benchmark == "dude" else "pockets.lmdb")
            if benchmark == "lit_pcba" and not pocket.is_file():
                pocket = target / "pocket.lmdb"
            for path in (molecules, pocket):
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"missing or unsafe benchmark LMDB: {path}")
            targets[benchmark, target.name] = (molecules, pocket)
    if not targets:
        raise ValueError("benchmark has no targets")
    return targets


def redact_lmdb(source: Path, destination: Path, kind: str,
                *, order: np.ndarray | None = None) -> int:
    """Whitelist model inputs, assigning fresh keys in the supplied order."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    origin = lmdb.open(str(source), subdir=False, readonly=True, lock=False,
                       readahead=False, meminit=False)
    output = lmdb.open(str(destination), subdir=False,
                       map_size=max(64 << 20, source.stat().st_size * 2))
    try:
        with origin.begin() as reader:
            count = reader.stat()["entries"]
            if count < 1:
                raise ValueError(f"empty LMDB: {source}")
            if order is not None and (order.dtype.kind not in "iu" or
                    order.shape != (count,) or
                    not np.array_equal(np.sort(order), np.arange(count))):
                raise ValueError("record order must be a complete permutation")
            labels = set()
            for start in range(0, count, 1024):
                with output.begin(write=True) as writer:
                    for index in range(start, min(start + 1024, count)):
                        key = str(index).encode("ascii")
                        original = index if order is None else int(order[index])
                        value = reader.get(str(original).encode("ascii"))
                        if value is None:
                            raise ValueError(f"noncontiguous LMDB keys: {source}")
                        record = pickle.loads(value)
                        if kind == "molecule":
                            label = record.get("label")
                            if label not in (0, 1):
                                raise ValueError(f"invalid benchmark label: {source}/{index}")
                            labels.add(label)
                        # A whitelist excludes labels, compound IDs, source paths,
                        # train/test metadata, and arbitrary fields added upstream.
                        clean = {field: record[field] for field in FIELDS[kind]}
                        writer.put(key, pickle.dumps(clean, protocol=4))
            if kind == "molecule" and labels != {0, 1}:
                raise ValueError(f"benchmark target needs both classes: {source}")
        output.sync()
    finally:
        output.close()
        origin.close()
    destination.with_name(destination.name + "-lock").unlink(missing_ok=True)
    return count


def build_pocket_derangement(targets: dict) -> dict[tuple[str, str], tuple[str, str]]:
    """Deterministically assign each target a donor pocket from a dissimilar target.

    Within each benchmark with >= 2 targets, sorted targets are shifted by half the
    roster (`N // 2`), guaranteeing a fixed-point-free derangement and separating
    alphabetically adjacent gene-family paralogs (e.g. akt1/akt2, esr1/esr2).
    If any benchmark has < 2 targets (e.g. minimal unit-test fixtures), derangement
    is performed across all discovered targets.
    """
    all_keys = sorted(targets.keys())
    if len(all_keys) < 2:
        raise ValueError("swapped-pocket counterfactual requires at least 2 targets")
    by_benchmark: dict[str, list[tuple[str, str]]] = {}
    for key in all_keys:
        by_benchmark.setdefault(key[0], []).append(key)
    mapping: dict[tuple[str, str], tuple[str, str]] = {}
    if all(len(group) >= 2 for group in by_benchmark.values()):
        for group in by_benchmark.values():
            n = len(group)
            shift = n // 2
            for idx, recipient in enumerate(group):
                donor = group[(idx + shift) % n]
                mapping[recipient] = donor
    else:
        n = len(all_keys)
        shift = n // 2
        for idx, recipient in enumerate(all_keys):
            donor = all_keys[(idx + shift) % n]
            mapping[recipient] = donor
    return mapping


def stage_inputs(root: Path, destination: Path, private_mapping: Path, *,
                 full_suite: bool = True, seed: bytes | None = None,
                 swap_pockets: bool = False) -> dict:
    """Shuffle each molecule target; keep the seed/index map outside inputs.

    A controller may reuse a private 32-byte seed for acceptance comparisons.
    Never supply this seed, manifest or original indices to candidate code.
    When `swap_pockets=True`, each target receives a deterministic donor pocket
    from a dissimilar target for label-free counterfactual specificity replay.
    """
    targets = discover_targets(root, full_suite=full_suite)
    pocket_donors = build_pocket_derangement(targets) if swap_pockets else None
    if (private_mapping.resolve().is_relative_to(destination.resolve()) or
            destination.resolve().is_relative_to(private_mapping.resolve())):
        raise ValueError("private mapping and inference inputs must be separate")
    seed = secrets.token_bytes(32) if seed is None else seed
    if not isinstance(seed, bytes) or len(seed) != 32:
        raise ValueError("permutation seed must contain exactly 32 bytes")
    if destination.exists():
        raise FileExistsError(destination)
    if private_mapping.exists():
        raise FileExistsError(private_mapping)
    destination.mkdir(parents=True)
    private_mapping.mkdir(parents=True, mode=0o700)
    counts = {}
    mappings = {}
    for key, (molecules, pocket) in targets.items():
        benchmark, target = key
        output = destination / "test" / benchmark / target
        origin = lmdb.open(str(molecules), subdir=False, readonly=True, lock=False)
        try:
            with origin.begin() as transaction:
                count = transaction.stat()["entries"]
        finally:
            origin.close()
        # Independent, deterministic streams per target with private entropy.
        entropy = hmac.digest(seed, f"{benchmark}/{target}".encode(), "sha256")
        generator = np.random.Generator(np.random.PCG64(int.from_bytes(entropy, "big")))
        order = generator.permutation(count)
        count = redact_lmdb(molecules, output / "mols.lmdb", "molecule", order=order)
        relative = f"{benchmark}/{target}.npy"
        mapping = private_mapping / relative
        mapping.parent.mkdir(parents=True, exist_ok=True)
        np.save(mapping, order, allow_pickle=False)
        mapping.chmod(0o600)
        mappings[f"{benchmark}/{target}"] = {
            "path": relative, "count": count,
            "sha256": hashlib.sha256(mapping.read_bytes()).hexdigest()}
        pocket_name = "pocket.lmdb" if benchmark == "dude" else "pockets.lmdb"
        pocket_source = targets[pocket_donors[key]][1] if pocket_donors is not None else pocket
        redact_lmdb(pocket_source, output / pocket_name, "pocket")
        counts[key] = count
    manifest = {"schema": "drugclip-private-order/v1", "seed_hex": seed.hex(),
                "algorithm": "HMAC-SHA256 target seed / NumPy PCG64 permutation",
                "numpy_version": np.__version__, "targets": mappings}
    if pocket_donors is not None:
        manifest["pocket_mode"] = "swapped_counterfactual"
        manifest["swapped_pocket_donors"] = {
            f"{b}/{t}": f"{db}/{dt}"
            for (b, t), (db, dt) in sorted(pocket_donors.items())
        }
    receipt = private_mapping / "manifest.json"
    receipt.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    receipt.chmod(0o600)
    return counts


def load_permutation(private_mapping: Path, benchmark: str, target: str,
                     count: int) -> np.ndarray:
    """Validate a trusted view-index -> original-index map before scoring."""
    try:
        manifest = json.loads((private_mapping / "manifest.json").read_text())
        if manifest["schema"] != "drugclip-private-order/v1":
            raise ValueError("unsupported mapping schema")
        item = manifest["targets"][f"{benchmark}/{target}"]
        path = private_mapping / item["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(private_mapping.resolve()):
            raise ValueError("mapping path escapes private directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("mapping checksum mismatch")
        order = np.load(path, allow_pickle=False)
        if (item["count"] != count or order.dtype.kind not in "iu" or
                order.shape != (count,) or
                not np.array_equal(np.sort(order), np.arange(count))):
            raise ValueError("mapping is not a complete permutation")
        return order
    except (OSError, ValueError, KeyError, TypeError) as error:
        # Trusted preparation corruption is infrastructure failure, not reward 0.
        raise RuntimeError(f"invalid private order mapping: {error}") from error
