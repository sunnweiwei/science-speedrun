"""Prepare fixed, bit-packed circuit-noise data before allocating training GPUs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .data import CircuitSampler, group_conditions


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_directory(root: Path, group: str, seed: int, shots: int) -> Path:
    return root / group / f"seed{seed}-shots{shots}"


def source_fingerprints(data_root: Path, group: str) -> dict[str, dict[str, str]]:
    conditions = group_conditions(data_root)[group]
    if [c.rounds for c in conditions] != list(range(1, 26, 2)):
        raise ValueError("Each group must contain all 13 odd lengths from 1 to 25")
    return {c.name: {name: sha256(c.path / name) for name in
                    ("circuit_ideal.stim", "circuit_noisy.stim")} for c in conditions}


def verify_cache(data_root: Path, output: Path, *, full_hashes: bool = True) -> dict[str, Any]:
    """Reject cache reuse after source, software, membership or payload changes."""
    import stim
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("version") != 2:
        raise ValueError("Cache lacks source provenance; explicitly attest or regenerate it")
    group = manifest["group"]
    conditions = group_conditions(data_root)[group]
    if manifest["source_circuits"] != source_fingerprints(data_root, group):
        raise ValueError("Cache source circuits differ from the current data release")
    if manifest["stim_version"] != stim.__version__:
        raise ValueError("Cache generation Stim version differs from the pinned runtime")
    if [x["condition"] for x in manifest["conditions"]] != [x.name for x in conditions]:
        raise ValueError("Cache condition membership/order mismatch")
    if sum(x["shots"] for x in manifest["conditions"]) != manifest["shots"]:
        raise ValueError("Cache shot count mismatch")
    if full_hashes:
        for entry in manifest["conditions"]:
            for filename, expected in entry["sha256"].items():
                if sha256(output / filename) != expected:
                    raise ValueError(f"Corrupt existing cache: {filename}")
    return manifest


def attest_source(data_root: Path, output: Path) -> None:
    """Upgrade the campaign's v1 caches only after source/sample consistency checks.

    Initial generation ran before source fingerprints were added. This explicit
    migration verifies every payload hash, the pinned Stim version, and the
    first generated 65,536-shot block for every condition against current
    circuits; it does not pretend source hashes were recorded during generation.
    """
    import stim
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") == 2:
        verify_cache(data_root, output)
        return
    if manifest.get("version") != 1 or manifest["stim_version"] != stim.__version__:
        raise ValueError("Cannot attest unknown cache/version")
    conditions = group_conditions(data_root)[manifest["group"]]
    if [x["condition"] for x in manifest["conditions"]] != [x.name for x in conditions]:
        raise ValueError("Cannot attest incomplete cache")
    for condition, entry in zip(conditions, manifest["conditions"], strict=True):
        for filename, expected in entry["sha256"].items():
            if sha256(output / filename) != expected:
                raise ValueError(f"Corrupt cache during attestation: {filename}")
        n = min(65_536, entry["shots"])
        expected_dets, expected_labels = CircuitSampler(
            condition, entry["seed"], noise_source="circuit").sample_packed(n)
        actual_dets = np.load(output / entry["files"]["train_detectors"], mmap_mode="r")
        actual_labels = np.load(output / entry["files"]["train_labels"], mmap_mode="r")
        if not np.array_equal(expected_dets, actual_dets[:n]) or not np.array_equal(expected_labels, actual_labels[:n]):
            raise ValueError(f"Source/sample mismatch: {condition.name}")
    manifest["version"] = 2
    manifest["source_circuits"] = source_fingerprints(data_root, manifest["group"])
    manifest["sampler_chunk_shots"] = 65_536
    manifest["source_attestation"] = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "method": "post-generation source fingerprints; all payload hashes and each first sampling block verified",
    }
    atomic_json(manifest_path, manifest)


def prepare(
    data_root: Path,
    cache_root: Path,
    group: str,
    seed: int,
    shots: int,
    validation_shots: int = 5120,
) -> Path:
    """Generate a fixed training population with independent synthetic validation."""
    conditions = group_conditions(data_root)[group]
    if shots < len(conditions) or validation_shots < 1:
        raise ValueError("Positive per-condition sample counts are required")
    output = cache_directory(cache_root, group, seed, shots)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = verify_cache(data_root, output)
        if (manifest["group"], manifest["seed"], manifest["shots"]) != (group, seed, shots):
            raise ValueError("Cache manifest identity differs from requested configuration")
        if manifest["validation_shots_per_condition"] != validation_shots:
            raise ValueError("Existing cache uses a different validation population")
        print(json.dumps({"reused_verified_cache": str(output)}), flush=True)
        return output

    start = time.monotonic()
    fingerprints = source_fingerprints(data_root, group)
    entries: list[dict[str, Any]] = []
    for index, condition in enumerate(conditions):
        count = shots // len(conditions) + int(index < shots % len(conditions))
        sampler_seed = seed * 100_000 + condition.rounds
        sampler = CircuitSampler(condition, seed=sampler_seed, noise_source="circuit")
        packed_width = (sampler.num_detectors + 7) // 8
        names = {
            "train_detectors": f"r{condition.rounds:02d}-train-detectors.npy",
            "train_labels": f"r{condition.rounds:02d}-train-labels.npy",
            "validation_detectors": f"r{condition.rounds:02d}-validation-detectors.npy",
            "validation_labels": f"r{condition.rounds:02d}-validation-labels.npy",
        }
        detector_file = np.lib.format.open_memmap(
            output / names["train_detectors"], mode="w+", dtype=np.uint8,
            shape=(count, packed_width),
        )
        label_file = np.lib.format.open_memmap(
            output / names["train_labels"], mode="w+", dtype=np.uint8, shape=(count, 1),
        )
        for offset in range(0, count, 65_536):
            n = min(65_536, count - offset)
            packed, labels = sampler.sample_packed(n)
            detector_file[offset:offset + n] = packed
            label_file[offset:offset + n] = labels
        detector_file.flush()
        label_file.flush()
        del detector_file, label_file
        validation = CircuitSampler(
            condition, seed=sampler_seed + 1_000_000_000, noise_source="circuit",
        )
        packed, labels = validation.sample_packed(validation_shots)
        np.save(output / names["validation_detectors"], packed, allow_pickle=False)
        np.save(output / names["validation_labels"], labels, allow_pickle=False)
        entry = {
            "condition": condition.name, "rounds": condition.rounds,
            "shots": count, "num_detectors": sampler.num_detectors,
            "seed": sampler_seed, "validation_seed": sampler_seed + 1_000_000_000,
            "files": names, "sha256": {name: sha256(output / name) for name in names.values()},
        }
        entries.append(entry)
        atomic_json(output / "progress.json", {
            "status": "generating", "completed_conditions": len(entries),
            "total_conditions": len(conditions), "elapsed_seconds": time.monotonic() - start,
        })
        print(json.dumps({"condition": condition.name, "shots": count,
                          "elapsed_seconds": time.monotonic() - start}), flush=True)
    import stim
    atomic_json(manifest_path, {
        "version": 2, "group": group, "seed": seed, "shots": shots,
        "distance": conditions[0].distance,
        "validation_shots_per_condition": validation_shots,
        "stim_version": stim.__version__, "noise_source": "released circuit_noisy.stim",
        "generation_seconds": time.monotonic() - start,
        "job_id": os.environ.get("SLURM_JOB_ID"), "conditions": entries,
        "source_circuits": fingerprints, "sampler_chunk_shots": 65_536,
        "protocol": "fixed independent synthetic train and validation; zero sweep reference",
    })
    atomic_json(output / "progress.json", {"status": "complete", "shots": shots})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--validation-shots", type=int, default=5120)
    args = parser.parse_args()
    print(prepare(args.data_root, args.cache_root, args.group, args.seed,
                  args.shots, args.validation_shots))


if __name__ == "__main__":
    main()
