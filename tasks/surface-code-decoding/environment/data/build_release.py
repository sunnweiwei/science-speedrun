#!/usr/bin/env python3
"""Deterministically package the audited source into separate role archives."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
from typing import Any
import numpy as np

TASK = Path(__file__).resolve().parents[2] / "environment" / "baseline" / "alphaqubit"  # the baseline project
sys.path.insert(0, str(TASK))
from qecbench.package import write_template
from baselines.data import discover_conditions, hardware_split_indices, load_hardware_packed
from qecbench.data import SCHEMA, sha256, write_json

RELEASE = "sycamore-hard-v1"


def sample_ids(condition: str, indices: np.ndarray) -> np.ndarray:
    # Original row indices are intentionally not part of the inference schema.
    return np.array([int.from_bytes(hashlib.blake2b(
        f"{RELEASE}|{condition}|{int(index)}".encode(), digest_size=8).digest(), "little")
        for index in indices], dtype=np.uint64)


def archive(source: Path, destination: Path) -> None:
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as tar:
                for path in sorted(source.rglob("*")):
                    if path.is_symlink():
                        raise ValueError("Release cannot contain symlinks")
                    name = path.relative_to(source.parent).as_posix()
                    info = tarfile.TarInfo(name)
                    info.type = tarfile.DIRTYPE if path.is_dir() else tarfile.REGTYPE
                    info.size = 0 if path.is_dir() else path.stat().st_size
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.mode = 0o755 if path.is_dir() or path.name.endswith(".sh") else 0o644
                    if path.is_file():
                        with path.open("rb") as handle:
                            tar.addfile(info, handle)
                    else:
                        tar.addfile(info)


def build(source: Path, reference_results: Path, output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Release output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    role_root = output / "materialized"
    development, verifier, grader = (role_root / role for role in ("development", "verifier", "grader"))
    conditions = discover_conditions(source)
    if len(conditions) != 130 or len({row.group_id for row in conditions}) != 10:
        raise ValueError("Require the complete audited 130-condition source")
    manifests = {split: {"schema": SCHEMA, "release": RELEASE, "split": split, "conditions": []}
                 for split in ("train", "validation", "test")}
    labels_manifest: dict[str, Any] = {"schema": "surface-code-decoding-labels/v1", "release": RELEASE, "conditions": []}
    for condition in conditions:
        for split in manifests:
            detectors, labels = load_hardware_packed(condition, split)
            ids = sample_ids(condition.name, hardware_split_indices(split))
            root = verifier if split == "test" else development / split
            folder = root / "conditions" / condition.name
            folder.mkdir(parents=True)
            payload = {"sample_ids": ids, "detectors": detectors}
            if split != "test":
                payload["labels"] = labels[:, 0]
            np.savez(folder / "arrays.npz", **payload)
            shutil.copyfile(condition.path / "circuit_ideal.stim", folder / "circuit_ideal.stim")
            if split != "test":
                shutil.copyfile(condition.path / "pij_from_even_for_odd.dem", folder / "pij_from_even_for_odd.dem")
            record = {"name": condition.name, "group": condition.group_id, "distance": condition.distance,
                      "basis": condition.basis, "rounds": condition.rounds, "shots": len(ids),
                      "num_detectors": condition.rounds * (condition.distance**2 - 1),
                      "arrays": f"conditions/{condition.name}/arrays.npz",
                      "arrays_sha256": sha256(folder / "arrays.npz"),
                      "circuit": f"conditions/{condition.name}/circuit_ideal.stim",
                      "circuit_sha256": sha256(folder / "circuit_ideal.stim")}
            if split != "test":
                record.update(noise=f"conditions/{condition.name}/pij_from_even_for_odd.dem",
                              noise_sha256=sha256(folder / "pij_from_even_for_odd.dem"))
            manifests[split]["conditions"].append(record)
            if split == "test":
                target = grader / "conditions" / f"{condition.name}.npz"
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez(target, sample_ids=ids, labels=labels[:, 0])
                labels_manifest["conditions"].append({"name": condition.name, "shots": len(ids),
                    "arrays": target.relative_to(grader).as_posix(), "arrays_sha256": sha256(target)})
    for split, manifest in manifests.items():
        root = verifier if split == "test" else development / split
        manifest["shots"] = sum(row["shots"] for row in manifest["conditions"])
        write_json(root / "manifest.json", manifest)
    labels_manifest["input_manifest_sha256"] = sha256(verifier / "manifest.json")
    write_json(grader / "manifest.json", labels_manifest)
    # No trained weights reach the agent: a weightless submission template plus the reference model's
    # results (validation predictions and metrics, test aggregates, training cost) under reference/.
    write_template(development / "reference" / "package-template")
    shutil.copytree(reference_results, development / "reference" / "results")
    shutil.copyfile(TASK / "score_config.json", grader / "score_config.json")
    write_json(development / "release.json", {"release": RELEASE, "source": "https://zenodo.org/records/6804040",
        "license": "CC-BY-4.0", "splits": {key: value["shots"] for key, value in manifests.items()},
        "calibration_exception": "Published even-fit DEM uses train and validation syndromes; no test syndromes or logical labels.",
        "source_is_public": True, "historical_test_is_not_secret": True,
        "trained_weights": "none; train from scratch (reference/ holds the reference model's results and a submission template)"})
    files = {}
    for role, filename in ((development, "development.tar.gz"), (verifier, "verifier-inputs.tar.gz"), (grader, "grader-labels.tar.gz")):
        inventory = {path.relative_to(role).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                     for path in sorted(role.rglob("*")) if path.is_file()}
        write_json(role / "inventory.json", {"schema": "surface-code-decoding-inventory/v1", "files": inventory})
        archive(role, output / filename)
        files[filename] = {"bytes": (output / filename).stat().st_size, "sha256": sha256(output / filename),
                           "expanded_bytes": sum(path.stat().st_size for path in role.rglob("*") if path.is_file()),
                           "role": role.name}
    result = {"schema": "surface-code-decoding-release/v1", "release": RELEASE, "files": files}
    write_json(output / "release-manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True,
                        help="directory written by make_reference_results.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.reference_results, args.output), indent=2))


if __name__ == "__main__":
    main()
