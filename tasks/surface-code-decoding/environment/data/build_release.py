#!/usr/bin/env python3
"""Deterministically package the audited source into separate role archives.

V4 Security Hardening:
- Replaces unkeyed Blake2b sample IDs with keyed HMAC-SHA256 tokens derived from an
  explicit operator secret (`--hmac-key-file` or `SSI_SURFACE_CODE_HMAC_KEY`, >= 16 bytes):
      hmac.new(secret_key, f"{folder}:{index}".encode("utf-8"), hashlib.sha256).hexdigest()[:32]
  (projected to unique uint64 sample_ids for the binary NPZ contract).
- Deterministically permutes test-split shot rows per condition using a keyed PRNG derived
  from `secret_key` so row order in `verifier-inputs.tar.gz` never leaks temporal shot sequence.
- Writes the private `(sample_id -> condition_folder, source_shot_index)` mapping ONLY to
  `grader/private_sample_id_map.json` inside `grader-labels.tar.gz` (`TASK_PRIVATE_DIR`),
  NEVER to `development.tar.gz` or `verifier-inputs.tar.gz`.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import hmac
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
from typing import Any
import numpy as np

TASK = Path(__file__).resolve().parents[2] / "environment" / "baseline" / "alphaqubit"  # the baseline project
sys.path.insert(0, str(TASK))
from qecbench.package import write_template
from qecbench.data import SCHEMA, sha256, write_json

RELEASE = "sycamore-hard-v1"
HMAC_KEY_ENV = "SSI_SURFACE_CODE_HMAC_KEY"
MIN_HMAC_KEY_BYTES = 16


def resolve_secret_key(
    secret_key: bytes | str | None = None,
    hmac_key_file: Path | None = None,
) -> bytes:
    """Resolve and validate the operator HMAC key required in release mode."""
    raw: bytes
    if secret_key is not None:
        raw = secret_key.encode("utf-8") if isinstance(secret_key, str) else bytes(secret_key)
    elif hmac_key_file is not None:
        raw = Path(hmac_key_file).read_bytes().strip()
    else:
        raw = os.environ.get(HMAC_KEY_ENV, "").strip().encode("utf-8")
    if len(raw) < MIN_HMAC_KEY_BYTES:
        raise ValueError(
            f"Release build requires an explicit secret key of at least {MIN_HMAC_KEY_BYTES} bytes "
            f"via --hmac-key-file or {HMAC_KEY_ENV}"
        )
    return raw


def hmac_sample_token(secret_key: bytes, folder: str, index: int) -> str:
    """Compute keyed 128-bit hex token for `(folder, original_shot_index)`."""
    if not secret_key:
        raise ValueError("secret_key must be non-empty")
    return hmac.new(
        secret_key,
        f"{folder}:{int(index)}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:32]


def sample_ids_with_tokens(
    condition: str,
    indices: np.ndarray,
    secret_key: bytes,
) -> tuple[np.ndarray, list[str]]:
    """Derive collision-checked uint64 sample IDs and full 32-hex HMAC tokens."""
    key = resolve_secret_key(secret_key=secret_key)
    tokens = [hmac_sample_token(key, condition, int(index)) for index in indices]
    ids = np.array([int(token[:16], 16) for token in tokens], dtype=np.uint64)
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError(f"64-bit HMAC sample_id collision detected in condition {condition}")
    return ids, tokens


def sample_ids(condition: str, indices: np.ndarray, secret_key: bytes) -> np.ndarray:
    """Derive keyed uint64 sample IDs without exposing original shot indices."""
    ids, _ = sample_ids_with_tokens(condition, indices, secret_key)
    return ids


def development_subkey(secret_key: bytes) -> bytes:
    """Domain-separate public train/validation sample IDs from the sealed test HMAC key."""
    key = resolve_secret_key(secret_key=secret_key)
    return hmac.new(key, f"development-split:{RELEASE}".encode("utf-8"), hashlib.sha256).digest()


def test_shuffle_permutation(secret_key: bytes, condition: str, num_shots: int) -> np.ndarray:
    """Deterministically permute test shots per condition using a keyed PRNG."""
    key = resolve_secret_key(secret_key=secret_key)
    seed_bytes = hmac.new(
        key,
        f"test-shuffle:{RELEASE}:{condition}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    rng = np.random.default_rng(np.frombuffer(seed_bytes, dtype=np.uint32))
    return rng.permutation(int(num_shots))


def prepare_test_condition_split(
    condition: str,
    detectors: np.ndarray,
    labels_1d: np.ndarray,
    raw_indices: np.ndarray,
    secret_key: bytes,
) -> dict[str, Any]:
    """Key and shuffle a single condition's test split while recording its sealed mapping."""
    key = resolve_secret_key(secret_key=secret_key)
    raw_indices_arr = np.asarray(raw_indices, dtype=np.int64)
    labels_arr = np.asarray(labels_1d, dtype=np.uint8)
    if len(detectors) != len(raw_indices_arr) or len(labels_arr) != len(raw_indices_arr):
        raise ValueError(f"Mismatched test shot arrays for {condition}")
    perm = test_shuffle_permutation(key, condition, len(raw_indices_arr))
    shuffled_indices = raw_indices_arr[perm]
    shuffled_detectors = detectors[perm]
    shuffled_labels = labels_arr[perm]
    ids, tokens = sample_ids_with_tokens(condition, shuffled_indices, key)
    return {
        "sample_ids": ids,
        "sample_ids_hmac_hex32": tokens,
        "detectors": shuffled_detectors,
        "labels": shuffled_labels,
        "source_shot_indices": [int(idx) for idx in shuffled_indices],
        "permutation": perm,
    }


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


def build(
    source: Path,
    reference_results: Path,
    output: Path,
    *,
    secret_key: bytes | str | None = None,
    hmac_key_file: Path | None = None,
) -> dict[str, Any]:
    from baselines.data import discover_conditions, hardware_split_indices, load_hardware_packed

    key = resolve_secret_key(secret_key=secret_key, hmac_key_file=hmac_key_file)
    dev_key = development_subkey(key)
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
    private_id_map: dict[str, Any] = {
        "schema": "surface-code-decoding-private-id-map/v1",
        "release": RELEASE,
        "kdf": "hmac-sha256(secret_key, f'{folder}:{index}')[:32]",
        "uint64_projection": "int(hex32[:16], 16)",
        "test_row_order": "keyed-prng-shuffled",
        "conditions": {},
    }
    for condition in conditions:
        for split in manifests:
            detectors, labels = load_hardware_packed(condition, split)
            split_indices = hardware_split_indices(split)
            if split == "test":
                prepared = prepare_test_condition_split(
                    condition.name,
                    detectors,
                    labels[:, 0],
                    split_indices,
                    key,
                )
                ids = prepared["sample_ids"]
                detectors_out = prepared["detectors"]
                labels_1d = prepared["labels"]
                private_id_map["conditions"][condition.name] = {
                    "sample_ids_uint64": [int(x) for x in ids],
                    "sample_ids_hmac_hex32": prepared["sample_ids_hmac_hex32"],
                    "source_shot_indices": prepared["source_shot_indices"],
                }
            else:
                ids = sample_ids(condition.name, split_indices, dev_key)
                detectors_out = detectors
                labels_1d = labels[:, 0]
            root = verifier if split == "test" else development / split
            folder = root / "conditions" / condition.name
            folder.mkdir(parents=True)
            payload = {"sample_ids": ids, "detectors": detectors_out}
            if split != "test":
                payload["labels"] = labels_1d
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
                np.savez(target, sample_ids=ids, labels=labels_1d)
                labels_manifest["conditions"].append({"name": condition.name, "shots": len(ids),
                    "arrays": target.relative_to(grader).as_posix(), "arrays_sha256": sha256(target)})
    for split, manifest in manifests.items():
        root = verifier if split == "test" else development / split
        manifest["shots"] = sum(row["shots"] for row in manifest["conditions"])
        write_json(root / "manifest.json", manifest)
    labels_manifest["input_manifest_sha256"] = sha256(verifier / "manifest.json")
    write_json(grader / "manifest.json", labels_manifest)
    # Private `(sample_id -> folder, index)` lookup exists ONLY in the sealed grader archive (`TASK_PRIVATE_DIR`).
    write_json(grader / "private_sample_id_map.json", private_id_map)
    # No trained weights reach the agent: a weightless submission template plus the reference model's
    # results (validation predictions and metrics, test aggregates, training cost) under reference/.
    write_template(development / "reference" / "package-template")
    shutil.copytree(reference_results, development / "reference" / "results")
    shutil.copyfile(TASK / "score_config.json", grader / "score_config.json")
    write_json(development / "release.json", {"release": RELEASE, "source": "https://zenodo.org/records/6804040",
        "license": "CC-BY-4.0", "splits": {key_name: value["shots"] for key_name, value in manifests.items()},
        "calibration_exception": "Published even-fit DEM uses train and validation syndromes; no test syndromes or logical labels.",
        "source_is_public": True, "historical_test_is_not_secret": True,
        "test_sample_ids": "keyed HMAC-SHA256 with keyed shot-order permutation; private mapping in grader role only",
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
    parser.add_argument("--hmac-key-file", type=Path, default=None,
                        help=f"Path to secret HMAC key (>=16 bytes); falls back to ${HMAC_KEY_ENV}")
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.reference_results, args.output,
                           hmac_key_file=args.hmac_key_file), indent=2))


if __name__ == "__main__":
    main()
