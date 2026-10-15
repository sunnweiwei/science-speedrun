"""Strict schemas and portable-package validation for CrystalGen."""

import csv
import hashlib
import json
from pathlib import Path
import re
import struct


REQUEST_SCHEMA = "crystalgen-request/v0"
MODEL_SCHEMA = "crystalgen-model/v0"
MP20_TRAIN_SHA256 = "133bab58dba02d316f9f91174839f17802dce648be4e8659d1a42b32b97fa01d"
MAX_PARAMETERS = 70_000_000
MAX_SUBMISSION_BYTES = 2 * 1024 * 1024 * 1024
MAX_SUBMISSION_FILES = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safetensors_parameter_count(path: Path) -> int:
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError("truncated safetensors checkpoint")
        header_length = struct.unpack("<Q", raw_length)[0]
        if not 2 <= header_length <= 100 * 1024 * 1024:
            raise ValueError("invalid safetensors header length")
        header = json.loads(stream.read(header_length))
    if not isinstance(header, dict):
        raise ValueError("invalid safetensors header")
    total = 0
    tensors = 0
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(spec, dict) or not isinstance(spec.get("shape"), list):
            raise ValueError("invalid safetensors tensor declaration")
        count = 1
        for dimension in spec["shape"]:
            if type(dimension) is not int or dimension < 0:
                raise ValueError("invalid safetensors tensor shape")
            count *= dimension
        total += count
        tensors += 1
    if tensors == 0:
        raise ValueError("safetensors checkpoint has no tensors")
    return total


def json_smoke_parameter_count(path: Path, model_config: dict) -> int:
    if model_config != {"architecture": "aggregate-statistics-smoke-generator"}:
        raise ValueError("JSON checkpoints are reserved for the interface smoke baseline")
    model = json.loads(path.read_text())
    if model.get("schema") != "crystalgen-smoke-model/v0":
        raise ValueError("invalid smoke checkpoint")
    elements = model.get("element_counts")
    atoms = model.get("atom_counts")
    if not isinstance(elements, dict) or not isinstance(atoms, dict):
        raise ValueError("invalid smoke checkpoint statistics")
    return len(elements) + len(atoms) + 2


def contained(root: Path, relative: str, *, require_file: bool = True) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError("expected a relative path")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("path escapes its declared root")
    if require_file and (not resolved.is_file() or resolved.is_symlink()):
        raise ValueError("path is missing, not a regular file, or a symlink")
    return resolved


def load_requests(input_root: Path) -> dict[str, dict]:
    path = input_root / "requests.jsonl"
    requests = {}
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            request = json.loads(line)
            required = {"schema_version", "request_id", "mode", "condition", "n_samples", "seed"}
            if set(request) != required or request["schema_version"] != REQUEST_SCHEMA:
                raise ValueError(f"invalid request fields/schema at line {line_number}")
            request_id = request["request_id"]
            if (not isinstance(request_id, str)
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", request_id)
                    or request_id in requests):
                raise ValueError("unsafe or duplicate request ID")
            if type(request["n_samples"]) is not int or not 1 <= request["n_samples"] <= 10_000:
                raise ValueError("n_samples must be an integer from 1 through 10000")
            if type(request["seed"]) is not int or request["seed"] < 0:
                raise ValueError("seed must be a nonnegative integer")
            if request["mode"] != "unconditional" or request["condition"] is not None:
                raise ValueError("only unconditional requests with condition=null are supported")
            requests[request_id] = request
    if not requests:
        raise ValueError("empty request suite")
    return requests


def expected_samples(requests: dict[str, dict]) -> set[tuple[str, int]]:
    return {
        (request_id, sample_index)
        for request_id, request in requests.items()
        for sample_index in range(request["n_samples"])
    }


def load_output(output_root: Path, requests: dict[str, dict]) -> tuple[dict[tuple[str, int], Path], list[str]]:
    expected = expected_samples(requests)
    manifest = output_root / "manifest.csv"
    if not manifest.is_file() or manifest.is_symlink():
        return {}, ["manifest.csv is missing"]
    records = {}
    errors = []
    try:
        with manifest.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["request_id", "sample_index", "structure_path"]:
                return {}, ["manifest.csv has the wrong columns"]
            used_paths = {}
            seen_keys = set()
            invalid_keys = set()
            for line_number, row in enumerate(reader, 2):
                if set(row) != {"request_id", "sample_index", "structure_path"}:
                    errors.append(f"line {line_number}: malformed manifest row")
                    continue
                try:
                    sample_index = int(row["sample_index"])
                except (TypeError, ValueError):
                    errors.append(f"line {line_number}: invalid sample_index")
                    continue
                key = (row["request_id"], sample_index)
                if key not in expected:
                    errors.append(f"line {line_number}: unknown sample")
                    continue
                if key in seen_keys:
                    errors.append(f"line {line_number}: duplicate sample")
                    invalid_keys.add(key)
                    records.pop(key, None)
                    continue
                seen_keys.add(key)
                try:
                    path = contained(output_root, row["structure_path"])
                    if path.suffix.lower() != ".cif" or path.stat().st_size == 0:
                        raise ValueError("structure is not a nonempty CIF")
                    relative = path.relative_to(output_root.resolve()).as_posix()
                    if relative in used_paths:
                        previous = used_paths[relative]
                        invalid_keys.update({previous, key})
                        records.pop(previous, None)
                        raise ValueError("structure path is reused; both samples are invalid")
                    used_paths[relative] = key
                    if key in invalid_keys:
                        continue
                    records[key] = path
                except (OSError, ValueError) as error:
                    errors.append(f"line {line_number}: {error}")
    except (OSError, csv.Error, UnicodeDecodeError) as error:
        return {}, [f"manifest.csv could not be read: {error}"]
    missing = expected - set(records)
    if missing:
        errors.append(f"{len(missing)} requested structures are missing or invalid")
    return records, errors


def validate_submission(root: Path) -> dict:
    root = root.resolve()
    if not (root / "generate.sh").is_file():
        raise FileNotFoundError("submission/generate.sh is required")
    file_count = 0
    byte_count = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("submission symlinks are not allowed")
        if path.is_file():
            file_count += 1
            byte_count += path.stat().st_size
    if file_count > MAX_SUBMISSION_FILES or byte_count > MAX_SUBMISSION_BYTES:
        raise ValueError("submission exceeds its file-count or byte limit")
    manifest = json.loads((root / "model_manifest.json").read_text())
    required = {
        "schema", "checkpoint", "parameter_count", "one_parameter_set",
        "trained_from_scratch", "training_data", "model_config",
    }
    if set(manifest) != required or manifest["schema"] != MODEL_SCHEMA:
        raise ValueError("invalid model manifest fields/schema")
    checkpoint_spec = manifest["checkpoint"]
    if (not isinstance(checkpoint_spec, dict)
            or set(checkpoint_spec) != {"path", "sha256", "format"}
            or checkpoint_spec["format"] not in {"json", "safetensors"}):
        raise ValueError("invalid checkpoint declaration")
    checkpoint = contained(root, checkpoint_spec["path"])
    if sha256(checkpoint) != checkpoint_spec["sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    count = manifest["parameter_count"]
    if type(count) is not int or not 1 <= count <= MAX_PARAMETERS:
        raise ValueError("parameter_count is outside the allowed range")
    if manifest["one_parameter_set"] is not True or manifest["trained_from_scratch"] is not True:
        raise ValueError("one from-scratch parameter set is required")
    expected_data = {"name": "MP-20", "split": "train", "source_sha256": MP20_TRAIN_SHA256}
    if manifest["training_data"] != expected_data:
        raise ValueError("training-data declaration does not match pinned MP-20 train")
    if not isinstance(manifest["model_config"], dict) or not manifest["model_config"]:
        raise ValueError("model_config must be a nonempty object")
    if checkpoint_spec["format"] == "safetensors":
        observed_count = safetensors_parameter_count(checkpoint)
    else:
        observed_count = json_smoke_parameter_count(checkpoint, manifest["model_config"])
    if count != observed_count:
        raise ValueError("parameter_count does not match the checkpoint tensor shapes")
    checkpoint_like = {
        path.resolve() for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {
            ".bin", ".ckpt", ".npy", ".npz", ".pt", ".pth", ".safetensors"
        }
    }
    if checkpoint_like != ({checkpoint.resolve()} if checkpoint.suffix.lower() in {
        ".bin", ".ckpt", ".npy", ".npz", ".pt", ".pth", ".safetensors"
    } else set()):
        raise ValueError("submission contains an undeclared checkpoint-like file")
    return manifest
