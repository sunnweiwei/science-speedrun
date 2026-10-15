"""Shared verifier helpers for protein-sequence-modeling."""
import hashlib
import json
from pathlib import Path

from metrics import TASKS

SCHEMA = "peer-protein-input/v1"
PRIVATE_SCHEMA = "peer-protein-private-labels/v1"
SNAPSHOT_SCHEMA = "peer-protein-prediction-snapshot/v1"
PUBLIC_FIELDS = {"example_id", "task", "features"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def contained(root, name):
    resolved = (root.resolve() / name).resolve()
    if (Path(name).is_absolute() or not resolved.is_relative_to(root.resolve())
            or not resolved.is_file() or resolved.is_symlink()):
        raise ValueError("unsafe path")
    return resolved


def jsonl(path, max_bytes=512 * 1024 * 1024, max_line=2 * 1024 * 1024):
    if not path.is_file() or path.is_symlink() or path.stat().st_size > max_bytes:
        raise ValueError(f"unsafe JSONL: {path.name}")
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            if len(line.encode()) > max_line:
                raise ValueError(f"oversized line: {path.name}:{number}")
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON: {path.name}:{number}") from error


def validate_input(root):
    manifest = json.loads((root / "manifest.json").read_text())
    names = {f"{task}.jsonl" for task in TASKS}
    if (set(manifest) != {"schema", "view", "files", "counts"} or manifest["schema"] != SCHEMA
            or manifest["view"] != "verifier" or set(manifest["files"]) != names
            or set(manifest["counts"]) != set(TASKS)):
        raise ValueError("invalid input manifest")
    all_ids = set()
    expected = {}
    for task in TASKS:
        path = contained(root, f"{task}.jsonl")
        if sha256(path) != manifest["files"][path.name]:
            raise ValueError("input hash mismatch")
        ids = set()
        count = 0
        for row in jsonl(path, 2 * 1024 ** 3):
            count += 1
            if (set(row) != PUBLIC_FIELDS or row.get("task") != task
                    or not isinstance(row.get("example_id"), str)
                    or not isinstance(row.get("features"), dict)
                    or row["example_id"] in all_ids):
                raise ValueError(f"invalid row: {task}")
            ids.add(row["example_id"])
            all_ids.add(row["example_id"])
        if count != manifest["counts"][task]:
            raise ValueError("input count mismatch")
        expected[task] = ids
    return manifest, expected
