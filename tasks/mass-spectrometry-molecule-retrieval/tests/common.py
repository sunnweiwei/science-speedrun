"""Trusted validation helpers shared by replay and grading."""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


QUERY_FIELDS = {
    "query_id", "mzs", "intensities", "precursor_mz", "adduct",
    "instrument_type", "collision_energy", "candidate_pool_id",
}


def compact_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, value):
    # The other verifier service polls for these files. Existence must mean
    # complete JSON, including on shared volumes.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(compact_json(value) + "\n")
            stream.flush()
            # A host-side runner may read bind-mounted verifier logs as a
            # different UID. The containing directory controls write access.
            os.fchmod(stream.fileno(), 0o644)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("path must be a nonempty string")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("path escapes its declared root")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def jsonl(path: Path):
    values = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            values.append(value)
    return values


def candidate_id(smiles: str) -> str:
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()


def numeric(value, *, positive=False, nullable=False):
    if nullable and value is None:
        return
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("expected a finite JSON number")
    if positive and value <= 0:
        raise ValueError("expected a positive number")


def validate_input(root: Path):
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "massspecgym-retrieval-input/v1":
        raise ValueError("invalid input manifest schema")
    for name in ("queries.jsonl", "candidate_pools.jsonl"):
        if manifest.get("files", {}).get(name) != sha256(root / name):
            raise ValueError(f"input hash mismatch: {name}")

    pools = {}
    for row in jsonl(root / "candidate_pools.jsonl"):
        if set(row) != {"candidate_pool_id", "candidates"}:
            raise ValueError("invalid candidate pool fields")
        pool_id = row["candidate_pool_id"]
        if not isinstance(pool_id, str) or not pool_id or pool_id in pools:
            raise ValueError("missing or duplicate candidate pool ID")
        candidates = row["candidates"]
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 256:
            raise ValueError("invalid candidate pool length")
        by_id = {}
        for candidate in candidates:
            if set(candidate) != {"candidate_id", "smiles"}:
                raise ValueError("invalid candidate fields")
            identifier, smiles = candidate["candidate_id"], candidate["smiles"]
            if not isinstance(smiles, str) or identifier != candidate_id(smiles):
                raise ValueError("candidate ID/SMILES mismatch")
            if identifier in by_id:
                raise ValueError("duplicate candidate ID")
            by_id[identifier] = smiles
        pools[pool_id] = by_id

    queries = {}
    for row in jsonl(root / "queries.jsonl"):
        if set(row) != QUERY_FIELDS:
            raise ValueError("invalid query fields")
        query_id = row["query_id"]
        if not isinstance(query_id, str) or not query_id or query_id in queries:
            raise ValueError("missing or duplicate query ID")
        if row["candidate_pool_id"] not in pools:
            raise ValueError("query references unknown candidate pool")
        mzs, intensities = row["mzs"], row["intensities"]
        if not isinstance(mzs, list) or not mzs or len(mzs) != len(intensities):
            raise ValueError("invalid peak arrays")
        for mz in mzs:
            numeric(mz, positive=True)
        for intensity in intensities:
            numeric(intensity)
            if intensity < 0:
                raise ValueError("negative intensity")
        if max(intensities) <= 0 or mzs != sorted(mzs):
            raise ValueError("invalid peak ordering or normalization")
        numeric(row["precursor_mz"], positive=True)
        numeric(row["collision_energy"], nullable=True)
        if not isinstance(row["adduct"], str) or not isinstance(row["instrument_type"], str):
            raise ValueError("invalid metadata strings")
        queries[query_id] = row

    counts = manifest.get("counts", {})
    if counts.get("queries") != len(queries) or counts.get("candidate_pools") != len(pools):
        raise ValueError("input manifest count mismatch")
    return manifest, queries, pools
