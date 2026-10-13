"""Build and migrate the current release layout without rematerializing targets."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import ContractError
from .protocol import INFERENCE_SCHEMA, RELEASE_SCHEMA, TARGET_STORAGE_SCHEMA
from .splicing import junction_coordinates
from .zarr_store import close_store, open_group


LEGACY_JUNCTION_QUERY_COUNT = 8_192
SUPERVISED_QUERIES_PER_TRACK = 1
JUNCTION_QUERY_BUDGET = LEGACY_JUNCTION_QUERY_COUNT + 734 * SUPERVISED_QUERIES_PER_TRACK
SCORING_PROTOCOL = "legacy8192-plus-one-positive-per-track/v1"
PUBLIC_COLUMNS = (
    "example_id", "split", "chromosome", "input_start", "input_end",
    "input_pad_left", "input_pad_right", "target_start", "target_end",
)
LABEL_COLUMNS = ("example_id", "target_shard", "target_group", "splice_site_positions")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _store_hash(path: Path) -> str:
    if not path.exists():
        raise ContractError(f"missing target store: {path}")
    if path.is_file():
        return _sha256(path)
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(str(item.relative_to(path)).encode())
            digest.update(bytes.fromhex(_sha256(item)))
    return digest.hexdigest()


def _referenced_target_hashes(root: Path, manifests: Path) -> dict[str, str]:
    references = set()
    for path in sorted(manifests.glob("*_materialized.parquet")):
        rows = pd.read_parquet(path, columns=["target_shard"])
        references.update(rows.target_shard.astype(str))
    references = sorted(references)
    def hash_reference(reference: str) -> tuple[str, str]:
        canonical = reference.removeprefix("v2/")
        current = root / canonical
        legacy = root / "v2" / canonical
        if current.exists() and legacy.exists() and _store_hash(current) != _store_hash(legacy):
            raise ContractError(f"target exists with different contents at both paths: {canonical}")
        path = current if current.exists() else legacy
        return canonical, _store_hash(path)

    with ThreadPoolExecutor(max_workers=min(4, max(1, len(references)))) as pool:
        result: dict[str, str] = {}
        for name, digest in pool.map(hash_reference, references):
            if name in result and result[name] != digest:
                raise ContractError(f"conflicting target references: {name}")
            result[name] = digest
        return result


def _promote_targets(source: Path, destination: Path) -> None:
    """Move target files, retaining duplicate legacy copies until final verification."""
    if not source.exists():
        return
    for path in sorted(source.rglob("*")):
        target = destination / path.relative_to(source)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not target.is_file() or _sha256(path) != _sha256(target):
                    raise ContractError(f"layout migration collision: {target}")
            else:
                os.replace(path, target)


def _finish_legacy_tree(root: Path) -> None:
    legacy_targets = root / "v2/targets"
    if legacy_targets.exists():
        for path in sorted(legacy_targets.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        legacy_targets.rmdir()
    for legacy in (root / "v2", root / "v3"):
        if legacy.exists():
            for path in sorted(legacy.rglob("*"), reverse=True):
                if path.is_dir():
                    path.rmdir()
            legacy.rmdir()


def _store_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _promote_directory(source: Path, destination: Path) -> None:
    """Move a legacy directory into the current layout, merging identical files."""
    if not source.exists():
        return
    if not destination.exists():
        os.replace(source, destination)
        return
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif target.exists():
            if not target.is_file() or _sha256(path) != _sha256(target):
                raise ContractError(f"layout migration collision: {target}")
            path.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, target)
    for path in sorted(source.rglob("*"), reverse=True):
        if path.is_dir():
            path.rmdir()
    source.rmdir()


def _rewrite_target_shards(path: Path) -> None:
    if not path.is_file():
        return
    rows = pd.read_parquet(path)
    if "target_shard" not in rows:
        return
    rewritten = rows.target_shard.astype(str).str.replace(
        r"^v2/targets/", "targets/", regex=True
    )
    if not rewritten.equals(rows.target_shard.astype(str)):
        rows["target_shard"] = rewritten
        _atomic_parquet(rows, path)


def _rewrite_smoke_receipts(root: Path) -> None:
    for path in sorted((root / "smoke").glob("SMOKE_*_VALIDATED.json")):
        receipt = json.loads(path.read_text())
        output = receipt.get("output")
        if isinstance(output, str):
            receipt["output"] = output.replace("/v2/smoke/", "/smoke/")
        if "receipt_sha256" in receipt:
            receipt.pop("receipt_sha256")
            receipt["receipt_sha256"] = hashlib.sha256(
                json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        _atomic_json(path, receipt)


def _rewrite_zipstore_receipt(root: Path) -> None:
    path = root / "zipstore_conversion.json"
    if not path.is_file():
        return
    receipt = json.loads(path.read_text())
    for artifact in receipt.get("artifacts", []):
        for key in ("path", "source"):
            value = artifact.get(key)
            if isinstance(value, str):
                artifact[key] = value.removeprefix("v2/")
    _atomic_json(path, receipt)


def migrate_layout(data_root: str | Path) -> dict[str, object]:
    """Promote a v2/v3 tree to the single current layout without decoding data."""
    root = Path(data_root).resolve()
    legacy_splits = root / "v2/splits"
    observed = _referenced_target_hashes(
        root, legacy_splits if legacy_splits.exists() else root / "splits"
    )
    journal = root / ".layout_migration_targets.json"
    if journal.exists():
        saved = json.loads(journal.read_text())
        if saved.get("schema") != "alphagenome-target-layout-migration/v1" or saved.get("target_hashes") != observed:
            raise ContractError("target migration journal differs from observed targets")
        before = saved["target_hashes"]
    else:
        before = observed
        _atomic_json(journal, {
            "schema": "alphagenome-target-layout-migration/v1",
            "target_hashes": before,
        })
    _promote_targets(root / "v2/targets", root / "targets")
    for name in ("metadata", "splits", "smoke", "source_index_parts"):
        _promote_directory(root / "v2" / name, root / name)
    legacy_validation = root / "v2/validation_report.json"
    if legacy_validation.exists():
        if (root / "validation_report.json").exists():
            raise ContractError("both legacy and current validation reports exist")
        os.replace(legacy_validation, root / "validation_report.json")
    for path in sorted((root / "v2").glob("*")):
        if not path.is_file():
            continue
        destination = root / path.name
        if destination.exists():
            if not destination.is_file() or _sha256(path) != _sha256(destination):
                raise ContractError(f"layout migration collision: {destination}")
            path.unlink()
        else:
            os.replace(path, destination)
    for name in ("public", "labels", "scoring"):
        _promote_directory(root / "v3" / name, root / name)
    legacy_release = root / "v3/release.json"
    if legacy_release.exists():
        if (root / "release.json").exists():
            raise ContractError("both legacy and current release manifests exist")
        os.replace(legacy_release, root / "release.json")
    for path in sorted((root / "splits").glob("*_materialized.parquet")):
        _rewrite_target_shards(path)
    for path in sorted((root / "labels").glob("*.parquet")):
        _rewrite_target_shards(path)
    _rewrite_smoke_receipts(root)
    _rewrite_zipstore_receipt(root)
    after = _referenced_target_hashes(root, root / "splits")
    if before != after:
        raise ContractError("target file hashes changed during layout migration")
    _finish_legacy_tree(root)
    journal.unlink()
    return {
        "status": "PASS",
        "schema": RELEASE_SCHEMA,
        "target_files": len(after),
        "target_bytes": sum(_store_size(root / name) for name in after),
        "target_hashes_sha256": hashlib.sha256(
            "\n".join(f"{name} {after[name]}" for name in sorted(after)).encode()
        ).hexdigest(),
    }


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def junction_queries(example_id: str, count: int = LEGACY_JUNCTION_QUERY_COUNT) -> np.ndarray:
    """Frozen compact-benchmark query indices; coordinates stay trusted."""

    seed = int.from_bytes(hashlib.sha256(example_id.encode()).digest()[:8], "little")
    index = np.arange(count, dtype=np.uint64)
    return np.stack(
        ((index * 131 + seed) % 512, (index * 137 + (seed >> 9)) % 512,
         index % 734), axis=-1,
    ).astype(np.int64)


def _junction_targets(
    group, queries: np.ndarray
) -> tuple[np.ndarray, str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    row = np.asarray(group["splice_junctions/row"], np.int64)
    column = np.asarray(group["splice_junctions/column"], np.int64)
    track = np.asarray(group["splice_junctions/track"], np.int64)
    source_value = np.asarray(group["splice_junctions/value"], np.float32)
    value = source_value
    keys = (row * 512 + column) * 734 + track
    order = np.argsort(keys)
    keys, value = keys[order], value[order]
    wanted = (queries[:, 0] * 512 + queries[:, 1]) * 734 + queries[:, 2]
    target = np.zeros(len(wanted), np.float32)
    if len(keys):
        at = np.searchsorted(keys, wanted)
        clipped = np.minimum(at, len(keys) - 1)
        hit = (at < len(keys)) & (keys[clipped] == wanted)
        target[hit] = value[at[hit]]
    digest = hashlib.sha256()
    for item in (np.asarray(group["splice_junctions/site_positions"], np.int32),
                 row, column, track, source_value):
        digest.update(str(item.dtype).encode())
        digest.update(np.asarray(item.shape, np.int64).tobytes())
        digest.update(np.ascontiguousarray(item).tobytes())
    return target, digest.hexdigest(), (row, column, track, source_value)


def _junction_track_table(data_root: Path) -> pd.DataFrame:
    tracks = pd.read_parquet(data_root / "metadata/tracks.parquet")
    junctions = tracks.loc[tracks.family == "SPLICE_JUNCTIONS"].sort_values("tensor_index")
    if len(junctions) != 734 or "junction_channel_half" not in junctions:
        raise ContractError("junction track metadata does not define 734 channel halves")
    halves = junctions.junction_channel_half.to_numpy(np.int64)
    if not np.isin(halves, (0, 1)).all():
        raise ContractError("junction_channel_half must contain only 0/1")
    return junctions.loc[:, ["tensor_index", "junction_channel_half"]].reset_index(drop=True)


def _scoring_part(data_root: Path, row, halves: np.ndarray) -> tuple[pd.DataFrame, str]:
    root, store = open_group(data_root / str(row.target_shard), mode="r")
    try:
        group = root[str(row.target_group)]
        positions = np.asarray(group["splice_junctions/site_positions"], np.int32)
        queries = junction_queries(str(row.example_id))
        query_halves = halves[queries[:, 2]]
        donor, acceptor, valid = junction_coordinates(
            positions, queries[:, 0], queries[:, 1], query_halves
        )
        target, source_hash, sparse = _junction_targets(group, queries)

        # The frozen 8,192 legacy queries are retained verbatim.  The full-data
        # audit showed that they undersample validation supervision, so append
        # at most one real nonzero event per available track.  These coordinates
        # remain trusted and missing predicted endpoints still score as zero.
        sparse_row, sparse_column, sparse_track, sparse_value = sparse
        sparse_donor, sparse_acceptor, sparse_valid = junction_coordinates(
            positions, sparse_row, sparse_column, halves[sparse_track]
        )
        candidates = np.flatnonzero(sparse_valid & (sparse_value != 0))
        if len(candidates):
            pair = sparse_row[candidates] * 512 + sparse_column[candidates]
            order = candidates[np.lexsort((pair, sparse_track[candidates]))]
            first = np.r_[True, sparse_track[order][1:] != sparse_track[order][:-1]]
            selected = order[first]
            legacy = {
                (int(d), int(a), int(t))
                for d, a, t, keep in zip(donor, acceptor, queries[:, 2], valid)
                if keep
            }
            selected = np.asarray([
                index for index in selected
                if (int(sparse_donor[index]), int(sparse_acceptor[index]),
                    int(sparse_track[index])) not in legacy
            ], dtype=np.int64)
            donor = np.concatenate((donor, sparse_donor[selected]))
            acceptor = np.concatenate((acceptor, sparse_acceptor[selected]))
            valid = np.concatenate((valid, np.ones(len(selected), dtype=bool)))
            queries = np.concatenate((queries, np.stack((
                sparse_row[selected], sparse_column[selected], sparse_track[selected]
            ), axis=-1)))
            target = np.concatenate((target, sparse_value[selected]))
    finally:
        close_store(store)
    return pd.DataFrame({
        "example_id": np.repeat(str(row.example_id), len(queries)),
        "donor_position": donor.astype(np.int32),
        "acceptor_position": acceptor.astype(np.int32),
        "track": queries[:, 2].astype(np.uint16),
        "target_value": target,
        "valid": valid,
    }), source_hash


def export_release(
    data_root: str | Path,
    splits: list[str] | tuple[str, ...],
    resume: bool = True,
) -> dict[str, object]:
    """Export current manifests/scoring using only existing target storage."""

    root = Path(data_root).resolve()
    existing_release_path = root / "release.json"
    existing_release = (
        json.loads(existing_release_path.read_text())
        if existing_release_path.is_file() else {}
    )
    if existing_release and existing_release.get("schema") != RELEASE_SCHEMA:
        raise ContractError("existing release.json has an incompatible schema")
    junction_tracks = _junction_track_table(root)
    halves = junction_tracks.junction_channel_half.to_numpy(np.int64)
    public_tracks = root / "public/junction_tracks.parquet"
    if not (resume and public_tracks.is_file()):
        _atomic_parquet(junction_tracks, public_tracks)
    split_reports: dict[str, object] = dict(existing_release.get("splits", {}))
    source_hashes: dict[str, str] = {
        key: value for key, value in existing_release.get("source_hashes", {}).items()
        if not key.startswith(("v2/", "v3/"))
    }
    tracks_source = "metadata/tracks.parquet"
    tracks_hash = _sha256(root / tracks_source)
    if (resume and source_hashes.get(tracks_source)
            and source_hashes[tracks_source] != tracks_hash):
        raise ContractError("track metadata changed since the resumable release export")
    source_hashes[tracks_source] = tracks_hash
    source_hashes["public/junction_tracks.parquet"] = _sha256(public_tracks)
    for split in splits:
        source = root / "splits" / f"{split}_materialized.parquet"
        if not source.is_file():
            raise ContractError(f"missing materialized manifest for {split}")
        source_key = str(source.relative_to(root))
        source_hash = _sha256(source)
        if (resume and source_hashes.get(source_key)
                and source_hashes[source_key] != source_hash):
            raise ContractError(f"{split} manifest changed since the resumable release export")
        source_hashes[source_key] = source_hash
        rows = pd.read_parquet(source).reset_index(drop=True)
        missing_public = set(PUBLIC_COLUMNS) - set(rows.columns)
        missing_labels = set(LABEL_COLUMNS) - set(rows.columns)
        if missing_public or missing_labels:
            raise ContractError(
                f"{split} manifest missing public={sorted(missing_public)} "
                f"labels={sorted(missing_labels)}"
            )
        public_path = root / "public" / f"{split}.parquet"
        labels_path = root / "labels" / f"{split}.parquet"
        if not (resume and public_path.is_file()):
            _atomic_parquet(rows.loc[:, PUBLIC_COLUMNS], public_path)
        if not (resume and labels_path.is_file()):
            _atomic_parquet(rows.loc[:, LABEL_COLUMNS], labels_path)

        scoring_dir = root / "scoring" / split
        scoring_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        query_n = np.zeros(734, np.int64)
        query_sum = np.zeros(734, np.float64)
        query_sum2 = np.zeros(734, np.float64)
        nonzero_targets = 0
        source_splice_hashes = []
        for index, row in enumerate(rows.itertuples(index=False)):
            destination = scoring_dir / f"part-{index:06d}.parquet"
            source_receipt = destination.with_suffix(".source.json")
            can_resume = resume and destination.is_file() and source_receipt.is_file()
            receipt = json.loads(source_receipt.read_text()) if can_resume else {}
            can_resume = can_resume and receipt.get("scoring_protocol") == SCORING_PROTOCOL
            if can_resume:
                scoring = pd.read_parquet(destination)
                if (receipt.get("example_id") != str(row.example_id)
                        or receipt.get("scoring_sha256") != _sha256(destination)):
                    raise ContractError(f"invalid resumable scoring part: {destination}")
                source_hash = receipt["source_splice_sha256"]
            else:
                scoring, source_hash = _scoring_part(root, row, halves)
                _atomic_parquet(scoring, destination)
                _atomic_json(source_receipt, {
                    "example_id": str(row.example_id),
                    "scoring_protocol": SCORING_PROTOCOL,
                    "source_splice_sha256": source_hash,
                    "scoring_sha256": _sha256(destination),
                })
            source_splice_hashes.append(source_hash)
            valid = scoring.valid.to_numpy(bool)
            track = scoring.track.to_numpy(np.int64)[valid]
            target = scoring.target_value.to_numpy(np.float64)[valid]
            query_n += np.bincount(track, minlength=734)
            query_sum += np.bincount(track, weights=target, minlength=734)
            query_sum2 += np.bincount(track, weights=target * target, minlength=734)
            nonzero_targets += int(np.count_nonzero(target))
            written += 1
        target_ss = np.maximum(query_n * query_sum2 - query_sum**2, 0)
        split_reports[split] = {
            "examples": len(rows),
            "public_sha256": _sha256(public_path),
            "labels_sha256": _sha256(labels_path),
            "scoring_parts": written,
            "valid_junction_queries": int(query_n.sum()),
            "nonzero_junction_targets": nonzero_targets,
            "junction_tracks_with_target_variance": int(((query_n >= 2) & (target_ss > 0)).sum()),
            "source_splice_hashes_sha256": hashlib.sha256(
                "\n".join(source_splice_hashes).encode()
            ).hexdigest(),
        }

    release = {
        "schema": RELEASE_SCHEMA,
        "inference_schema": INFERENCE_SCHEMA,
        "target_storage_schema": TARGET_STORAGE_SCHEMA,
        "coordinate_system": "zero-based-relative-to-1mb-input",
        "usage_coordinate_system": "zero-based-relative-to-central-target",
        "legacy_junction_query_count": LEGACY_JUNCTION_QUERY_COUNT,
        "supervised_queries_per_track": SUPERVISED_QUERIES_PER_TRACK,
        "junction_query_budget": JUNCTION_QUERY_BUDGET,
        "scoring_protocol": SCORING_PROTOCOL,
        "source_hashes": source_hashes,
        "splits": split_reports,
    }
    _atomic_json(root / "release.json", release)
    return release
