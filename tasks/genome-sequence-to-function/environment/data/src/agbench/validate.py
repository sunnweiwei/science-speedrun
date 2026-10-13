"""Structural and capacity validation for compact materializations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import ContractError
from .quantize import CONTACT_ENCODING, DENSE_ENCODING
from .protocol import (
    CONTACT_BINS,
    DENSE_BINS,
    RELEASE_MAX_GIB,
    DATASET_SCHEMA,
    SPLIT_BUDGETS,
    TRANSFORM_CONTRACT,
)
from .release import (
    JUNCTION_QUERY_BUDGET,
    LABEL_COLUMNS,
    LEGACY_JUNCTION_QUERY_COUNT,
    PUBLIC_COLUMNS,
    SCORING_PROTOCOL,
    SUPERVISED_QUERIES_PER_TRACK,
    _scoring_part,
    _sha256,
)
from .zarr_store import close_store, open_group


_DENSE_TRACKS = {
    "atac": 167,
    "dnase": 305,
    "procap": 12,
    "cage": 546,
    "rna_seq": 667,
    "chip_tf": 1_617,
    "chip_histone": 1_116,
}


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _referenced_target_size(data_dir: Path, materialized: pd.DataFrame) -> int:
    """Return the size of unique target stores referenced by a manifest.

    A materialization host may retain both an expanded ``.zarr`` work tree and
    its release ``.zarr.zip`` beside it.  Only manifest-referenced stores are
    part of the dataset contract and capacity budget.
    """
    shards = {str(value) for value in materialized["target_shard"].tolist()}
    return sum(_size(data_dir / shard) for shard in shards)


def _validate_example(example, example_id: str) -> None:
    if example.attrs.get("example_id") != example_id:
        raise ContractError(f"Example group ID mismatch for {example_id}")
    for family, tracks in _DENSE_TRACKS.items():
        values = example[f"{family}/values"]
        mask = example[f"{family}/mask"]
        quantized = example[family].attrs.get("encoding") == DENSE_ENCODING
        expected_dtype = np.dtype("uint8") if quantized else np.dtype("float32")
        if values.shape != (DENSE_BINS, tracks) or values.dtype != expected_dtype:
            raise ContractError(
                f"{example_id}/{family} has {values.shape} {values.dtype}; "
                f"expected {(DENSE_BINS, tracks)} {expected_dtype}"
            )
        if quantized and any(example[f"{family}/{name}"].shape != (tracks,)
                             for name in ("values_lo", "values_hi")):
            raise ContractError(f"{example_id}/{family} quantization ranges are invalid")
        if mask.shape not in {(tracks,), (DENSE_BINS, tracks)}:
            raise ContractError(f"{example_id}/{family} mask shape is {mask.shape}")
    if example["splice_sites/values"].shape != (196_608, 4):
        raise ContractError(f"{example_id} splice-sites shape is invalid")
    usage_positions = example["splice_site_usage/positions"]
    usage_values = example["splice_site_usage/values"]
    usage_mask = example["splice_site_usage/mask"]
    if usage_values.shape != (usage_positions.shape[0], 734):
        raise ContractError(f"{example_id} splice-usage event shape is invalid")
    if usage_mask.shape != usage_values.shape:
        raise ContractError(f"{example_id} splice-usage mask shape is invalid")
    positions = np.asarray(example["splice_site_positions"], dtype=np.int64)
    expected_usage = np.unique(positions[
        (positions >= 425_984) & (positions < 622_592)
    ] - 425_984)
    if positions.shape != (4, 512) or not np.array_equal(
        np.asarray(usage_positions), expected_usage
    ):
        raise ContractError(f"{example_id} splice-position coordinates are inconsistent")
    junction = example["splice_junctions"]
    lengths = {
        junction[name].shape[0] for name in ("row", "column", "track", "value")
    }
    if len(lengths) != 1 or junction["site_positions"].shape != (4, 512):
        raise ContractError(f"{example_id} sparse junction arrays are inconsistent")
    if list(junction.attrs.get("dense_shape", [])) != [512, 512, 734]:
        raise ContractError(f"{example_id} junction dense-shape metadata is invalid")
    if not np.array_equal(np.asarray(junction["site_positions"]), positions):
        raise ContractError(f"{example_id} junction positions differ from public positions")
    contact = example["contact_maps/upper_triangle"]
    contact_mask = example["contact_maps/mask"]
    triangle = CONTACT_BINS * (CONTACT_BINS + 1) // 2
    contact_quantized = example["contact_maps"].attrs.get("encoding") == CONTACT_ENCODING
    contact_dtype = np.dtype("uint8") if contact_quantized else np.dtype("float16")
    if contact.shape != (triangle, 28) or contact.dtype != contact_dtype:
        raise ContractError(f"{example_id} contact-map representation is invalid")
    if contact_mask.shape != contact.shape:
        raise ContractError(f"{example_id} contact-map mask shape is invalid")


def validate_release_metadata(data_dir: Path, *, splits: list[str]) -> dict[str, object]:
    """Validate public isolation and trusted coordinate indices without dense reads."""
    release_path = data_dir / "release.json"
    if not release_path.is_file():
        raise ContractError("Missing release.json; run `agbench migrate-release`")
    release = json.loads(release_path.read_text())
    if release.get("schema") != "alphagenome-human-compact-release/v3":
        raise ContractError("Invalid release schema")
    if (
        release.get("legacy_junction_query_count") != LEGACY_JUNCTION_QUERY_COUNT
        or release.get("supervised_queries_per_track") != SUPERVISED_QUERIES_PER_TRACK
        or release.get("junction_query_budget") != JUNCTION_QUERY_BUDGET
        or release.get("scoring_protocol") != SCORING_PROTOCOL
    ):
        raise ContractError("Invalid or stale junction scoring protocol")
    tracks = pd.read_parquet(data_dir / "metadata/tracks.parquet")
    junction_tracks = tracks.loc[tracks.family == "SPLICE_JUNCTIONS"].sort_values("tensor_index")
    halves = junction_tracks.junction_channel_half.to_numpy(np.int64)
    public_tracks = pd.read_parquet(data_dir / "public/junction_tracks.parquet")
    if tuple(public_tracks.columns) != ("tensor_index", "junction_channel_half"):
        raise ContractError("public junction track mapping has unexpected fields")
    if not np.array_equal(
        public_tracks.sort_values("tensor_index").junction_channel_half.to_numpy(np.int64), halves
    ):
        raise ContractError("public junction track mapping differs from source metadata")
    reports = {}
    for split in splits:
        public_path = data_dir / "public" / f"{split}.parquet"
        labels_path = data_dir / "labels" / f"{split}.parquet"
        public = pd.read_parquet(public_path)
        labels = pd.read_parquet(labels_path)
        split_release = release.get("splits", {}).get(split, {})
        if (split_release.get("examples") != len(public)
                or split_release.get("public_sha256") != _sha256(public_path)
                or split_release.get("labels_sha256") != _sha256(labels_path)):
            raise ContractError(f"{split} manifest hashes/count differ from release.json")
        if tuple(public.columns) != PUBLIC_COLUMNS:
            raise ContractError(f"{split} public manifest violates the field whitelist")
        if tuple(labels.columns) != LABEL_COLUMNS:
            raise ContractError(f"{split} labels manifest has unexpected fields")
        public_ids = public.example_id.astype(str).tolist()
        if public_ids != labels.example_id.astype(str).tolist() or len(public_ids) != len(set(public_ids)):
            raise ContractError(f"{split} IDs/order differ between public and labels")
        valid_queries = 0
        query_n = np.zeros(734, np.int64)
        query_sum = np.zeros(734, np.float64)
        query_sum2 = np.zeros(734, np.float64)
        nonzero_targets = 0
        source_splice_hashes = []
        for index, row in enumerate(labels.itertuples(index=False)):
            target_root, target_store = open_group(data_dir / str(row.target_shard), mode="r")
            target = target_root[str(row.target_group)]
            positions = np.stack([np.asarray(item, np.int32) for item in row.splice_site_positions])
            stored = np.asarray(target["splice_junctions/site_positions"], np.int32)
            close_store(target_store)
            if not np.array_equal(positions, stored):
                raise ContractError(f"{row.example_id} label positions differ from target storage")
            expected, source_splice_hash = _scoring_part(data_dir, row, halves)
            scoring_path = data_dir / "scoring" / split / f"part-{index:06d}.parquet"
            scoring = pd.read_parquet(scoring_path)
            source_receipt = json.loads(scoring_path.with_suffix(".source.json").read_text())
            source_splice_hashes.append(source_receipt.get("source_splice_sha256", ""))
            if (tuple(scoring.columns) != tuple(expected.columns)
                    or len(scoring) > JUNCTION_QUERY_BUDGET
                    or any(not np.array_equal(scoring[column].to_numpy(), expected[column].to_numpy())
                           for column in expected.columns)
                    or source_receipt.get("example_id") != str(row.example_id)
                    or source_receipt.get("scoring_protocol") != SCORING_PROTOCOL
                    or source_receipt.get("source_splice_sha256") != source_splice_hash
                    or source_receipt.get("scoring_sha256") != _sha256(scoring_path)):
                raise ContractError(f"{row.example_id} scoring coordinates do not round-trip")
            valid = scoring.valid.to_numpy(bool)
            track = scoring.track.to_numpy(np.int64)[valid]
            target_value = scoring.target_value.to_numpy(np.float64)[valid]
            valid_queries += int(valid.sum())
            query_n += np.bincount(track, minlength=734)
            query_sum += np.bincount(track, weights=target_value, minlength=734)
            query_sum2 += np.bincount(track, weights=target_value * target_value, minlength=734)
            nonzero_targets += int(np.count_nonzero(target_value))
        combined_source_hash = hashlib.sha256(
            "\n".join(source_splice_hashes).encode()
        ).hexdigest()
        target_ss = np.maximum(query_n * query_sum2 - query_sum**2, 0)
        tracks_with_variance = int(((query_n >= 2) & (target_ss > 0)).sum())
        if (split_release.get("scoring_parts") != len(public_ids)
                or split_release.get("valid_junction_queries") != valid_queries
                or split_release.get("nonzero_junction_targets") != nonzero_targets
                or split_release.get("junction_tracks_with_target_variance") != tracks_with_variance
                or split_release.get("source_splice_hashes_sha256") != combined_source_hash):
            raise ContractError(f"{split} scoring audit differs from release.json")
        reports[split] = {
            "examples": len(public_ids),
            "valid_junction_queries": valid_queries,
            "nonzero_junction_targets": nonzero_targets,
            "junction_tracks_with_target_variance": tracks_with_variance,
        }
    return {"status": "PASS", "schema": release["schema"], "splits": reports}


def validate_release(
    data_dir: Path,
    *,
    splits: list[str],
    require_complete: bool = True,
    require_release: bool = False,
) -> dict[str, object]:
    tracks_path = data_dir / "metadata" / "tracks.parquet"
    if not tracks_path.exists():
        raise ContractError("Missing track manifest")
    tracks = pd.read_parquet(tracks_path)
    if len(tracks) != 5_930 or tracks["track_id"].duplicated().any():
        raise ContractError("Track manifest must contain 5,930 unique track IDs")
    reports = {}
    all_bytes = 0
    for split in splits:
        selected_path = data_dir / "splits" / f"{split}_selected.parquet"
        materialized_path = data_dir / "splits" / f"{split}_materialized.parquet"
        if not selected_path.exists() or not materialized_path.exists():
            raise ContractError(f"Missing selected/materialized manifest for {split}")
        selected = pd.read_parquet(selected_path)
        materialized = pd.read_parquet(materialized_path)
        expected_selected = SPLIT_BUDGETS[split]
        if len(selected) != expected_selected:
            raise ContractError(
                f"{split} selection has {len(selected)} examples; "
                f"protocol requires {expected_selected}"
            )
        selected_ids = set(selected["example_id"].astype(str))
        materialized_ids = set(materialized["example_id"].astype(str))
        if not materialized_ids <= selected_ids:
            raise ContractError(f"{split} materialized examples are outside the selection")
        if require_complete and materialized_ids != selected_ids:
            raise ContractError(
                f"{split} is incomplete: {len(materialized_ids)}/{len(selected_ids)}"
            )
        opened = {}
        stores = {}
        for row in materialized.itertuples(index=False):
            shard = str(row.target_shard)
            if shard not in opened:
                root, store = open_group(data_dir / shard, mode="r")
                if root.attrs.get("schema") != DATASET_SCHEMA or root.attrs.get("transform_contract") != TRANSFORM_CONTRACT:
                    raise ContractError(f"Invalid schema in {shard}")
                opened[shard] = root
                stores[shard] = store
            _validate_example(opened[shard][str(row.target_group)], str(row.example_id))
        for store in stores.values():
            close_store(store)
        stored = _referenced_target_size(data_dir, materialized)
        all_bytes += stored
        reports[split] = {
            "selected_examples": len(selected_ids),
            "materialized_examples": len(materialized_ids),
            "stored_bytes": stored,
            "complete": materialized_ids == selected_ids,
        }
    if all_bytes > RELEASE_MAX_GIB * 2**30:
        raise ContractError(
            f"Targets use {all_bytes / 2**30:.2f} GiB, above "
            f"the {RELEASE_MAX_GIB}-GiB hard limit"
        )
    target_root = data_dir / "targets"
    support_paths = (
        "metadata", "splits", "smoke", "source_index_parts", "public",
        "labels", "scoring", "release.json", "dataset_manifest.json",
    )
    support_bytes = sum(_size(data_dir / name) for name in support_paths)
    dataset_bytes = support_bytes + all_bytes
    reference_dir = data_dir / "reference"
    reference_bytes = _size(reference_dir) if reference_dir.exists() else 0
    packaged_bytes = dataset_bytes + reference_bytes
    if require_complete and packaged_bytes > RELEASE_MAX_GIB * 2**30:
        raise ContractError(
            f"Packaged data uses {packaged_bytes / 2**30:.2f} GiB, above "
            f"the {RELEASE_MAX_GIB}-GiB hard limit"
        )
    report = {
        "schema": DATASET_SCHEMA,
        "status": "PASS",
        "track_count": len(tracks),
        "splits": reports,
        "target_bytes": all_bytes,
        "target_gib": all_bytes / 2**30,
        "dataset_bytes": dataset_bytes,
        "release_metadata_bytes": support_bytes,
        "reference_bytes": reference_bytes,
        "packaged_bytes": packaged_bytes,
        "packaged_gib": packaged_bytes / 2**30,
        "require_complete": require_complete,
    }
    if require_release or (data_dir / "release.json").exists():
        report["release"] = validate_release_metadata(data_dir, splits=splits)
    path = data_dir / "validation_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
