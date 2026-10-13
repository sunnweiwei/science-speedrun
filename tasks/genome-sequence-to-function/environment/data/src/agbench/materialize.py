"""Family-aware compact transforms and a real one-record smoke writer."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil

from numcodecs import Blosc
import numpy as np
import pandas as pd
import zarr

from .errors import ContractError
from .gcs import verify_object_current
from .protocol import (
    QUANTIZED_SPLITS,
    CONTACT_BINS,
    DENSE_BIN_BP,
    DENSE_BINS,
    INPUT_BP,
    TARGET_BP,
    DATASET_SCHEMA,
    TRANSFORM_CONTRACT,
    BUNDLES,
    RELEASE_MAX_GIB,
)
from .quantize import CONTACT_ENCODING, DENSE_ENCODING, encode
from .source_shards import aligned_shards
from .tfrecords import SourceRecord, iter_records


_COMPRESSOR = Blosc(cname="zstd", clevel=7, shuffle=Blosc.BITSHUFFLE)
_MASK_COMPRESSOR = Blosc(cname="zstd", clevel=7, shuffle=Blosc.SHUFFLE)
_DENSE_KEYS = {
    "ATAC": ("atac", "atac_mask"),
    "DNASE": ("dnase", "dnase_mask"),
    "PROCAP": ("procap", "procap_mask"),
    "CAGE": ("cage", "cage_mask"),
    "RNA_SEQ": ("rna_seq", "rna_seq_mask"),
    "CHIP_TF": ("chip_tf", "chip_tf_mask"),
    "CHIP_HISTONE": ("chip_histone", "chip_histone_mask"),
}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _track_indices(tracks: pd.DataFrame, family: str) -> np.ndarray:
    frame = tracks.loc[tracks["family"] == family].sort_values("track_index")
    if frame.empty:
        raise ContractError(f"No tracks for {family}")
    return frame["tensor_index"].to_numpy(dtype=np.int64)


def _target_start(record: SourceRecord, target_start: int | None) -> int:
    source_bp = record.end - record.start
    extra = source_bp - TARGET_BP
    if extra < 0 or extra % 2:
        raise ContractError(f"Cannot center target in {record.coordinate}")
    expected = record.start + extra // 2
    if target_start is not None and target_start != expected:
        raise ContractError(
            f"Target start {target_start} differs from centered source target {expected}"
        )
    return expected


def _input_start(record: SourceRecord, input_start: int | None) -> int:
    source_bp = record.end - record.start
    extra = source_bp - INPUT_BP
    if extra < 0 or extra % 2 or (extra // 2) % 2_048:
        raise ContractError(f"Cannot center 1-Mb input on source {record.coordinate}")
    expected = record.start + extra // 2
    if input_start is not None and input_start != expected:
        raise ContractError(
            f"Input start {input_start} differs from centered source input {expected}"
        )
    return expected


def _target_slice(record: SourceRecord, spatial_length: int, resolution_bp: int,
                  target_start: int | None = None) -> slice:
    target_start = _target_start(record, target_start)
    if TARGET_BP % resolution_bp:
        raise ContractError(f"Target length is not divisible by {resolution_bp} bp")
    target_bins = TARGET_BP // resolution_bp
    if spatial_length == target_bins:
        return slice(0, target_bins)
    source_bp = record.end - record.start
    if source_bp % resolution_bp or spatial_length != source_bp // resolution_bp:
        raise ContractError(
            f"Unexpected {resolution_bp}-bp source length {spatial_length} "
            f"for {record.coordinate}; expected {target_bins} target bins or "
            f"{source_bp // resolution_bp} source bins"
        )
    offset_bp = target_start - record.start
    if offset_bp % resolution_bp:
        raise ContractError(f"Target offset {offset_bp} is not {resolution_bp}-bp aligned")
    start_bin = offset_bp // resolution_bp
    return slice(start_bin, start_bin + target_bins)


def _model_input_slice(record: SourceRecord, spatial_length: int,
                       resolution_bp: int, input_start: int | None = None) -> slice:
    input_start = _input_start(record, input_start)
    model_bins = INPUT_BP // resolution_bp
    if spatial_length == model_bins:
        return slice(0, model_bins)
    source_bp = record.end - record.start
    if source_bp % resolution_bp or spatial_length != source_bp // resolution_bp:
        raise ContractError(f"Unexpected {resolution_bp}-bp input length {spatial_length}")
    offset_bp = input_start - record.start
    if offset_bp % resolution_bp:
        raise ContractError(f"Input offset {offset_bp} is not {resolution_bp}-bp aligned")
    start_bin = offset_bp // resolution_bp
    return slice(start_bin, start_bin + model_bins)


def _public_site_positions(raw: np.ndarray, record: SourceRecord,
                           input_start: int | None = None) -> np.ndarray:
    positions = np.asarray(raw, dtype=np.int32).copy()
    if positions.shape != (4, 512):
        raise ContractError(f"Unexpected splice-site-position shape {positions.shape}")
    offset = _input_start(record, input_start) - record.start
    valid = positions >= 0
    positions[valid] -= offset
    positions[valid & ((positions < 0) | (positions >= INPUT_BP))] = -1
    return positions


def _validate_alignment(
    record: SourceRecord,
    *,
    expected_coordinate: tuple[str, int, int] | None,
    expected_dna_hash: str | None,
) -> tuple[tuple[str, int, int], str]:
    if expected_coordinate is not None and record.coordinate != expected_coordinate:
        raise ContractError(
            f"Record coordinates differ: {record.coordinate} != {expected_coordinate}"
        )
    if expected_dna_hash is not None and record.dna_serialized_sha256 != expected_dna_hash:
        raise ContractError("Record DNA tensor differs across bundles")
    return record.coordinate, record.dna_serialized_sha256


def _write_dense(group, bundle: str, record: SourceRecord, indices: np.ndarray,
                 target_start: int | None = None, *, quantize: bool = False) -> dict:
    value_key, mask_key = _DENSE_KEYS[bundle]
    raw_values = record.tensors[value_key]
    resolution = BUNDLES[bundle].native_resolution
    values_slice = _target_slice(record, raw_values.shape[0], resolution, target_start)
    values = np.asarray(raw_values[values_slice][:, indices], dtype=np.float32)
    source_mask = np.asarray(record.tensors[mask_key], dtype=bool)
    if source_mask.ndim != 2 or source_mask.shape[1] <= int(indices.max()):
        raise ContractError(f"Invalid {bundle} mask shape {source_mask.shape}")
    if source_mask.shape[0] != 1:
        mask_slice = _target_slice(record, source_mask.shape[0], resolution, target_start)
        source_mask = source_mask[mask_slice]
    if values.shape[0] == DENSE_BINS:
        compact = values
        spatial_mask = source_mask[:, indices]
    elif values.shape[0] == DENSE_BINS * DENSE_BIN_BP:
        reshaped = values.reshape(DENSE_BINS, DENSE_BIN_BP, len(indices))
        if source_mask.shape[0] == 1:
            spatial_mask = source_mask[:, indices]
            compact = reshaped.sum(axis=1, dtype=np.float32)
        elif source_mask.shape[0] == values.shape[0]:
            mask = source_mask[:, indices].reshape(
                DENSE_BINS, DENSE_BIN_BP, len(indices)
            )
            compact = np.where(mask, reshaped, 0).sum(axis=1, dtype=np.float32)
            spatial_mask = mask.all(axis=1)
        else:
            raise ContractError(f"Invalid spatial {bundle} mask shape {source_mask.shape}")
    else:
        raise ContractError(f"Unexpected {bundle} spatial length {values.shape[0]}")
    if spatial_mask.shape[0] == 1:
        stored_mask = spatial_mask[0]
        mask_kind = "track"
        compact[:, ~stored_mask] = 0
    else:
        stored_mask = spatial_mask
        mask_kind = "spatial_track"
        compact = np.where(stored_mask, compact, 0)
    finite = np.isfinite(compact)
    if not finite.all():
        if stored_mask.ndim == 1:
            stored_mask = np.broadcast_to(stored_mask[None, :], compact.shape).copy()
            mask_kind = "spatial_track"
        stored_mask &= finite
        compact = np.where(stored_mask, compact, 0)
    family = group.create_group(bundle.lower())
    chunks = (min(256, compact.shape[0]), min(64, compact.shape[1]))
    if quantize:
        codes, lo, hi = encode(compact, asinh=True)
        family.create_dataset("values", data=codes, chunks=chunks, compressor=_COMPRESSOR)
        family.create_dataset("values_lo", data=lo, compressor=_COMPRESSOR)
        family.create_dataset("values_hi", data=hi, compressor=_COMPRESSOR)
        family.attrs["encoding"] = DENSE_ENCODING
    else:
        family.create_dataset(
            "values", data=compact.astype(np.float32), chunks=chunks, compressor=_COMPRESSOR,
        )
    family.create_dataset(
        "mask",
        data=stored_mask,
        chunks=tuple(min(256, dim) for dim in stored_mask.shape),
        compressor=_MASK_COMPRESSOR,
    )
    if bundle == "RNA_SEQ":
        strand = np.asarray(record.tensors["rna_seq_strand"][:, indices], dtype=np.int8)
        family.create_dataset("strand", data=strand[0], compressor=_COMPRESSOR)
    family.attrs.update(
        {
            "representation": "dense_128",
            "resolution_bp": 128,
            "track_count": len(indices),
            "mask_kind": mask_kind,
        }
    )
    return {"shape": list(compact.shape), "mask_shape": list(stored_mask.shape)}


def _valid_splice_positions(raw: np.ndarray, target_length: int) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.int32)
    valid = raw[(raw >= 0) & (raw < target_length)]
    return np.unique(valid)


def _write_splice_sites(group, record: SourceRecord, indices: np.ndarray,
                        target_start: int | None = None) -> dict:
    raw = record.tensors["splice_sites"]
    span = _target_slice(record, raw.shape[0], 1, target_start)
    values = np.asarray(raw[span][:, indices], dtype=bool)
    family = group.create_group("splice_sites")
    family.create_dataset(
        "values",
        data=values,
        chunks=(min(4096, values.shape[0]), values.shape[1]),
        compressor=_MASK_COMPRESSOR,
    )
    family.attrs.update({"representation": "base", "resolution_bp": 1})
    return {"shape": list(values.shape), "positive_values": int(values.sum())}


def _write_splice_usage(
    group, record: SourceRecord, indices: np.ndarray, raw_positions: np.ndarray,
    target_start: int | None = None,
) -> dict:
    source = record.tensors["splice_site_usage"]
    target_offset = _target_start(record, target_start) - record.start
    raw_events = np.unique(np.asarray(raw_positions, dtype=np.int64))
    raw_events = raw_events[(raw_events >= target_offset) &
                            (raw_events < target_offset + TARGET_BP)]
    positions = (raw_events - target_offset).astype(np.int32)
    if source.shape[0] == record.end - record.start:
        source_indices = raw_events
    elif source.shape[0] == TARGET_BP:
        source_indices = positions
    else:
        raise ContractError(f"Unexpected splice-usage length {source.shape[0]}")
    values = np.asarray(source[np.ix_(source_indices, indices)], dtype=np.float16)
    mask = np.isfinite(values)
    values = np.where(mask, values, 0)
    family = group.create_group("splice_site_usage")
    family.create_dataset("positions", data=positions.astype(np.int32), compressor=_COMPRESSOR)
    family.create_dataset(
        "values",
        data=values,
        chunks=(min(256, max(1, len(positions))), min(64, values.shape[1])),
        compressor=_COMPRESSOR,
    )
    family.create_dataset(
        "mask",
        data=mask,
        chunks=(min(256, max(1, len(positions))), min(64, values.shape[1])),
        compressor=_MASK_COMPRESSOR,
    )
    family.attrs.update({"representation": "event", "track_count": len(indices)})
    return {"events": len(positions), "values_shape": list(values.shape)}


def _write_sparse_junctions(group, record: SourceRecord, raw_positions: np.ndarray) -> dict:
    source = record.tensors["splice_junctions"]
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    tracks: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for start in range(0, source.shape[0], 8):
        block = np.asarray(source[start : start + 8])
        nz = np.nonzero(block)
        if nz[0].size:
            rows.append((nz[0] + start).astype(np.uint16))
            columns.append(nz[1].astype(np.uint16))
            tracks.append(nz[2].astype(np.uint16))
            values.append(block[nz].astype(np.float32))
    row = np.concatenate(rows) if rows else np.empty(0, np.uint16)
    column = np.concatenate(columns) if columns else np.empty(0, np.uint16)
    track = np.concatenate(tracks) if tracks else np.empty(0, np.uint16)
    value = np.concatenate(values) if values else np.empty(0, np.float32)
    finite = np.isfinite(value)
    row, column, track, value = (
        item[finite] for item in (row, column, track, value)
    )
    family = group.create_group("splice_junctions")
    family.create_dataset("row", data=row, chunks=(min(262144, max(1, len(row))),), compressor=_COMPRESSOR)
    family.create_dataset("column", data=column, chunks=(min(262144, max(1, len(row))),), compressor=_COMPRESSOR)
    family.create_dataset("track", data=track, chunks=(min(262144, max(1, len(row))),), compressor=_COMPRESSOR)
    family.create_dataset("value", data=value, chunks=(min(262144, max(1, len(row))),), compressor=_COMPRESSOR)
    family.create_dataset("site_positions", data=np.asarray(raw_positions, np.int32), compressor=_COMPRESSOR)
    family.attrs.update(
        {"representation": "sparse_coo", "dense_shape": list(source.shape), "implicit_value": 0.0}
    )
    return {"dense_shape": list(source.shape), "nonzero_values": len(value)}


def _write_contact_maps(group, record: SourceRecord, indices: np.ndarray,
                        input_start: int | None = None, *, quantize: bool = False) -> dict:
    source = record.tensors["contact_maps"]
    if source.shape[0] != source.shape[1]:
        raise ContractError(f"Unexpected contact-map shape {source.shape}")
    span = _model_input_slice(record, source.shape[0], 2_048, input_start)
    row, column = np.triu_indices(CONTACT_BINS)
    original = np.asarray(source[row + span.start, column + span.start][:, indices], dtype=np.float32)
    finite = np.isfinite(original)
    compact = np.where(finite, original, 0).astype(np.float16)
    max_error = float(np.max(np.abs(original[finite] - compact[finite]))) if finite.any() else 0.0
    family = group.create_group("contact_maps")
    chunks = (min(8192, len(row)), min(28, len(indices)))
    if quantize:
        codes, lo, hi = encode(np.where(finite, original, 0), asinh=False)
        family.create_dataset("upper_triangle", data=codes, chunks=chunks, compressor=_COMPRESSOR)
        family.create_dataset("upper_triangle_lo", data=lo, compressor=_COMPRESSOR)
        family.create_dataset("upper_triangle_hi", data=hi, compressor=_COMPRESSOR)
        family.attrs["encoding"] = CONTACT_ENCODING
    else:
        family.create_dataset("upper_triangle", data=compact, chunks=chunks, compressor=_COMPRESSOR)
    family.create_dataset(
        "mask",
        data=finite,
        chunks=(min(8192, len(row)), min(28, len(indices))),
        compressor=_MASK_COMPRESSOR,
    )
    family.attrs.update(
        {
            "representation": "symmetric_upper_triangle",
            "matrix_bins": CONTACT_BINS,
            "resolution_bp": 2048,
            "track_count": len(indices),
        }
    )
    return {
        "upper_triangle_shape": list(compact.shape),
        "float16_max_abs_error": max_error,
    }


def materialize_smoke(data_dir: Path, *, split: str = "valid") -> dict[str, object]:
    tracks_path = data_dir / "metadata" / "tracks.parquet"
    if not tracks_path.exists():
        raise ContractError("Run bootstrap before smoke")
    tracks = pd.read_parquet(tracks_path)
    shard_set = aligned_shards(split)[0]
    output = data_dir / "smoke" / split / "first-record.zarr"
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    if output.exists():
        shutil.rmtree(output)
    partial.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(partial), mode="w")
    expected_coordinate = None
    expected_dna_hash = None
    report: dict[str, object] = {}

    positions_record = next(
        iter(iter_records(shard_set["SPLICE_SITE_POSITIONS"].gs_uri, "SPLICE_SITE_POSITIONS", compact_target=True))
    )
    expected_coordinate, expected_dna_hash = _validate_alignment(
        positions_record,
        expected_coordinate=expected_coordinate,
        expected_dna_hash=expected_dna_hash,
    )
    raw_positions = positions_record.tensors["splice_site_positions"]
    del positions_record

    for bundle in _DENSE_KEYS:
        record = next(iter(iter_records(shard_set[bundle].gs_uri, bundle, compact_target=True)))
        expected_coordinate, expected_dna_hash = _validate_alignment(
            record,
            expected_coordinate=expected_coordinate,
            expected_dna_hash=expected_dna_hash,
        )
        report[bundle] = _write_dense(group, bundle, record, _track_indices(tracks, bundle))
        del record

    record = next(iter(iter_records(shard_set["SPLICE_SITES"].gs_uri, "SPLICE_SITES", compact_target=True)))
    _validate_alignment(record, expected_coordinate=expected_coordinate, expected_dna_hash=expected_dna_hash)
    report["SPLICE_SITES"] = _write_splice_sites(
        group, record, _track_indices(tracks, "SPLICE_SITES")
    )
    del record

    record = next(iter(iter_records(shard_set["SPLICE_SITE_USAGE"].gs_uri, "SPLICE_SITE_USAGE", compact_target=True)))
    _validate_alignment(record, expected_coordinate=expected_coordinate, expected_dna_hash=expected_dna_hash)
    report["SPLICE_SITE_USAGE"] = _write_splice_usage(
        group, record, _track_indices(tracks, "SPLICE_SITE_USAGE"), raw_positions
    )
    del record

    record = next(iter(iter_records(shard_set["SPLICE_JUNCTIONS"].gs_uri, "SPLICE_JUNCTIONS", compact_target=True)))
    _validate_alignment(record, expected_coordinate=expected_coordinate, expected_dna_hash=expected_dna_hash)
    report["SPLICE_JUNCTIONS"] = _write_sparse_junctions(
        group, record, _public_site_positions(raw_positions, record)
    )
    del record

    record = next(iter(iter_records(shard_set["CONTACT_MAPS"].gs_uri, "CONTACT_MAPS", compact_target=True)))
    _validate_alignment(record, expected_coordinate=expected_coordinate, expected_dna_hash=expected_dna_hash)
    report["CONTACT_MAPS"] = _write_contact_maps(
        group, record, _track_indices(tracks, "CONTACT_MAPS")
    )
    del record

    group.attrs.update(
        {
            "schema": DATASET_SCHEMA,
            "transform_contract": TRANSFORM_CONTRACT,
            "split": split,
            "interval": list(expected_coordinate or ()),
            "dna_serialized_sha256": expected_dna_hash,
            "track_manifest_sha256": hashlib.sha256(tracks_path.read_bytes()).hexdigest(),
            "source_objects": {bundle: item.__dict__ for bundle, item in shard_set.items()},
        }
    )
    partial.replace(output)
    stored_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    receipt = {
        "schema": DATASET_SCHEMA,
        "status": "PASS",
        "split": split,
        "interval": list(expected_coordinate or ()),
        "stored_bytes": stored_bytes,
        "stored_mib": stored_bytes / 2**20,
        "families": report,
        "output": str(output),
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    receipt_path = data_dir / "smoke" / f"SMOKE_{split.upper()}_VALIDATED.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _json_row(row) -> dict[str, object]:
    result = {}
    for key, value in row._asdict().items():
        if isinstance(value, np.generic):
            value = value.item()
        result[key] = value
    return result


def _read_selected_record(
    *,
    source,
    bundle: str,
    selected_by_index: dict[int, dict[str, object]],
    callback,
) -> None:
    remaining = set(selected_by_index)
    verify_object_current(source)
    for record in iter_records(
        source.gs_uri, bundle, record_indices=remaining, compact_target=True
    ):
        if record.source_record_index is None:
            raise ContractError(f"{bundle} reader omitted its source record index")
        record_index = record.source_record_index
        callback(record_index, record, selected_by_index[record_index])
        remaining.remove(record_index)
        if not remaining:
            break
    verify_object_current(source)
    if remaining:
        raise ContractError(
            f"{bundle} shard ended before selected record indices {sorted(remaining)[:5]}"
        )


def _part_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _materialize_part(
    *,
    data_dir: Path,
    split: str,
    shard_set,
    selected_rows: pd.DataFrame,
    tracks: pd.DataFrame,
) -> tuple[Path, list[dict[str, object]], int]:
    source = shard_set["SPLICE_SITE_POSITIONS"]
    target_dir = data_dir / "targets" / split
    target_dir.mkdir(parents=True, exist_ok=True)
    output = target_dir / f"part-{source.shard:04d}-of-{source.shard_count:04d}.zarr"
    partial = output.with_name(output.name + ".partial")
    if output.exists():
        root = zarr.open_group(str(output), mode="r")
        if root.attrs.get("schema") != DATASET_SCHEMA or root.attrs.get("transform_contract") != TRANSFORM_CONTRACT:
            raise ContractError(f"Existing part has incompatible schema: {output}")
        rows = list(root.attrs.get("manifest_rows", []))
        observed_ids = sorted(str(row["example_id"]) for row in rows)
        expected_ids = sorted(selected_rows["example_id"].astype(str).tolist())
        if observed_ids != expected_ids:
            raise ContractError(
                f"Existing part selection differs from the frozen manifest: {output}"
            )
        return output, rows, _part_size(output)
    if partial.exists():
        shutil.rmtree(partial)
    root = zarr.open_group(str(partial), mode="w")
    selected_by_index = {
        int(row.source_record_index): _json_row(row)
        for row in selected_rows.itertuples(index=False)
    }
    expected: dict[str, tuple[tuple[str, int, int], str]] = {}
    manifest_rows: list[dict[str, object]] = []

    def write_positions(record_index: int, record: SourceRecord, row: dict[str, object]):
        coordinate, dna_hash = _validate_alignment(
            record, expected_coordinate=None, expected_dna_hash=None
        )
        if dna_hash != row["dna_serialized_sha256"]:
            raise ContractError(
                f"Indexed DNA hash changed for {row['example_id']} in shard {source.shard}"
            )
        example_id = str(row["example_id"])
        example = root.create_group(f"examples/{example_id}")
        positions = _public_site_positions(
            record.tensors["splice_site_positions"], record, int(row["input_start"])
        )
        example.create_dataset("splice_site_positions", data=positions, compressor=_COMPRESSOR)
        example.attrs.update(
            {
                "example_id": example_id,
                "interval": list(coordinate),
                "input_start": int(row["input_start"]),
                "input_end": int(row["input_end"]),
                "target_start": int(row["target_start"]),
                "target_end": int(row["target_end"]),
                "dna_serialized_sha256": dna_hash,
            }
        )
        expected[example_id] = (coordinate, dna_hash)
        materialized_row = dict(row)
        materialized_row.update(
            {
                "target_shard": str(output.relative_to(data_dir)),
                "target_group": f"examples/{example_id}",
                "splice_site_positions": positions.tolist(),
            }
        )
        manifest_rows.append(materialized_row)

    _read_selected_record(
        source=source,
        bundle="SPLICE_SITE_POSITIONS",
        selected_by_index=selected_by_index,
        callback=write_positions,
    )

    def aligned_example(record: SourceRecord, row: dict[str, object]):
        example_id = str(row["example_id"])
        coordinate, dna_hash = expected[example_id]
        _validate_alignment(
            record, expected_coordinate=coordinate, expected_dna_hash=dna_hash
        )
        return root[f"examples/{example_id}"]

    for bundle in _DENSE_KEYS:
        def write_dense_record(record_index, record, row, current_bundle=bundle):
            del record_index
            example = aligned_example(record, row)
            _write_dense(
                example,
                current_bundle,
                record,
                _track_indices(tracks, current_bundle),
                int(row["target_start"]),
                quantize=split in QUANTIZED_SPLITS,
            )

        _read_selected_record(
            source=shard_set[bundle],
            bundle=bundle,
            selected_by_index=selected_by_index,
            callback=write_dense_record,
        )

    def write_splice_sites_record(record_index, record, row):
        del record_index
        example = aligned_example(record, row)
        _write_splice_sites(
            example, record, _track_indices(tracks, "SPLICE_SITES"),
            int(row["target_start"]),
        )

    _read_selected_record(
        source=shard_set["SPLICE_SITES"],
        bundle="SPLICE_SITES",
        selected_by_index=selected_by_index,
        callback=write_splice_sites_record,
    )

    def write_usage_record(record_index, record, row):
        del record_index
        example = aligned_example(record, row)
        raw_positions = np.asarray(example["splice_site_positions"], dtype=np.int64)
        raw_positions[raw_positions >= 0] += int(row["input_start"]) - record.start
        _write_splice_usage(
            example,
            record,
            _track_indices(tracks, "SPLICE_SITE_USAGE"),
            raw_positions,
            int(row["target_start"]),
        )

    _read_selected_record(
        source=shard_set["SPLICE_SITE_USAGE"],
        bundle="SPLICE_SITE_USAGE",
        selected_by_index=selected_by_index,
        callback=write_usage_record,
    )

    def write_junction_record(record_index, record, row):
        del record_index
        example = aligned_example(record, row)
        positions = np.asarray(example["splice_site_positions"])
        _write_sparse_junctions(example, record, positions)

    _read_selected_record(
        source=shard_set["SPLICE_JUNCTIONS"],
        bundle="SPLICE_JUNCTIONS",
        selected_by_index=selected_by_index,
        callback=write_junction_record,
    )

    def write_contact_record(record_index, record, row):
        del record_index
        example = aligned_example(record, row)
        _write_contact_maps(
            example, record, _track_indices(tracks, "CONTACT_MAPS"),
            int(row["input_start"]), quantize=split in QUANTIZED_SPLITS,
        )

    _read_selected_record(
        source=shard_set["CONTACT_MAPS"],
        bundle="CONTACT_MAPS",
        selected_by_index=selected_by_index,
        callback=write_contact_record,
    )

    root.attrs.update(
        {
            "schema": DATASET_SCHEMA,
            "transform_contract": TRANSFORM_CONTRACT,
            "split": split,
            "source_shard": source.shard,
            "source_shard_count": source.shard_count,
            "source_objects": {bundle: item.__dict__ for bundle, item in shard_set.items()},
            "manifest_rows": manifest_rows,
        }
    )
    partial.replace(output)
    return output, manifest_rows, _part_size(output)


def _materialize_job(task) -> tuple[int, Path, list[dict[str, object]], int]:
    data_dir, split, shard_set, selected_rows, tracks_path = task
    shard = shard_set["SPLICE_SITE_POSITIONS"].shard
    output, rows, size = _materialize_part(
        data_dir=data_dir,
        split=split,
        shard_set=shard_set,
        selected_rows=selected_rows,
        tracks=pd.read_parquet(tracks_path),
    )
    return shard, output, rows, size


def materialize_split(
    data_dir: Path,
    *,
    split: str,
    max_shards: int | None = None,
    workers: int = 1,
) -> dict[str, object]:
    if workers < 1:
        raise ContractError("Materialization workers must be at least 1")
    selected_path = data_dir / "splits" / f"{split}_selected.parquet"
    tracks_path = data_dir / "metadata" / "tracks.parquet"
    if not selected_path.exists() or "source_record_index" not in pd.read_parquet(
        selected_path, columns=None
    ).columns:
        raise ContractError(f"Run index for {split} before materialization")
    selected = pd.read_parquet(selected_path)
    by_shard = {int(key): frame for key, frame in selected.groupby("source_shard")}
    shard_sets = [
        item
        for item in aligned_shards(split)
        if item["SPLICE_SITE_POSITIONS"].shard in by_shard
    ]
    if max_shards is not None:
        shard_sets = shard_sets[:max_shards]
    completed: dict[int, tuple[list[dict[str, object]], int]] = {}
    total_bytes = 0

    def record_completion(result: tuple[int, Path, list[dict[str, object]], int]) -> None:
        nonlocal total_bytes
        shard, output, rows, size = result
        if shard in completed:
            raise ContractError(f"Shard {shard} was materialized more than once")
        completed[shard] = (rows, size)
        total_bytes += size
        print(
            f"completed {split} shard {shard}: {len(rows)} examples, "
            f"{size / 2**20:.2f} MiB -> {output}",
            flush=True,
        )
        projected = size / max(1, len(rows)) * len(selected)
        if projected > RELEASE_MAX_GIB * 2**30:
            raise ContractError(
                f"Projected {split} storage alone is {projected / 2**30:.1f} GiB, "
                f"above the full-release {RELEASE_MAX_GIB}-GiB hard limit"
            )

    if workers == 1:
        tracks = pd.read_parquet(tracks_path)
        for shard_set in shard_sets:
            shard = shard_set["SPLICE_SITE_POSITIONS"].shard
            output, rows, size = _materialize_part(
                data_dir=data_dir,
                split=split,
                shard_set=shard_set,
                selected_rows=by_shard[shard],
                tracks=tracks,
            )
            record_completion((shard, output, rows, size))
    else:
        # Spawn isolates TensorFlow's memory allocator per shard. One process
        # handles one shard, so completed parts remain atomic and resumable.
        jobs = (
            (data_dir, split, shard_set,
             by_shard[shard_set["SPLICE_SITE_POSITIONS"].shard], tracks_path)
            for shard_set in shard_sets
        )
        context = multiprocessing.get_context("spawn")
        with context.Pool(processes=workers, maxtasksperchild=1) as pool:
            for result in pool.imap_unordered(_materialize_job, jobs, chunksize=1):
                record_completion(result)

    all_rows = [row for shard in sorted(completed) for row in completed[shard][0]]
    materialized = pd.DataFrame(all_rows)
    materialized_path = data_dir / "splits" / f"{split}_materialized.parquet"
    materialized.to_parquet(materialized_path, index=False)
    complete = max_shards is None and len(all_rows) == len(selected)
    receipt = {
        "schema": DATASET_SCHEMA,
        "status": "PASS",
        "split": split,
        "complete": complete,
        "selected_examples": len(selected),
        "materialized_examples": len(all_rows),
        "materialized_shards": len(completed),
        "workers": workers,
        "stored_bytes": total_bytes,
        "stored_mib_per_example": total_bytes / max(1, len(all_rows)) / 2**20,
    }
    receipt_path = data_dir / "metadata" / f"materialization_{split}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    # New materializations and migrated releases share the same exporter.
    from .release import export_release
    export_release(data_dir, [split], resume=False)
    return receipt
