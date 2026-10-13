"""Stream the small positions bundle to build authoritative source manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .errors import ContractError
from .gcs import verify_object_current
from .protocol import INPUT_BP, SPLIT_BUDGETS, TARGET_BP
from .source_shards import aligned_shards
from .manifests import stratified_select
from .tfrecords import iter_records


_INDEX_PART_FORMAT = "agbench-v2-source-index-v3"
_LEGACY_INDEX_PART_FORMAT = "agbench-v2-source-index-v2"


def _centered_span(start: int, end: int, length: int, label: str) -> tuple[int, int]:
    difference = end - start - length
    if difference < 0 or difference % 2:
        raise ContractError(
            f"Cannot center {length} {label} bases in source interval {start}-{end}"
        )
    centered_start = start + difference // 2
    return centered_start, centered_start + length


def _target_span(start: int, end: int) -> tuple[int, int]:
    return _centered_span(start, end, TARGET_BP, "target")


def _model_input_span(start: int, end: int) -> tuple[int, int]:
    """Crop the official source context symmetrically to the frozen 1-Mb input."""

    return _centered_span(start, end, INPUT_BP, "model-input")


def _coordinate_manifest(data_dir: Path, split: str) -> pd.DataFrame:
    split_dir = data_dir / "splits"
    path = (
        split_dir / "train_eligible.parquet"
        if split == "train"
        else split_dir / f"{split}_full.parquet"
    )
    if not path.exists():
        raise ContractError("Run bootstrap before index")
    frame = pd.read_parquet(path)
    return frame


def index_source_split(data_dir: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    coordinate = _coordinate_manifest(data_dir, split)
    lookup = {
        (row.chromosome, int(row.target_start), int(row.target_end)): row._asdict()
        for row in coordinate.itertuples(index=False)
    }
    rows: list[dict] = []
    unmatched: list[tuple[str, int, int]] = []
    excluded_by_policy: list[tuple[str, int, int]] = []
    shard_sets = aligned_shards(split)
    part_dir = data_dir / "source_index_parts" / split
    part_dir.mkdir(parents=True, exist_ok=True)
    for shard_set in shard_sets:
        source = shard_set["SPLICE_SITE_POSITIONS"]
        stem = f"part-{source.shard:04d}-of-{source.shard_count:04d}"
        part_path = part_dir / f"{stem}.parquet"
        part_receipt_path = part_dir / f"{stem}.json"
        if part_path.exists() and part_receipt_path.exists():
            part_receipt = json.loads(part_receipt_path.read_text())
            expected_identity = {
                "name": source.name,
                "generation": source.generation,
                "size": source.size,
            }
            if part_receipt.get("source") != expected_identity:
                raise ContractError(
                    f"Source-index checkpoint has stale object identity: {part_path}"
                )
            part_format = part_receipt.get("format")
            if part_format not in {_INDEX_PART_FORMAT, _LEGACY_INDEX_PART_FORMAT}:
                raise ContractError(
                    f"Source-index checkpoint has unknown format: {part_path}"
                )
            part = pd.read_parquet(part_path)
            if part_format == _LEGACY_INDEX_PART_FORMAT:
                left = part["input_start"] - part["source_interval_start"]
                right = part["source_interval_end"] - part["input_end"]
                if (
                    (left < 0).any()
                    or (right < 0).any()
                    or (left != right).any()
                    or ((part["input_end"] - part["input_start"]) != INPUT_BP).any()
                ):
                    raise ContractError(
                        f"Legacy source-index checkpoint cannot be upgraded safely: {part_path}"
                    )
                part["source_context_bp"] = (
                    part["source_interval_end"] - part["source_interval_start"]
                )
                part["source_crop_left"] = left
                part["source_crop_right"] = right
                temporary_part = part_path.with_suffix(".parquet.tmp")
                part.to_parquet(temporary_part, index=False)
                temporary_part.replace(part_path)
                part_receipt["format"] = _INDEX_PART_FORMAT
                temporary_receipt = part_receipt_path.with_suffix(".json.tmp")
                temporary_receipt.write_text(
                    json.dumps(part_receipt, indent=2, sort_keys=True) + "\n"
                )
                temporary_receipt.replace(part_receipt_path)
                action = "upgraded"
            else:
                action = "resumed"
            rows.extend(part.to_dict(orient="records"))
            unmatched.extend([("checkpoint", 0, 0)] * int(part_receipt["unmatched"]))
            excluded_by_policy.extend(
                [("checkpoint", 0, 0)]
                * int(part_receipt.get("excluded_by_policy", 0))
            )
            print(
                f"{action} {split} source index shard {source.shard}/"
                f"{source.shard_count}: {len(part)} records",
                flush=True,
            )
            continue
        shard_rows: list[dict] = []
        shard_unmatched: list[tuple[str, int, int]] = []
        shard_excluded: list[tuple[str, int, int]] = []
        verify_object_current(source)
        for record_index, record in enumerate(
            iter_records(source.gs_uri, "SPLICE_SITE_POSITIONS")
        ):
            target_start, target_end = _target_span(record.start, record.end)
            input_start, input_end = _model_input_span(record.start, record.end)
            key = (record.chromosome, target_start, target_end)
            base = lookup.get(key)
            if base is None:
                shard_unmatched.append(key)
                continue
            if (
                int(base["input_start"]) != input_start
                or int(base["input_end"]) != input_end
            ):
                raise ContractError(
                    f"Cropped official source input differs from the coordinate "
                    f"manifest for {key}: {input_start}-{input_end} from source "
                    f"{record.start}-{record.end} vs "
                    f"{base['input_start']}-{base['input_end']}"
                )
            if bool(base.get("excluded_context_hits_heldout_target", False)):
                shard_excluded.append(key)
                continue
            row = dict(base)
            row.update(
                {
                    "source_interval_start": record.start,
                    "source_interval_end": record.end,
                    "source_context_bp": record.end - record.start,
                    "source_crop_left": input_start - record.start,
                    "source_crop_right": record.end - input_end,
                    "source_shard": source.shard,
                    "source_shard_count": source.shard_count,
                    "source_record_index": record_index,
                    "source_positions_object": source.name,
                    "source_positions_generation": source.generation,
                    "dna_serialized_sha256": record.dna_serialized_sha256,
                }
            )
            shard_rows.append(row)
        verify_object_current(source)
        part = pd.DataFrame(shard_rows)
        temporary_part = part_path.with_suffix(".parquet.tmp")
        part.to_parquet(temporary_part, index=False)
        temporary_part.replace(part_path)
        part_receipt = {
            "format": _INDEX_PART_FORMAT,
            "split": split,
            "shard": source.shard,
            "shard_count": source.shard_count,
            "source": {
                "name": source.name,
                "generation": source.generation,
                "size": source.size,
            },
            "matched": len(shard_rows),
            "unmatched": len(shard_unmatched),
            "excluded_by_policy": len(shard_excluded),
        }
        temporary_receipt = part_receipt_path.with_suffix(".json.tmp")
        temporary_receipt.write_text(
            json.dumps(part_receipt, indent=2, sort_keys=True) + "\n"
        )
        temporary_receipt.replace(part_receipt_path)
        rows.extend(shard_rows)
        unmatched.extend(shard_unmatched)
        excluded_by_policy.extend(shard_excluded)
        print(
            f"completed {split} source index shard {source.shard}/"
            f"{source.shard_count}: {len(shard_rows)} matched, "
            f"{len(shard_excluded)} excluded, {len(shard_unmatched)} unmatched",
            flush=True,
        )
    source_frame = pd.DataFrame(rows)
    if source_frame.empty:
        raise ContractError(f"No {split} source records matched the official coordinate manifest")
    if source_frame["example_id"].duplicated().any():
        duplicates = source_frame.loc[
            source_frame["example_id"].duplicated(), "example_id"
        ].head().tolist()
        raise ContractError(f"Duplicate source-backed {split} examples: {duplicates}")
    source_frame = source_frame.sort_values(
        ["source_shard", "source_record_index"]
    ).reset_index(drop=True)
    selected = stratified_select(source_frame, SPLIT_BUDGETS[split]).reset_index(drop=True)
    split_dir = data_dir / "splits"
    source_frame.to_parquet(split_dir / f"{split}_source_full.parquet", index=False)
    selected.to_parquet(split_dir / f"{split}_selected.parquet", index=False)
    report = {
        "split": split,
        "coordinate_candidates": len(coordinate),
        "source_backed_records": len(source_frame),
        "selected_records": len(selected),
        "unmatched_source_records": len(unmatched),
        "source_records_excluded_by_policy": len(excluded_by_policy),
        "source_shards": len(shard_sets),
    }
    report_path = data_dir / "metadata" / f"source_index_{split}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return source_frame, selected, report


def index_source(data_dir: Path, splits: list[str]) -> dict[str, object]:
    reports = {}
    for split in splits:
        _, _, reports[split] = index_source_split(data_dir, split)
    for report_path in (data_dir / "metadata").glob("source_index_*.json"):
        recovered = json.loads(report_path.read_text())
        recovered_split = recovered.get("split")
        if recovered_split in {"train", "valid", "test"}:
            reports.setdefault(recovered_split, recovered)
    manifest_path = data_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_index"] = manifest.get("source_index", {}) | reports
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    receipt_path = data_dir / "metadata" / "source_index_report.json"
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    merged = previous.get("splits", {}) | reports
    receipt = {"status": "PASS", "splits": merged}
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
