"""Source-aligned coordinate, split, and track manifests for benchmark protocol."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from .errors import ContractError
from .metadata import parse_all_track_tables
from .protocol import (
    BUNDLES,
    INPUT_BP,
    DATASET_SCHEMA,
    SELECTION_SEED,
    SPLIT_BUDGETS,
)


def _stable_hash(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def _read_fai(path: Path) -> dict[str, int]:
    result = {}
    for line in path.read_text().splitlines():
        fields = line.split("\t")
        if len(fields) >= 2:
            result[fields[0]] = int(fields[1])
    if not result:
        raise ContractError(f"Empty reference index: {path}")
    return result


def build_full_manifest(fold_bed: Path, fai: Path) -> pd.DataFrame:
    chrom_lengths = _read_fai(fai)
    rows: list[dict[str, object]] = []
    with gzip.open(fold_bed, "rt") as handle:
        for source_row, line in enumerate(handle):
            fields = line.rstrip().split("\t")
            if len(fields) < 4:
                raise ContractError(f"Malformed fold BED row {source_row + 1}")
            chrom, start_text, end_text, fold = fields[:4]
            start, end = int(start_text), int(end_text)
            if chrom not in chrom_lengths or start < 0 or end <= start:
                raise ContractError(f"Invalid fold BED interval: {fields[:4]}")
            center = (start + end) // 2
            context_start = center - INPUT_BP // 2
            context_end = context_start + INPUT_BP
            split = "valid" if fold == "fold0" else "test" if fold == "fold1" else "train"
            example_id = "agv2-" + _stable_hash(chrom, start, end, fold)[:24]
            rows.append(
                {
                    "example_id": example_id,
                    "split": split,
                    "source_fold": fold,
                    "source_row": source_row,
                    "chromosome": chrom,
                    "anchor": center,
                    "input_start": context_start,
                    "input_end": context_end,
                    "input_pad_left": max(0, -context_start),
                    "input_pad_right": max(0, context_end - chrom_lengths[chrom]),
                    "target_start": start,
                    "target_end": end,
                    "selection_hash": _stable_hash(SELECTION_SEED, split, example_id),
                }
            )
    frame = pd.DataFrame(rows)
    if set(frame["source_fold"]) != {f"fold{i}" for i in range(8)}:
        raise ContractError("Fold BED does not contain fold0 through fold7")
    if frame["example_id"].duplicated().any():
        raise ContractError("Coordinate manifest contains duplicate example IDs")
    return frame


def _overlaps_any(frame: pd.DataFrame, heldout: pd.DataFrame) -> pd.Series:
    by_chrom = {
        chrom: sorted(zip(group.target_start.astype(int), group.target_end.astype(int)))
        for chrom, group in heldout.groupby("chromosome", sort=False)
    }
    flags = []
    for row in frame.itertuples(index=False):
        hit = False
        for start, end in by_chrom.get(row.chromosome, []):
            if start >= row.input_end:
                break
            if end > row.input_start:
                hit = True
                break
        flags.append(hit)
    return pd.Series(flags, index=frame.index, dtype=bool)


def apply_train_exclusion(full: pd.DataFrame) -> pd.DataFrame:
    heldout = full.loc[full["split"].isin(["valid", "test"])]
    train = full.loc[full["split"] == "train"].copy()
    train["excluded_context_hits_heldout_target"] = _overlaps_any(train, heldout)
    return train


def stratified_select(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if count >= len(frame):
        return frame.sort_values("selection_hash").copy()
    sizes = frame.groupby("chromosome", sort=True).size()
    raw = sizes * count / len(frame)
    quotas = raw.map(math.floor).astype(int)
    remainder = count - int(quotas.sum())
    order = sorted(sizes.index, key=lambda c: (-(raw[c] - quotas[c]), c))
    for chrom in order[:remainder]:
        quotas[chrom] += 1
    parts = []
    for chrom, quota in quotas.items():
        if quota:
            group = frame.loc[frame["chromosome"] == chrom]
            parts.append(group.sort_values("selection_hash").head(int(quota)))
    selected = pd.concat(parts, ignore_index=False).sort_values("selection_hash")
    if len(selected) != count:
        raise ContractError(f"Stratified selection produced {len(selected)} rows, expected {count}")
    return selected.copy()


def select_splits(full: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    train_audit = apply_train_exclusion(full)
    eligible = train_audit.loc[~train_audit["excluded_context_hits_heldout_target"]].copy()
    selected = {
        "train": stratified_select(eligible, SPLIT_BUDGETS["train"]),
        "valid": stratified_select(full.loc[full["split"] == "valid"], SPLIT_BUDGETS["valid"]),
        "test": stratified_select(full.loc[full["split"] == "test"], SPLIT_BUDGETS["test"]),
    }
    audit = {
        "train_candidates": len(train_audit),
        "train_excluded_context_hits_heldout_target": int(train_audit["excluded_context_hits_heldout_target"].sum()),
        "train_eligible": len(eligible),
        **{f"{split}_selected": len(frame) for split, frame in selected.items()},
    }
    return selected, audit


def build_track_manifest(textproto: Path) -> pd.DataFrame:
    tables = parse_all_track_tables(textproto.read_text())
    parts = []
    global_index = 0
    for bundle, spec in BUNDLES.items():
        if spec.family is None:
            continue
        try:
            frame = tables[spec.family].copy()
        except KeyError as exc:
            raise ContractError(f"Missing output metadata for {spec.family}") from exc
        # Junction metadata describes tissues while the tensor has two stranded
        # channels per tissue. Preserve the source metadata and expand explicitly.
        if spec.family == "SPLICE_JUNCTIONS" and len(frame) * 2 == spec.real_tracks:
            frame = pd.concat(
                [frame.assign(junction_channel_half=half) for half in (0, 1)],
                ignore_index=True,
            )
            frame["track_index"] = range(len(frame))
            frame["tensor_index"] = range(len(frame))
        else:
            frame["tensor_index"] = frame["source_index"].astype(int)
        if len(frame) != spec.real_tracks:
            raise ContractError(
                f"{spec.family} has {len(frame)} real metadata tracks; expected {spec.real_tracks}"
            )
        frame.insert(0, "global_track_index", range(global_index, global_index + len(frame)))
        frame.insert(1, "track_id", [f"{spec.family.lower()}:{i}" for i in range(len(frame))])
        frame["family"] = spec.family
        frame["bundle"] = bundle
        frame["representation"] = spec.representation
        frame["native_resolution_bp"] = spec.native_resolution
        frame["stored_resolution_bp"] = spec.stored_resolution
        global_index += len(frame)
        parts.append(frame)
    result = pd.concat(parts, ignore_index=True)
    if result["track_id"].duplicated().any():
        raise ContractError("Track IDs are not unique")
    return result


def write_manifests(
    data_dir: Path, *, fold_bed: Path, fai: Path, textproto: Path
) -> dict[str, object]:
    root = data_dir
    split_dir = root / "splits"
    metadata_dir = root / "metadata"
    split_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    full = build_full_manifest(fold_bed, fai)
    selected, audit = select_splits(full)
    tracks = build_track_manifest(textproto)
    for split in ("valid", "test"):
        full.loc[full["split"] == split].to_parquet(split_dir / f"{split}_full.parquet", index=False)
    eligible = apply_train_exclusion(full)
    eligible.to_parquet(split_dir / "train_eligible.parquet", index=False)
    for split, frame in selected.items():
        frame.to_parquet(split_dir / f"{split}_selected.parquet", index=False)
    tracks.to_parquet(metadata_dir / "tracks.parquet", index=False)
    manifest = {
        "schema": DATASET_SCHEMA,
        "input_bp": INPUT_BP,
        "selection_seed": SELECTION_SEED,
        "split_audit": audit,
        "track_count": len(tracks),
        "track_counts": tracks.groupby("family").size().astype(int).to_dict(),
    }
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
