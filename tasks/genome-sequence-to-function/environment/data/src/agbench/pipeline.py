"""Bootstrap and real-source audit entry points for benchmark protocol."""

from __future__ import annotations

import json
from pathlib import Path

from .constants import (
    FOLD_BED_SHA256,
    FOLD_BED_URL,
    OUTPUT_METADATA_SHA256,
    OUTPUT_METADATA_URL,
)
from .downloads import download_verified
from .errors import ContractError
from .reference import fetch_fai
from .protocol import BUNDLES, DATASET_SCHEMA
from .source_shards import aligned_shards, build_inventory
from .manifests import write_manifests
from .tfrecords import iter_records


def bootstrap(data_dir: Path) -> dict[str, object]:
    shared = data_dir / "metadata"
    shared.mkdir(parents=True, exist_ok=True)
    fold_bed = download_verified(
        FOLD_BED_URL,
        shared / "sequences_human.bed.gz",
        expected_sha256=FOLD_BED_SHA256,
    )
    textproto = download_verified(
        OUTPUT_METADATA_URL,
        shared / "human_output_metadata.textproto",
        expected_sha256=OUTPUT_METADATA_SHA256,
    )
    fai = fetch_fai(data_dir)
    return write_manifests(data_dir, fold_bed=fold_bed, fai=fai, textproto=textproto)


def inspect(data_dir: Path, splits: list[str], *, probe: bool) -> dict[str, object]:
    inventory = build_inventory(splits)
    probes: list[dict[str, object]] = []
    alignments: list[dict[str, object]] = []
    if probe:
        for split in splits:
            shard_set = aligned_shards(split)[0]
            expected_coordinate = None
            expected_dna_hash = None
            for bundle, source in shard_set.items():
                record = next(iter(iter_records(source.gs_uri, bundle, decode_dna=False)))
                if expected_coordinate is None:
                    expected_coordinate = record.coordinate
                    expected_dna_hash = record.dna_serialized_sha256
                if record.coordinate != expected_coordinate:
                    raise ContractError(
                        f"First-record coordinate mismatch in {split}: "
                        f"{bundle}={record.coordinate}, expected={expected_coordinate}"
                    )
                if record.dna_serialized_sha256 != expected_dna_hash:
                    raise ContractError(f"First-record DNA mismatch in {split}: {bundle}")
                probes.append(
                    {
                        "split": split,
                        "bundle": bundle,
                        "uri": source.gs_uri,
                        "generation": source.generation,
                        "interval": list(record.coordinate),
                        "fields": list(record.feature_names),
                        "tensors": {
                            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                            for name, value in record.tensors.items()
                        },
                    }
                )
                del record
            alignments.append(
                {
                    "split": split,
                    "shard": next(iter(shard_set.values())).shard,
                    "interval": list(expected_coordinate or ()),
                    "bundle_count": len(shard_set),
                    "dna_serialized_sha256": expected_dna_hash,
                }
            )
    report = {
        "schema": DATASET_SCHEMA,
        "status": "PASS",
        "bundles": list(BUNDLES),
        "inventory": inventory,
        "record_probes": probes,
        "alignment_probes": alignments,
    }
    path = data_dir / "metadata" / "upstream_inspection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
