from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr

from agbench.errors import ContractError
from agbench import release as release_module
from agbench.release import (
    JUNCTION_QUERY_BUDGET,
    LEGACY_JUNCTION_QUERY_COUNT,
    PUBLIC_COLUMNS,
    export_release,
    migrate_layout,
    _store_hash,
)


def test_release_export_reuses_targets_and_keeps_coordinates_trusted(tmp_path: Path):
    (tmp_path / "metadata").mkdir(parents=True)
    (tmp_path / "splits").mkdir(parents=True)
    tracks = pd.DataFrame({
        "family": ["SPLICE_JUNCTIONS"] * 734,
        "tensor_index": np.arange(734),
        "junction_channel_half": np.repeat([0, 1], 367),
    })
    tracks.to_parquet(tmp_path / "metadata/tracks.parquet", index=False)
    target_path = tmp_path / "targets/train/part.zarr"
    target = zarr.open_group(str(target_path), mode="w").create_group("examples/example-1")
    positions = np.stack([np.arange(512, dtype=np.int32) + offset
                          for offset in (1_000, 2_000, 3_000, 4_000)])
    junction = target.create_group("splice_junctions")
    junction.create_dataset("site_positions", data=positions)
    junction.create_dataset("row", data=np.array([2], np.uint16))
    junction.create_dataset("column", data=np.array([3], np.uint16))
    junction.create_dataset("track", data=np.array([0], np.uint16))
    junction.create_dataset("value", data=np.array([5], np.float32))
    row = {
        "example_id": "example-1", "split": "train", "chromosome": "chr1",
        "input_start": 10, "input_end": 1_048_586, "input_pad_left": 0,
        "input_pad_right": 0, "target_start": 425_994, "target_end": 622_602,
        "target_shard": "targets/train/part.zarr",
        "target_group": "examples/example-1", "splice_site_positions": positions.tolist(),
        "source_row": 99,
    }
    pd.DataFrame([row]).to_parquet(tmp_path / "splits/train_materialized.parquet", index=False)

    release = export_release(tmp_path, ["train"])

    public = pd.read_parquet(tmp_path / "public/train.parquet")
    labels = pd.read_parquet(tmp_path / "labels/train.parquet")
    scoring = pd.read_parquet(tmp_path / "scoring/train/part-000000.parquet")
    assert tuple(public.columns) == PUBLIC_COLUMNS
    assert "splice_site_positions" not in public
    assert "source_row" not in public
    assert labels.example_id.tolist() == public.example_id.tolist() == ["example-1"]
    assert release["target_storage_schema"].endswith("/v2")
    assert release["legacy_junction_query_count"] == LEGACY_JUNCTION_QUERY_COUNT
    assert release["junction_query_budget"] == JUNCTION_QUERY_BUDGET
    assert len(scoring) == LEGACY_JUNCTION_QUERY_COUNT + 1
    assert (scoring.target_value == 5).any()
    assert target_path.exists()
    positive = scoring.loc[scoring.track < 367]
    negative = scoring.loc[scoring.track >= 367]
    assert positive.donor_position.between(1_000, 1_511).all()
    assert positive.acceptor_position.between(2_000, 2_511).all()
    assert negative.donor_position.between(3_000, 3_511).all()
    assert negative.acceptor_position.between(4_000, 4_511).all()


def test_legacy_layout_migration_preserves_target_hashes_and_rewrites_manifests(
    tmp_path: Path,
):
    target = tmp_path / "v2/targets/train/part.zarr.zip"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"immutable-target")
    (tmp_path / "v2/splits").mkdir(parents=True)
    (tmp_path / "v3/labels").mkdir(parents=True)
    rows = pd.DataFrame({
        "example_id": ["example"],
        "target_shard": ["v2/targets/train/part.zarr.zip"],
    })
    rows.to_parquet(tmp_path / "v2/splits/train_materialized.parquet", index=False)
    rows.to_parquet(tmp_path / "v3/labels/train.parquet", index=False)
    (tmp_path / "v2/zipstore_conversion.json").write_text(json.dumps({
        "artifacts": [{
            "path": "v2/targets/train/part.zarr.zip",
            "source": "v2/targets/train/part.zarr",
            "sha256": "example",
        }]
    }))

    report = migrate_layout(tmp_path)

    assert report["status"] == "PASS"
    assert report["target_files"] == 1
    assert (tmp_path / "targets/train/part.zarr.zip").read_bytes() == b"immutable-target"
    for path in (
        tmp_path / "splits/train_materialized.parquet",
        tmp_path / "labels/train.parquet",
    ):
        assert pd.read_parquet(path).target_shard.tolist() == [
            "targets/train/part.zarr.zip"
        ]
    assert not (tmp_path / "v2").exists()
    assert not (tmp_path / "v3").exists()
    artifact = json.loads((tmp_path / "zipstore_conversion.json").read_text())["artifacts"][0]
    assert artifact["path"] == "targets/train/part.zarr.zip"
    assert artifact["source"] == "targets/train/part.zarr"


def test_migration_resumes_after_targets_move(tmp_path: Path, monkeypatch):
    target = tmp_path / "v2/targets/train/part.zarr.zip"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"unchanged-target")
    manifests = tmp_path / "v2/splits"
    manifests.mkdir(parents=True)
    pd.DataFrame({
        "example_id": ["one"],
        "target_shard": ["v2/targets/train/part.zarr.zip"],
    }).to_parquet(manifests / "train_materialized.parquet", index=False)
    original = release_module._promote_targets

    def interrupt(source, destination):
        original(source, destination)
        raise RuntimeError("injected interruption after target move")

    with monkeypatch.context() as patcher:
        patcher.setattr(release_module, "_promote_targets", interrupt)
        with pytest.raises(RuntimeError, match="injected interruption"):
            migrate_layout(tmp_path)
    assert not target.exists()
    assert (tmp_path / ".layout_migration_targets.json").exists()
    report = migrate_layout(tmp_path)
    assert report["status"] == "PASS"
    assert (tmp_path / "targets/train/part.zarr.zip").read_bytes() == b"unchanged-target"
    assert not (tmp_path / ".layout_migration_targets.json").exists()


def test_missing_target_store_is_an_error(tmp_path: Path):
    with pytest.raises(ContractError, match="missing target store"):
        _store_hash(tmp_path / "absent.zarr")
