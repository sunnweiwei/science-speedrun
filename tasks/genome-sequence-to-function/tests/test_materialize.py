from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
import zarr

from agbench import materialize as materialize_module
from agbench import release
from agbench.errors import ContractError
from agbench.materialize import (
    _track_indices,
    _valid_splice_positions,
    _write_contact_maps,
    _write_dense,
    _write_splice_sites,
    _write_splice_usage,
    _public_site_positions,
)
from agbench.tfrecords import SourceRecord
from agbench.validate import _referenced_target_size


def test_track_indices_use_tensor_mapping():
    tracks = pd.DataFrame(
        {
            "family": ["ATAC", "ATAC"],
            "track_index": [0, 1],
            "tensor_index": [3, 7],
        }
    )
    assert _track_indices(tracks, "ATAC").tolist() == [3, 7]


def test_dense_1bp_sum_pool_and_track_mask(tmp_path):
    values = np.ones((196_608, 3), dtype=np.float32)
    mask = np.array([[True, False, True]], dtype=bool)
    record = SourceRecord(
        chromosome="chr1",
        start=0,
        end=1_048_576,
        tensors={"atac": values, "atac_mask": mask},
        dna=None,
        dna_serialized_sha256="x",
        feature_names=(),
    )
    root = zarr.open_group(str(tmp_path / "target.zarr"), mode="w")
    report = _write_dense(root, "ATAC", record, np.array([0, 1, 2]))
    assert report == {"shape": [1536, 3], "mask_shape": [3]}
    assert np.all(root["atac/values"][:, 0] == 128)
    assert root["atac/mask"][:].tolist() == [True, False, True]


def test_valid_splice_positions_are_unique_and_in_target():
    raw = np.array([[4, 4, -1], [196_607, 196_608, -1]], dtype=np.int32)
    assert _valid_splice_positions(raw, 196_608).tolist() == [4, 196_607]


def test_contact_nonfinite_values_are_masked(tmp_path):
    values = np.zeros((512, 512, 1), dtype=np.float32)
    values[0, 0, 0] = np.nan
    record = SourceRecord(
        chromosome="chr1",
        start=0,
        end=1_048_576,
        tensors={"contact_maps": values},
        dna=None,
        dna_serialized_sha256="x",
        feature_names=(),
    )
    root = zarr.open_group(str(tmp_path / "contact.zarr"), mode="w")
    _write_contact_maps(root, record, np.array([0]))
    assert not root["contact_maps/mask"][0, 0]
    assert root["contact_maps/upper_triangle"][0, 0] == 0


def test_padded_train_source_crops_all_spatial_families(tmp_path):
    source_bp = 1_052_672
    target_offset = (source_bp - 196_608) // 2
    input_offset = (source_bp - 1_048_576) // 2
    raw = np.zeros((source_bp, 1), dtype=np.float32)
    raw[target_offset:target_offset + 196_608] = 2
    raw[:target_offset] = 7
    sites = np.zeros((source_bp, 1), dtype=bool)
    sites[target_offset, 0] = True
    usage = np.zeros((source_bp, 1), dtype=np.float16)
    usage[target_offset, 0] = 3
    contact = np.zeros((514, 514, 1), dtype=np.float32)
    contact[1, 1, 0] = 5
    record = SourceRecord(
        chromosome="chr1", start=0, end=source_bp,
        tensors={"atac": raw, "atac_mask": np.ones((1, 1), bool),
                 "splice_sites": sites, "splice_site_usage": usage,
                 "contact_maps": contact},
        dna=None, dna_serialized_sha256="x", feature_names=(),
    )
    root = zarr.open_group(str(tmp_path / "padded.zarr"), mode="w")
    _write_dense(root, "ATAC", record, np.array([0]), target_offset)
    _write_splice_sites(root, record, np.array([0]), target_offset)
    raw_positions = np.full((4, 512), -1, np.int32)
    raw_positions[0, 0] = target_offset
    raw_positions[0, 1] = target_offset - 1
    raw_positions[0, 2] = source_bp - 1
    _write_splice_usage(root, record, np.array([0]), raw_positions, target_offset)
    _write_contact_maps(root, record, np.array([0]), input_offset)
    assert np.all(root["atac/values"][:] == 256)
    assert root["splice_sites/values"][0, 0]
    assert root["splice_site_usage/positions"][:].tolist() == [0]
    assert root["splice_site_usage/values"][0, 0] == 3
    assert root["contact_maps/upper_triangle"][0, 0] == 5
    public = _public_site_positions(raw_positions, record, input_offset)
    assert public[0, 0] == target_offset - input_offset
    assert public[0, 1] == target_offset - input_offset - 1
    assert public[0, 2] == -1


def test_parallel_shards_have_deterministic_manifest_and_receipt(tmp_path, monkeypatch):
    splits = tmp_path / "splits"
    metadata = tmp_path / "metadata"
    splits.mkdir(parents=True)
    metadata.mkdir(parents=True)
    pd.DataFrame({"source_shard": [2, 1], "example_id": ["second", "first"],
                  "source_record_index": [0, 0]}).to_parquet(splits / "train_selected.parquet")
    pd.DataFrame({"track_id": ["x"]}).to_parquet(metadata / "tracks.parquet")
    shard_sets = [{"SPLICE_SITE_POSITIONS": SimpleNamespace(shard=i)} for i in (1, 2)]
    monkeypatch.setattr(materialize_module, "aligned_shards", lambda split: shard_sets)

    def fake_job(task):
        shard = task[2]["SPLICE_SITE_POSITIONS"].shard
        return shard, tmp_path / f"part-{shard}.zarr", [
            {"example_id": str(task[3].iloc[0].example_id), "source_shard": shard}
        ], 1024

    monkeypatch.setattr(materialize_module, "_materialize_job", fake_job)
    monkeypatch.setattr(release, "export_release", lambda *args, **kwargs: {})

    class ReversedPool:
        def __init__(self, *, processes, maxtasksperchild):
            assert processes == 2 and maxtasksperchild == 1

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def imap_unordered(self, callback, jobs, *, chunksize):
            assert chunksize == 1
            return iter([callback(job) for job in reversed(list(jobs))])

    monkeypatch.setattr(
        materialize_module.multiprocessing, "get_context",
        lambda method: SimpleNamespace(Pool=ReversedPool) if method == "spawn" else None,
    )
    receipt = materialize_module.materialize_split(tmp_path, split="train", workers=2)
    materialized = pd.read_parquet(splits / "train_materialized.parquet")
    assert materialized.example_id.tolist() == ["first", "second"]
    assert receipt["materialized_shards"] == 2
    assert receipt["workers"] == 2
    assert receipt["complete"]


def test_materialize_rejects_zero_workers(tmp_path):
    with pytest.raises(ContractError, match="at least 1"):
        materialize_module.materialize_split(tmp_path, split="train", workers=0)


def test_capacity_counts_only_manifest_referenced_target_stores(tmp_path):
    target_dir = tmp_path / "targets" / "train"
    referenced = target_dir / "part-0000.zarr.zip"
    expanded_duplicate = target_dir / "part-0000.zarr"
    referenced.parent.mkdir(parents=True)
    referenced.write_bytes(b"zip-release")
    expanded_duplicate.mkdir()
    (expanded_duplicate / "payload").write_bytes(b"expanded-working-copy")
    manifest = pd.DataFrame(
        {
            "target_shard": [
                "targets/train/part-0000.zarr.zip",
                "targets/train/part-0000.zarr.zip",
            ]
        }
    )
    assert _referenced_target_size(tmp_path, manifest) == len(b"zip-release")
