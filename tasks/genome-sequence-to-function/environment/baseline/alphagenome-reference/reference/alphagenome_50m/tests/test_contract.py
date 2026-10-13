import numpy as np
import pandas as pd
import torch
from pathlib import Path
import zarr

from reference.alphagenome_50m.dataset import AlphaGenomeDataset, DENSE_FAMILIES, normalize_splice_positions
from reference.alphagenome_50m.train_eval import _match_coordinate_queries, _model_inputs, _resume_position


def test_training_matches_the_complete_release_scoring_part():
    positions = torch.full((4, 512), -1, dtype=torch.int64)
    positions[0, 17], positions[1, 19] = 1017, 2019
    count = 8_193
    scoring = {
        "donor_position": torch.full((count,), -1, dtype=torch.int64),
        "acceptor_position": torch.full((count,), -1, dtype=torch.int64),
        "track": torch.zeros(count, dtype=torch.int64),
        "target_value": torch.zeros(count),
        "valid": torch.zeros(count, dtype=torch.bool),
    }
    scoring["donor_position"][-1] = 1017
    scoring["acceptor_position"][-1] = 2019
    scoring["target_value"][-1] = 7
    scoring["valid"][-1] = True
    queries, valid, missing = _match_coordinate_queries(
        scoring, positions, torch.zeros(734, dtype=torch.int64)
    )
    assert queries.shape == (1, count, 3)
    assert queries[0, -1].tolist() == [17, 19, 0]
    assert valid.sum().item() == 1
    assert not missing[-1]
    inputs, target, training_valid = _model_inputs({
        "dna": torch.zeros(8, 4),
        "usage_positions": torch.tensor([1]),
        "site_positions": positions,
        "junction_track_halves": torch.zeros(734, dtype=torch.int64),
        "junction_scoring": scoring,
    })
    assert inputs["junction_queries"].shape == (1, count, 3)
    assert training_valid.sum().item() == 1
    assert torch.isclose(target[0, -1], torch.asinh(torch.tensor(7.0)))


def test_nested_arrow_splice_positions_are_normalized():
    nested = np.empty(4, dtype=object)
    for index in range(4):
        nested[index] = np.arange(512, dtype=np.int32) + index
    observed = normalize_splice_positions(nested)
    assert observed.shape == (4, 512)
    assert observed.dtype == np.int64


def test_inference_dataset_needs_no_labels_or_splice_positions(tmp_path):
    public = tmp_path / "public"
    public.mkdir(parents=True)
    pd.DataFrame([{
        "example_id": "example", "split": "test", "chromosome": "chr1",
        "input_start": 0, "input_end": 8, "input_pad_left": 0,
        "input_pad_right": 0, "target_start": 2, "target_end": 6,
    }]).to_parquet(public / "test.parquet", index=False)
    pd.DataFrame({
        "tensor_index": np.arange(734),
        "junction_channel_half": np.repeat([0, 1], 367),
    }).to_parquet(public / "junction_tracks.parquet", index=False)
    dataset = AlphaGenomeDataset(tmp_path, "test", include_targets=False)
    dataset._sequence = lambda row: (np.zeros((8, 4), np.float32), np.ones(8, bool))
    sample = dataset[0]
    assert set(sample) == {"example_id", "dna", "input_mask", "junction_track_halves"}


def test_training_dataset_reads_scoring_part_at_item_index(tmp_path):
    (tmp_path / "public").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "scoring/train").mkdir(parents=True)
    positions = np.full((4, 512), -1, dtype=np.int32)
    positions[0, 0], positions[1, 0] = 10, 20
    public = pd.DataFrame([{
        "example_id": "example", "split": "train", "chromosome": "chr1",
        "input_start": 0, "input_end": 8, "input_pad_left": 0,
        "input_pad_right": 0, "target_start": 2, "target_end": 6,
    }])
    public.to_parquet(tmp_path / "public/train.parquet", index=False)
    pd.DataFrame({
        "tensor_index": np.arange(734),
        "junction_channel_half": np.zeros(734, dtype=np.int8),
    }).to_parquet(tmp_path / "public/junction_tracks.parquet", index=False)
    pd.DataFrame([{
        "example_id": "example", "target_shard": "targets/train/part.zarr",
        "target_group": "examples/example", "splice_site_positions": positions.tolist(),
    }]).to_parquet(tmp_path / "labels/train.parquet", index=False)
    pd.DataFrame([{
        "example_id": "example", "donor_position": 10, "acceptor_position": 20,
        "track": 0, "target_value": 3.0, "valid": True,
    }]).to_parquet(tmp_path / "scoring/train/part-000000.parquet", index=False)
    group = zarr.open_group(str(tmp_path / "targets/train/part.zarr"), mode="w")
    example = group.create_group("examples/example")
    for family in DENSE_FAMILIES:
        example.create_dataset(f"{family}/values", data=np.zeros((1, 1), np.float32))
        example.create_dataset(f"{family}/mask", data=np.ones((1, 1), bool))
    example.create_dataset("splice_sites/values", data=np.zeros((4, 4), np.float32))
    example.create_dataset("splice_site_usage/positions", data=np.array([0], np.int32))
    example.create_dataset("splice_site_usage/values", data=np.zeros((1, 1), np.float32))
    example.create_dataset("splice_site_usage/mask", data=np.ones((1, 1), bool))
    for name, data in {
        "row": np.array([0], np.int32), "column": np.array([0], np.int32),
        "track": np.array([0], np.int32), "value": np.array([3], np.float32),
        "site_positions": positions,
    }.items():
        example.create_dataset(f"splice_junctions/{name}", data=data)
    example.create_dataset("contact_maps/upper_triangle", data=np.zeros((1, 1), np.float32))
    example.create_dataset("contact_maps/mask", data=np.ones((1, 1), bool))
    dataset = AlphaGenomeDataset(tmp_path, "train", include_targets=True)
    dataset._sequence = lambda row: (np.zeros((8, 4), np.float32), np.ones(8, bool))
    sample = dataset[0]
    assert sample["junction_scoring"]["target_value"].tolist() == [3.0]


def test_resume_accepts_only_complete_epoch_checkpoints():
    assert _resume_position({"epoch": 2, "global_step": 20,
                             "checkpoint_kind": "epoch", "examples": 10,
                             "schedule_epochs": 5}, examples=10, epochs=5) == (2, 20)
    for payload in ({"epoch": 1, "global_step": 15},
                    {"epoch": 1, "global_step": 10, "checkpoint_kind": "progress"}):
        try:
            _resume_position(payload, examples=10, epochs=5)
        except ValueError:
            pass
        else:
            raise AssertionError("partial checkpoint was accepted for resume")


def test_ddp_runner_works_offline_and_scores_with_the_verifier():
    runner = Path(__file__).parents[1] / "run_ddp.sh"
    text = runner.read_text()
    command = next(line for line in text.splitlines() if line.startswith("torchrun "))
    assert "--master-addr 127.0.0.1" in command and "--standalone" not in command
    assert "/workspace/evaluation/score.py" in text and "--split valid" in text
