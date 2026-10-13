"""Prediction contract v4: sparse splice usage and junctions, shared by grader and agent."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment" / "evaluation"))
from scoring import (FIXED_SHAPES, MAX_JUNCTION_PAIRS, PerTrackMoments,  # noqa: E402
                     _junction_prediction, _usage_prediction, validate_example)


def test_oracle_moments_score_one():
    target = np.array([[0, 2], [1, 1], [2, 0]], np.float32)
    moments = PerTrackMoments(2)
    moments.update(target, target)
    assert np.isclose(moments.report()["mean_pearson"], 1.0)


def _example(tmp_path: Path, pairs=((0, 104, 209),), usage_sites=(5, 17)):
    group = zarr.open_group(str(tmp_path / "p.zarr"), mode="w")
    for name, shape in FIXED_SHAPES.items():
        group.create_dataset(name, shape=shape, dtype="f2", fill_value=0, chunks=(4096, 64))
    group.create_dataset("splice_usage_positions", data=np.asarray(usage_sites, np.int32))
    usage = np.zeros((len(usage_sites), 734), np.float16); usage[1, 3] = 2
    group.create_dataset("splice_usage", data=usage)
    half, donor, acceptor = (np.asarray(v) for v in zip(*pairs)) if pairs else ([], [], [])
    group.create_dataset("junction_half", data=np.asarray(half, np.int8))
    group.create_dataset("junction_donor", data=np.asarray(donor, np.int32))
    group.create_dataset("junction_acceptor", data=np.asarray(acceptor, np.int32))
    values = np.zeros((len(pairs), 734), np.float16)
    if len(pairs):
        values[0, 3] = 7
    group.create_dataset("junctions", data=values)
    return group


def test_usage_at_unlisted_hidden_site_is_a_zero_prediction(tmp_path: Path):
    group = _example(tmp_path)
    values, missing = _usage_prediction(group, np.array([17, 9]))
    assert values[0, 3] == 2 and not values[1].any()
    assert missing.tolist() == [False, True]


def test_junction_query_matches_strand_half_and_both_endpoints(tmp_path: Path):
    group = _example(tmp_path)
    halves = np.repeat([0, 1], 367)  # track 3 is a plus-strand (half 0) track
    queries = pd.DataFrame({"donor_position": [104, 104, 105], "acceptor_position": [209, 209, 209],
                            "track": [3, 400, 3]})
    values, missing = _junction_prediction(group, queries, halves)
    assert values.tolist() == [7.0, 0.0, 0.0]
    assert missing.tolist() == [False, True, True]


def test_validation_accepts_the_contract_and_rejects_bad_keys(tmp_path: Path):
    validate_example(_example(tmp_path), "ok")
    with pytest.raises(ValueError, match="unique"):
        validate_example(_example(tmp_path / "a", pairs=((0, 1, 2), (0, 1, 2))), "dup")
    with pytest.raises(ValueError, match="sorted"):
        validate_example(_example(tmp_path / "b", usage_sites=(17, 5)), "order")
    with pytest.raises(ValueError, match="out of range"):
        validate_example(_example(tmp_path / "c", pairs=((2, 1, 2),)), "half")
    group = _example(tmp_path / "d", pairs=())
    validate_example(group, "empty")  # a model may list no junction pairs at all
    assert MAX_JUNCTION_PAIRS == 16_384
