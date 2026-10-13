from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from agbench.splicing import junction_coordinates


def _common_module():
    # The metric moved to environment/evaluation/scoring.py (shared by grader and agent).
    path = Path(__file__).resolve().parents[1] / "environment" / "evaluation" / "scoring.py"
    spec = importlib.util.spec_from_file_location("verifier_scoring", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_junction_coordinates_do_not_mix_strands():
    positions = np.full((4, 512), -1, np.int32)
    positions[0, 3], positions[1, 7] = 103, 207
    positions[2, 3], positions[3, 7] = 303, 407
    donor, acceptor, valid = junction_coordinates(
        positions, np.array([3, 3]), np.array([7, 7]), np.array([0, 1])
    )
    assert donor.tolist() == [103, 303]
    assert acceptor.tolist() == [207, 407]
    assert valid.tolist() == [True, True]
    positions[3, 7] = -1
    assert junction_coordinates(positions, 3, 7, 1)[2].item() is False


def test_constant_prediction_is_zero_not_a_dropped_track():
    moments = _common_module().PerTrackMoments(1)
    moments.update(np.ones((4, 1)), np.arange(4, dtype=np.float32)[:, None])
    report = moments.report()
    assert report["valid_tracks"] == 1
    assert report["mean_pearson"] == 0.0
    assert report["per_track_pearson"] == [0.0]


def test_unvarying_target_is_the_only_reason_to_drop_track():
    moments = _common_module().PerTrackMoments(2)
    moments.update(
        np.array([[0, 0], [0, 1], [0, 2]], np.float32),
        np.array([[1, 0], [1, 1], [1, 2]], np.float32),
    )
    report = moments.report()
    assert report["valid_tracks"] == 1
    assert report["per_track_pearson"][0] is None
    assert report["per_track_pearson"][1] == 1.0
