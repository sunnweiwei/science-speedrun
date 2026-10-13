"""The verifier must map the raw reward onto this track's 0-1 reference scale."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import main_verifier as mv  # noqa: E402

BOUNDS = ROOT / "environment" / "evaluation" / "score_bounds.json"


def test_bounds_file_is_well_formed():
    b = json.loads(BOUNDS.read_text())
    assert b["schema"] == "benchmark-score-bounds/v1"
    assert b["upper"]["reward"] > b["lower"]["reward"]
    for side in ("lower", "upper"):
        assert b[side]["what"] and b[side]["source"]


def test_floor_is_zero_and_ceiling_is_one():
    b = json.loads(BOUNDS.read_text())
    assert mv.normalised(b["lower"]["reward"])["normalised_score"] == pytest.approx(0.0, abs=1e-12)
    assert mv.normalised(b["upper"]["reward"])["normalised_score"] == pytest.approx(1.0, abs=1e-12)


def test_below_the_floor_and_above_the_ceiling_are_clipped():
    assert mv.normalised(0.0)["normalised_score"] == 0.0
    assert mv.normalised(1.0)["normalised_score"] == 1.0


def test_measured_systems_land_where_expected():
    """The replayed checkpoints, against the numbers the README reports."""
    for reward, want in ((0.1530, 0.298), (0.3254, 0.652), (0.4121, 0.830),
                         (0.4180, 0.842), (0.4465, 0.901)):
        assert mv.normalised(reward)["normalised_score"] == pytest.approx(want, abs=2e-3)


def test_this_tracks_ceiling_is_reachable():
    """Unlike pde-surrogate, the ceiling here is a real system a submission can match."""
    assert json.loads(BOUNDS.read_text())["upper"]["achievable_by_a_submission"] is True


def test_inverted_bounds_are_rejected(tmp_path, monkeypatch):
    bad = tmp_path / "score_bounds.json"
    bad.write_text(json.dumps({"lower": {"reward": 0.9}, "upper": {"reward": 0.1}}))
    monkeypatch.setattr(mv, "BOUNDS", bad)
    with pytest.raises(ValueError):
        mv.normalised(0.5)


def test_a_missing_bounds_file_still_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(mv, "BOUNDS", tmp_path / "absent.json")
    assert mv.normalised(0.5) is None
