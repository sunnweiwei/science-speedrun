"""Score bounds metadata and V4 primary score/reward emission checks."""
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


def test_bounds_anchors_match_v4_hierarchical_receipts():
    b = json.loads(BOUNDS.read_text())
    assert b["lower"]["reward"] == pytest.approx(0.0036919698345322383, abs=1e-12)
    assert b["upper"]["reward"] == pytest.approx(0.44001043721132516, abs=1e-12)
    assert b["upper"]["achievable_by_a_submission"] is True


def test_main_verifier_omits_normalised_score_and_keeps_score_and_reward():
    assert not hasattr(mv, "normalised")
    payload = mv.harbor_reward({"score": 0.19347315873897375, "reward": 0.19347315873897375, "note": "ignored"})
    assert payload == {"score": 0.19347315873897375, "reward": 0.19347315873897375}
    assert "normalised_score" not in payload
