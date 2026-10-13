"""Primary reward: panel case detection at a fixed alert budget."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment" / "evaluation"))
from panel_scoring import (MIN_POSITIVES, PANEL, PRIMARY, has_family_history,  # noqa: E402
                           history_code, panel_score, recall_at)


def test_recall_at_alert_budget_counts_cases_in_top_share():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0] * 10)
    p = np.where(y == 1, 0.9, 0.1).astype(float)
    p[0] = 0.05  # one case ranked last
    recall, ppv = recall_at(y, p, 0.05)  # top 5 of 100
    assert recall == pytest.approx(5 / 20) and ppv == 1.0


def test_ties_score_their_expectation_so_constant_prediction_is_chance():
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.1).astype(int)
    recall, _ = recall_at(y, np.full(1000, 0.3), 0.05)
    assert recall == pytest.approx(0.05)
    # Row order cannot be exploited: putting all cases first does not change a tied score.
    order = np.argsort(-y, kind="stable")
    assert recall_at(y[order], np.full(1000, 0.3), 0.05)[0] == pytest.approx(0.05)


def test_family_history_exclusion_matches_icd10_exactly_and_icd9_by_prefix():
    family = PANEL["I50"][1]
    assert has_family_history({"I11"}, family)
    assert has_family_history({"ICD9:42831"}, family)
    assert not has_family_history({"I10", "ICD9:4019"}, family)
    assert history_code("I509") == "I50" and history_code("ICD9:42831") == "ICD9:42831"


def _panel_inputs(n=2000, seed=1):
    codes = list(PANEL)
    rng = np.random.default_rng(seed)
    labels = (rng.random((n, len(codes))) < 0.03).astype(np.uint8)
    valid = np.ones_like(labels, dtype=bool)
    complete = np.ones(n, dtype=bool)
    return codes, labels, valid, complete, [set() for _ in range(n)]


def test_perfect_ranking_scores_one_and_reward_set_ignores_predictions():
    codes, labels, valid, complete, history = _panel_inputs()
    perfect = panel_score(labels, valid, labels.astype(float), complete, codes, history)
    noisy = panel_score(labels, valid, np.random.default_rng(2).random(labels.shape),
                        complete, codes, history)
    assert perfect[PRIMARY] == pytest.approx(1.0)
    assert noisy[PRIMARY] < 0.2
    assert perfect["endpoints_in_reward"] == noisy["endpoints_in_reward"]


def test_rare_endpoints_and_ineligible_patients_stay_out_of_the_reward():
    codes, labels, valid, complete, history = _panel_inputs(n=400)
    labels[:, codes.index("C67")] = 0
    labels[: MIN_POSITIVES - 1, codes.index("C67")] = 1
    complete[:50] = False
    for h in history[50:100]:
        h.add("I11")
    report = panel_score(labels, valid, np.random.default_rng(3).random(labels.shape),
                         complete, codes, history)
    assert "C67" not in report["endpoints_in_reward"]
    hf = report["endpoints"]["I50"]
    assert hf["eligible"] == 300 and hf["family_history_excluded"] == 50
