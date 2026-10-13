"""Independent expansion policy tests; no model training or test-fold reads."""
from __future__ import annotations

import json
import copy
import math
from pathlib import Path

import pytest

from baselines.alphaqubit_gate import GROUPS, _validation_evidence, checked_fit, policy_decision, run_gate


def evidence(**changes):
    return {"global_steps": 15000, "seen_samples": 5_000_000, "complete_validations": 5,
            "accepted_best_fit": True, "first_main_BCE": 0.65, "candidate_main_BCE": 0.50,
            "selected_LER": 0.034, "matching_LER": 0.02,
            "selected_errors": 18000, "no_correction_errors": 24000,
            "total_shots": 66560, **changes}


def test_learning_and_already_strong_are_independent_positive_routes() -> None:
    learning = policy_decision(evidence())
    assert learning["passed"] and not learning["measurements"]["already_strong"]
    strong = policy_decision(evidence(candidate_main_BCE=0.64, selected_LER=0.024))
    assert strong["passed"] and strong["measurements"]["already_strong"]
    assert strong["measurements"]["bce_relative_improvement"] < 0.10


@pytest.mark.parametrize("changes,reason", [
    ({"global_steps": 9999}, "fewer_than_10000_optimizer_steps"),
    ({"seen_samples": 3_999_999}, "fewer_than_4M_training_examples"),
    ({"complete_validations": 2}, "fewer_than_three_complete_validations"),
    ({"accepted_best_fit": False}, "no_accepted_best_validation_fit"),
    ({"selected_LER": 0.0401}, "selected_LER_exceeds_twice_matching_or_fit_unavailable"),
    ({"candidate_main_BCE": 0.64}, "insufficient_BCE_improvement_and_not_already_strong"),
    ({"selected_errors": 21601}, "less_than_10_percent_shot_error_improvement_over_no_correction"),
    ({"matching_LER": None}, "selected_LER_exceeds_twice_matching_or_fit_unavailable"),
])
def test_each_required_gate_rejects_independently(changes, reason) -> None:
    result = policy_decision(evidence(**changes))
    assert not result["passed"] and reason in result["rejection_reasons"]


def test_zero_matching_does_not_create_an_infinite_or_spurious_pass() -> None:
    result = policy_decision(evidence(matching_LER=0.0))
    assert not result["passed"] and result["measurements"]["model_to_matching_LER_ratio"] is None
    assert policy_decision(evidence(matching_LER=0.0, selected_LER=0.0))["passed"]


def test_missing_artifacts_always_write_non_scientific_no_expansion(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "gate.json"
    result = run_gate(tmp_path, output)
    assert json.loads(output.read_text()) == result
    assert not result["passed"] and result["decision"] == "no_expansion"
    assert set(result["groups"]) == set(GROUPS)
    assert all(record["status"] == "insufficient_evidence" for record in result["groups"].values())
    assert "not a negative scientific result" in result["interpretation"]


def fit_snapshot(step: int, bce: float) -> dict:
    rows = []
    for rounds in range(1, 26, 2):
        errors = round((1 - math.exp(0.003 + rounds * math.log(0.96))) * 5120 / 2)
        rows.append({"condition": f"surface_code_bX_d3_r{rounds:02d}_center_3_5", "rounds": rounds,
                     "shots": 5120, "errors": errors, "main_error": errors / 5120,
                     "main_loss": bce, "aux_loss": 0.2, "loss": bce + 0.004})
    fit = checked_fit(rows)
    assert fit["accepted"]
    return {"complete": True, "global_steps": step, "seen_samples": step * 256,
            "shots": 66560, "per_condition": rows, "fit": fit,
            "main_error": sum(row["errors"] for row in rows) / 66560,
            "main_loss": bce, "aux_loss": 0.2, "loss": bce + 0.004}


def test_validation_evidence_recomputes_fit_and_rejects_count_drift() -> None:
    history = [fit_snapshot(4000, 0.65), fit_snapshot(9000, 0.5), fit_snapshot(16000, 0.55)]
    progress = {"global_steps": 16000, "latest_validation": history[-1]}
    best = {"validation": history[1]}
    complete, candidate = _validation_evidence(history, best, progress)
    assert len(complete) == 3 and candidate["global_steps"] == 9000
    modified = copy.deepcopy(history)
    modified[0]["per_condition"][0]["errors"] += 1
    with pytest.raises(ValueError, match="raw counts"):
        _validation_evidence(modified, best, progress)
    modified = copy.deepcopy(history)
    modified[1]["fit"]["logical_error_per_round"] *= 0.5
    with pytest.raises(ValueError, match="does not reproduce"):
        _validation_evidence(modified, best, progress)
    with pytest.raises(ValueError, match="duplicated"):
        _validation_evidence([history[0], *history], best, progress)
    modified = copy.deepcopy(history)
    modified[0]["main_loss"] += 0.1
    with pytest.raises(ValueError, match="aggregate"):
        _validation_evidence(modified, best, progress)
    with pytest.raises(ValueError, match="absent"):
        _validation_evidence(history, {"validation": fit_snapshot(8000, 0.5)}, progress)
    with pytest.raises(ValueError, match="absent"):
        _validation_evidence(history, best, {**progress, "latest_validation": history[0]})
