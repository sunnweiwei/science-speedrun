"""Rendering contracts for partial evidence and finalized score artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1] / "environment" / "baseline" / "alphaqubit" if (Path(__file__).resolve().parents[1] / "environment" / "baseline" / "alphaqubit").is_dir() else Path(__file__).resolve().parents[1]  # the baseline project: environment/baseline/alphaqubit here, /opt/task in the verifier image
sys.path.insert(0, str(PROJECT))
from baselines.report import render_report


def test_partial_report_does_not_claim_neural_completion_or_zero_actual_usage(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (tmp_path / "campaign.json").write_text(json.dumps({
        "max_actual_allocated_gpu_hours": 384, "max_concurrent_gpus": 8,
        "jobs": [{"id": "1", "name": "training", "gpu_count": 1,
                  "array_tasks": 2, "max_hours_per_task": 2}],
    }))
    html = render_report(tmp_path, "run", evidence_root=evidence)
    assert "held-out evaluation pending" in html
    assert "full held-out scores available" not in html
    assert "partial; missing actuals are not zero" in html
    assert "4.00 GPU-hours" in html
    assert "This page does not poll jobs or run evaluation" in html


def test_final_scores_and_untrusted_values_render_without_raw_markup_or_nonfinite_numbers(tmp_path: Path) -> None:
    evidence, campaign = tmp_path / "evidence", tmp_path / "run"
    evidence.mkdir()
    campaign.mkdir()
    attack = "<script id='untrusted'>alert(1)</script>"
    (evidence / "classical-summary.json").write_text(json.dumps({
        "verification": {"all_130_conditions_verified": True},
        "methods": {attack: {"macro": {}, "group_fits": {"group": {
            "logical_error_per_round": float("nan"),
            "zero_round_fidelity": float("inf"), "diagnostic_flags": [attack],
        }}}},
    }))
    score = {
        "split": "test", "conditions": 130, "total_shots": 3_250_000,
        "summary": {"status": "ok", "macro": {
            "d3": {"logical_error_per_round": 0.03, "ci95": [0.029, 0.031]},
            "d5": {"logical_error_per_round": 0.02, "ci95": [0.019, 0.021]},
        }, "per_group": {}},
        "calibration": {"macro": {"d3": {"negative_log_likelihood": 0.12, "brier_score": 0.04}}},
        "paired_comparisons": {"pymatching_correlated_selected": {
            "status": "ok", "comparisons": {"d3": {
                "logical_error_per_round_a_minus_b": -0.003,
                "ci95": [-0.004, -0.002],
            }},
        }},
    }
    (campaign / "scores-pretrain.json").write_text(json.dumps(score))
    html = render_report(tmp_path, "run", evidence_root=evidence)
    assert attack not in html and "&lt;script id=&#x27;untrusted&#x27;&gt;" in html
    assert "full held-out scores available; inspect fit diagnostics" in html
    assert "held-out evaluation pending" in html  # Fine-tuning remains pending.
    assert "3.000" in html and "[2.900, 3.100]" in html
    assert "-0.300" in html and "[-0.400, -0.200]" in html
    assert "0.12000" in html and "0.04000" in html
    assert "invalid" in html
    assert ">nan<" not in html and ">inf<" not in html


def test_accounting_snapshot_overrides_incomplete_submission_ledger_and_marks_job_states(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (tmp_path / "campaign.json").write_text(json.dumps({
        "max_actual_allocated_gpu_hours": 384, "max_concurrent_gpus": 8,
        "jobs": [{"id": "1", "name": "pretrain", "gpu_count": 1,
                  "array_tasks": 3, "max_hours_per_task": 24}],
    }))
    (tmp_path / "accounting.json").write_text(json.dumps({
        "updated_utc": "2026-09-19T06:00:00+00:00",
        "actual_allocated_gpu_hours": 12.375, "active_allocated_gpus": 0,
        "declared_allocation_ceiling_gpu_hours": 72, "remaining_actual_gpu_hours": 371.625,
        "records": [
            {"job_id": "1_0", "state": "FAILED", "gpu_hours": 12.375},
            {"job_id": "1_1", "state": "CANCELLED", "gpu_hours": 0},
            {"job_id": "1_2", "state": "PENDING", "reason": "DependencyNeverSatisfied", "gpu_hours": 0},
        ],
    }))
    html = render_report(tmp_path, "run", evidence_root=evidence)
    assert "Actual allocated GPU-hours: 12.375" in html
    assert "2026-09-19T06:00:00+00:00" in html
    assert "Remaining actual budget: 371.625" in html
    assert "Recorded actuals sum to 0.000" not in html
    assert "Failed / interrupted · FAILED" in html
    assert "Cancelled · CANCELLED" in html
    assert "Dependency blocked" in html
    assert "held-out evaluation pending" in html


def test_incomplete_representative_gate_is_waiting_not_failed(tmp_path: Path) -> None:
    evidence, campaign = tmp_path / "evidence", tmp_path / "run"
    evidence.mkdir()
    campaign.mkdir()
    (campaign / "representative-gate.json").write_text(json.dumps({
        "passed": False, "checked_utc": "2026-09-19T06:00:00Z",
        "groups": {"group": {"passed": False, "checks": {"stage_stopped": False}}},
    }))
    html = render_report(tmp_path, "run", evidence_root=evidence)
    assert "waiting_for_completed_stages" in html
    assert "this waiting result is not a failed training job" in html
    assert "<strong class='state-failed'>failed</strong>" not in html
    assert "held-out evaluation pending" in html
