"""Rendering checks for incomplete progress, resource scope, and safe values."""
from __future__ import annotations

import json
from pathlib import Path

from baselines.alphaqubit_report import render_report, write_report


def save(root: Path, name: str, value: dict) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_partial_report_keeps_sunk_cost_and_diagnostics_separate_from_quality(tmp_path: Path) -> None:
    save(tmp_path, "alphaqubit-v1/pilot/runtime.json", {
        "group": "d5_X", "config": {"dtype": "bfloat16"}, "batch_size": 256,
        "rounds": 25, "samples_per_second": 1500., "gradient_norm": 1.,
        "master_dtype": "float32", "first_update_compile_seconds": 76.,
    })
    save(tmp_path, "alphaqubit-v1/overfit.json", {"overfit_passed": True,
        "latest_metrics": {"main_loss": 0.05, "main_error": 0.01}})
    html = render_report(tmp_path)
    assert "145.90194 GPU-hours" in html
    assert "Missing usage is not zero" in html
    assert "finite update / throughput recorded" in html
    assert "fixed-batch overfit passed" in html
    assert html.count("held-out evaluation pending") == 2
    assert "full odd-test coverage recorded" not in html
    assert "not official Google code" in html


def test_accounting_and_stage_validation_render_without_implying_convergence(tmp_path: Path) -> None:
    save(tmp_path, "accounting.json", {
        "updated_utc": "2026-09-20T03:00:00Z", "actual_allocated_gpu_hours": 150.,
        "remaining_actual_gpu_hours": 234., "records": [{"job_id": "old_lstm", "state": "CANCELLED", "gpu_hours": 145.90194}],
    })
    save(tmp_path, "alphaqubit-v1/models/d5_X/pretrain/progress.json", {
        "status": "stopped", "stop_reason": "budget", "global_steps": 1234,
        "seen_samples": 315904, "updated_at": "2026-09-20T03:01:00Z",
        "latest_validation": {"complete": True, "main_loss": .24,
            "fit": {"accepted": False, "logical_error_per_round": -.02,
                    "rejection_reasons": ["unphysical_ler"]}},
    })
    html = render_report(tmp_path)
    assert "150.00000" in html and "234.00000" in html
    assert "not added a second time" in html
    assert "CANCELLED" in html
    assert "stopped; convergence not established" in html
    assert "315,904" in html and "1,234" in html
    assert "rejected validation fit" in html and "unphysical_ler" in html
    assert html.count("held-out evaluation pending") == 2


def test_final_scores_escape_markup_reject_nonfinite_display_and_write_atomically(tmp_path: Path) -> None:
    attack = "<script>alert('unsafe')</script>"
    save(tmp_path, "alphaqubit-v1/scores-finetune.json", {
        "split": "test", "conditions": 130, "total_shots": 3250000,
        "summary": {"macro": {"d3": {"logical_error_per_round": .03, "ci95": [.029, .031]}},
                    "per_group": {attack: {"logical_error_per_round": float("nan"), "zero_round_fidelity": float("inf")}}},
        "calibration": {"macro": {"d3": {"negative_log_likelihood": .12, "brier_score": .04}}},
        "paired_comparisons": {"pymatching_correlated_selected": {"comparisons": {"d3": {
            "logical_error_per_round_a_minus_b": -.003, "ci95": [-.004, -.002]}}}},
    })
    path = write_report(tmp_path)
    html = path.read_text()
    assert path == tmp_path / "alphaqubit-v1/report.html"
    assert "full odd-test coverage recorded" in html
    assert "3.000" in html and "[2.900, 3.100]" in html
    assert "-0.300" in html and "[-0.400, -0.200]" in html
    assert "0.12000" in html and "0.04000" in html
    assert attack not in html and "&lt;script&gt;" in html
    assert "invalid" in html and ">nan<" not in html and ">inf<" not in html
    assert sorted(item.name for item in path.parent.iterdir()) == ["report.html", "scores-finetune.json"]


def test_expansion_gate_displays_allocation_evidence_without_promoting_test_claim(tmp_path: Path) -> None:
    save(tmp_path, "alphaqubit-v1/expansion-gate.json", {
        "passed": True, "decision": "expand", "evaluated_at": "2026-09-20T05:00:00Z",
        "interpretation": "Representative learning evidence supports further allocation.",
        "thresholds": {"minimum_seen_samples": 1000000},
        "groups": {"d5_X": {"passed": True, "status": "eligible", "rejection_reasons": [],
            "measurements": {"seen_samples": 2000000, "complete_validations": 4,
                "selected_LER": .035, "matching_LER": .03, "model_to_matching_LER_ratio": 7 / 6,
                "first_main_BCE": .5, "candidate_main_BCE": .3, "bce_relative_improvement": .4,
                "selected_errors": 500, "no_correction_errors": 1000, "total_shots": 10000,
                "shot_error_relative_improvement": .5}}},
    })
    html = render_report(tmp_path)
    assert "Expansion allowed by the recorded gate" in html
    assert "2,000,000" in html and "1.167" in html
    assert "40.00" in html and "50.00" in html
    assert "Synthetic MWPM LER" in html and "not a scientific success/failure verdict" in html
    assert html.count("held-out evaluation pending") == 2
    assert "full odd-test coverage recorded" not in html


def test_closed_expansion_gate_preserves_diagnosis_reasons_and_escapes_values(tmp_path: Path) -> None:
    attack = "<img src=x onerror=alert(1)>"
    save(tmp_path, "alphaqubit-v1/expansion-gate.json", {
        "passed": False, "decision": "no_expansion", "groups": {
            "d3_X": {"status": "insufficient_evidence", "rejection_reasons": ["stage_not_finalized"]},
            "d5_X": {"status": "needs_diagnosis", "rejection_reasons": [attack],
                      "measurements": {"selected_LER": float("nan")}},
        },
    })
    html = render_report(tmp_path)
    assert "No expansion: inspect missing evidence or diagnosis reasons" in html
    assert "stage_not_finalized" in html and "needs_diagnosis" in html
    assert attack not in html and "&lt;img" in html
    assert "invalid" in html
    assert "Expansion allowed by the recorded gate" not in html
