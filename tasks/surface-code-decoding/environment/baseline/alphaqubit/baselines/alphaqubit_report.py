"""Render an editable AlphaQubit evidence snapshot using only local JSON files."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


PRIOR_LSTM_GPU_HOURS = 145.90194444444444
METHODS = {
    "pymatching_mwpm_selected": "PyMatching MWPM",
    "pymatching_correlated_selected": "PyMatching correlated",
    "released_belief_matching": "Released belief matching",
    "released_tensor_network": "Released tensor network",
}


def text(value: Any) -> str:
    return escape(str(value), quote=True)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def number(value: Any, digits: int = 3, scale: float = 1.) -> str:
    if value is None:
        return "—"
    return f"{value * scale:,.{digits}f}" if finite(value) else "invalid"


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def interval(value: Any, ci: Any, scale: float = 100.) -> str:
    result = number(value, scale=scale)
    if value is None:
        return result
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        return f"{result} <span class='muted'>[{number(ci[0], scale=scale)}, {number(ci[1], scale=scale)}]</span>"
    return result + " <span class='muted'>(CI unavailable)</span>"


def table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "<p class='muted'>No evidence recorded yet.</p>"
    head = "".join(f"<th scope='col'>{text(item)}</th>" for item in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


class Evidence:
    def __init__(self) -> None:
        self.sources: list[tuple[Path, str]] = []
        self.warnings: list[str] = []

    def read(self, path: Path, *, history: bool = False) -> Any:
        if not path.is_file():
            return [] if history else {}
        try:
            raw = path.read_bytes()
            self.sources.append((path, hashlib.sha256(raw).hexdigest()))
            if history:
                rows = []
                for index, line in enumerate(raw.splitlines(), 1):
                    try:
                        value = json.loads(line)
                        if not isinstance(value, dict):
                            raise ValueError("expected an object")
                        rows.append(value)
                    except (ValueError, UnicodeDecodeError) as error:
                        self.warnings.append(f"{path}, line {index}: {error}")
                return rows
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("expected an object")
            return value
        except (OSError, ValueError) as error:
            self.warnings.append(f"Could not read {path}: {error}")
            return [] if history else {}


def state(value: Any, reason: Any = None) -> str:
    label = str(value or "not recorded")
    if reason:
        label += "; " + str(reason)
    upper = label.upper()
    css = ""
    if any(word in upper for word in ("FAIL", "TIMEOUT", "OUT_OF_MEMORY", "DEPENDENCYNEVERSATISFIED")):
        css = "bad"
    elif any(word in upper for word in ("CANCEL", "PENDING", "DEPENDENCY", "WAIT")):
        css = "pending"
    return f"<span class='{css}'>{text(label)}</span>"


def resources(accounting: dict[str, Any]) -> str:
    result = "<h2>Resource budget</h2>"
    actual = accounting.get("actual_allocated_gpu_hours")
    if finite(actual):
        result += (f"<p>Actual allocated GPU-hours: <strong>{number(actual, 5)}</strong>. "
                   f"Remaining actual budget: <strong>{number(accounting.get('remaining_actual_gpu_hours'), 5)}</strong> GPU-hours. "
                   f"Snapshot: {text(accounting.get('updated_utc', 'timestamp unavailable'))}.</p>")
        if actual < PRIOR_LSTM_GPU_HOURS - 1e-5:
            result += "<p class='bad'>Accounting total is below the recorded prior LSTM expenditure; reconcile this snapshot before relying on its remaining budget.</p>"
    else:
        result += (f"<p class='pending'>Current actual accounting is unavailable. Known prior expenditure is at least "
                   f"{number(PRIOR_LSTM_GPU_HOURS, 5)} GPU-hours; at most {number(384 - PRIOR_LSTM_GPU_HOURS, 5)} "
                   "GPU-hours remained before AlphaQubit work. Missing usage is not zero.</p>")
    result += (f"<p>The earlier LSTM campaign consumed <strong>{number(PRIOR_LSTM_GPU_HOURS, 5)} GPU-hours</strong> "
               "and remains included in the same 384 GPU-hour budget; it is not added a second time to the accounting total. "
               "The task limit is eight concurrent GPUs. Card-hours on different GPU models are not H200-equivalent hours.</p>")
    result += table(["Accounting field", "Recorded value"], [
        ["Hard task budget, GPU-hours", number(accounting.get("max_actual_allocated_gpu_hours", 384), 2)],
        ["Declared allocation ceiling, GPU-hours", number(accounting.get("declared_allocation_ceiling_gpu_hours"), 5)],
        ["Active allocated GPUs at snapshot", number(accounting.get("active_allocated_gpus"), 0)],
        ["Accounting scope", text(accounting.get("scope", "not recorded"))],
    ])
    records = accounting.get("records", [])
    if isinstance(records, list) and records:
        result += "<details><summary>Recorded Slurm states and actual allocation</summary>"
        result += table(["Job", "State / reason", "GPU-hours", "Nodes"], [[
            text(row.get("job_id", "—")), state(row.get("state"), row.get("reason")),
            number(row.get("gpu_hours"), 5), text(row.get("nodes", "—")),
        ] for row in records if isinstance(row, dict)]) + "</details>"
    return result + "<p class='note'>Accounting may lag running jobs. Retired LSTM cancellations and deliberate pilot cancellations do not establish AlphaQubit failure. Allocation ceilings are not actual consumption.</p>"


def runtime_rows(pilots: list[tuple[Path, dict[str, Any]]]) -> str:
    rows = []
    for path, record in pilots:
        config, memory = mapping(record.get("config")), mapping(record.get("memory"))
        speed, gradient = record.get("samples_per_second"), record.get("gradient_norm")
        passed = finite(speed) and speed > 0 and finite(gradient) and gradient > 0 and record.get("master_dtype") == "float32"
        label = "finite update / throughput recorded" if passed else "runtime check not established"
        rows.append([text(path.name), text(record.get("group", "—")), text(config.get("dtype", "—")),
                     number(record.get("batch_size"), 0), number(record.get("rounds"), 0),
                     number(speed, 1), number(record.get("first_update_compile_seconds"), 1),
                     number(memory.get("peak_bytes_in_use"), 2, 1 / 2**30), text(label)])
    result = "<h2>Runtime and overfit diagnostics</h2>"
    result += table(["Artifact", "Group", "Compute", "Batch", "Rounds", "Examples/s", "First update + compile, s", "Peak memory, GiB", "Evidence"], rows)
    result += "<p class='note'>Throughput probes exclude fresh simulation generation unless their recorded scope says otherwise. Passing a runtime or fixed-batch overfit check does not establish generalization, convergence, or held-out decoding accuracy.</p>"
    if pilots:
        result += "<details><summary>Hardware and timing scope</summary>" + table(["Artifact", "GPU", "Timing scope"], [[
            text(path.name), text(record.get("gpu", "not recorded")), text(record.get("timing_scope", "not recorded")),
        ] for path, record in pilots]) + "</details>"
    return result


def overfit_rows(records: list[tuple[Path, dict[str, Any]]]) -> str:
    if not records:
        return "<p class='pending'>Fixed-batch overfit diagnostic pending.</p>"
    rows = []
    for path, record in records:
        passed = record.get("overfit_passed")
        metrics = mapping(record.get("latest_metrics"))
        verdict = "fixed-batch overfit passed" if passed is True else "fixed-batch overfit failed" if passed is False else "overfit outcome not recorded"
        rows.append([text(path.name), state(verdict), number(metrics.get("main_loss"), 5),
                     number(metrics.get("main_error"), 3, 100), number(record.get("steady_steps"), 0)])
    return table(["Artifact", "Diagnostic outcome", "Training BCE", "Training error, %", "Steps"], rows)


def expansion_gate(gate: dict[str, Any]) -> str:
    result = "<h2>Expansion allocation gate</h2>"
    if not gate:
        return result + "<p class='pending'>Expansion decision pending: no gate artifact recorded.</p>"
    expand = gate.get("passed") is True and gate.get("decision") == "expand"
    decision = "Expansion allowed by the recorded gate" if expand else "No expansion: inspect missing evidence or diagnosis reasons"
    result += (f"<p><strong>{text(decision)}</strong>. Evaluated: {text(gate.get('evaluated_at', 'timestamp unavailable'))}.</p>"
               f"<p>{text(gate.get('interpretation', ''))}</p>"
               "<p class='note'>This gate governs further GPU allocation. It is not a scientific success/failure verdict, "
               "evidence of convergence, or a held-out test result. Insufficient evidence and a model needing diagnosis "
               "both keep expansion closed.</p>")
    rows, progress_rows = [], []
    for group, record in mapping(gate.get("groups")).items():
        record = mapping(record)
        measured = mapping(record.get("measurements"))
        reasons = record.get("rejection_reasons", [])
        reasons = reasons if isinstance(reasons, list) else [reasons]
        rows.append([text(group), text(record.get("status", "not established")),
                     number(measured.get("seen_samples"), 0), number(measured.get("complete_validations"), 0),
                     number(measured.get("selected_LER"), 3, 100), number(measured.get("matching_LER"), 3, 100),
                     number(measured.get("model_to_matching_LER_ratio"), 3),
                     text("; ".join(map(str, reasons)) or "none recorded")])
        progress_rows.append([text(group), number(measured.get("global_steps"), 0),
                              number(measured.get("first_main_BCE"), 5), number(measured.get("candidate_main_BCE"), 5),
                              number(measured.get("bce_relative_improvement"), 2, 100),
                              number(measured.get("selected_errors"), 0), number(measured.get("no_correction_errors"), 0),
                              number(measured.get("total_shots"), 0), number(measured.get("shot_error_relative_improvement"), 2, 100)])
    result += table(["Representative group", "Gate evidence state", "Seen samples", "Complete validations", "Selected validation LER, %", "Synthetic MWPM LER, %", "Model / MWPM LER", "Reasons"], rows)
    result += "<details><summary>Learning and no-correction evidence</summary>" + table([
        "Group", "Steps", "Initial validation BCE", "Candidate validation BCE", "BCE improvement, %",
        "Selected errors", "No-correction errors", "Validation shots", "Shot-error improvement, %",
    ], progress_rows) + "</details>"
    thresholds = mapping(gate.get("thresholds"))
    if thresholds:
        def threshold_value(value: Any) -> str:
            if isinstance(value, bool):
                return "yes" if value else "no"
            return number(value, 5) if isinstance(value, (int, float)) else text(value)
        result += "<details><summary>Recorded allocation thresholds</summary>" + table([
            "Threshold", "Value",
        ], [[text(key), threshold_value(value)] for key, value in thresholds.items()]) + "</details>"
    return result


def training_rows(campaign: Path, evidence: Evidence) -> str:
    rows, validation_rows = [], []
    directories = sorted(path for stage in ("pretrain", "finetune") for path in (campaign / "models").glob(f"*/{stage}") if path.is_dir())
    for directory in directories:
        progress = evidence.read(directory / "progress.json")
        best = evidence.read(directory / "best.json")
        history = evidence.read(directory / "history.jsonl", history=True)
        latest = mapping(progress.get("latest_validation"))
        if not latest:
            latest = next((row for row in reversed(history) if row.get("complete") is True), {})
        fit = mapping(latest.get("fit"))
        accepted = fit.get("accepted")
        validity = "accepted validation fit" if accepted is True else "rejected validation fit" if accepted is False else "validation pending"
        status = progress.get("status", "progress not recorded")
        if status == "stopped":
            status = "stopped; convergence not established"
        rows.append([text(directory.parent.name), text(directory.name), state(status, progress.get("stop_reason")),
                     number(progress.get("global_steps"), 0), number(progress.get("seen_samples"), 0),
                     number(progress.get("elapsed_seconds"), 2, 1 / 3600),
                     number(progress.get("steady_samples_per_second"), 1),
                     number(mapping(progress.get("last_training_metrics")).get("main_loss"), 5),
                     text(progress.get("updated_at", "—"))])
        validation_rows.append([text(directory.parent.name), text(directory.name), text(validity),
                                number(latest.get("main_loss"), 5), number(fit.get("logical_error_per_round"), 3, 100),
                                number(fit.get("selection_r_squared"), 4), number(fit.get("zero_round_fidelity"), 4),
                                number(best.get("selected_LER", progress.get("best_LER")), 3, 100),
                                text("; ".join(map(str, fit.get("rejection_reasons", []))))])
    result = "<h2>AlphaQubit training progress</h2>"
    result += table(["Group", "Stage", "Recorded state", "Steps", "Seen samples", "Elapsed h", "Training examples/s", "Latest training BCE", "Updated"], rows)
    result += table(["Group", "Stage", "Validation", "Main BCE", "Latest LER, %", "Selection R²", "F₀", "Best accepted LER, %", "Fit rejection reasons"], validation_rows)
    return result + "<p class='note'>Pretraining validation is independently seeded synthetic data; finetuning selection uses the even development split. These development measurements are separate from odd held-out scores. A stopped stage or saved checkpoint is not proof of convergence.</p>"


def result_row(name: str, summary: dict[str, Any], coverage: str) -> list[str]:
    macro = mapping(summary.get("macro"))
    d3, d5 = mapping(macro.get("d3")), mapping(macro.get("d5"))
    ratio = mapping(macro.get("error_suppression_d3_over_d5"))
    return [text(name), text(coverage), interval(d3.get("logical_error_per_round"), d3.get("ci95")),
            interval(d5.get("logical_error_per_round"), d5.get("ci95")), interval(ratio.get("value"), ratio.get("ci95"), 1)]


def score_tables(classical: dict[str, Any], neural: list[tuple[str, dict[str, Any]]]) -> str:
    rows, fits, calibration, paired = [], [], [], []
    for key, name in METHODS.items():
        summary = mapping(mapping(classical.get("scores")).get(key))
        if summary:
            rows.append(result_row(name, summary, "completed classical artifact" if classical.get("completed_utc") else "classical artifact present"))
            for group, fit in mapping(summary.get("per_group")).items():
                fits.append([text(name), text(group), interval(fit.get("logical_error_per_round"), fit.get("logical_error_per_round_ci95")),
                             number(fit.get("zero_round_fidelity"), 4), number(fit.get("weighted_log_fidelity_r_squared"), 4),
                             text("; ".join(map(str, fit.get("diagnostic_flags", []))))])
    for stage, score in neural:
        coverage = "held-out evaluation pending"
        if score:
            if score.get("split") == "test" and score.get("conditions") == 130 and score.get("total_shots") == 3_250_000:
                coverage = "full odd-test coverage recorded"
            else:
                coverage = f"partial / other split: {score.get('split', 'unknown')}, {score.get('conditions', '?')} conditions"
            rows.append(result_row("AlphaQubit " + stage, mapping(score.get("summary")), coverage))
            for group, fit in mapping(mapping(score.get("summary")).get("per_group")).items():
                fits.append([text("AlphaQubit " + stage), text(group), interval(fit.get("logical_error_per_round"), fit.get("logical_error_per_round_ci95")),
                             number(fit.get("zero_round_fidelity"), 4), number(fit.get("weighted_log_fidelity_r_squared"), 4),
                             text("; ".join(map(str, fit.get("diagnostic_flags", []))) or fit.get("reason", ""))])
            for group, metrics in mapping(mapping(score.get("calibration")).get("macro")).items():
                calibration.append([text(stage), text(group), number(metrics.get("negative_log_likelihood"), 5), number(metrics.get("brier_score"), 5)])
            for method, comparison in mapping(score.get("paired_comparisons")).items():
                for group, value in mapping(comparison.get("comparisons")).items():
                    paired.append([text(stage), text(METHODS.get(method, method)), text(group),
                                   interval(value.get("logical_error_per_round_a_minus_b"), value.get("ci95"))])
        else:
            rows.append(result_row("AlphaQubit " + stage, {}, coverage))
    result = "<h2>Held-out results and classical comparators</h2>"
    result += table(["Decoder / stage", "Evidence coverage", "d3 LER, % [95% CI]", "d5 LER, % [95% CI]", "d3 / d5 suppression ratio"], rows)
    result += "<p class='note'>LER is a fitted per-round quantity from rounds 3,5,…,25, with a free intercept, separately for each basis and patch. Distance macros average group fits. F₀ above one is an extrapolated intercept and must be interpreted alongside fit quality. Public odd shots are held out by this workflow, not a secret benchmark test set.</p>"
    result += "<details><summary>Per-group fit diagnostics</summary>" + table(["Decoder", "Group", "LER, % [95% CI]", "F₀", "Weighted log-fidelity R²", "Flags"], fits) + "</details>"
    result += "<h3>Probability calibration</h3>" + table(["Stage", "Scope", "NLL", "Brier score"], calibration)
    result += "<h3>Paired comparisons</h3>" + table(["Stage", "Comparator", "Scope", "AlphaQubit minus comparator LER, percentage points [95% CI]"], paired)
    return result + "<p class='note'>Negative paired differences favor AlphaQubit. Runtime probes and development scores are not entered into this comparison table.</p>"


def render_report(asset_root: Path) -> str:
    evidence = Evidence()
    campaign = asset_root / "alphaqubit-v1"
    accounting = evidence.read(asset_root / "accounting.json")
    gate = evidence.read(campaign / "expansion-gate.json")
    classical = evidence.read(asset_root / "classical-v1" / "scores.json")
    pilots = [(path, evidence.read(path)) for path in sorted((campaign / "pilot").glob("*.json"))]
    pilots = [(path, value) for path, value in pilots if value]
    overfits = [(path, record) for path, record in pilots if record.get("fixed_batch_training_diagnostic") is True]
    for path in (campaign / "overfit.json", campaign / "pilot" / "overfit.json"):
        if path.is_file() and all(path != previous for previous, _ in overfits):
            record = evidence.read(path)
            if record:
                overfits.append((path, record))
    neural = [(stage, evidence.read(campaign / f"scores-{stage}.json")) for stage in ("pretrain", "finetune")]
    body = ("<h1>AlphaQubit reproduction: evidence snapshot</h1>"
            f"<p class='muted'>Generated {text(datetime.now(timezone.utc).isoformat())}. Local artifacts only; this page does not poll jobs or run evaluation.</p>"
            "<p>This campaign implements AlphaQubit independently from the Nature 2024 pseudocode and Methods. "
            "It uses documented reconstruction assumptions and a bounded Sycamore d3/d5 protocol. "
            "It is not official Google code, the complete paper ensemble, or the d3–11 Pauli+ experiment. "
            "Earlier LSTM experiments remain secondary work and are included in the resource budget.</p>")
    if not any(score for _, score in neural):
        body += "<p class='pending'><strong>No AlphaQubit odd-test scores have been recorded.</strong> Runtime, overfit, and development measurements below establish separate levels of evidence.</p>"
    body += resources(accounting) + runtime_rows(pilots) + overfit_rows(overfits) + expansion_gate(gate)
    body += training_rows(campaign, evidence) + score_tables(classical, neural)
    body += ("<h2>Provenance and limits</h2><p>Architecture and protocol sources: "
             "<a href='https://www.nature.com/articles/s41586-024-08148-8'>Nature 2024 paper</a> and "
             "<a href='https://media.springernature.com/full/springer-static/esm/art%3A10.1038%2Fs41586-024-08148-8/MediaObjects/41586_2024_8148_MOESM1_ESM.pdf'>supplementary pseudocode</a>. "
             "The repository records reconstruction choices in reproduction/alphaqubit-source-audit.md and "
             "reproduction/alphaqubit-implementation-options.md. Choices include unspecified residual-block internals, "
             "cycle encoding, value-head dimension, optimizer conventions, and the interpretation of schedule units. "
             "Paper-level accuracy requires measured evidence; implementation and runtime checks alone do not establish it.</p>")
    if evidence.warnings:
        body += "<h3>Artifact-read warnings</h3><ul>" + "".join(f"<li>{text(item)}</li>" for item in evidence.warnings) + "</ul>"
    body += "<details><summary>Source files and SHA-256</summary>" + table(["Path", "SHA-256"], [
        [text(path), f"<code>{text(digest)}</code>"] for path, digest in evidence.sources
    ]) + "</details>"
    css = """body{max-width:1260px;margin:48px auto;padding:0 24px;color:#20272d;background:white;font:15px/1.6 system-ui,sans-serif}h1{font-size:28px;font-weight:600}h2{font-size:21px;margin-top:36px;border-top:1px solid #d9dfe3;padding-top:18px}h3{font-size:17px;margin-top:26px}p{max-width:1040px}table{width:100%;border-collapse:collapse;font-size:13px;margin:14px 0}th,td{text-align:left;vertical-align:top;padding:9px 10px;border-bottom:1px solid #d9dfe3}th{font-weight:600;background:#f5f6f7}td{font-variant-numeric:tabular-nums}.scroll{overflow-x:auto}.muted,.note{color:#5d6972}.note{font-size:13px}.bad{color:#9e2828}.pending{color:#79581d}a{color:#244c62}summary{cursor:pointer;font-weight:500}details{margin:18px 0}code{font-size:11px;overflow-wrap:anywhere}@media print{body{margin:12px;max-width:none}.scroll{overflow:visible}}"""
    return f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>AlphaQubit reproduction evidence</title><style>{css}</style></head><body>{body}</body></html>"


def write_report(asset_root: Path, output: Path | None = None) -> Path:
    destination = output or asset_root / "alphaqubit-v1" / "report.html"
    document = render_report(asset_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as stream:
            temporary = stream.name
            stream.write(document)
        os.replace(temporary, destination)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(write_report(args.asset_root, args.output))


if __name__ == "__main__":
    main()
