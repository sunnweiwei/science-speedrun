"""Render an editable, standalone evidence snapshot without running evaluation.

Usage: python -m baselines.report --asset-root ASSET --campaign native-v1
                                 --output status.html
"""
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


STAGES = {"pretrain": "Neural: synthetic pretraining", "finetune": "Neural: hardware fine-tuning"}
METHOD_NAMES = {
    "pymatching_mwpm_selected": "PyMatching MWPM",
    "pymatching_correlated_selected": "PyMatching correlated",
    "released_belief_matching": "Released belief matching",
    "released_tensor_network": "Released tensor network",
}


def text(value: Any) -> str:
    return escape(str(value), quote=True)


def number(value: Any, digits: int = 3, *, scale: float = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "invalid"
    return f"{value * scale:,.{digits}f}"


def interval(value: Any, limits: Any, *, percent: bool = False) -> str:
    scale = 100 if percent else 1
    estimate = number(value, 3, scale=scale)
    if value is None:
        return estimate
    if not isinstance(limits, (list, tuple)) or len(limits) != 2:
        return f"{estimate} <span class='muted'>(CI unavailable)</span>"
    return f"{estimate} <span class='ci'>[{number(limits[0], 3, scale=scale)}, {number(limits[1], 3, scale=scale)}]</span>"


def table(headers: list[str], rows: list[list[str]], *, caption: str | None = None) -> str:
    if not rows:
        return "<p class='muted'>No evidence recorded yet.</p>"
    caption_html = f"<caption>{text(caption)}</caption>" if caption else ""
    head = "".join(f"<th scope='col'>{text(label)}</th>" for label in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<div class='table-wrap'><table>{caption_html}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


class Evidence:
    def __init__(self) -> None:
        self.files: list[dict[str, Any]] = []
        self.warnings: list[str] = []

    def read(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            payload = path.read_bytes()
            value = json.loads(payload)
            if not isinstance(value, dict):
                raise ValueError("expected a JSON object")
            self.files.append({"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()})
            return value
        except (OSError, ValueError) as error:
            self.warnings.append(f"Could not read {path}: {error}")
            return None

    def history(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            payload = path.read_bytes()
        except OSError as error:
            self.warnings.append(f"Could not read {path}: {error}")
            return []
        self.files.append({"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()})
        rows: list[dict[str, Any]] = []
        for index, line in enumerate(payload.splitlines(), start=1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
                if row.get("complete") is True:
                    rows.append(row)
            except (ValueError, UnicodeDecodeError) as error:
                self.warnings.append(f"History line {index} unavailable in {path}: {error}")
        return rows


def _fit_rows(label: str, groups: dict[str, Any]) -> list[list[str]]:
    return [[
        text(label), text(group),
        interval(fit.get("logical_error_per_round"), fit.get("logical_error_per_round_ci95"), percent=True),
        number(fit.get("zero_round_fidelity")),
        number(fit.get("weighted_log_fidelity_r_squared"), 4),
        number(fit.get("reduced_chi_square")),
        text("; ".join(fit.get("diagnostic_flags", [])) or fit.get("reason") or "none"),
    ] for group, fit in sorted(groups.items())]


def _result_row(label: str, role: str, summary: dict[str, Any]) -> list[str]:
    macro = summary.get("macro", {})
    d3, d5 = macro.get("d3", {}), macro.get("d5", {})
    suppression = macro.get("error_suppression_d3_over_d5", {})
    groups = summary.get("group_fits", summary.get("per_group", {}))
    return [
        text(label), text(role),
        interval(d3.get("logical_error_per_round"), d3.get("ci95"), percent=True),
        interval(d5.get("logical_error_per_round"), d5.get("ci95"), percent=True),
        interval(suppression.get("value"), suppression.get("ci95")),
        text(len(groups)) if groups else "—",
    ]


def _score_state(score: dict[str, Any] | None) -> str:
    if score is None:
        return "held-out evaluation pending"
    coverage = score.get("conditions") == 130 and score.get("total_shots") == 3_250_000
    if score.get("split") != "test" or not coverage:
        return "score artifact present; full held-out coverage not established"
    if score.get("summary", {}).get("status") != "ok":
        return "full held-out scoring available; some decay fits are undefined"
    return "full held-out scores available; inspect fit diagnostics"


def _loss_curve(history: list[dict[str, Any]], label: str) -> str:
    points = []
    for row in history:
        x, y = row.get("elapsed_seconds"), row.get("loss")
        if all(isinstance(v, (int, float)) and math.isfinite(v) for v in (x, y)):
            points.append((float(x) / 3600, float(y)))
    if len(points) < 2:
        return ""
    points.sort()
    if len(points) > 400:
        points = [points[index * (len(points) - 1) // 399] for index in range(400)]
    low_x, high_x = points[0][0], points[-1][0]
    low_y, high_y = min(y for _, y in points), max(y for _, y in points)
    dx, dy = max(high_x - low_x, 1e-9), max(high_y - low_y, 1e-6)
    coords = " ".join(f"{50 + 530 * (x-low_x)/dx:.2f},{135 - 100 * (y-low_y)/dy:.2f}" for x, y in points)
    return (
        f"<figure><figcaption>{text(label)} · validation objective (main BCE + 0.5 auxiliary BCE)</figcaption>"
        f"<svg viewBox='0 0 620 180' role='img' aria-label='{text(label)} validation loss'>"
        "<path d='M50 30 V145 H585' fill='none' stroke='#aab1b7'/>"
        f"<polyline points='{coords}' fill='none' stroke='#244c62' stroke-width='1.6'/>"
        f"<text x='7' y='39'>{high_y:.3f}</text><text x='7' y='138'>{low_y:.3f}</text>"
        f"<text x='50' y='166'>{low_x:.2f} h</text><text x='540' y='166'>{high_x:.2f} h</text>"
        "</svg></figure>"
    )


def _job_status(state: Any, reason: Any = None) -> str:
    label = str(state or "not recorded")
    if reason:
        label += "; " + str(reason)
    upper = label.upper()
    if "DEPENDENCYNEVERSATISFIED" in upper or "INVALID_DEPEND" in upper:
        return f"<strong class='state-failed'>Dependency blocked · {text(label)}</strong>"
    if "DEPENDENCY" in upper:
        return f"<span class='state-waiting'>Waiting on dependency · {text(label)}</span>"
    if any(token in upper for token in ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED")):
        return f"<strong class='state-failed'>Failed / interrupted · {text(label)}</strong>"
    if "CANCELLED" in upper:
        return f"<span class='state-cancelled'>Cancelled · {text(label)}</span>"
    return text(label)


def _gate_status(gate: dict[str, Any]) -> str:
    if gate.get("status"):
        return str(gate["status"])
    if gate.get("passed") is True:
        return "passed"
    # Older snapshots had only passed=false. An unfinished stage must not be
    # described as failed merely because its completion check is still false.
    groups = gate.get("groups", {})
    if groups and any(not group.get("checks", {}).get("stage_stopped") for group in groups.values()):
        return "waiting_for_completed_stages"
    return "failed" if groups else "not established"


def _operational_status(
    accounting: dict[str, Any] | None, ledger: dict[str, Any] | None,
    gate: dict[str, Any] | None,
) -> str:
    result = ""
    if gate:
        status = _gate_status(gate)
        style = "state-failed" if status == "failed" else "state-waiting"
        result += (
            f"<p>Representative learning gate: <strong class='{style}'>{text(status)}</strong>. "
            f"Checked: {text(gate.get('checked_utc', gate.get('updated_utc', 'timestamp not recorded')))}. "
            "This gate concerns synthetic-validation learning and completed development stages, not test accuracy or convergence.</p>"
        )
        if status == "waiting_for_completed_stages":
            result += "<p class='note'>Training stages are still incomplete; this waiting result is not a failed training job.</p>"
    if accounting:
        jobs = {str(job.get("id")): job for job in (ledger or {}).get("jobs", [])}
        exceptions = []
        for row in accounting.get("records", []):
            formatted = _job_status(row.get("state"), row.get("reason"))
            if "class='state-" in formatted:
                job = jobs.get(str(row.get("job_id", "")).split("_", 1)[0], {})
                exceptions.append([
                    text(row.get("job_id", "—")), text(job.get("name", "—")), formatted,
                    text(job.get("status", job.get("note", "—"))),
                ])
        if exceptions:
            result += f"<p>Job exceptions and dependency waits in the accounting snapshot at {text(accounting.get('updated_utc', 'unknown time'))}:</p>"
            result += table(["Job", "Purpose", "Slurm state", "Campaign note"], exceptions)
            result += "<p class='note'>Cancellation may be deliberate, for example replacing a pending allocation. Job states never substitute for missing neural test scores.</p>"
    return result


def _resources(
    ledger: dict[str, Any] | None, pilot: dict[str, Any] | None,
    accounting: dict[str, Any] | None = None,
) -> str:
    if accounting is not None:
        jobs = {str(job.get("id")): job for job in (ledger or {}).get("jobs", [])}
        rows = []
        for record in accounting.get("records", []):
            job = jobs.get(str(record.get("job_id", "")).split("_", 1)[0], {})
            rows.append([
                text(record.get("job_id", "—")), text(job.get("name", "—")),
                _job_status(record.get("state"), record.get("reason")),
                number(record.get("allocated_gpus"), 0),
                number(record.get("elapsed_seconds"), 2, scale=1/3600),
                number(record.get("gpu_hours"), 3), text(record.get("nodes", "—")),
            ])
        result = (
            f"<p>Slurm accounting snapshot: <strong>{text(accounting.get('updated_utc', 'timestamp not recorded'))}</strong>. "
            f"{text(accounting.get('scope', 'Registered campaign jobs only.'))}.</p>"
            f"<p><strong>Actual allocated GPU-hours: {number(accounting.get('actual_allocated_gpu_hours'), 3)}</strong>; "
            f"active allocated GPUs at that snapshot: {number(accounting.get('active_allocated_gpus'), 0)}. "
            f"Remaining actual budget: {number(accounting.get('remaining_actual_gpu_hours'), 3)} GPU-hours.</p>"
            f"<p>Declared allocation caps: {number(accounting.get('declared_allocation_ceiling_gpu_hours'), 2)} GPU-hours. "
            f"Policy ceilings: {number(accounting.get('max_actual_allocated_gpu_hours', (ledger or {}).get('max_actual_allocated_gpu_hours')), 0)} GPU-hours "
            f"and {number(accounting.get('max_concurrent_gpus', (ledger or {}).get('max_concurrent_gpus')), 0)} concurrent GPUs. "
            "This accounting snapshot takes precedence over missing per-job actuals in the submission ledger.</p>"
        )
        result += table(["Job", "Purpose", "Slurm state", "Allocated GPUs", "Allocated elapsed · h", "Actual · GPU-h", "Node"], rows)
        result += f"<p class='note'>{text(accounting.get('accounting_caveat', 'Slurm accounting may lag.'))}. Training elapsed time is not a substitute for allocated time; no H200-equivalence conversion is used.</p>"
        return result
    if ledger is None:
        return "<p>No campaign accounting file is available. Missing actual usage is not zero.</p>"
    rows: list[list[str]] = []
    cap_total = actual_total = 0.0
    actual_count = gpu_jobs = 0
    for job in ledger.get("jobs", []):
        gpus = job.get("gpu_count", 0)
        tasks = job.get("array_tasks", 1)
        hours = job.get("max_hours_per_task")
        numeric = all(isinstance(value, (int, float)) and math.isfinite(value) for value in (gpus, tasks, hours))
        cap = gpus * tasks * hours if numeric else None
        status = str(job.get("status", "not recorded"))
        if status.lower().startswith("cancelled pending"):
            cap = 0
        actual = job.get("actual_gpu_hours")
        if isinstance(gpus, (int, float)) and gpus > 0:
            gpu_jobs += 1
            if isinstance(actual, (int, float)) and math.isfinite(actual):
                actual_total += actual
                actual_count += 1
        if cap is not None:
            cap_total += cap
        rows.append([
            text(job.get("id", "—")), text(job.get("name", "—")), text(status),
            number(tasks, 0), number(gpus, 0), number(cap, 2), number(actual, 3),
        ])
    completeness = "complete for listed GPU jobs" if actual_count == gpu_jobs else "partial; missing actuals are not zero"
    result = (
        f"<p>Policy ceiling: <strong>{number(ledger.get('max_actual_allocated_gpu_hours'), 0)} allocated GPU-hours</strong>; "
        f"at most {number(ledger.get('max_concurrent_gpus'), 0)} concurrent GPUs. "
        "Accounting uses actual card-hours, with no H200-equivalence conversion.</p>"
        f"<p>Listed allocation caps sum to {number(cap_total, 2)} GPU-hours. "
        f"Recorded actuals sum to {number(actual_total, 3)} GPU-hours across {actual_count}/{gpu_jobs} GPU job records "
        f"({text(completeness)}). Training elapsed time is not a substitute for Slurm allocation accounting.</p>"
    )
    result += table(["Job", "Purpose", "Recorded status", "Tasks", "GPU / task", "Cap · GPU-h", "Actual · GPU-h"], rows)
    if pilot:
        result += f"<p class='note'>The separate pilot audit records {number(pilot.get('allocated_gpu_hours'), 4)} GPU-hours for its two pilots; this is not added again to the campaign ledger.</p>"
    return result


def render_report(asset_root: Path, campaign: str, *, evidence_root: Path | None = None) -> str:
    if Path(campaign).name != campaign or campaign in {"", ".", ".."}:
        raise ValueError("campaign must be one directory name")
    evidence_root = evidence_root or Path(__file__).resolve().parents[1] / "reproduction"
    evidence = Evidence()
    classical = evidence.read(evidence_root / "classical-summary.json")
    pilot = evidence.read(evidence_root / "pilot-summary.json")
    ledger = evidence.read(asset_root / "campaign.json")
    accounting = evidence.read(asset_root / "accounting.json")
    campaign_root = asset_root / campaign
    gate = evidence.read(campaign_root / "representative-gate.json")
    if gate is None:
        gate = evidence.read(campaign_root / "gate.json")
    scores = {stage: evidence.read(campaign_root / f"scores-{stage}.json") for stage in STAGES}
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    verified = bool(classical and classical.get("verification", {}).get("all_130_conditions_verified"))
    classical_status = "Classical replay independently verified" if verified else "Classical verification evidence unavailable"
    body = [
        f"<header><p class='kicker'>Reproduction evidence · {text(campaign)}</p><h1>Sycamore surface-code decoding</h1>"
        f"<p class='lead'>{classical_status}. Neural results are reported only when saved held-out score artifacts exist.</p>"
        f"<p class='muted'>Snapshot generated {text(generated)}. This page does not poll jobs or run evaluation.</p></header>",
        "<section><h2>Evidence status</h2>",
        table(["Stage", "Evidence available"], [
            ["Classical baselines", classical_status],
            *[[text(label), text(_score_state(scores[stage]))] for stage, label in STAGES.items()],
        ]),
        _operational_status(accounting, ledger, gate),
        "<p class='note'>A submitted job, saved checkpoint, or completed runtime pilot does not establish convergence or completed reproduction. Test scores and artifact review remain separate requirements.</p></section>",
        "<section><h2>Held-out logical error rates</h2><p>Logical errors per round are shown as percentages; brackets are 95% confidence intervals. Lower is better. Each patch and basis is fitted separately across rounds 3–25 before averaging within a distance.</p>",
    ]
    result_rows: list[list[str]] = []
    group_rows: list[list[str]] = []
    if classical:
        for method, summary in classical.get("methods", {}).items():
            label = METHOD_NAMES.get(method, method)
            result_rows.append(_result_row(label, summary.get("role", "classical reference"), summary))
            group_rows.extend(_fit_rows(label, summary.get("group_fits", {})))
    for stage, label in STAGES.items():
        score = scores[stage]
        if score is None:
            result_rows.append([text(label), "evaluation pending", "—", "—", "—", "—"])
            continue
        summary = score.get("summary", {})
        role = f"{score.get('split', 'unknown split')}; {score.get('conditions', '?')}/130 conditions; fits {summary.get('status', 'unknown')}"
        result_rows.append(_result_row(label, role, summary))
        group_rows.extend(_fit_rows(label, summary.get("per_group", {})))
    body.append(table(["Method / stage", "Evidence type", "d3 · % / round", "d5 · % / round", "d3 / d5", "Groups"], result_rows))
    body.append("<p class='note'>Distance-3 averages eight patch/basis fits; distance-5 averages two. The suppression ratio is descriptive across different measured patches and does not establish a fault-tolerance threshold. Released BP/TN predictions are rescored references, not newly reproduced decoder implementations.</p>")
    body.append("<details><summary>Group fits and diagnostics</summary>" + table(
        ["Method / stage", "Group", "LER · % / round [95% CI]", "F0", "Weighted R²", "Reduced χ²", "Flags / reason"], group_rows,
    ) + "</details>")
    body.append("<p class='note'>F0 = exp(intercept) extrapolates the free-intercept fit to round zero. F0 slightly above one can reflect extrapolation across omitted initialization/readout boundaries; it is not an observed probability. Fits are not clipped or filtered. Large reduced χ² can indicate departures from the exponential model even when R² is high.</p></section>")

    body.append("<section><h2>Probabilities and paired comparisons</h2>")
    calibration_rows: list[list[str]] = []
    paired_rows: list[list[str]] = []
    for stage, label in STAGES.items():
        score = scores[stage] or {}
        for distance, record in score.get("calibration", {}).get("macro", {}).items():
            calibration_rows.append([text(label), text(distance), number(record.get("negative_log_likelihood"), 5), number(record.get("brier_score"), 5)])
        for comparator, comparison in score.get("paired_comparisons", {}).items():
            estimates = comparison.get("comparisons", {})
            d3, d5 = estimates.get("d3", {}), estimates.get("d5", {})
            paired_rows.append([
                text(label), text(METHOD_NAMES.get(comparator, comparator)),
                interval(d3.get("logical_error_per_round_a_minus_b"), d3.get("ci95"), percent=True),
                interval(d5.get("logical_error_per_round_a_minus_b"), d5.get("ci95"), percent=True),
                text(comparison.get("reason") or comparison.get("status", "unknown")),
            ])
    body.append(table(["Neural stage", "Distance / aggregate", "NLL", "Brier"], calibration_rows))
    body.append("<p class='note'>These proper scores use official-path probabilities. Lower is better. Aggregation weights conditions equally; the all-condition aggregate contains more d3 conditions. Classical artifacts contain hard decisions, so no comparable confidence-calibration result is claimed. NLL/Brier combine accuracy and calibration and do not replace reliability diagnostics.</p>")
    body.append(table(["Neural stage", "Comparator", "d3 difference · percentage points", "d5 difference · percentage points", "Status"], paired_rows))
    body.append("<p class='note'>Paired differences are neural minus comparator on the same shots; negative values favor the neural model. Confidence intervals resample matched error outcomes within each condition.</p></section>")

    body.append("<section><h2>Training snapshots</h2><p>Rows below show the latest recorded progress, not a live Slurm state. Pretraining validation is synthetic; fine-tuning validation uses the held-out even hardware subset. These validation errors are shot errors, not the test logical-error-per-round metric above.</p>")
    curves: list[str] = []
    for stage, label in STAGES.items():
        rows = []
        for path in sorted((campaign_root / "models").glob(f"*/{stage}/progress.json")):
            progress = evidence.read(path)
            if progress is None:
                continue
            metadata = evidence.read(path.parent / "metadata.json") or {}
            history = evidence.history(path.parent / "history.jsonl")
            latest = progress.get("latest_validation") or (history[-1] if history else {})
            group = path.parent.parent.name
            state = str(progress.get("status", "unknown"))
            if progress.get("stop_reason"):
                state += "; " + str(progress["stop_reason"])
            rows.append([
                text(group), text(state), number(progress.get("global_steps"), 0),
                number(progress.get("epoch"), 0), number(progress.get("seen_samples"), 0),
                number(metadata.get("protocol", {}).get("training_samples_per_epoch"), 0),
                number(progress.get("best_validation_loss"), 5),
                number(latest.get("main_error"), 2, scale=100),
                number(progress.get("elapsed_seconds"), 2, scale=1/3600),
                text(progress.get("updated_at", "—")),
            ])
            curve = _loss_curve(history, f"{label} · {group}")
            if curve:
                curves.append(curve)
        body.append(f"<h3>{text(label)} · {len(rows)}/10 groups with progress records</h3>")
        body.append(table(["Group", "Recorded status", "Steps", "Completed epochs", "Samples seen", "Population / epoch", "Best objective", "Latest validation error · %", "Stage elapsed · h", "Last update · UTC"], rows))
    if curves:
        body.append("<details><summary>Validation objective histories</summary>" + "".join(curves) + "</details>")
    body.append("</section><section><h2>GPU pilot evidence</h2>")
    pilot_rows = [[
        text(item.get("group", "—")), text(item.get("gpu", "—")),
        number(item.get("parameters"), 0), number(item.get("batch_size"), 0),
        number(item.get("optimizer_steps"), 0), number(item.get("steady_samples_per_second"), 1),
        number(item.get("fraction_of_epoch_seen"), 3, scale=100),
        text("established" if item.get("convergence_established") is True else "not established"),
    ] for item in (pilot or {}).get("pilots", [])]
    body.append(table(["Group", "GPU", "Parameters", "Batch", "Pilot steps", "Steady samples / s", "Epoch seen · %", "Convergence"], pilot_rows))
    body.append("<p class='note'>Pilot throughput excludes warmup, loading, validation, checkpoint writes and Slurm startup. A short pilot establishes execution and approximate throughput only; it does not establish hardware decoding accuracy.</p></section>")
    body.append("<section><h2>Resource accounting</h2>" + _resources(ledger, pilot, accounting))
    if classical:
        resources = classical.get("resources", {})
        body.append(f"<p>Classical replay: {number(resources.get('application_wall_seconds'), 1)} s application wall time on {number(resources.get('allocated_cpu_cores'), 0)} allocated CPU cores; Slurm job {text(resources.get('slurm_job_id', '—'))}. Offline batch timing is not streaming decoder latency.</p>")
    body.append("</section>")
    body.append("""<section><h2>Protocol and limits</h2><ul>
<li>Dataset: 130 Sycamore conditions, four d3 patches and one d5 patch, X/Z basis, rounds 1–25. Each condition has 50,000 shots; all 25,000 odd-indexed shots are test data.</li>
<li>Hardware development uses the first 19,880 even shots for training and the remaining 5,120 for validation. Matching calibration is selected on validation before odd-shot scoring. Released even-calibrated DEMs include validation syndromes, but no validation logical labels.</li>
<li>Native QRENND uses two LSTM layers with widths64/253 for d3/d5. Both heads learn all-data measurement-flip parity. A circuit-derived exact GF(2) transform maps labels and predictions to/from the official path observable. Initial absent off-basis detectors are zero-padded.</li>
<li>Fixed simulated training populations use the released approximate Pauli circuits, not AlphaQubit's Pauli+ leakage simulator. Training budget, data generation and runtime are adaptations; no exact reproduction of the paper's final scores is claimed.</li>
<li>Bootstrap intervals condition on these circuits and assume independent shots. They omit hardware drift, shared events, model-seed variability and exponential-fit misspecification.</li>
<li>Odd labels and historical predictions are public. This is a documented held-out development protocol, not a blind contamination-resistant benchmark.</li>
<li>These d3/d5 experiments do not establish behavior at distance11, on another processor, with soft readout, or in real-time fault-tolerant control.</li>
</ul></section>""")
    if evidence.warnings:
        body.append("<section><h2>Unavailable evidence</h2><ul>" + "".join(f"<li>{text(warning)}</li>" for warning in evidence.warnings) + "</ul></section>")
    body.append("""<section><h2>Sources and provenance</h2><p>
<a href="https://doi.org/10.1103/PhysRevResearch.7.013029">QRENND paper · PRR2025</a> ·
<a href="https://arxiv.org/html/2307.03280v2">QRENND architecture and protocol</a> ·
<a href="https://zenodo.org/records/6804040">Sycamore data release</a> ·
<a href="https://github.com/oscarhiggott/PyMatching">PyMatching</a> ·
<a href="https://www.nature.com/articles/s41586-024-08148-8">AlphaQubit · Nature2024</a></p>
<p>AlphaQubit motivates the task; the neural baseline here is QRENND, not an official AlphaQubit implementation. This report reads saved evidence and does not decode shots, train models or alter checkpoints.</p>""")
    if classical:
        body.append(f"<p>Verified classical scores SHA256: <code>{text(classical.get('sources', {}).get('scores_sha256', 'unavailable'))}</code>.</p>")
    body.append("<details><summary>Snapshot input files and SHA256</summary>" + table(
        ["Input evidence", "SHA256"], [[f"<code>{text(row['path'])}</code>", f"<code>{text(row['sha256'])}</code>"] for row in evidence.files],
    ) + "</details></section>")
    css = """
:root{color-scheme:light;font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#20262b;background:#fff;font-size:15px;line-height:1.55}
.state-failed{color:#8b3028;font-weight:600}.state-cancelled{color:#795927;font-weight:600}.state-waiting{color:#315d76;font-weight:600}
body{margin:0}main{max-width:1250px;margin:auto;padding:36px 32px 64px}header{padding-bottom:20px;border-bottom:1px solid #cbd0d4}h1{font-size:30px;font-weight:600;letter-spacing:-.025em;margin:8px 0 14px}h2{font-size:20px;font-weight:600;margin:0 0 12px}h3{font-size:16px;margin:22px 0 9px}p{margin:10px 0}section{margin-top:36px}.kicker{font-size:12px;letter-spacing:.07em;text-transform:uppercase;color:#5b6570}.lead{font-size:17px}.muted,.note,.ci{color:#65717b}.note{font-size:13px}.table-wrap{overflow-x:auto;margin:14px 0}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}th,td{text-align:left;vertical-align:top;padding:10px 10px;border-bottom:1px solid #dce1e4}th{font-weight:600;border-top:1px solid #aeb7be;border-bottom:1px solid #aeb7be;background:#f7f8f9}td .ci{white-space:nowrap;font-size:12px}caption{text-align:left;margin-bottom:8px}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:11px;overflow-wrap:anywhere}a{color:#244c62;text-underline-offset:3px}li{margin:9px 0}details{margin:18px 0}summary{cursor:pointer;color:#244c62;font-size:14px}figure{margin:22px 0;max-width:680px}figcaption{font-size:13px;color:#4d5963}svg{width:100%;height:auto}svg text{font-size:10px;fill:#5b6570}.toolbar{max-width:1250px;margin:0 auto;padding:16px 32px 0;font-size:12px;color:#65717b}button{font:inherit;color:#244c62;background:#fff;border:1px solid #b9c2c9;padding:5px 9px;cursor:pointer;margin-right:10px}main[contenteditable=true]{outline:1px dashed #aeb7be;outline-offset:-8px}@media(max-width:700px){main{padding:24px 16px}h1{font-size:25px}.toolbar{padding:12px 16px 0}}@media print{.toolbar{display:none}main{max-width:none;padding:0}section{break-inside:avoid}.table-wrap{overflow:visible}details>summary{display:none}details>*{display:block}a{color:inherit}}
"""
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Sycamore baseline evidence · {text(campaign)}</title><style>{css}</style></head><body>"
        "<div class='toolbar'><button id='edit' type='button'>Enable text editing</button>Standalone snapshot; editing this page does not change source evidence.</div>"
        "<main id='report' contenteditable='false' spellcheck='false'>" + "".join(body) + "</main>"
        "<script>document.getElementById('edit').addEventListener('click',function(){const r=document.getElementById('report');const active=r.contentEditable==='true';r.contentEditable=active?'false':'true';this.textContent=active?'Enable text editing':'Finish text editing';});</script>"
        "</body></html>\n"
    )


def write_report(asset_root: Path, campaign: str, output: Path) -> None:
    if output.suffix.lower() != ".html":
        raise ValueError("Report output must have an .html suffix")
    content = render_report(asset_root, campaign)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--campaign", default="native-v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_report(args.asset_root, args.campaign, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "source": "saved evidence only; no training or evaluation"}))


if __name__ == "__main__":
    main()
