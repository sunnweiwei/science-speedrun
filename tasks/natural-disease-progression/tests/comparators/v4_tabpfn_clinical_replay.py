#!/usr/bin/env python3
"""Sealed-side V4 TabPFN-3.5 longitudinal clinical comparator & paired bootstrap replay.

Fixes BOTH methodology flaws identified in the V3 audit (`tabular_baseline.py` & `tab-auto.py`):
1. Eliminates `hash(code) % 128` / `% 64` feature collisions AND `tab-auto.py:24` lab-stat
   truncation by fitting an explicit, collision-free train-only vocabulary:
   - Top frequent exact diagnosis codes by train patient prevalence (default 256, zero collisions)
   - Top frequent exact medication codes by train patient prevalence (default 128, zero collisions)
   - All 26 ICD chapter log-counts (`A`..`Z`)
   - All 7 lab summary statistics (`last, mean, std, min, max, log_count, missing`) for each of the
     top frequent train lab `(code, unit)` pairs (default 48)
   - Cutoff-bounded patient demographics and per-modality burden/recency summaries (13 features)
2. Eliminates severe `3,072`-row training support truncation (`min(512, pos)` + `3072 - sp` neg)
   by keeping ALL eligible complete-year positive cases plus stratified negatives up to
   `--max-support` (default `10000`; set `0` to use the full ~19k eligible complete-year train
   cohort on 80 GB H100) with `TabPFNClassifier(n_estimators="auto", fit_mode="fit_with_cache",
   ignore_pretraining_limits=True)` and exact Bayes prior-odds correction, scoring all `11,424`
   held-out test patients and computing the 2,000-draw paired patient-level bootstrap 95% CI.

Kept strictly under `natural-disease-progression/tests/comparators/` so it never enters `/workspace`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TASK_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = TASK_ROOT / "environment" / "evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from panel_scoring import (  # noqa: E402
    MIN_POSITIVES,
    PANEL,
    PRIMARY,
    has_family_history,
    history_code,
    panel_score,
)
from risk_scoring import HORIZONS, _full_year  # noqa: E402

CHAPTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LAB_METRICS = ("last", "mean", "std", "min", "max", "log_count", "missing")
DEFAULT_TOP_DIAGNOSES = 256
DEFAULT_TOP_MEDICATIONS = 128
DEFAULT_TOP_LABS = 48
DEFAULT_MAX_SUPPORT = 10000
EXPECTED_TEST_PATIENTS = 11424
EXPECTED_ELIGIBLE_ENDPOINTS = 17


def enforce_offline_network() -> None:
    """Disable outbound AF_INET/AF_INET6 sockets during local restricted-data inference."""
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TABPFN_DISABLE_TELEMETRY="1",
        DO_NOT_TRACK="1",
    )
    orig_connect = socket.socket.connect
    if getattr(orig_connect, "_ndp_offline_patched", False):
        return

    def local_only(self: socket.socket, addr: Any) -> Any:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError("Outbound network access is disabled during sealed clinical replay")
        return orig_connect(self, addr)

    local_only._ndp_offline_patched = True  # type: ignore[attr-defined]
    socket.socket.connect = local_only  # type: ignore[method-assign]


def fit_collision_free_feature_spec(
    train_dir: Path,
    top_diagnoses: int = DEFAULT_TOP_DIAGNOSES,
    top_medications: int = DEFAULT_TOP_MEDICATIONS,
    top_labs: int = DEFAULT_TOP_LABS,
    min_lab_numeric_n: int = 100,
) -> dict[str, Any]:
    """Fit an exact, collision-free clinical vocabulary strictly on `train/events.parquet`."""
    events_path = str(train_dir / "events.parquet")
    db = duckdb.connect()
    try:
        diag_rows = db.execute(
            """
            SELECT code,
                   count(DISTINCT sample_id) AS patient_n,
                   count(*) AS event_n
            FROM read_parquet(?)
            WHERE event_type = 'diagnosis' AND code IS NOT NULL AND code <> ''
            GROUP BY code
            ORDER BY patient_n DESC, event_n DESC, code ASC
            LIMIT ?
            """,
            [events_path, int(top_diagnoses)],
        ).fetchall()

        med_rows = db.execute(
            """
            SELECT code,
                   count(DISTINCT sample_id) AS patient_n,
                   count(*) AS event_n
            FROM read_parquet(?)
            WHERE event_type IN ('prescription', 'administration')
              AND code IS NOT NULL AND code <> ''
            GROUP BY code
            ORDER BY patient_n DESC, event_n DESC, code ASC
            LIMIT ?
            """,
            [events_path, int(top_medications)],
        ).fetchall()

        lab_rows = db.execute(
            """
            SELECT code,
                   coalesce(unit, '') AS unit,
                   count(try_cast(value AS DOUBLE)) AS numeric_n
            FROM read_parquet(?)
            WHERE event_type = 'lab' AND code IS NOT NULL AND code <> ''
            GROUP BY code, coalesce(unit, '')
            HAVING numeric_n >= ?
            ORDER BY numeric_n DESC, code ASC, unit ASC
            LIMIT ?
            """,
            [events_path, int(min_lab_numeric_n), int(top_labs)],
        ).fetchall()
    finally:
        db.close()

    diag_codes = [str(code) for code, _, _ in diag_rows]
    med_codes = [str(code) for code, _, _ in med_rows]
    if len(set(diag_codes)) != len(diag_codes):
        raise AssertionError("Duplicate diagnosis code in exact vocabulary")
    if len(set(med_codes)) != len(med_codes):
        raise AssertionError("Duplicate medication code in exact vocabulary")

    return {
        "schema": "ndp-tabular-features-collision-free/v4",
        "fit_source": "train/events.parquet only",
        "hash_collisions": False,
        "lab_summary_metrics": list(LAB_METRICS),
        "top_diagnoses": [
            {"code": str(code), "train_patient_n": int(pat_n), "train_event_n": int(ev_n)}
            for code, pat_n, ev_n in diag_rows
        ],
        "top_medications": [
            {"code": str(code), "train_patient_n": int(pat_n), "train_event_n": int(ev_n)}
            for code, pat_n, ev_n in med_rows
        ],
        "top_labs": [
            {"code": str(code), "unit": str(unit), "train_numeric_n": int(num_n)}
            for code, unit, num_n in lab_rows
        ],
    }


def build_collision_free_features(
    input_dir: Path,
    spec: dict[str, Any],
    output: Path | None = None,
) -> dict[str, Any]:
    """Materialize collision-free patient feature matrix from cutoff-bounded public events.

    `sample_id` is used strictly as a keyed row alignment identifier and never enters `features`.
    """
    if spec.get("schema") != "ndp-tabular-features-collision-free/v4":
        raise ValueError(f"Unsupported feature spec schema: {spec.get('schema')}")

    started = time.monotonic()
    samples = sorted(
        pq.read_table(input_dir / "samples.parquet").to_pylist(),
        key=lambda row: row["sample_id"],
    )
    ids = [str(row["sample_id"]) for row in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample_id in samples.parquet")
    row_index = {sid: i for i, sid in enumerate(ids)}

    basic_names = ["age", "sex_f", "log_events", "log_stays", "history_days"]
    for modality in ("diagnosis", "lab", "prescription", "administration"):
        basic_names.extend((f"log_{modality}_events", f"days_since_last_{modality}"))

    diag_vocab = [str(item["code"]) for item in spec["top_diagnoses"]]
    med_vocab = [str(item["code"]) for item in spec["top_medications"]]
    lab_vocab = spec["top_labs"]
    if len(set(diag_vocab)) != len(diag_vocab) or len(set(med_vocab)) != len(med_vocab):
        raise ValueError("Vocabulary contains duplicate codes")

    feature_names: list[str] = list(basic_names)
    feature_names.extend(f"diagnosis_chapter_{ch}" for ch in CHAPTERS)
    feature_names.extend(f"diagnosis_exact_{i}_{code}" for i, code in enumerate(diag_vocab))
    feature_names.extend(f"medication_exact_{i}_{code}" for i, code in enumerate(med_vocab))
    feature_names.extend(
        f"lab_{i}_{metric}" for i in range(len(lab_vocab)) for metric in LAB_METRICS
    )

    x = np.zeros((len(ids), len(feature_names)), dtype=np.float32)
    x[:, 0] = [float(row["approx_age"]) for row in samples]
    x[:, 1] = [1.0 if row["gender"] == "F" else 0.0 for row in samples]
    # Initialize days_since_last_* to -1.0 when a patient has zero events in that modality.
    for m in range(4):
        x[:, 6 + 2 * m] = -1.0

    chapter_offset = len(basic_names)
    diag_offset = chapter_offset + len(CHAPTERS)
    med_offset = diag_offset + len(diag_vocab)
    lab_offset = med_offset + len(med_vocab)

    if len(lab_vocab) > 0:
        x[:, lab_offset + 6 :: len(LAB_METRICS)] = 1.0

    event_file = str(input_dir / "events.parquet")
    db = duckdb.connect()
    try:
        aggregate = db.execute(
            """
            SELECT sample_id,
                   count(*) AS n,
                   count(DISTINCT stay_key) AS stays,
                   max(seconds_before_cutoff) / 86400.0 AS history_days,
                   count(*) FILTER (WHERE event_type = 'diagnosis') AS diagnosis_n,
                   min(seconds_before_cutoff) FILTER (WHERE event_type = 'diagnosis') / 86400.0 AS diagnosis_recent,
                   count(*) FILTER (WHERE event_type = 'lab') AS lab_n,
                   min(seconds_before_cutoff) FILTER (WHERE event_type = 'lab') / 86400.0 AS lab_recent,
                   count(*) FILTER (WHERE event_type = 'prescription') AS prescription_n,
                   min(seconds_before_cutoff) FILTER (WHERE event_type = 'prescription') / 86400.0 AS prescription_recent,
                   count(*) FILTER (WHERE event_type = 'administration') AS administration_n,
                   min(seconds_before_cutoff) FILTER (WHERE event_type = 'administration') / 86400.0 AS administration_recent
            FROM read_parquet(?)
            GROUP BY sample_id
            """,
            [event_file],
        ).fetchall()
        for row in aggregate:
            sid = str(row[0])
            if sid not in row_index:
                continue
            i = row_index[sid]
            x[i, 2:5] = (np.log1p(row[1]), np.log1p(row[2]), float(row[3] or 0.0))
            for m in range(4):
                cnt = row[4 + 2 * m] or 0
                recent = row[5 + 2 * m]
                x[i, 5 + 2 * m] = np.log1p(cnt)
                x[i, 6 + 2 * m] = float(recent) if recent is not None else -1.0

        chapter_index = {ch: j for j, ch in enumerate(CHAPTERS)}
        chapters = db.execute(
            """
            SELECT sample_id, left(code, 1) AS chapter, count(*) AS n
            FROM read_parquet(?)
            WHERE event_type = 'diagnosis' AND code IS NOT NULL
            GROUP BY sample_id, left(code, 1)
            """,
            [event_file],
        ).fetchall()
        for sid, chapter, n in chapters:
            sid = str(sid)
            if sid in row_index and chapter in chapter_index:
                x[row_index[sid], chapter_offset + chapter_index[chapter]] = np.log1p(n)

        if diag_vocab:
            db.register(
                "diag_vocab",
                pa.table(
                    {
                        "code": pa.array(diag_vocab, type=pa.string()),
                        "col_idx": np.arange(len(diag_vocab), dtype=np.int32),
                    }
                ),
            )
            diag_counts = db.execute(
                """
                SELECT e.sample_id, v.col_idx, count(*) AS n
                FROM read_parquet(?) e
                JOIN diag_vocab v ON e.code = v.code
                WHERE e.event_type = 'diagnosis'
                GROUP BY e.sample_id, v.col_idx
                """,
                [event_file],
            ).fetchall()
            for sid, col_idx, n in diag_counts:
                sid = str(sid)
                if sid in row_index:
                    x[row_index[sid], diag_offset + int(col_idx)] = np.log1p(n)

        if med_vocab:
            db.register(
                "med_vocab",
                pa.table(
                    {
                        "code": pa.array(med_vocab, type=pa.string()),
                        "col_idx": np.arange(len(med_vocab), dtype=np.int32),
                    }
                ),
            )
            med_counts = db.execute(
                """
                SELECT e.sample_id, v.col_idx, count(*) AS n
                FROM read_parquet(?) e
                JOIN med_vocab v ON e.code = v.code
                WHERE e.event_type IN ('prescription', 'administration')
                GROUP BY e.sample_id, v.col_idx
                """,
                [event_file],
            ).fetchall()
            for sid, col_idx, n in med_counts:
                sid = str(sid)
                if sid in row_index:
                    x[row_index[sid], med_offset + int(col_idx)] = np.log1p(n)

        if lab_vocab:
            db.register(
                "lab_vocab",
                pa.table(
                    {
                        "code": pa.array([str(r["code"]) for r in lab_vocab], type=pa.string()),
                        "unit": pa.array([str(r["unit"]) for r in lab_vocab], type=pa.string()),
                        "lab_idx": np.arange(len(lab_vocab), dtype=np.int32),
                    }
                ),
            )
            lab_rows = db.execute(
                """
                SELECT e.sample_id,
                       s.lab_idx,
                       arg_min(try_cast(e.value AS DOUBLE), e.seconds_before_cutoff) AS last_val,
                       avg(try_cast(e.value AS DOUBLE)) AS mean_val,
                       stddev_pop(try_cast(e.value AS DOUBLE)) AS std_val,
                       min(try_cast(e.value AS DOUBLE)) AS min_val,
                       max(try_cast(e.value AS DOUBLE)) AS max_val,
                       count(try_cast(e.value AS DOUBLE)) AS numeric_n
                FROM read_parquet(?) e
                JOIN lab_vocab s ON e.code = s.code AND coalesce(e.unit, '') = s.unit
                WHERE e.event_type = 'lab'
                GROUP BY e.sample_id, s.lab_idx
                """,
                [event_file],
            ).fetchall()
            for sid, lab_idx, last_val, mean_val, std_val, min_val, max_val, num_n in lab_rows:
                sid = str(sid)
                if sid not in row_index or not num_n or last_val is None or mean_val is None:
                    continue
                base = lab_offset + int(lab_idx) * len(LAB_METRICS)
                x[row_index[sid], base : base + 7] = (
                    float(last_val),
                    float(mean_val),
                    float(std_val or 0.0),
                    float(min_val),
                    float(max_val),
                    float(np.log1p(num_n)),
                    0.0,
                )
    finally:
        db.close()

    if not np.isfinite(x).all():
        raise ValueError("Non-finite value materialized in collision-free feature matrix")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            ids=np.asarray(ids),
            features=x,
            feature_names=np.asarray(feature_names),
        )

    return {
        "ids": np.asarray(ids),
        "features": x,
        "feature_names": np.asarray(feature_names),
        "samples": len(ids),
        "num_features": int(x.shape[1]),
        "duration_seconds": round(time.monotonic() - started, 2),
    }


def select_support_indices(
    eligible_mask: np.ndarray,
    y_col: np.ndarray,
    max_support: int | None = DEFAULT_MAX_SUPPORT,
    seed: int = 20261103,
) -> tuple[np.ndarray, dict[str, int]]:
    """Select training support patients without the V3 3,072-row / 512-positive truncation bug.

    - If `max_support` is None, <= 0, or >= total eligible complete-year patients (~19k on MIMIC-IV
      train), ALL eligible positives and negatives are kept.
    - Otherwise (default `max_support = 10000`), ALL positives are retained whenever
      `n_pos < max_support`, and the remaining capacity is filled with uniformly sampled eligible
      negatives without replacement (corrected analytically via `prior_odds_correction`).
    """
    eligible_b = np.asarray(eligible_mask, dtype=bool)
    y_b = np.asarray(y_col, dtype=bool)
    pos = np.flatnonzero(eligible_b & y_b)
    neg = np.flatnonzero(eligible_b & ~y_b)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError(f"Endpoint requires both positive ({len(pos)}) and negative ({len(neg)}) train cases")

    rng = np.random.default_rng(seed)
    total_eligible = len(pos) + len(neg)
    if max_support is None or max_support <= 0 or total_eligible <= max_support:
        sp = pos.copy()
        sn = neg.copy()
    elif len(pos) < max_support:
        sp = pos.copy()
        sn = rng.choice(neg, size=max_support - len(sp), replace=False)
    else:
        target_pos = min(max(1, round(max_support * len(pos) / total_eligible)), max_support - 1)
        sp = rng.choice(pos, size=target_pos, replace=False)
        sn = rng.choice(neg, size=max_support - len(sp), replace=False)

    support = np.r_[sp, sn]
    rng.shuffle(support)
    meta = {
        "n_eligible_total": int(total_eligible),
        "n_pos_total": int(len(pos)),
        "n_neg_total": int(len(neg)),
        "n_support_total": int(len(support)),
        "n_pos_support": int(len(sp)),
        "n_neg_support": int(len(sn)),
    }
    return support, meta


def prior_odds_correction(
    prob: np.ndarray,
    n_pos_total: int,
    n_neg_total: int,
    n_pos_support: int,
    n_neg_support: int,
) -> np.ndarray:
    """Exact monotone Bayes prior-odds adjustment for case-control support sampling.

    Preserves patient ranking strictly (`Recall@5%` and `AUROC` invariant) while restoring
    population prevalence calibration.
    """
    p = np.asarray(prob, dtype=np.float64)
    if not np.isfinite(p).all() or (p < 0.0).any() or (p > 1.0).any():
        raise ValueError("Predicted probabilities must be finite and within [0, 1]")
    prior = n_pos_total / (n_pos_total + n_neg_total)
    context = n_pos_support / (n_pos_support + n_neg_support)
    if not (0.0 < prior < 1.0 and 0.0 < context < 1.0):
        raise ValueError("Invalid prior or support prevalence")
    if abs(prior - context) <= 1e-15:
        return p.copy()
    odds_ratio = (prior / (1.0 - prior)) / (context / (1.0 - context))
    denom = (1.0 - p) + p * odds_ratio
    calibrated = np.where(denom > 0.0, (p * odds_ratio) / denom, 0.0)
    return np.clip(calibrated, 0.0, 1.0)


def load_label_grid_365(
    label_dir: Path,
    ids: list[str],
    diseases: list[str],
    boundaries_parquet: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load keyed 365-day labels, validity masks, and full-year indicators."""
    sample_idx = {sid: i for i, sid in enumerate(ids)}
    disease_idx = {d: j for j, d in enumerate(diseases)}
    y = np.zeros((len(ids), len(diseases)), dtype=bool)
    valid = np.zeros_like(y)
    full = np.zeros(len(ids), dtype=bool)
    seen = 0

    con = duckdb.connect()
    try:
        rows = con.execute(
            """
            SELECT sample_id,
                   disease_code,
                   disease_valid_365,
                   disease_label_365,
                   CAST(cutoff AS DATE) + 365 <= CAST(diagnosis_observation_end AS DATE)
                     AND CAST(cutoff AS DATE) + 365 <= CAST(death_observation_end AS DATE) AS full_year
            FROM read_parquet(?)
            WHERE disease_code IN (SELECT unnest(?))
            """,
            [str(label_dir / "label_windows.parquet"), list(diseases)],
        ).fetchall()
    finally:
        con.close()

    for sid, d, v_flag, y_flag, full_flag in rows:
        sid = str(sid)
        if sid not in sample_idx or d not in disease_idx:
            continue
        i = sample_idx[sid]
        j = disease_idx[d]
        y[i, j] = bool(y_flag)
        valid[i, j] = bool(v_flag)
        full[i] = bool(full_flag)
        seen += 1

    if seen != len(ids) * len(diseases):
        raise ValueError(f"Expected {len(ids) * len(diseases)} keyed 365d cells, found {seen}")

    if boundaries_parquet is not None and boundaries_parquet.is_file():
        b_rows = pq.read_table(boundaries_parquet).to_pylist()
        by_sid = {str(r["sample_id"]): _full_year(r) for r in b_rows}
        expected_full = np.array([by_sid[s] for s in ids], dtype=bool)
        if not np.array_equal(full, expected_full):
            raise ValueError("label_windows full_year disagrees with sample_boundaries.parquet")

    return y, valid, full


def load_diagnosis_history(input_dir: Path, ids: list[str]) -> list[set[str]]:
    """Extract normalized ICD-10 category / ICD-9 prefix history sets per sample_id."""
    lookup = {sid: i for i, sid in enumerate(ids)}
    hist: list[set[str]] = [set() for _ in ids]
    con = duckdb.connect()
    try:
        rows = con.execute(
            """
            SELECT DISTINCT sample_id, code
            FROM read_parquet(?)
            WHERE event_type = 'diagnosis' AND code IS NOT NULL
            """,
            [str(input_dir / "events.parquet")],
        ).fetchall()
    finally:
        con.close()
    for sid, code in rows:
        sid = str(sid)
        if sid in lookup:
            hist[lookup[sid]].add(history_code(str(code)))
    return hist


def load_365d_predictions(
    pred_path: Path,
    expected_ids: list[str],
    target_diseases: list[str],
    manifest_path: Path | None = None,
) -> np.ndarray:
    """Load 365-day predictions aligned to `(expected_ids, target_diseases)` from `.parquet` or `.npz`."""
    if pred_path.suffix == ".npz":
        archive = np.load(pred_path, allow_pickle=False)
        src_ids = [str(s) for s in archive["ids"].tolist()]
        src_diseases = [str(d) for d in archive["diseases"].tolist()]
        probs = np.asarray(archive["predictions"], dtype=np.float64)
        if probs.shape != (len(src_ids), len(src_diseases)):
            raise ValueError("Invalid predictions shape in .npz archive")
        id_map = {s: i for i, s in enumerate(src_ids)}
        if set(id_map) != set(expected_ids) or len(id_map) != len(expected_ids):
            raise ValueError("Prediction sample_ids in .npz do not match expected cohort")
        d_idx = [src_diseases.index(d) for d in target_diseases]
        row_idx = [id_map[s] for s in expected_ids]
        aligned = probs[np.ix_(row_idx, d_idx)]
    elif pred_path.suffix == ".parquet":
        if manifest_path is None or not manifest_path.is_file():
            raise ValueError("manifest_path is required when reading sealed predictions.parquet")
        manifest = json.loads(manifest_path.read_text())
        all_diseases = [str(d) for d in manifest["scored_diseases"]]
        horizons = manifest.get("horizons_days", manifest.get("horizons", list(HORIZONS)))
        if 365 not in horizons:
            raise ValueError("365-day horizon missing from manifest")
        k365 = list(horizons).index(365)
        table = pq.read_table(pred_path, columns=["sample_id", "disease_risk"])
        if table.num_rows != len(expected_ids):
            raise ValueError("predictions.parquet row count mismatch")
        src_ids = [str(s) for s in table["sample_id"].to_pylist()]
        id_map = {s: i for i, s in enumerate(src_ids)}
        if set(id_map) != set(expected_ids) or len(id_map) != len(expected_ids):
            raise ValueError("predictions.parquet sample_ids do not match expected cohort")
        flat = (
            table.column("disease_risk")
            .combine_chunks()
            .values.to_numpy()
            .reshape(len(src_ids), len(all_diseases), len(horizons))
        )
        probs365 = np.asarray(flat[:, :, k365], dtype=np.float64)
        d_idx = [all_diseases.index(d) for d in target_diseases]
        row_idx = [id_map[s] for s in expected_ids]
        aligned = probs365[np.ix_(row_idx, d_idx)]
    else:
        raise ValueError(f"Unsupported prediction format: {pred_path}")

    if not np.isfinite(aligned).all():
        raise ValueError("Non-finite 365-day predictions")
    return aligned


def weighted_panel_recall_at_5pct(
    rankings: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    weights: np.ndarray,
) -> float:
    """Compute tie-aware mean Recall@5% across pre-indexed endpoints for patient bootstrap weights."""
    recalls: list[float] = []
    for order, starts, labels in rankings:
        w = weights[order]
        total = np.add.reduceat(w, starts)
        hits = np.add.reduceat(w * labels, starts)
        n_eff = int(total.sum())
        pos_eff = int(hits.sum())
        if n_eff == 0 or pos_eff == 0:
            continue
        cumulative = total.cumsum()
        k = max(1, math.ceil(0.05 * n_eff))
        pos_group = int(cumulative.searchsorted(k, side="left"))
        prior = int(cumulative[pos_group - 1]) if pos_group > 0 else 0
        found = float(hits[:pos_group].sum()) + (k - prior) * float(hits[pos_group]) / float(
            total[pos_group]
        )
        recalls.append(found / float(pos_eff))
    if not recalls:
        raise ValueError("No scorable endpoints in bootstrap draw")
    return float(np.mean(recalls))


def paired_patient_bootstrap(
    labels365: np.ndarray,
    valid365: np.ndarray,
    complete_year: np.ndarray,
    disease_codes: list[str],
    history: list[set[str]],
    reference_pred365: np.ndarray,
    candidate_pred365: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 20261103,
    min_positives: int = MIN_POSITIVES,
) -> dict[str, Any]:
    """Paired patient-level bootstrap (default 2,000 draws) on the frozen eligible panel endpoints.

    Uses identical patient multiplicity vectors across both models and all eligible endpoints.
    """
    n_patients = len(complete_year)
    if reference_pred365.shape != (n_patients, len(disease_codes)) or candidate_pred365.shape != (
        n_patients,
        len(disease_codes),
    ):
        raise ValueError("Prediction matrix shape mismatch in paired_patient_bootstrap")

    ref_rankings: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    cand_rankings: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    scored_codes: list[str] = []

    for code, (_, family) in PANEL.items():
        if code not in disease_codes:
            continue
        col = disease_codes.index(code)
        no_hist = np.fromiter(
            (not has_family_history(h, family) for h in history), bool, count=n_patients
        )
        sel = np.flatnonzero(complete_year & valid365[:, col] & no_hist)
        if int(labels365[sel, col].sum()) < min_positives:
            continue
        scored_codes.append(code)
        for rankings_out, pred in (
            (ref_rankings, reference_pred365),
            (cand_rankings, candidate_pred365),
        ):
            ordered = sel[np.argsort(-pred[sel, col], kind="stable")]
            ordered_scores = pred[ordered, col]
            starts = np.r_[0, np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]) + 1]
            rankings_out.append((ordered, starts, labels365[ordered, col].astype(np.int64)))

    if not scored_codes:
        raise ValueError(f"No panel endpoints reached min_positives={min_positives}")

    unit_w = np.ones(n_patients, dtype=np.int64)
    ref_obs = weighted_panel_recall_at_5pct(ref_rankings, unit_w)
    cand_obs = weighted_panel_recall_at_5pct(cand_rankings, unit_w)

    # Cross-check exact point-estimate parity with panel_score when min_positives == MIN_POSITIVES.
    if min_positives == MIN_POSITIVES:
        ref_panel = panel_score(
            labels365, valid365, reference_pred365, complete_year, disease_codes, history
        )
        cand_panel = panel_score(
            labels365, valid365, candidate_pred365, complete_year, disease_codes, history
        )
        if abs(ref_obs - float(ref_panel[PRIMARY])) >= 1e-12:
            raise AssertionError("Reference point score disagrees with panel_score")
        if abs(cand_obs - float(cand_panel[PRIMARY])) >= 1e-12:
            raise AssertionError("Candidate point score disagrees with panel_score")

    rng = np.random.default_rng(seed)
    ref_draws = np.empty(n_resamples, dtype=np.float64)
    cand_draws = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        w = np.bincount(rng.integers(0, n_patients, size=n_patients), minlength=n_patients)
        ref_draws[b] = weighted_panel_recall_at_5pct(ref_rankings, w)
        cand_draws[b] = weighted_panel_recall_at_5pct(cand_rankings, w)

    delta = cand_draws - ref_draws
    return {
        "schema": "ndp-paired-patient-bootstrap-recall5pct/v4",
        "metric": PRIMARY,
        "n_patients": int(n_patients),
        "n_endpoints": len(scored_codes),
        "endpoints_in_reward": scored_codes,
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "reference_score": float(ref_obs),
        "candidate_score": float(cand_obs),
        "candidate_minus_reference": float(cand_obs - ref_obs),
        "difference_ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
        "reference_ci95": np.quantile(ref_draws, [0.025, 0.975]).tolist(),
        "candidate_ci95": np.quantile(cand_draws, [0.025, 0.975]).tolist(),
        "probability_candidate_better": float(np.mean(delta > 0.0)),
    }


def replay_tabpfn_panel(
    train_dir: Path,
    train_features_npz: Path,
    test_features_npz: Path,
    output_dir: Path,
    model_path: Path,
    max_support: int = DEFAULT_MAX_SUPPORT,
    device: str = "cuda",
    batch_size: int = 256,
    seed: int = 20261103,
    role: int = 0,
    num_roles: int = 1,
) -> dict[str, Any]:
    """Fit TabPFN-3.5 per PANEL endpoint on collision-free features and score all test patients."""
    enforce_offline_network()
    from tabpfn import TabPFNClassifier  # Lazy import so CPU unit tests do not require tabpfn

    tr = np.load(train_features_npz, allow_pickle=False)
    te = np.load(test_features_npz, allow_pickle=False)
    train_ids = [str(s) for s in tr["ids"].tolist()]
    test_ids = [str(s) for s in te["ids"].tolist()]
    x_train = np.asarray(tr["features"], dtype=np.float32)
    x_test = np.asarray(te["features"], dtype=np.float32)
    if list(tr["feature_names"].tolist()) != list(te["feature_names"].tolist()):
        raise ValueError("Train and test feature column names do not match")

    panel_codes = list(PANEL.keys())
    y_train, valid_train, full_train = load_label_grid_365(train_dir, train_ids, panel_codes)
    train_hist = load_diagnosis_history(train_dir, train_ids)

    per_endpoint_dir = output_dir / "per_endpoint"
    per_endpoint_dir.mkdir(parents=True, exist_ok=True)
    support_audit: dict[str, Any] = {}

    for j, code in enumerate(panel_codes):
        if j % num_roles != role:
            continue
        out_npy = per_endpoint_dir / f"{code}.npy"
        out_meta = per_endpoint_dir / f"{code}.meta.json"
        if out_npy.is_file() and out_meta.is_file():
            support_audit[code] = json.loads(out_meta.read_text())
            continue

        family = PANEL[code][1]
        excluded = np.fromiter(
            (has_family_history(h, family) for h in train_hist), bool, count=len(train_ids)
        )
        eligible = valid_train[:, j] & full_train & ~excluded
        support, meta = select_support_indices(
            eligible_mask=eligible,
            y_col=y_train[:, j],
            max_support=max_support,
            seed=seed + j,
        )

        clf = TabPFNClassifier(
            n_estimators="auto",
            model_path=str(model_path.resolve()),
            device=device,
            fit_mode="fit_with_cache",
            random_state=seed,
            ignore_pretraining_limits=True,
        )
        # Keep ALL collision-free features (including all 7 lab summary statistics).
        clf.fit(x_train[support], y_train[support, j].astype(int))
        raw_probs = np.concatenate(
            [
                clf.predict_proba(x_test[start : start + batch_size])[:, 1]
                for start in range(0, len(x_test), batch_size)
            ]
        )
        calibrated = prior_odds_correction(
            raw_probs,
            n_pos_total=meta["n_pos_total"],
            n_neg_total=meta["n_neg_total"],
            n_pos_support=meta["n_pos_support"],
            n_neg_support=meta["n_neg_support"],
        )
        if len(calibrated) != len(x_test) or not np.isfinite(calibrated).all():
            raise RuntimeError(f"Invalid TabPFN probability output for {code}")
        np.save(out_npy, calibrated.astype(np.float64))
        out_meta.write_text(json.dumps(meta, indent=2) + "\n")
        support_audit[code] = meta

    # Merge if all 26 PANEL endpoints have finished.
    if all((per_endpoint_dir / f"{code}.npy").is_file() for code in panel_codes):
        merged = np.column_stack(
            [np.load(per_endpoint_dir / f"{code}.npy") for code in panel_codes]
        )
        merged_npz = output_dir / "v4_tabpfn_test_predictions.npz"
        np.savez_compressed(
            merged_npz,
            ids=np.asarray(test_ids),
            diseases=np.asarray(panel_codes),
            predictions=merged,
        )
        return {
            "status": "merged",
            "predictions_npz": str(merged_npz),
            "n_test": len(test_ids),
            "num_features": int(x_train.shape[1]),
            "max_support": max_support,
            "support_audit": support_audit,
        }
    return {"status": "partial_shard_complete", "role": role, "num_roles": num_roles}


def evaluate_and_compare(
    test_public_dir: Path,
    test_trusted_dir: Path,
    tabpfn_npz: Path,
    output_json: Path,
    agent_predictions: Path | None = None,
    expected_test_patients: int = EXPECTED_TEST_PATIENTS,
    expected_endpoints: int = EXPECTED_ELIGIBLE_ENDPOINTS,
    n_resamples: int = 2000,
    seed: int = 20261103,
) -> dict[str, Any]:
    """Score TabPFN-3.5 on the full test cohort and optionally run paired bootstrap vs Agent."""
    public_ids = [
        str(s)
        for s in pq.read_table(test_public_dir / "samples.parquet", columns=["sample_id"])
        .column(0)
        .to_pylist()
    ]
    if expected_test_patients > 0 and len(public_ids) != expected_test_patients:
        raise ValueError(
            f"Expected {expected_test_patients} held-out test patients, got {len(public_ids)}"
        )

    panel_codes = list(PANEL.keys())
    tab_probs = load_365d_predictions(tabpfn_npz, public_ids, panel_codes)
    y_test, v_test, full_test = load_label_grid_365(
        test_trusted_dir,
        public_ids,
        panel_codes,
        boundaries_parquet=test_trusted_dir / "sample_boundaries.parquet",
    )
    test_hist = load_diagnosis_history(test_public_dir, public_ids)
    panel_report = panel_score(y_test, v_test, tab_probs, full_test, panel_codes, test_hist)

    eligible_recalls = {
        code: float(entry["recall_at_5pct"])
        for code, entry in panel_report["endpoints"].items()
        if entry["in_reward"]
    }
    if expected_endpoints > 0 and len(eligible_recalls) != expected_endpoints:
        raise ValueError(
            f"Expected {expected_endpoints} eligible endpoints, got {len(eligible_recalls)}"
        )

    score_val = float(panel_report[PRIMARY])
    mean_check = float(sum(eligible_recalls.values()) / len(eligible_recalls))
    if abs(score_val - mean_check) >= 1e-12:
        raise AssertionError("panel_recall_at_5pct differs from equal mean of eligible endpoints")

    result: dict[str, Any] = {
        "schema": "ndp-v4-tabpfn-clinical-comparator/v4",
        "score": score_val,
        "reward": score_val,
        "panel_recall_at_5pct": score_val,
        "panel_recall_at_1pct": panel_report["panel_recall_at_1pct"],
        "panel_ppv_at_5pct": panel_report["panel_ppv_at_5pct"],
        "panel_auroc": panel_report["panel_auroc"],
        "panel_auprc": panel_report["panel_auprc"],
        "n_test": len(public_ids),
        "complete_year": int(full_test.sum()),
        "n_endpoints": len(eligible_recalls),
        "endpoint_recall_at_5pct": eligible_recalls,
    }

    if agent_predictions is not None:
        agent_probs = load_365d_predictions(
            agent_predictions,
            public_ids,
            panel_codes,
            manifest_path=test_public_dir / "manifest.json",
        )
        result["paired_bootstrap_vs_agent"] = paired_patient_bootstrap(
            labels365=y_test,
            valid365=v_test,
            complete_year=full_test,
            disease_codes=panel_codes,
            history=test_hist,
            reference_pred365=tab_probs,
            candidate_pred365=agent_probs,
            n_resamples=n_resamples,
            seed=seed,
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_feat = sub.add_parser("build-features", help="Fit/apply collision-free clinical features")
    p_feat.add_argument("--train-dir", type=Path, required=True)
    p_feat.add_argument("--test-dir", type=Path, required=True)
    p_feat.add_argument("--output-dir", type=Path, required=True)
    p_feat.add_argument("--top-diagnoses", type=int, default=DEFAULT_TOP_DIAGNOSES)
    p_feat.add_argument("--top-medications", type=int, default=DEFAULT_TOP_MEDICATIONS)
    p_feat.add_argument("--top-labs", type=int, default=DEFAULT_TOP_LABS)
    p_feat.add_argument("--min-lab-numeric-n", type=int, default=100)

    p_run = sub.add_parser("replay-tabpfn", help="Run TabPFN-3.5 across the 26 PANEL endpoints")
    p_run.add_argument("--train-dir", type=Path, required=True)
    p_run.add_argument("--train-features", type=Path, required=True)
    p_run.add_argument("--test-features", type=Path, required=True)
    p_run.add_argument("--model-path", type=Path, required=True)
    p_run.add_argument("--output-dir", type=Path, required=True)
    p_run.add_argument(
        "--max-support",
        type=int,
        default=DEFAULT_MAX_SUPPORT,
        help="Max support rows per endpoint (default 10000; pass 0 for full ~19k train cohort)",
    )
    p_run.add_argument("--device", default="cuda")
    p_run.add_argument("--batch-size", type=int, default=256)
    p_run.add_argument("--seed", type=int, default=20261103)
    p_run.add_argument("--role", type=int, default=0)
    p_run.add_argument("--num-roles", type=int, default=1)

    p_eval = sub.add_parser(
        "score-and-compare", help="Score on all 11,424 test patients + 2,000-draw paired bootstrap"
    )
    p_eval.add_argument("--test-public", type=Path, required=True)
    p_eval.add_argument("--test-trusted", type=Path, required=True)
    p_eval.add_argument("--tabpfn-npz", type=Path, required=True)
    p_eval.add_argument("--agent-predictions", type=Path, default=None)
    p_eval.add_argument("--output", type=Path, required=True)
    p_eval.add_argument("--expected-test-patients", type=int, default=EXPECTED_TEST_PATIENTS)
    p_eval.add_argument("--expected-endpoints", type=int, default=EXPECTED_ELIGIBLE_ENDPOINTS)
    p_eval.add_argument("--n-resamples", type=int, default=2000)
    p_eval.add_argument("--seed", type=int, default=20261103)

    args = parser.parse_args()
    if args.command == "build-features":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        spec = fit_collision_free_feature_spec(
            args.train_dir,
            top_diagnoses=args.top_diagnoses,
            top_medications=args.top_medications,
            top_labs=args.top_labs,
            min_lab_numeric_n=args.min_lab_numeric_n,
        )
        spec_path = args.output_dir / "v4_feature_spec.json"
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")
        tr_summary = build_collision_free_features(
            args.train_dir, spec, output=args.output_dir / "train-features-v4.npz"
        )
        te_summary = build_collision_free_features(
            args.test_dir, spec, output=args.output_dir / "test-features-v4.npz"
        )
        print(
            json.dumps(
                {
                    "spec": str(spec_path),
                    "train": {k: v for k, v in tr_summary.items() if isinstance(v, (int, float))},
                    "test": {k: v for k, v in te_summary.items() if isinstance(v, (int, float))},
                },
                indent=2,
            )
        )
    elif args.command == "replay-tabpfn":
        out = replay_tabpfn_panel(
            train_dir=args.train_dir,
            train_features_npz=args.train_features,
            test_features_npz=args.test_features,
            output_dir=args.output_dir,
            model_path=args.model_path,
            max_support=args.max_support,
            device=args.device,
            batch_size=args.batch_size,
            seed=args.seed,
            role=args.role,
            num_roles=args.num_roles,
        )
        print(json.dumps(out, indent=2))
    elif args.command == "score-and-compare":
        out = evaluate_and_compare(
            test_public_dir=args.test_public,
            test_trusted_dir=args.test_trusted,
            tabpfn_npz=args.tabpfn_npz,
            output_json=args.output,
            agent_predictions=args.agent_predictions,
            expected_test_patients=args.expected_test_patients,
            expected_endpoints=args.expected_endpoints,
            n_resamples=args.n_resamples,
            seed=args.seed,
        )
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
