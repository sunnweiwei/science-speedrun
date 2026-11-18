"""CPU unit tests for V4 collision-free TabPFN clinical features, bootstrap CI, and V3 score parity."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

TASK_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = TASK_ROOT / "environment" / "evaluation"
COMP_DIR = TASK_ROOT / "tests" / "comparators"
for _p in (str(EVAL_DIR), str(COMP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from panel_scoring import PANEL, PRIMARY, panel_score, recall_at  # noqa: E402
from risk_scoring import HORIZONS  # noqa: E402
from v4_tabpfn_clinical_replay import (  # noqa: E402
    LAB_METRICS,
    build_collision_free_features,
    evaluate_and_compare,
    fit_collision_free_feature_spec,
    load_365d_predictions,
    paired_patient_bootstrap,
    prior_odds_correction,
    select_support_indices,
)


def _load_ssi_index_v3():
    v3_path = next(p for p in Path(__file__).resolve().parents if (p / "harness" / "ssi_index_v3.py").is_file()) / "harness" / "ssi_index_v3.py"
    if not v3_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("ssi_index_v3", v3_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_collision_free_vocabulary_and_full_7_lab_summaries(tmp_path: Path) -> None:
    """Distinct ICD/med codes that collide under DuckDB hash()%128/%64 map to disjoint columns."""
    db = duckdb.connect()
    # Find two diagnosis codes that collide under DuckDB hash(code) % 128, and two med codes under % 64.
    diag_buckets: dict[int, list[str]] = {}
    for idx in range(500):
        code = f"I{idx:03d}"
        b = int(db.execute("SELECT hash(?) % 128", [code]).fetchone()[0])
        diag_buckets.setdefault(b, []).append(code)
    colliding_diags = next(codes[:2] for codes in diag_buckets.values() if len(codes) >= 2)

    med_buckets: dict[int, list[str]] = {}
    for idx in range(300):
        code = f"MED_{idx:04d}"
        b = int(db.execute("SELECT hash(?) % 64", [code]).fetchone()[0])
        med_buckets.setdefault(b, []).append(code)
    colliding_meds = next(codes[:2] for codes in med_buckets.values() if len(codes) >= 2)
    db.close()

    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()

    # Build 300 distinct diagnosis codes and 150 distinct med codes on train so top-256 / top-128 cuts apply.
    samples_train = [
        {"sample_id": f"tr_{i:04d}", "approx_age": 50 + (i % 30), "gender": "F" if i % 2 == 0 else "M"}
        for i in range(40)
    ]
    pq.write_table(pa.Table.from_pylist(samples_train), train_dir / "samples.parquet")

    events_train: list[dict] = []
    # Give colliding_diags[0] prevalence 10, colliding_diags[1] prevalence 9 so both enter top-256.
    for p_idx in range(10):
        events_train.append({
            "sample_id": f"tr_{p_idx:04d}",
            "stay_key": "s1",
            "event_type": "diagnosis",
            "code": colliding_diags[0],
            "value": None,
            "unit": None,
            "seconds_before_cutoff": 86400.0,
        })
    for p_idx in range(9):
        events_train.append({
            "sample_id": f"tr_{p_idx + 10:04d}",
            "stay_key": "s1",
            "event_type": "diagnosis",
            "code": colliding_diags[1],
            "value": None,
            "unit": None,
            "seconds_before_cutoff": 172800.0,
        })
    for extra in range(300):
        c = f"Z{extra:03d}"
        if c in colliding_diags:
            continue
        events_train.append({
            "sample_id": f"tr_{extra % 3:04d}",
            "stay_key": "s1",
            "event_type": "diagnosis",
            "code": c,
            "value": None,
            "unit": None,
            "seconds_before_cutoff": 3600.0,
        })

    # Medications: give colliding_meds[0] prevalence 8, colliding_meds[1] prevalence 7.
    for p_idx in range(8):
        events_train.append({
            "sample_id": f"tr_{p_idx:04d}",
            "stay_key": "s1",
            "event_type": "prescription",
            "code": colliding_meds[0],
            "value": None,
            "unit": None,
            "seconds_before_cutoff": 7200.0,
        })
    for p_idx in range(7):
        events_train.append({
            "sample_id": f"tr_{p_idx + 10:04d}",
            "stay_key": "s1",
            "event_type": "administration",
            "code": colliding_meds[1],
            "value": None,
            "unit": None,
            "seconds_before_cutoff": 14400.0,
        })
    for extra in range(150):
        mc = f"RX_{extra:04d}"
        events_train.append({
            "sample_id": f"tr_{extra % 2:04d}",
            "stay_key": "s1",
            "event_type": "prescription",
            "code": mc,
            "value": None,
            "unit": None,
            "seconds_before_cutoff": 3600.0,
        })

    # Lab measurements with >= 5 numeric measurements (min_lab_numeric_n=5 in test).
    for k, val in enumerate([1.0, 2.0, 3.0, 4.0, 10.0, 12.0]):
        events_train.append({
            "sample_id": f"tr_{k:04d}",
            "stay_key": "s1",
            "event_type": "lab",
            "code": "CREATININE",
            "value": str(val),
            "unit": "mg/dL",
            "seconds_before_cutoff": float((k + 1) * 1000),
        })

    pq.write_table(pa.Table.from_pylist(events_train), train_dir / "events.parquet")

    spec = fit_collision_free_feature_spec(
        train_dir, top_diagnoses=256, top_medications=128, top_labs=48, min_lab_numeric_n=5
    )
    assert spec["hash_collisions"] is False
    assert len(spec["top_diagnoses"]) == 256
    assert len(spec["top_medications"]) == 128
    assert len(spec["top_labs"]) == 1
    diag_codes = [d["code"] for d in spec["top_diagnoses"]]
    med_codes = [m["code"] for m in spec["top_medications"]]
    assert len(set(diag_codes)) == 256
    assert len(set(med_codes)) == 128
    assert colliding_diags[0] in diag_codes and colliding_diags[1] in diag_codes
    assert colliding_meds[0] in med_codes and colliding_meds[1] in med_codes

    # Test patients: p0 has colliding_diags[0] + colliding_meds[0] + multi-point lab; p1 has colliding_diags[1] + colliding_meds[1] + missing lab.
    samples_test = [
        {"sample_id": "te_0", "approx_age": 65, "gender": "F"},
        {"sample_id": "te_1", "approx_age": 65, "gender": "F"},
    ]
    events_test = [
        {"sample_id": "te_0", "stay_key": "k1", "event_type": "diagnosis", "code": colliding_diags[0], "value": None, "unit": None, "seconds_before_cutoff": 86400.0},
        {"sample_id": "te_0", "stay_key": "k1", "event_type": "prescription", "code": colliding_meds[0], "value": None, "unit": None, "seconds_before_cutoff": 86400.0},
        # Three lab points at 100s (newest=4.0), 500s (1.0), 900s (7.0) -> mean=4.0, min=1.0, max=7.0, std=sqrt(6), log_count=log1p(3), missing=0.0
        {"sample_id": "te_0", "stay_key": "k1", "event_type": "lab", "code": "CREATININE", "value": "4.0", "unit": "mg/dL", "seconds_before_cutoff": 100.0},
        {"sample_id": "te_0", "stay_key": "k1", "event_type": "lab", "code": "CREATININE", "value": "1.0", "unit": "mg/dL", "seconds_before_cutoff": 500.0},
        {"sample_id": "te_0", "stay_key": "k1", "event_type": "lab", "code": "CREATININE", "value": "7.0", "unit": "mg/dL", "seconds_before_cutoff": 900.0},
        {"sample_id": "te_1", "stay_key": "k2", "event_type": "diagnosis", "code": colliding_diags[1], "value": None, "unit": None, "seconds_before_cutoff": 86400.0},
        {"sample_id": "te_1", "stay_key": "k2", "event_type": "administration", "code": colliding_meds[1], "value": None, "unit": None, "seconds_before_cutoff": 86400.0},
    ]
    pq.write_table(pa.Table.from_pylist(samples_test), test_dir / "samples.parquet")
    pq.write_table(pa.Table.from_pylist(events_test), test_dir / "events.parquet")

    built = build_collision_free_features(test_dir, spec)
    names = built["feature_names"].tolist()
    feats = built["features"]
    assert feats.shape == (2, 13 + 26 + 256 + 128 + 7)

    # Verify zero collision between the two colliding diagnosis codes and two colliding medication codes.
    col_d0 = next(i for i, n in enumerate(names) if n.endswith(f"_{colliding_diags[0]}") and n.startswith("diagnosis_exact_"))
    col_d1 = next(i for i, n in enumerate(names) if n.endswith(f"_{colliding_diags[1]}") and n.startswith("diagnosis_exact_"))
    assert col_d0 != col_d1
    assert feats[0, col_d0] == pytest.approx(np.log1p(1)) and feats[0, col_d1] == 0.0
    assert feats[1, col_d0] == 0.0 and feats[1, col_d1] == pytest.approx(np.log1p(1))

    col_m0 = next(i for i, n in enumerate(names) if n.endswith(f"_{colliding_meds[0]}") and n.startswith("medication_exact_"))
    col_m1 = next(i for i, n in enumerate(names) if n.endswith(f"_{colliding_meds[1]}") and n.startswith("medication_exact_"))
    assert col_m0 != col_m1
    assert feats[0, col_m0] == pytest.approx(np.log1p(1)) and feats[0, col_m1] == 0.0
    assert feats[1, col_m0] == 0.0 and feats[1, col_m1] == pytest.approx(np.log1p(1))

    # Verify ALL 7 lab summary statistics are materialized and preserved (fixing tab-auto.py:24).
    lab_cols = [names.index(f"lab_0_{m}") for m in LAB_METRICS]
    assert feats[0, lab_cols].tolist() == pytest.approx(
        [4.0, 4.0, float(np.sqrt(6.0)), 1.0, 7.0, float(np.log1p(3)), 0.0], rel=1e-5
    )
    assert feats[1, lab_cols].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def test_support_selection_and_prior_odds_preserve_ranks() -> None:
    """Eliminates 3,072-row / 512-positive truncation and preserves Recall@5% ranks."""
    rng = np.random.default_rng(42)
    n_train = 19000
    eligible = np.ones(n_train, dtype=bool)
    eligible[:500] = False
    y = np.zeros(n_train, dtype=bool)
    # 850 eligible positives (> old 512 cap!).
    pos_candidates = np.flatnonzero(eligible)[:850]
    y[pos_candidates] = True

    support_10k, meta_10k = select_support_indices(eligible, y, max_support=10000, seed=7)
    assert meta_10k["n_support_total"] == 10000
    assert meta_10k["n_pos_support"] == 850  # All 850 positives kept!
    assert meta_10k["n_neg_support"] == 9150
    assert len(set(support_10k.tolist())) == 10000

    # Full ~19k mode when max_support=0.
    support_full, meta_full = select_support_indices(eligible, y, max_support=0, seed=7)
    assert meta_full["n_support_total"] == 18500
    assert len(support_full) == 18500

    # Prior-odds correction strictly preserves rankings (including exact ties) for Recall@5%.
    raw_p = rng.uniform(0.001, 0.999, size=2000)
    raw_p[10:25] = 0.777  # intentional tie block
    y_eval = (rng.random(2000) < 0.08).astype(np.int64)
    cal_p = prior_odds_correction(
        raw_p,
        n_pos_total=meta_10k["n_pos_total"],
        n_neg_total=meta_10k["n_neg_total"],
        n_pos_support=meta_10k["n_pos_support"],
        n_neg_support=meta_10k["n_neg_support"],
    )
    assert np.all(np.diff(np.sort(raw_p)) > 0) == np.all(np.diff(np.sort(cal_p)) > 0)
    rec_raw, ppv_raw = recall_at(y_eval, raw_p, 0.05)
    rec_cal, ppv_cal = recall_at(y_eval, cal_p, 0.05)
    assert abs(rec_raw - rec_cal) < 1e-15
    assert abs(ppv_raw - ppv_cal) < 1e-15


def test_paired_bootstrap_and_ssi_index_v3_parity(tmp_path: Path) -> None:
    """Verify paired 2,000-draw bootstrap CI and < 1e-12 parity with ssi_index_v3.disease()."""
    rng = np.random.default_rng(20261115)
    panel_codes = list(PANEL.keys())  # 26 panel diseases; make exactly 17 have >= 20 positives
    n_patients = 600
    ids = [f"p_{i:04d}" for i in range(n_patients)]
    labels365 = np.zeros((n_patients, len(panel_codes)), dtype=bool)
    valid365 = np.ones_like(labels365, dtype=bool)
    complete_year = np.ones(n_patients, dtype=bool)
    complete_year[-40:] = False  # 560 complete-year patients
    history = [set() for _ in range(n_patients)]

    for j in range(len(panel_codes)):
        n_pos = 35 if j < 17 else 5  # First 17 endpoints eligible (>=20), remaining 9 <20
        chosen = rng.choice(560, size=n_pos, replace=False)
        labels365[chosen, j] = True

    ref_pred = np.clip(labels365.astype(float) * 0.35 + rng.uniform(0.0, 0.5, labels365.shape), 0.0, 1.0)
    strong_pred = np.clip(labels365.astype(float) * 0.80 + rng.uniform(0.0, 0.2, labels365.shape), 0.0, 1.0)

    # 1. Self-comparison yields exact zero difference and [0.0, 0.0] CI.
    same_boot = paired_patient_bootstrap(
        labels365, valid365, complete_year, panel_codes, history, ref_pred, ref_pred, n_resamples=200, seed=11
    )
    assert same_boot["n_endpoints"] == 17
    assert same_boot["candidate_minus_reference"] == 0.0
    assert same_boot["difference_ci95"] == [0.0, 0.0]

    # 2. Strong vs reference across 2,000 draws yields strictly positive 95% CI.
    diff_boot = paired_patient_bootstrap(
        labels365, valid365, complete_year, panel_codes, history, ref_pred, strong_pred, n_resamples=2000, seed=20261103
    )
    assert diff_boot["n_endpoints"] == 17
    assert diff_boot["n_resamples"] == 2000
    assert diff_boot["difference_ci95"][0] > 0.0

    # 3. Verify < 1e-12 parity with harness/ssi_index_v3.disease() across normal, chance, and perfect fixtures.
    v3 = _load_ssi_index_v3()
    chance_pred = np.full_like(ref_pred, 0.25)
    perfect_pred = labels365.astype(float)
    for pred_matrix in (ref_pred, strong_pred, chance_pred, perfect_pred):
        rep = panel_score(labels365, valid365, pred_matrix, complete_year, panel_codes, history)
        endpoints_17 = {
            code: entry["recall_at_5pct"]
            for code, entry in rep["endpoints"].items()
            if entry["in_reward"]
        }
        assert len(endpoints_17) == 17
        container_score = float(rep[PRIMARY])
        if v3 is not None:
            v3_res = v3.disease(endpoints_17)
            assert abs(container_score - v3_res["score"]) < 1e-12

    # 4. End-to-end evaluate_and_compare + parquet/npz prediction alignment check.
    pub_dir = tmp_path / "public"
    tru_dir = tmp_path / "trusted"
    pub_dir.mkdir()
    tru_dir.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [{"sample_id": sid, "approx_age": 60, "gender": "F", "cutoff": date(2160, 1, 1)} for sid in ids]
        ),
        pub_dir / "samples.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([{"sample_id": ids[0], "event_type": "diagnosis", "code": "Z00"}]),
        pub_dir / "events.parquet",
    )
    (pub_dir / "manifest.json").write_text(
        json.dumps({"horizons_days": list(HORIZONS), "scored_diseases": panel_codes, "split": "test"})
    )
    lw_rows = []
    for i, sid in enumerate(ids):
        end_dt = date(2162, 1, 1) if complete_year[i] else date(2160, 6, 1)
        for j, code in enumerate(panel_codes):
            lw_rows.append({
                "sample_id": sid,
                "disease_code": code,
                "disease_valid_365": bool(valid365[i, j]),
                "disease_label_365": int(labels365[i, j]),
                "cutoff": date(2160, 1, 1),
                "diagnosis_observation_end": end_dt,
                "death_observation_end": end_dt,
            })
    pq.write_table(pa.Table.from_pylist(lw_rows), tru_dir / "label_windows.parquet")
    pq.write_table(
        pa.Table.from_pylist([
            {
                "sample_id": sid,
                "split": "test",
                "cutoff": date(2160, 1, 1),
                "diagnosis_observation_end": date(2162, 1, 1) if complete_year[i] else date(2160, 6, 1),
                "death_observation_end": date(2162, 1, 1) if complete_year[i] else date(2160, 6, 1),
            }
            for i, sid in enumerate(ids)
        ]),
        tru_dir / "sample_boundaries.parquet",
    )

    tab_npz = tmp_path / "tab.npz"
    np.savez_compressed(tab_npz, ids=np.asarray(ids), diseases=np.asarray(panel_codes), predictions=ref_pred)

    # Build a valid 13-horizon predictions.parquet for the agent candidate.
    agent_parquet = tmp_path / "predictions.parquet"
    parquet_rows = []
    for i, sid in enumerate(ids):
        grid13 = np.outer(strong_pred[i], np.linspace(0.1, 1.0, len(HORIZONS))).astype(np.float32)
        parquet_rows.append({
            "sample_id": sid,
            "disease_risk": grid13.reshape(-1).tolist(),
            "death_risk": np.linspace(0.01, 0.05, len(HORIZONS), dtype=np.float32).tolist(),
        })
    pq.write_table(pa.Table.from_pylist(parquet_rows), agent_parquet)
    loaded_from_pq = load_365d_predictions(agent_parquet, ids, panel_codes, pub_dir / "manifest.json")
    assert np.allclose(loaded_from_pq, strong_pred, atol=1e-6)

    out_json = tmp_path / "comparator_report.json"
    summary = evaluate_and_compare(
        test_public_dir=pub_dir,
        test_trusted_dir=tru_dir,
        tabpfn_npz=tab_npz,
        output_json=out_json,
        agent_predictions=agent_parquet,
        expected_test_patients=n_patients,
        expected_endpoints=17,
        n_resamples=200,
        seed=20261103,
    )
    assert summary["score"] == summary["reward"] == summary["panel_recall_at_5pct"]
    if v3 is not None:
        assert abs(summary["score"] - v3.disease(summary["endpoint_recall_at_5pct"])["score"]) < 1e-12


def test_reward_json_has_score_and_reward_and_tests_stay_sealed() -> None:
    """Verify grader.py, main_verifier.py, and score.py emit both score & reward, and tests/ stays out of /workspace."""
    grader_src = (TASK_ROOT / "tests" / "grader.py").read_text(encoding="utf-8")
    main_src = (TASK_ROOT / "tests" / "main_verifier.py").read_text(encoding="utf-8")
    score_src = (TASK_ROOT / "environment" / "evaluation" / "score.py").read_text(encoding="utf-8")
    assert '{"score": reward, "reward": reward}' in grader_src
    assert '{"score": 0.0, "reward": 0.0}' in grader_src
    assert '{"score": 0.0, "reward": 0.0}' in main_src
    assert '{"score": reward, "reward": reward}' in score_src

    dev_dockerfile = (TASK_ROOT / "environment" / "Dockerfile").read_text(encoding="utf-8")
    dev_compose = (TASK_ROOT / "environment" / "docker-compose.yaml").read_text(encoding="utf-8")
    assert "tests" not in dev_dockerfile
    assert "comparators" not in dev_dockerfile
    assert "tests" not in dev_compose
