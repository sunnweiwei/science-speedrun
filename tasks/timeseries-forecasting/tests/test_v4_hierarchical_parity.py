"""CPU parity tests between timeseries-forecasting/environment/evaluation/score.py and harness/ssi_index_v3.py,
plus Starter/Reference identity manifest verification (§7.1.1) and sandbox visibility checks."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = next(p for p in Path(__file__).resolve().parents if (p / "harness" / "ssi_index_v3.py").is_file()) / "harness"
sys.path.insert(0, str(ROOT / "environment" / "evaluation"))
sys.path.insert(0, str(HARNESS))

import ssi_index_v3 as v3  # noqa: E402
import ts_io  # noqa: E402
from score import TIMESERIES_ROSTER, hierarchical_time_series, score, score_forecast_rows  # noqa: E402


def test_frozen_97_roster_matches_ssi_index_v3():
    v3_roster = v3.INPUTS["timeseries_roster"]
    assert set(TIMESERIES_ROSTER) == set(v3_roster) and len(TIMESERIES_ROSTER) == 97
    for k, (dom, src, hor, m, q) in TIMESERIES_ROSTER.items():
        v_dom, v_src, v_hor, v_m, v_q = v3_roster[k]
        assert (dom, src, hor) == (v_dom, v_src, v_hor)
        assert abs(m - v_m) == 0.0 and abs(q - v_q) == 0.0


def test_hierarchical_time_series_parity_normal_and_boundaries():
    rng = np.random.default_rng(42)
    keys = sorted(TIMESERIES_ROSTER)
    # Normal random ratios
    ratios_normal = {k: (float(rng.uniform(0.2, 2.5)), float(rng.uniform(0.15, 1.8))) for k in keys}
    # Boundary clipping (< 1e-3 and > 10.0)
    ratios_boundary = {
        k: ((1e-6, 1e-5) if i % 3 == 0 else (25.0, 50.0) if i % 3 == 1 else (1.0, 1.0))
        for i, k in enumerate(keys)
    }
    # All failed (10.0, 10.0)
    ratios_failed = {k: (10.0, 10.0) for k in keys}

    for ratios in (ratios_normal, ratios_boundary, ratios_failed):
        got = hierarchical_time_series(ratios)
        want = v3.time_series(ratios)
        assert abs(got["score"] - want["score"]) < 1e-12
        assert abs(got["signed_score"] - want["signed_score"]) < 1e-12
        for d in want["components"]:
            assert abs(got["domains"][d] - want["components"][d]) < 1e-12
        for s in want["diagnostics"]:
            assert abs(got["sources"][s] - want["diagnostics"][s]) < 1e-12


def test_end_to_end_hdf5_score_parity_full_partial_and_failed(tmp_path):
    """Build a synthetic 97-config HDF5 suite covering normal, partial (fraction=0.5), and failed configs,
    and verify < 1e-12 parity between score() and ssi_index_v3.forecast_rows()."""
    split = json.loads((ROOT / "environment" / "data" / "split.json").read_text())
    ref = json.loads((ROOT / "environment" / "evaluation" / "seasonal_naive_reference.json").read_text())["configs"]
    rng = np.random.default_rng(7)

    truth_dir = tmp_path / "truth"
    out_dir = tmp_path / "output"
    staged = []
    coverage = {}

    for idx, cfg in enumerate(split["configs"]):
        cid = cfg["config_id"]
        n_inst, n_var, pred_len = 4, 1, 6
        labels = rng.uniform(5.0, 20.0, size=(n_inst, n_var, pred_len)).astype(np.float32)
        se = rng.uniform(0.5, 3.0, size=(n_inst, n_var)).astype(np.float32)
        meta = {
            "dataset": cfg["dataset"],
            "freq": cfg["freq"],
            "term": cfg["term"],
            "config_id": cid,
            "leaderboard_key": cfg["leaderboard_key"],
            "prediction_length": pred_len,
            "seasonality": cfg["seasonality"],
            "windows": 1,
            "n_variates": n_var,
            "domain": cfg["domain"],
            "split": "test",
        }
        ts_io.write_truth_file(truth_dir / "cases" / f"{cid}.h5", labels, se, meta)
        staged.append({
            "config_id": cid,
            "leaderboard_key": cfg["leaderboard_key"],
            "dataset": cfg["dataset"],
            "term": cfg["term"],
            "freq": cfg["freq"],
            "prediction_length": pred_len,
            "seasonality": cfg["seasonality"],
            "n_variates": n_var,
            "n_instances": n_inst,
            "domain": cfg["domain"],
        })

        if idx % 11 == 0:
            # Missing prediction -> status == "failed" -> (10.0, 10.0)
            continue
        noise = rng.normal(0.0, 0.8, size=labels.shape).astype(np.float32)
        mean = labels + noise
        quantiles = np.stack([mean + (q - 0.5) * 1.5 for q in ts_io.QUANTILE_LEVELS], axis=2).astype(np.float32)
        ts_io.write_prediction_file(out_dir / "cases" / f"{cid}.h5", mean, quantiles, {"model": "test"})
        if idx % 7 == 0:
            # Partial coverage: exact 2/4 = 0.5
            coverage[cid] = [True, True, False, False]

    (truth_dir / "index.json").write_text(json.dumps({"contract": ts_io.CONTRACT, "split": "test", "configs": staged}))

    rep = score(truth_dir, out_dir, baseline=ref, coverage=coverage)
    assert rep["n_configs"] == 97
    assert rep["n_failed"] > 0 and rep["n_partial"] > 0
    assert rep["score"] == rep["reward"]
    assert "flat_97_skill_diagnostic" in rep

    # Verify < 1e-12 parity against ssi_index_v3.forecast_rows on:
    # (a) full report["configs"] (uses raw metrics + single-application partial blend),
    # (b) pre-blended relative-only rows (needs_partial_blend=False),
    # (c) unblended rel_mase/rel_wql rows + status="partial" (needs_partial_blend=True).
    v3_from_configs = v3.forecast_rows(rep["configs"])
    assert abs(rep["score"] - v3_from_configs["score"]) < 1e-12
    assert abs(rep["signed_score"] - v3_from_configs["signed_score"]) < 1e-12

    sc_from_configs = score_forecast_rows(rep["configs"])
    assert abs(sc_from_configs["score"] - v3_from_configs["score"]) < 1e-12

    rel_only_rows = [
        {"leaderboard_key": c["leaderboard_key"], "status": c["status"], "relative": c["relative"]}
        for c in rep["configs"]
    ]
    v3_from_rel_only = v3.forecast_rows(rel_only_rows)
    assert abs(rep["score"] - v3_from_rel_only["score"]) < 1e-12

    unblended_rows = []
    for c in rep["configs"]:
        if c["status"] == "failed":
            unblended_rows.append({"leaderboard_key": c["leaderboard_key"], "status": "failed"})
        else:
            _, _, _, bm, bq = TIMESERIES_ROSTER[c["leaderboard_key"]]
            row = {
                "leaderboard_key": c["leaderboard_key"],
                "status": c["status"],
                "rel_mase": c["metrics"]["mase"] / bm,
                "rel_wql": c["metrics"]["wql"] / bq,
            }
            if c["status"] == "partial":
                row["coverage"] = c["coverage"]
            unblended_rows.append(row)
    v3_from_unblended = v3.forecast_rows(unblended_rows)
    sc_from_unblended = score_forecast_rows(unblended_rows)
    assert abs(rep["score"] - v3_from_unblended["score"]) < 1e-12
    assert abs(sc_from_unblended["score"] - v3_from_unblended["score"]) < 1e-12


def test_starter_and_reference_manifest_matches_committed_receipts():
    manifest = json.loads((ROOT / "tests" / "starter_and_reference_manifest.json").read_text())

    # Official B_19 must be strictly the 6.4M starter, never conflated with 41.8M or 50.9M
    b19 = manifest["official_starter_B19"]
    assert b19["parameters"] == 6404864 and b19["parameter_label"] == "6.4M" and b19["eligible_for_B19"] is True

    entries_with_rows = [
        manifest["trivial_control_T19"],
        manifest["normalizer_replay_control"],
        manifest["official_starter_B19"],
        manifest["named_scaled_controls"][0],
        manifest["external_frontier_single_system"],
        *manifest["companion_local_zero_shot_replays"],
    ]
    for entry in entries_with_rows:
        p = ROOT / entry["receipt_path"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == entry["receipt_sha256"], entry["receipt_path"]
        data = json.loads(p.read_text())
        rows = data.get("per_config") or data.get("configs")
        assert isinstance(rows, list) and len(rows) == 97
        v3_res = v3.forecast_rows(rows)
        sc_res = score_forecast_rows(rows)
        assert abs(sc_res["score"] - v3_res["score"]) < 1e-12
        assert abs(entry["score"] - v3_res["score"]) < 1e-12
        if "signed_score" in entry and entry["signed_score"] is not None:
            assert abs(entry["signed_score"] - v3_res["signed_score"]) < 1e-12

    # 50.9M summary receipt lacks 97-config rows and must be explicitly marked ineligible for B_19
    tfm50 = manifest["named_scaled_controls"][1]
    p50 = ROOT / tfm50["receipt_path"]
    assert hashlib.sha256(p50.read_bytes()).hexdigest() == tfm50["receipt_sha256"]
    assert tfm50["parameters"] == 50918592 and tfm50["has_97_config_rows"] is False and tfm50["eligible_for_B19"] is False
    assert tfm50["score"] is None


def test_docs_and_dockerfile_visibility_boundary():
    dockerfile = (ROOT / "environment" / "Dockerfile").read_text()
    compose = (ROOT / "environment" / "docker-compose.yaml").read_text()
    for line in dockerfile.splitlines():
        s = line.strip()
        if s.startswith("COPY "):
            assert "tests/" not in s and " tests " not in s, f"tests/ leaked into Dockerfile: {s}"
    assert "/tests" not in compose and "tests/" not in compose

    for doc_rel in ("instruction.md", "environment/contract.md"):
        txt = (ROOT / doc_rel).read_text()
        assert "7 domains -> 23 sources -> horizon" in txt
        assert "flat_97_skill_diagnostic" in txt

    readme = (ROOT / "tests" / "README.md").read_text()
    for heading in (
        "## Scientific & Literature Justification of Verifier Design",
        "### 1. Primary Metric & Upstream Paper Provenance",
        "### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted",
        "### 3. Diagnostic Signals Retained in `report.json`",
    ):
        assert heading in readme
