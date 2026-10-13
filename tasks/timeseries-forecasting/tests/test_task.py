"""Unit tests for the timeseries-forecasting grader, I/O and protocol tables (no real data, no GPU)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environment" / "evaluation"))
sys.path.insert(0, str(ROOT / "environment" / "evaluation"))
import ts_io  # noqa: E402
from score import config_metrics, score, sort_quantiles  # noqa: E402
from make_example import make  # noqa: E402


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    out = tmp_path_factory.mktemp("example")
    make(out, seed=0)
    return out


def test_protocol_tables():
    assert ts_io.prediction_length("electricity/H", "H", "short") == 48
    assert ts_io.prediction_length("electricity/15T", "15T", "long") == 720
    assert ts_io.prediction_length("m4_weekly", "W-SUN", "short") == 13
    assert ts_io.prediction_length("m4_yearly", "A-DEC", "short") == 6
    assert [ts_io.seasonality(f) for f in ("10S", "5T", "10T", "15T", "H", "D", "W-SUN", "M", "Q-DEC", "A-DEC")] == [360, 288, 144, 96, 24, 1, 1, 12, 4, 1]
    assert ts_io.n_windows("electricity/H", 35064, 48) == 20 and ts_io.n_windows("m4_hourly", 748, 48) == 1
    assert ts_io.n_windows("hospital", 84, 12) == 1 and ts_io.n_windows("us_births/D", 7305, 30) == 20
    assert ts_io.test_cutoffs(100, 10, 3) == [70, 80, 90] and ts_io.valid_cutoff(100, 10, 3) == 60
    assert ts_io.config_id("electricity/H", "H", "short") == "electricity__H__short"


def test_split_manifest_frozen():
    man = json.loads((ROOT / "environment" / "data" / "split.json").read_text())
    assert man["n_configs"] == 97 and len(man["configs"]) == 97
    assert len({c["config_id"] for c in man["configs"]}) == 97 and len({c["leaderboard_key"] for c in man["configs"]}) == 97
    assert man["n_test_instances"] == 326490
    by_key = {c["leaderboard_key"]: c for c in man["configs"]}
    assert by_key["electricity/H/short"]["windows"] == 20 and by_key["electricity/H/short"]["prediction_length"] == 48
    assert by_key["ett1/15T/long"]["prediction_length"] == 720 and by_key["ett1/15T/long"]["n_variates"] == 7
    assert by_key["m4_yearly/A/short"]["windows"] == 1 and by_key["kdd_cup_2018/H/long"]["prediction_length"] == 720
    for c in man["configs"]:
        assert c["prediction_length"] == ts_io.prediction_length(c["dataset"], c["freq"], c["term"])
        assert c["seasonality"] == ts_io.seasonality(c["freq"])
        assert c["n_test_instances"] == c["n_series"] * c["windows"]


def test_seasonal_error_masked_mean():
    h = np.array([[1.0, np.nan, 3.0, 5.0, np.nan, 9.0]])
    # lag 2 pairs: (3-1)=2, (5-nan)=nan, (nan-3)=nan, (9-5)=4 -> mean of valid = 3
    assert ts_io.seasonal_error(h, 2)[0] == pytest.approx(3.0)
    assert ts_io.seasonal_error(np.array([[1.0, 2.0, 4.0]]), 10)[0] == pytest.approx(1.5)  # m > T falls back to lag 1
    assert np.isnan(ts_io.seasonal_error(np.array([[np.nan, np.nan, 1.0]]), 1)[0])


def test_config_metrics_against_hand_computation():
    labels = np.array([[[1.0, 2.0, np.nan, 4.0]]], dtype=np.float32)  # (1, 1, 4)
    se = np.array([[2.0]])
    q = np.zeros((1, 1, 9, 4), np.float32)
    for j, lv in enumerate(ts_io.QUANTILE_LEVELS):
        q[0, 0, j] = np.array([1.0, 2.0, 3.0, 4.0]) + (lv - 0.5) * 2  # median exact, symmetric spread
    m = config_metrics(labels, se, q[:, :, 4], q)
    assert m["mase"] == pytest.approx(0.0) and m["n_valid_labels"] == 3
    # WQL: for level lv with pred = y + (lv-0.5)*2: 2*|(y-pred)*(1{pred>=y}-lv)| ; sum over 3 valid labels / sum|y| = 7
    exp = []
    for lv in ts_io.QUANTILE_LEVELS:
        d = (lv - 0.5) * 2
        exp.append(3 * 2 * abs((-d) * ((d >= 0) - lv)) / 7.0)
    assert m["wql"] == pytest.approx(np.mean(exp))


def test_mase_drops_zero_seasonal_error_like_gluonts():
    labels = np.ones((2, 1, 3), np.float32) * 5
    se = np.array([[1.0], [0.0]])
    q = np.zeros((2, 1, 9, 3), np.float32) + 4.0
    m = config_metrics(labels, se, q[:, :, 4], q)
    assert m["mase"] == pytest.approx(1.0) and m["n_mase_elements"] == 3


def test_sort_quantiles_counts_crossings():
    q = np.zeros((1, 1, 9, 2)); q[0, 0, :, 0] = np.arange(9)[::-1]; q[0, 0, :, 1] = np.arange(9)
    s, frac = sort_quantiles(q)
    assert frac == 0.5 and np.all(np.diff(s, axis=2) >= 0)


def test_example_scores_and_reward_bounds(example):
    base = {c: {"mase": 1.0, "wql": 0.1} for c in ("example_hourly__H__short", "example_daily__D__short", "example_multi__T__medium")}
    rep = score(example / "truth", example / "output", base)
    assert rep["n_configs"] == 3 and rep["n_failed"] == 0
    assert 0.0 <= rep["reward"] <= 1.0
    for c in rep["configs"]:
        assert np.isfinite(c["metrics"]["mase"]) and np.isfinite(c["metrics"]["wql"]) and c["status"] == "ok"
    # a perfect forecast is reward 1, seasonal-naive-level (rel 1) is reward 0
    perfect = example / "perfect"
    for c in rep["configs"]:
        labels, _, _ = ts_io.read_truth_file(example / "truth" / "cases" / f"{c['config_id']}.h5")
        lab = np.nan_to_num(labels)
        ts_io.write_prediction_file(perfect / "cases" / f"{c['config_id']}.h5", lab, np.repeat(lab[:, :, None], 9, axis=2))
    rep_p = score(example / "truth", perfect, base)
    assert rep_p["reward"] == pytest.approx(1.0, abs=2e-3) and rep_p["rel_mase_gmean"] == pytest.approx(1e-3)
    same = {c["config_id"]: {"mase": c["metrics"]["mase"], "wql": c["metrics"]["wql"]} for c in rep["configs"]}
    assert score(example / "truth", example / "output", same)["reward"] == pytest.approx(0.0)


def test_case_file_roundtrip(example):
    case = ts_io.CaseFile(example / "input" / "cases" / "example_multi__T__medium.h5")
    assert case.n_variates == 7 and case.prediction_length == 480 and case.n == int(case.meta["windows"])
    c0 = case.context(0)
    assert c0.shape[0] == 7 and c0.shape[1] == case.context_length[0] <= ts_io.CONTEXT_CAP
    assert list(case.contexts())[0].shape == c0.shape
    case.close()


def test_seasonal_naive_reference_runs_on_example(example, tmp_path):
    out = tmp_path / "snaive"
    subprocess.run(["bash", str(ROOT / "environment" / "baseline" / "seasonal_naive" / "infer.sh"), str(example / "input"), str(out)], check=True)
    rep = score(example / "truth", out, None, allow_missing_baseline=True)
    assert rep["n_failed"] == 0 and all(np.isfinite(c["metrics"]["mase"]) for c in rep["configs"])
    mean, quant = ts_io.read_prediction_file(out / "cases" / "example_hourly__H__short.h5")
    assert np.all(np.diff(quant, axis=2) >= 0) and np.array_equal(quant[:, :, 4], mean)


def test_bundled_reference_matches_leaderboard_snapshot():
    ref = json.loads((ROOT / "environment" / "evaluation" / "seasonal_naive_reference.json").read_text())
    man = json.loads((ROOT / "environment" / "data" / "split.json").read_text())
    assert set(ref["configs"]) == {c["config_id"] for c in man["configs"]}
    assert ref["configs"]["m4_weekly__W__short"]["mase"] == pytest.approx(2.777295, abs=1e-5)
    assert ref["configs"]["solar__H__short"]["wql"] == pytest.approx(0.591656, abs=1e-5)


def test_cohorts_keep_labels_out_of_contexts(example):
    """Within a cohort, no instance's context may contain another instance's label region, whichever item it
    belongs to: every instance of a cohort shares the cutoff timestamp."""
    for cid in ("example_hourly__H__short", "example_multi__T__medium"):
        case = ts_io.CaseFile(example / "input" / "cases" / f"{cid}.h5")
        L = case.prediction_length
        by_item, by_cohort = {}, {}
        for i in range(case.n):
            by_item.setdefault(case.item_id[i], []).append(i)
            by_cohort.setdefault(int(case.cohort[i]), set()).add(str(case.forecast_start[i]))
        for item, idx in by_item.items():
            for a in idx:
                for b in idx:
                    if a != b and case.cohort[a] == case.cohort[b]:
                        # label of a spans history_length[a] .. +L; context of b ends at history_length[b]
                        assert case.history_length[b] <= case.history_length[a], (cid, item, a, b)
        assert all(len(starts) == 1 for starts in by_cohort.values()), (cid, by_cohort)
        assert len(set(case.cohort.tolist())) == int(case.meta["windows"])
        case.close()


def test_cohort_ranks_are_absolute_time_except_synthetic_directories():
    """Two items whose windows fall at different times never share a cohort (kdd_cup's stations are correlated);
    M4's placeholder timestamps rank each item's own windows instead, so its call count does not explode."""
    sys.path.insert(0, str(ROOT / "environment" / "evaluation"))
    import make_cases
    times = {"kdd_cup_2018_with_missing": {"a": {"2018-03-01T00:00:00", "2018-03-03T00:00:00"},
                                           "b": {"2018-03-02T00:00:00", "2018-03-03T00:00:00"}},
             "m4_daily": {"a": {"1900-01-01T00:00:00", "1900-01-15T00:00:00"}, "b": {"1950-06-01T00:00:00", "1950-06-15T00:00:00"}}}
    table = make_cases.rank_cohorts(times)
    assert table["kdd_cup_2018_with_missing"] == {"a": {"2018-03-01T00:00:00": 0, "2018-03-03T00:00:00": 2},
                                                  "b": {"2018-03-02T00:00:00": 1, "2018-03-03T00:00:00": 2}}
    assert table["m4_daily"] == {"a": {"1900-01-01T00:00:00": 0, "1900-01-15T00:00:00": 1},
                                 "b": {"1950-06-01T00:00:00": 0, "1950-06-15T00:00:00": 1}}
    assert make_cases.synthetic_time("m4_hourly") and not make_cases.synthetic_time("electricity")


def test_starter_inference_is_batch_independent():
    """A package's forecast for an instance must not depend on which other instances share its infer.sh call."""
    import torch
    sys.path.insert(0, str(ROOT / "environment" / "baseline" / "starter"))
    from model import PatchForecaster
    torch.manual_seed(0)
    m = PatchForecaster(d_model=32, layers=1, heads=2, max_context=256).eval()
    a, b = torch.randn(1, 100), torch.randn(1, 250)
    alone, _ = m.predict(a, 24)
    batched, _ = m.predict(torch.cat([torch.nn.functional.pad(a, (150, 0), value=float("nan")), b]), 24)
    assert torch.allclose(alone[0], batched[0], atol=1e-5)


@pytest.mark.parametrize("package", ["environment/baseline/seasonal_naive", "docs/validation/released_models/trivial_context_quantiles",
                                     "environment/baseline/starter"])
def test_packages_carry_their_own_reader(package):
    """A package must be self-contained: the verifier refuses a symlink that leaves it, so `ts_io.py` is a real copy."""
    p = ROOT / package / "ts_io.py"
    assert p.is_file() and not p.is_symlink(), f"{package}/ts_io.py must be a copy, not a link"
    assert p.read_bytes() == (ROOT / "environment" / "evaluation" / "ts_io.py").read_bytes(), \
        f"{package}/ts_io.py has drifted from environment/evaluation/ts_io.py"
