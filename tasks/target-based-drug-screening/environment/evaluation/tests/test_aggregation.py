import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

EVALUATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATION))
from aggregation import SCORING_PROTOCOL, summarize_targets


def suite(dude=0.8, lit=0.2):
    return [dict(benchmark=name, target_id=f"target-{i}",
                 roc_auc=0.7, **{"bedroc_80.5": value, "ef_0.005": 10,
                 "ef_0.01": 8, "ef_0.02": 6, "ef_0.05": 3})
            for name, count, value in [("dude", 102, dude), ("lit_pcba", 15, lit)]
            for i in range(count)]


def test_datasets_have_equal_weight_and_raw_means_remain_available():
    result = summarize_targets(suite(), require_full_suite=True)
    assert result["dataset_balanced_bedroc"] == pytest.approx(0.5)
    assert result["overall_mean_bedroc_80.5"] == pytest.approx((102*.8 + 15*.2)/117)
    assert result["overall_mean_ef_0.01"] == 8
    assert result["scoring_protocol"] == SCORING_PROTOCOL
    # A fixed increase in either dataset has the same overall effect.
    assert summarize_targets(suite(.9, .2))["dataset_balanced_bedroc"] == pytest.approx(.55)
    assert summarize_targets(suite(.8, .3))["dataset_balanced_bedroc"] == pytest.approx(.55)
    assert summarize_targets(suite(0, 0))["dataset_balanced_bedroc"] == 0
    assert summarize_targets(suite(1, 1))["dataset_balanced_bedroc"] == 1


def test_incomplete_evaluation_can_report_diagnostics_but_not_a_reward():
    rows = suite()[:-1]
    partial = summarize_targets(rows)
    assert not partial["complete_benchmark"]
    assert "dataset_balanced_bedroc" not in partial
    with pytest.raises(ValueError, match="requires 102"):
        summarize_targets(rows, require_full_suite=True)
    rows.append({**rows[0], "target_id": "extra-dude"})  # 117 total is insufficient.
    with pytest.raises(ValueError, match="requires 102"):
        summarize_targets(rows, require_full_suite=True)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.01, True, "0.5"])
def test_malformed_bedroc_never_produces_a_primary_score(bad):
    rows = suite()
    rows[0]["bedroc_80.5"] = bad
    with pytest.raises(ValueError):
        summarize_targets(rows, require_full_suite=True)


def test_duplicate_targets_are_rejected():
    rows = suite()
    rows[1] = rows[0].copy()
    with pytest.raises(ValueError, match="duplicate"):
        summarize_targets(rows, require_full_suite=True)


def test_historical_reaggregation_uses_targets_and_preserves_source(tmp_path):
    script = EVALUATION.parents[1] / "tests/reaggregate.py"
    spec = importlib.util.spec_from_file_location("reaggregate", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = dict(valid=True, reward=8/9, per_target=suite(),
               primary_metric="equal-target mean EF at 1%",
               benchmarks={"incorrect_cached_summary": 99})
    original = copy.deepcopy(old)
    result = module.reaggregate(old, "source-digest")
    assert old == original
    assert result["reward"] == pytest.approx(.5)
    assert result["migration"]["source_reward"] == 8/9
    assert result["migration"]["source_report_sha256"] == "source-digest"
    assert set(result["benchmarks"]) == {"dude", "lit_pcba"}
    with pytest.raises(ValueError, match="only valid"):
        module.reaggregate({**old, "valid": False}, "digest")
    source = tmp_path / "old.json"
    source.write_text(json.dumps(old))
    before = source.read_bytes()
    assert subprocess.run([sys.executable, str(script), str(source), str(source)],
                          capture_output=True).returncode != 0
    assert source.read_bytes() == before
