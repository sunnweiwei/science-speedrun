from copy import deepcopy

import pytest

from aggregate import (
    ATOM_ZERO_FORCE_ANCHORS,
    DISCOVERY_F1_BASE,
    REFERENCE,
    REFERENCE_V2,
    SUITE_INPUTS_V2,
    aggregate_score,
    expanded_score,
    ssi_v4_score,
)


def reference_metrics():
    return {
        "discovery": {"F1": REFERENCE["F1"]},
        "phonons": {"avg_freq": {"MAE": REFERENCE["avg_freq_MAE_K"]}},
        "elasticity": {
            "bulk_modulus_vrh": {"MAE": REFERENCE["bulk_MAE_GPa"]},
            "shear_modulus_vrh": {"MAE": REFERENCE["shear_MAE_GPa"]},
        },
    }


def test_reference_perfect_and_invalid_scores():
    metrics = reference_metrics()
    assert aggregate_score(metrics)["score"] == pytest.approx(50)
    metrics["discovery"]["F1"] = 1.0
    metrics["phonons"]["avg_freq"]["MAE"] = 0.0
    for value in metrics["elasticity"].values():
        value["MAE"] = 0.0
    assert aggregate_score(metrics)["score"] == 100
    assert aggregate_score(metrics, valid=False)["score"] == 0
    assert aggregate_score({"phonons": metrics["phonons"]})["score"] is None


def test_equal_suite_weights_and_monotonic_errors():
    base = reference_metrics()
    results = []
    for suite in base:
        metrics = deepcopy(base)
        if suite == "discovery":
            metrics[suite]["F1"] = 1 - 2 * (1 - REFERENCE["F1"])
        else:
            for item in metrics[suite].values():
                item["MAE"] *= 2
        result = aggregate_score(metrics)
        assert 0 < result["score"] < 50
        results.append(result["score"])
    assert results == pytest.approx([results[0]] * 3)
    # Correlated diagnostics and case counts must not silently change weighting.
    base["phonons"].update(entropy={"MAE": 1e8}, n=100000)
    assert aggregate_score(base)["score"] == pytest.approx(50)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, None])
def test_invalid_errors_rejected(value):
    metrics = reference_metrics()
    metrics["phonons"]["avg_freq"]["MAE"] = value
    with pytest.raises(ValueError, match="primary metric"):
        aggregate_score(metrics)


def test_eight_suite_score_reference_ideal_and_equal_weights():
    assert expanded_score(REFERENCE_V2)["score"] == pytest.approx(50)
    perfect = dict.fromkeys(REFERENCE_V2, 0.0)
    perfect["discovery_F1"] = 1.0
    assert expanded_score(perfect)["score"] == 100
    scores = []
    for keys in SUITE_INPUTS_V2.values():
        metrics = dict(REFERENCE_V2)
        for key in keys:
            metrics[key] = (
                1 - 2 * (1 - metrics[key])
                if key == "discovery_F1"
                else 2 * metrics[key]
            )
        scores.append(expanded_score(metrics)["score"])
    assert scores == pytest.approx([scores[0]] * 8)
    assert 0 < scores[0] < 50
    # Diagnostic metrics and per-suite case counts do not change weights.
    assert expanded_score(
        {**REFERENCE_V2, "phonon_entropy_MAE": 1e9, "cleavage_n": 100000}
    )["score"] == pytest.approx(50)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1, True])
def test_expanded_score_invalid_or_missing_suite_is_zero(value):
    metrics = {**REFERENCE_V2, "kappa_SRME": value}
    assert expanded_score(metrics)["score"] == 0
    assert expanded_score({})["score"] == 0
    assert expanded_score(REFERENCE_V2, valid=False)["score"] == 0


def test_expanded_score_grader_consumes_flat_metrics(tmp_path, monkeypatch):
    import json
    import sys

    import grader

    report = {
        "physical": {
            "valid": True,
            "metrics": {
                "discovery": {"F1": 0.856},
                "phonons": {"avg_freq": {"MAE": 18.2}},
                "elasticity": {
                    "bulk_modulus_vrh": {"MAE": 5.560},
                    "shear_modulus_vrh": {"MAE": 7.277},
                },
            },
        },
        "kappa": {"kappa_SRME": 0.455},
        "mof": {"bulk_modulus_MAPE_pct": 26.6},
        "amorphous": {"energy_RelRMSE_pct_mean": 8.780, "force_RelRMSE_pct_mean": 14.8},
        "cleavage": {"cleavage_relMAE_pct": 6.598},
        "neb": {"barrier_MAE_eV": 0.147},
    }
    assert expanded_score(grader.flatten(report))["score"] == pytest.approx(50)

    # Exercise the output boundary: keep the aggregate and omit unavailable raw metrics.
    sealed, labels, output = (tmp_path / name for name in ("sealed", "labels", "output"))
    sealed.mkdir()
    snapshot = {"schema": grader.SCHEMA, "runs": [], "predictions": {}, "inputs": {}}
    for suite in report:
        prediction = sealed / suite
        inputs = labels / suite / "inputs"
        prediction.mkdir()
        inputs.mkdir(parents=True)
        (prediction / "run.json").write_text("{}")
        snapshot["predictions"][suite] = grader.tree_sha256(prediction)
        snapshot["inputs"][suite] = grader.tree_sha256(inputs)
    (sealed / "snapshot.json").write_text(json.dumps(snapshot))
    (tmp_path / "READY").write_text(grader.sha256(sealed / "snapshot.json"))
    monkeypatch.setattr(grader, "score_physical", lambda *args: report["physical"])
    monkeypatch.setattr(grader, "score_extra", lambda suite, *args: report[suite])
    monkeypatch.setattr(sys, "argv", ["grader", "--sealed", str(sealed),
                                    "--labels", str(labels), "--output", str(output)])
    grader.main()
    reward = json.loads((output / "reward.json").read_text())
    report_out = json.loads((output / "report.json").read_text())
    expected_v4 = ssi_v4_score(grader.flatten(report))
    assert reward["score"] == pytest.approx(expected_v4["score"], abs=1e-12)
    assert reward["reward"] == pytest.approx(expected_v4["score"], abs=1e-12)
    assert reward["signed_score"] == pytest.approx(expected_v4["signed_score"], abs=1e-12)
    assert reward["crystal_potential_score"] == pytest.approx(50)
    assert reward["discovery_F1"] == report["physical"]["metrics"]["discovery"]["F1"]
    assert "phonon_entropy_MAE" not in reward
    assert all(isinstance(value, (int, float)) for value in reward.values())
    assert report_out["ssi_v4"]["score"] == pytest.approx(expected_v4["score"], abs=1e-12)
    assert report_out["aggregate"]["score"] == pytest.approx(50)


def test_ssi_v4_score_parity_with_harness_v3():
    import importlib.util
    from pathlib import Path

    harness_path = next(p for p in Path(__file__).resolve().parents if (p / "harness" / "ssi_index_v3.py").is_file()) / "harness" / "ssi_index_v3.py"
    spec = importlib.util.spec_from_file_location("ssi_index_v3", harness_path)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)

    perfect = {k: 0.0 for k in ATOM_ZERO_FORCE_ANCHORS}
    perfect["discovery_F1"] = 1.0

    zero_force = {"discovery_F1": DISCOVERY_F1_BASE, **ATOM_ZERO_FORCE_ANCHORS}

    partial_negative_suite = {
        **REFERENCE_V2,
        "mof_bulk_modulus_MAPE_pct": 180.0,  # worse than zero-force -> negative suite term, positive mean
    }

    failed_all = {
        "discovery_F1": 0.0,
        "phonon_avg_freq_MAE_K": 600.0,
        "bulk_modulus_MAE_GPa": 150.0,
        "shear_modulus_MAE_GPa": 80.0,
        "kappa_SRME": 2.0,
        "mof_bulk_modulus_MAPE_pct": 200.0,
        "amorphous_energy_RelRMSE_pct": 200.0,
        "amorphous_force_RelRMSE_pct": 200.0,
        "cleavage_relMAE_pct": 200.0,
        "neb_barrier_MAE_eV": 2.0,
    }

    for fixture in (perfect, zero_force, REFERENCE_V2, partial_negative_suite, failed_all):
        got = ssi_v4_score(fixture)
        ref = harness.atom(fixture)
        assert abs(got["score"] - ref["score"]) < 1e-12
        assert abs(got["signed_score"] - ref["signed_score"]) < 1e-12
        assert set(got["components"]) == set(ref["components"])
        for comp_name, comp_val in ref["components"].items():
            assert abs(got["components"][comp_name] - comp_val) < 1e-12

    assert abs(ssi_v4_score(perfect)["score"] - 1.0) < 1e-15
    zf_res = ssi_v4_score(zero_force)
    assert abs(zf_res["score"]) < 1e-15
    assert abs(zf_res["signed_score"]) < 1e-15
    assert all(abs(v) < 1e-15 for v in zf_res["components"].values())
    assert ssi_v4_score(failed_all)["signed_score"] < 0.0
    assert ssi_v4_score(failed_all)["score"] == 0.0


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1, True])
def test_ssi_v4_score_invalid_or_missing_suite_is_zero(value):
    metrics = {**REFERENCE_V2, "kappa_SRME": value}
    assert ssi_v4_score(metrics)["score"] == 0.0
    assert ssi_v4_score({})["score"] == 0.0
    assert ssi_v4_score(REFERENCE_V2, valid=False)["score"] == 0.0


def test_dev_dockerfile_and_compose_exclude_sealed_tests():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "environment" / "Dockerfile").read_text()
    compose = (root / "environment" / "docker-compose.yaml").read_text()
    assert "tests" not in dockerfile
    assert "tests" not in compose

    readme = (root / "tests" / "README.md").read_text()
    assert "## Scientific & Literature Justification of Verifier Design" in readme
    assert "### 1. Primary Metric & Upstream Paper Provenance" in readme
    assert "### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted" in readme
    assert "### 3. Diagnostic Signals Retained in report.json" in readme


def test_main_verifier_invalid_submission_writes_zero_v4_score(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).with_name("main_verifier.py")
    output = tmp_path / "output"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--submission",
            str(tmp_path / "missing"),
            "--input",
            str(tmp_path / "inputs"),
            "--sealed",
            str(tmp_path / "sealed"),
            "--output",
            str(output),
        ],
        check=True,
    )
    report = json.loads((output / "report.json").read_text())
    reward = json.loads((output / "reward.json").read_text())
    assert report["ssi_v4"]["version"] == "ssi-atomistic-score-v4"
    assert report["ssi_v4"]["score"] == 0.0
    assert report["aggregate"]["version"] == "crystal-potential-score-v2"
    assert reward["score"] == reward["reward"] == reward["crystal_potential_score"] == 0.0
    assert reward["invalid_submission"] is True
