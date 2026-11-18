"""CPU unit tests for V3/V4 weather metric parity and container visibility isolation."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import xarray as xr

TASK_ROOT = Path(__file__).resolve().parents[1]
WN2_DIR = TASK_ROOT / "environment" / "weathernext" / "wn2"
HARNESS_PATH = next(p for p in Path(__file__).resolve().parents if (p / "harness" / "ssi_index_v3.py").is_file()) / "harness" / "ssi_index_v3.py"

sys.path.insert(0, str(WN2_DIR))
from merge_scores import reward as wn2_reward  # noqa: E402
from verify_track import primary_reward  # noqa: E402


def _load_ssi_v3():
    spec = importlib.util.spec_from_file_location("ssi_index_v3", HARNESS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SSI_V3 = _load_ssi_v3()


def _build_scales(protocol: dict, rng: np.random.Generator) -> xr.Dataset:
    levels = protocol["pressure_levels_hpa"]
    data_vars = {}
    for name, spec in protocol["variables"].items():
        if spec["pressure_levels"]:
            vals = rng.uniform(0.5, 50.0, size=len(levels))
            data_vars[name] = ("level", vals)
        else:
            data_vars[name] = float(rng.uniform(0.5, 25.0))
    return xr.Dataset(data_vars, coords={"level": levels})


def _build_full_protocol_reports(
    protocol: dict,
    scales: xr.Dataset,
    mode: str,
    rng: np.random.Generator,
) -> list[dict]:
    reports = []
    for date in protocol["evaluation_initializations"]:
        per_case = []
        for var_name, spec in protocol["variables"].items():
            levels = protocol["pressure_levels_hpa"] if spec["pressure_levels"] else [None]
            for level in levels:
                scale_val = (
                    float(scales[var_name].sel(level=level))
                    if level is not None
                    else float(scales[var_name])
                )
                for lead in protocol["lead_hours"]:
                    if mode == "zero":
                        crps = 0.0
                    elif mode == "starter_like":
                        crps = 0.469183 * scale_val
                    elif mode == "degraded":
                        crps = float(rng.uniform(3.0, 12.0) * scale_val)
                    else:
                        lead_factor = 0.15 + 0.75 * (lead / 360.0)
                        crps = float(lead_factor * rng.uniform(0.7, 1.3) * scale_val)
                    per_case.append(
                        {
                            "case": date,
                            "variable": var_name,
                            "level_hpa": level,
                            "lead_hours": lead,
                            "crps": crps,
                        }
                    )
        reports.append(
            {
                "status": "passed",
                "members": protocol["members"],
                "per_case": per_case,
            }
        )
    return reports


@pytest.mark.parametrize("mode", ["normal", "starter_like", "zero", "degraded"])
def test_primary_reward_matches_ssi_index_v3_weather_full_10_var_panel(mode: str) -> None:
    protocol = json.loads((WN2_DIR / "protocol.json").read_text())
    assert len(protocol["variables"]) == 10
    rng = np.random.default_rng(20261115)
    scales = _build_scales(protocol, rng)
    reports = _build_full_protocol_reports(protocol, scales, mode, rng)

    container_out = primary_reward(reports, protocol, scales)
    assert "score" in container_out and "reward" in container_out
    assert abs(container_out["score"] - container_out["reward"]) < 1e-15
    assert set(container_out["normalized_crps_by_variable"]) == set(protocol["variables"])

    v3_out = SSI_V3.weather(container_out)
    assert abs(container_out["score"] - v3_out["score"]) < 1e-12
    assert abs(container_out["reward"] - v3_out["score"]) < 1e-12

    # Also check fallback branch in ssi_index_v3.weather when only normalized_crps is passed
    v3_scalar_only = SSI_V3.weather({"normalized_crps": container_out["normalized_crps"]})
    assert abs(container_out["score"] - v3_scalar_only["score"]) < 1e-12

    if mode == "zero":
        assert abs(container_out["score"] - 1.0) < 1e-12
    elif mode == "starter_like":
        assert abs(container_out["normalized_crps"] - 0.469183) < 1e-12
        assert abs(container_out["score"] - (1.0 / 1.469183)) < 1e-12


def test_empirical_crps_fair_false_and_evaluator_digest_preserved() -> None:
    eval_path = WN2_DIR / "evaluate.py"
    text = eval_path.read_text()
    assert 'CRPSEnsemble(ensemble_dim="member", fair=False, use_sort=True' in text
    assert 'estimator="empirical CRPS; fair=False"' in text
    digest = hashlib.sha256(eval_path.read_bytes()).hexdigest()
    assert digest == "e109bd45a248012aa91388be585d741fa6a416de1dfdf7bf02aa55778c8cc427"


def test_failed_verification_writes_both_score_and_reward_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import verify_track

    protocol_path = WN2_DIR / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    package = tmp_path / "package"
    package.mkdir()
    (package / "package.json").write_text("{}")
    benchmark = tmp_path / "benchmark"
    (benchmark / "tracker" / "truth").mkdir(parents=True)
    (benchmark / "tracker" / "named-truth.json").write_text("{}")
    scales = _build_scales(protocol, np.random.default_rng(1))
    scales.to_netcdf(benchmark / "training-scales.nc")
    output = tmp_path / "out"
    output.mkdir()
    (output / "forecast-status.json").write_text(
        json.dumps({"status": "failed", "predictions_retained": False})
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_track",
            "--package",
            str(package),
            "--benchmark",
            str(benchmark),
            "--output",
            str(output),
            "--phase",
            "score",
            "--legacy-case-mode",
        ],
    )
    with pytest.raises(ValueError):
        verify_track.main()
    reward_payload = json.loads((output / "reward.json").read_text())
    assert reward_payload["score"] == 0.0
    assert reward_payload["reward"] == 0.0


def test_docs_and_visibility_boundary() -> None:
    instruction_text = (TASK_ROOT / "instruction.md").read_text()
    contract_text = (TASK_ROOT / "environment" / "contract.md").read_text()
    assert "mean_10_variables(normalized_crps)" in instruction_text
    assert "fair=False" in instruction_text
    assert "mean_10_variables(normalized_crps)" in contract_text
    assert "fair=False" in contract_text

    env_dockerfile = (TASK_ROOT / "environment" / "Dockerfile").read_text()
    env_compose = (TASK_ROOT / "environment" / "docker-compose.yaml").read_text()
    for line in env_dockerfile.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert "tests/" not in stripped and "/tests" not in stripped, f"Leak in Dockerfile: {line}"
    for line in env_compose.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert "tests/" not in stripped and "/tests" not in stripped, f"Leak in docker-compose: {line}"
