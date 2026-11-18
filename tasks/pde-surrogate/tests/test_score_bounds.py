"""Verify V4 PDE surrogate 4-group score parity with ssi_index_v3.pde() and removal of normalised_score."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "environment" / "evaluation"))
import main_verifier as mv  # noqa: E402
import score as S  # noqa: E402

BOUNDS = ROOT / "environment" / "evaluation" / "score_bounds.json"


def _load_ssi_index_v3():
    v3_path = next(p for p in Path(__file__).resolve().parents if (p / "harness" / "ssi_index_v3.py").is_file()) / "harness" / "ssi_index_v3.py"
    if not v3_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("ssi_index_v3", v3_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bounds_file_is_well_formed():
    b = json.loads(BOUNDS.read_text())
    assert b["schema"] == "benchmark-score-bounds/v1"
    assert b["upper"]["reward"] > b["lower"]["reward"]
    for side in ("lower", "upper"):
        assert b[side]["what"] and b[side]["source"]


def test_obsolete_normalised_score_removed_from_main_verifier():
    assert not hasattr(mv, "normalised")
    src = (ROOT / "tests" / "main_verifier.py").read_text(encoding="utf-8")
    assert "normalised_score" not in src


def _stage_four_group_fixture(tmp_path: Path):
    rng = np.random.default_rng(20261115)
    inp, truth, out = tmp_path / "in", tmp_path / "truth", tmp_path / "out"
    (inp / "cases").mkdir(parents=True)
    (truth / "cases").mkdir(parents=True)
    (out / "cases").mkdir(parents=True)

    specs = [
        ("c_ape_0", "ape2d/burgers", 4),
        ("c_ape_1", "ape2d/kdv", 4),
        ("c_pdea", "pdearena/navier_stokes_2d", 4),
        ("c_am", "well/active_matter", 4),
        ("c_trl", "well/turbulent_radiative_layer_2D", 4),
    ]
    index_cases = []
    targets = {}
    for cid, ds, k_steps in specs:
        target = rng.normal(size=(k_steps, 2, 8, 8)).astype(np.float32)
        targets[cid] = target
        with h5py.File(truth / "cases" / f"{cid}.h5", "w") as f:
            f.create_dataset("target", data=target)
            f.attrs["fields"] = ["u", "v"]
            f.attrs["field_scheme"] = "VV"
            f.attrs["dataset"] = ds
        index_cases.append({"case_id": cid, "dataset": ds, "n_steps_out": k_steps})

    (inp / "index.json").write_text(json.dumps({"contract": S.CONTRACT, "cases": index_cases}))
    return inp, truth, out, targets


@pytest.mark.parametrize("mode", ["normal", "partial_failure", "all_failed", "exact_continuation"])
def test_four_group_score_matches_ssi_index_v3_within_1e12(tmp_path: Path, mode: str):
    inp, truth, out, targets = _stage_four_group_fixture(tmp_path)
    rng = np.random.default_rng(99)
    for idx, (cid, target) in enumerate(targets.items()):
        if mode == "all_failed":
            continue
        if mode == "partial_failure" and idx in (1, 3):
            continue
        pred = target if mode == "exact_continuation" else (target + (0.15 * (idx + 1)) * rng.normal(size=target.shape)).astype(np.float32)
        with h5py.File(out / "cases" / f"{cid}.h5", "w") as f:
            f.create_dataset("prediction", data=pred)

    rep = S.score(inp, truth, out)
    expected_groups = {"ape2d", "pdearena", "well/active_matter", "well/turbulent_radiative_layer_2D"}
    assert set(rep["groups"].keys()) == expected_groups
    assert rep["score"] == rep["reward"]

    # Verify exact mathematical formula (1 - log10(clip(v, 0.01, 10))) / 3 per group.
    for g_name, g_entry in rep["groups"].items():
        v_clipped = min(10.0, max(0.01, g_entry["vrmse"]))
        formula_val = (1.0 - math.log10(v_clipped)) / 3.0
        assert abs(g_entry["reward"] - formula_val) < 1e-12, g_name

    v3 = _load_ssi_index_v3()
    if v3 is not None:
        v3_res = v3.pde(rep["groups"])
        assert abs(rep["score"] - v3_res["score"]) < 1e-12
        assert abs(rep["reward"] - v3_res["score"]) < 1e-12

    if mode == "all_failed":
        assert rep["score"] == 0.0
    elif mode == "exact_continuation":
        assert rep["score"] == 1.0


def test_dev_container_excludes_tests_tree():
    dev_dockerfile = (ROOT / "environment" / "Dockerfile").read_text(encoding="utf-8")
    dev_compose = (ROOT / "environment" / "docker-compose.yaml").read_text(encoding="utf-8")
    assert "COPY tests" not in dev_dockerfile
    assert "/tests" not in dev_dockerfile
    assert "tests" not in dev_compose
