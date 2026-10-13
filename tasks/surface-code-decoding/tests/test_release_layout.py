"""The AlphaQubit training code reads a development release root (the agent's /workspace/data) exactly as it
reads the raw Zenodo directory: same conditions, same train/validation rows in the same order."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from baselines import alphaqubit_data, data

NAME = "surface_code_bX_d3_r01_center_3_5"
WIDTH = 1  # 1 round x 8 detectors -> one packed byte per shot


@pytest.fixture()
def roots(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    detectors = rng.integers(0, 256, size=(50_000, WIDTH), dtype=np.uint8)
    labels = rng.integers(0, 2, size=50_000, dtype=np.uint8)
    raw = tmp_path / "raw" / NAME
    raw.mkdir(parents=True)
    detectors.tofile(raw / "detection_events.b8")
    (raw / "obs_flips_actual.01").write_bytes(b"".join(b"1\n" if bit else b"0\n" for bit in labels))
    for name in ("circuit_ideal.stim", "pij_from_even_for_odd.dem"):
        (raw / name).write_text(f"# {name}\n")
    release = tmp_path / "release"
    for split in ("train", "validation"):
        rows = data.hardware_split_indices(split)
        folder = release / split / "conditions" / NAME
        folder.mkdir(parents=True)
        np.savez(folder / "arrays.npz", sample_ids=rows.astype(np.uint64), detectors=detectors[rows], labels=labels[rows])
        for name in ("circuit_ideal.stim", "pij_from_even_for_odd.dem"):
            (folder / name).write_text(f"# {name}\n")
        (release / split / "manifest.json").write_text(json.dumps({"split": split, "conditions": [{"name": NAME}]}))
    return tmp_path / "raw", release


def test_release_root_is_recognised(roots: tuple[Path, Path]) -> None:
    raw, release = roots
    assert data.is_release_root(release) and not data.is_release_root(raw)
    (condition,) = data.discover_conditions(release)
    assert (condition.name, condition.group_id, condition.rounds) == (NAME, "d3_bX_center_3_5", 1)
    assert condition.path == release / "train" / "conditions" / NAME


@pytest.mark.parametrize("loader", [data.load_hardware_packed, alphaqubit_data.load_hardware_packed])
@pytest.mark.parametrize("split", ["train", "validation"])
def test_release_rows_equal_raw_rows(roots: tuple[Path, Path], loader, split: str) -> None:
    raw, release = roots
    (from_raw,), (from_release,) = data.discover_conditions(raw), data.discover_conditions(release)
    (raw_det, raw_lab), (rel_det, rel_lab) = loader(from_raw, split), loader(from_release, split)
    assert raw_det.dtype == rel_det.dtype == np.uint8 and rel_lab.shape == (len(rel_det), 1)
    assert np.array_equal(raw_det, rel_det) and np.array_equal(raw_lab, rel_lab)


def test_release_loader_refuses_test_rows(roots: tuple[Path, Path]) -> None:
    (condition,) = data.discover_conditions(roots[1])
    with pytest.raises(ValueError):
        data.load_release_packed(condition, "test")


def test_package_template_has_the_submission_layout_without_weights(tmp_path: Path) -> None:
    from baselines.alphaqubit_artifact import EXPECTED_GROUPS, INFERENCE_SOURCES
    from qecbench.package import write_template
    template = tmp_path / "package-template"
    write_template(template)
    for name in ("infer.sh", "README.md", "model_config.json", "pyproject.toml", "uv.lock", "requirements.lock"):
        assert (template / name).is_file(), name
    assert {path.name for path in (template / "src/baselines").iterdir()} >= set(INFERENCE_SOURCES) | {"__init__.py", "metrics.py"}
    assert (template / "src/qecbench/inference.py").is_file()
    config = json.loads((template / "model_config.json").read_text())
    assert set(config["models"]) == set(EXPECTED_GROUPS)
    assert not (template / "weights").exists() and not (template / "model_manifest.json").exists()
    assert not any(template.rglob("*.npz"))
