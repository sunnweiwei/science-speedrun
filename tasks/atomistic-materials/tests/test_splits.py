"""Scientific split invariants, including cross-suite and historical leakage."""

import copy
from collections import Counter

import pytest

from splits import assign, composition_key, load_split, validate_assignment


def row(suite, material_id, formula, n_sites=2):
    return {
        "id": f"{suite}/{material_id}",
        "suite": suite,
        "material_id": material_id,
        "composition": composition_key(formula),
        "n_sites": n_sites,
    }


def test_canonical_composition_handles_polymorphs_supercells_and_dimers():
    assert composition_key("Fe2O3") == composition_key("Fe4 O6")
    assert composition_key("O") == composition_key("O2")
    assert composition_key("LiFePO4") != composition_key("LiFePO3")


def test_group_assignment_is_order_independent_and_protects_shared_ids():
    rows = [
        row("phonons", "mp-1", "Fe2O3"),
        row("elasticity", "mp-1", "FeO"),
        row("discovery", "wbm-1", "Fe4O6"),
        row("discovery", "wbm-2", "SiO2"),
    ]
    exposure = {"compositions": [composition_key("FeO")], "material_ids": []}
    result, _ = assign(rows, exposure)
    reversed_result, _ = assign(rows[::-1], exposure)
    assert result == reversed_result
    assert all(r["split"] == "val" for r in result if r["material_id"] != "wbm-2")
    validate_assignment(result, exposure)


def test_overlap_and_prior_exposure_are_rejected():
    exposure = {"compositions": [composition_key("FeO")], "material_ids": []}
    rows, _ = assign(
        [row("phonons", "mp-1", "FeO"), row("elasticity", "mp-1", "FeO")], exposure
    )
    corrupted = copy.deepcopy(rows)
    corrupted[0]["split"] = "test"
    with pytest.raises(ValueError, match="overlap"):
        validate_assignment(corrupted, exposure)
    for record in corrupted:
        record["split"] = "test"
    with pytest.raises(ValueError, match="exposed"):
        validate_assignment(corrupted, exposure)


def test_frozen_release_covers_every_case_and_is_reproducible():
    import json
    from splits import DEFAULT_SPLITS

    if not (DEFAULT_SPLITS / "manifest.json").exists():
        pytest.skip("Materialize --components protocol for frozen-release checks")
    rows, manifest = load_split()
    assert Counter(r["suite"] for r in rows) == {
        "discovery": 256963,
        "phonons": 9958,
        "elasticity": 12122,
    }
    exposure = json.loads((DEFAULT_SPLITS / "exposure.json").read_text())
    inputs = [{k: v for k, v in r.items() if k not in ("group", "split")} for r in rows]
    rebuilt, _ = assign(
        inputs, exposure, manifest["target_validation_fraction"], manifest["seed"]
    )
    assert rebuilt == rows


def test_manifest_tampering_is_rejected(tmp_path):
    import shutil
    from splits import DEFAULT_SPLITS

    if not (DEFAULT_SPLITS / "manifest.json").exists():
        pytest.skip("Materialize --components protocol for frozen-release checks")
    directory = tmp_path / "split"
    shutil.copytree(DEFAULT_SPLITS, directory)
    with (directory / "assignments.jsonl.gz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_split(directory)
