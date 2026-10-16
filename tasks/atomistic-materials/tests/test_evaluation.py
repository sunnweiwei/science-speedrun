import copy

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase import units

from common import decode_atoms, encode_atoms
from physics import elasticity, make_phonon, phonons, relax
from score import corrected_formation, discovery_metrics, score_rows


def test_no_reference_labels_in_atoms():
    atoms = bulk("Cu")
    atoms.info["energy"] = -9999
    atoms.set_initial_magnetic_moments([17])
    atoms.calc = EMT()
    atoms.get_potential_energy()
    restored = decode_atoms(encode_atoms(atoms))
    assert restored.info == {}
    assert restored.calc is None
    assert set(restored.arrays) == {"numbers", "positions"}


def sample_rows():
    targets = [
        {
            "id": f"elasticity/{i}",
            "suite": "elasticity",
            "bulk_modulus_vrh": 100 + 50 * i,
            "shear_modulus_vrh": 30 + 10 * i,
        }
        for i in range(2)
    ]
    predictions = [dict(row, status="ok") for row in targets]
    predictions[0]["bulk_modulus_vrh"] += 10
    predictions[1]["bulk_modulus_vrh"] -= 20
    return targets, predictions


def test_scores_match_by_id_not_row_order():
    targets, predictions = sample_rows()
    score = score_rows(targets, predictions)
    assert score == score_rows(targets, predictions[::-1])
    assert score["elasticity"]["bulk_modulus_vrh"]["MAE"] == 15


@pytest.mark.parametrize("mutation", ["duplicate", "unknown"])
def test_malformed_prediction_sets_are_rejected(mutation):
    targets, predictions = sample_rows()
    if mutation == "duplicate":
        predictions.append(copy.deepcopy(predictions[0]))
    else:
        predictions[0]["id"] = "elasticity/unknown"
    with pytest.raises(ValueError):
        score_rows(targets, predictions)


@pytest.mark.parametrize("mutation", ["missing", "nan", "failed", "suite"])
def test_failed_cases_are_scored_and_cannot_improve_score(mutation):
    targets, predictions = sample_rows()
    clean = score_rows(targets, predictions)["elasticity"]
    if mutation == "missing":
        predictions.pop(0)
    elif mutation == "nan":
        predictions[0]["bulk_modulus_vrh"] = float("nan")
    elif mutation == "failed":
        predictions[0]["status"] = "failed"
    else:
        predictions[0]["suite"] = "phonons"
    scored = score_rows(targets, predictions)["elasticity"]
    assert scored["failed"] == 1
    for field in ("bulk_modulus_vrh", "shear_modulus_vrh"):
        assert scored[field]["MAE"] >= clean[field]["MAE"]


def test_signed_hull_threshold_and_units():
    metrics = discovery_metrics([-0.1, 0, 0.2, 0.3], [-0.2, 0.01, -0.1, 0.3])
    assert (metrics["TP"], metrics["FN"], metrics["FP"], metrics["TN"]) == (1, 1, 1, 1)
    assert metrics["F1"] == 0.5
    assert metrics["hull_MAE_meV_per_atom"] == pytest.approx(102.5)


def test_wrong_correction_runtime_is_rejected():
    with pytest.raises(ValueError, match="pymatgen 2023.5.10"):
        corrected_formation({}, {})


def test_elastic_bulk_matches_independent_energy_curvature():
    atoms = bulk("Cu")
    calculator = EMT()
    result = elasticity({"atoms": encode_atoms(atoms)}, calculator)
    relax(atoms, calculator)
    energies = []
    delta = 0.001
    for volume_scale in (1 - delta, 1, 1 + delta):
        probe = atoms.copy()
        probe.set_cell(atoms.cell * volume_scale ** (1 / 3), scale_atoms=True)
        probe.calc = EMT()
        energies.append(probe.get_potential_energy())
    curvature = (energies[0] - 2 * energies[1] + energies[2]) / (
        delta**2 * atoms.get_volume()
    )
    assert result["bulk_modulus_vrh"] == pytest.approx(curvature / units.GPa, rel=0.02)
    assert 20 < result["shear_modulus_vrh"] < 200


def test_phonon_pipeline_has_no_force_constants_at_input():
    atoms = bulk("Cu")
    case = {
        "atoms": encode_atoms(atoms),
        "phonon": {
            "masses": atoms.get_masses().tolist(),
            "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            "primitive_matrix": None,
        },
    }
    assert make_phonon(case).force_constants is None
    result = phonons(case, EMT())
    assert 100 < result["max_freq"] < 600
    assert 0 < result["heat_capacity"] < 3 * 8.3145
    assert np.isfinite(result["free_energy"])
    assert result["diagnostics"]["supercell_atoms"] == 8
