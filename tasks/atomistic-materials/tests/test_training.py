import copy
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from pymatgen.core import Structure

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment/data"))
from records import formula, mptrj_record, validate
from splits import stable_hash


def test_raw_total_energy_and_vasp_stress_conversion():
    atoms = bulk("Cu", cubic=True)
    row = dict(
        structure=Structure.from_ase_atoms(atoms).as_dict(),
        uncorrected_total_energy=-12.0,
        corrected_total_energy=-14.0,
        energy_per_atom=-3.5,
        force=np.zeros((4, 3)).tolist(),
        stress=(10 * np.eye(3)).tolist(),
    )
    result = mptrj_record("mp-1", "mp-1-0-0", row)
    assert result["energy"] == -12.0
    assert np.allclose(result["stress"], -np.eye(3) * units.GPa)
    assert validate(result) == "Cu"
    row["energy_per_atom"] = -12.0
    with pytest.raises(ValueError, match="identity"):
        mptrj_record("mp-1", "frame", row)


def test_composition_exclusion_covers_cell_changes():
    small = bulk("Cu")
    large = small.repeat((2, 2, 2))
    large.rattle(0.01, seed=5)
    assert formula(small) == formula(large)
    assert stable_hash(formula(small)) == stable_hash(formula(large))
