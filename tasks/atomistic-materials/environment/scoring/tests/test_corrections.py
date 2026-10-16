import pytest
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.entries.computed_entries import ComputedStructureEntry

from common import encode_atoms
from score import corrected_formation


def test_formation_energy_is_per_atom_and_composition_preserved():
    atoms = bulk("Al").repeat((2, 1, 1))
    entry = ComputedStructureEntry(
        AseAtomsAdaptor.get_structure(atoms),
        -6,
        parameters={"run_type": "GGA", "potcar_symbols": ["PAW_PBE Al 04Jan2001"]},
    )
    target = {"entry": entry.as_dict(), "elemental_refs": {"Al": -4}}
    prediction = {"energy": -6, "atoms": encode_atoms(atoms)}
    assert corrected_formation(prediction, target) == pytest.approx(1)
    prediction["atoms"] = encode_atoms(bulk("Cu"))
    with pytest.raises(ValueError, match="composition"):
        corrected_formation(prediction, target)
