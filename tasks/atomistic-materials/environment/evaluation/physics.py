"""Fixed protocols adapted from FAIRChem's MIT-licensed UMA benchmark recipes.

See THIRD_PARTY_NOTICES.md and PROTOCOLS.md for source revision and differences.
The calculator receives geometry only. These routines never open target files.
"""

from __future__ import annotations

import numpy as np
from ase import units
from ase.build import make_supercell
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from phonopy import Phonopy
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.analysis.elasticity import DeformedStructureSet, ElasticTensor, Strain
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_pmg_structure, get_phonopy_structure

from common import decode_atoms, encode_atoms

THZ_TO_K = units._hplanck * 1e12 / units._k


def relax(atoms, calculator, *, fmax=0.02, cell=True):
    atoms.calc = calculator
    optimizer = FIRE(FrechetCellFilter(atoms) if cell else atoms, logfile=None)
    converged = optimizer.run(fmax=fmax, steps=500)
    return {"converged": bool(converged), "steps": optimizer.nsteps}


def discovery(case, calculator):
    atoms = decode_atoms(case["atoms"])
    diagnostics = relax(atoms, calculator)
    return {
        "energy": float(atoms.get_potential_energy()),
        "atoms": encode_atoms(atoms),
        "diagnostics": diagnostics,
    }


def make_phonon(case):
    atoms = decode_atoms(case["atoms"])
    cell = PhonopyAtoms(
        numbers=atoms.numbers,
        positions=atoms.positions,
        cell=atoms.cell.array,
        masses=case["phonon"]["masses"],
    )
    return Phonopy(
        cell,
        supercell_matrix=case["phonon"]["supercell_matrix"],
        primitive_matrix=case["phonon"]["primitive_matrix"],
        symprec=1e-4,
    )


def phonon_observables(phonon):
    # Same commensurate q-points as upstream; retain negative frequencies.
    qpoints = get_commensurate_points(phonon.supercell_matrix)
    phonon.run_qpoints(qpoints)
    frequencies = phonon.get_qpoints_dict()["frequencies"] * THZ_TO_K
    phonon.run_mesh((20, 20, 20))
    phonon.run_thermal_properties(t_min=0, t_max=600, t_step=75)
    thermal = phonon.get_thermal_properties_dict()
    index = list(thermal["temperatures"]).index(300)
    return {
        "max_freq": float(frequencies.max()),
        "avg_freq": float(frequencies.mean()),
        **{
            key: float(thermal[key][index])
            for key in ("entropy", "heat_capacity", "free_energy")
        },
    }


def phonons(case, calculator):
    reference_geometry = make_phonon(case)  # no DFT force constants
    primitive = get_pmg_structure(reference_geometry.primitive).to_ase_atoms()
    diagnostics = relax(primitive, calculator, fmax=0.005)
    transform = (
        np.linalg.inv(reference_geometry.primitive_matrix.T)
        if reference_geometry.primitive_matrix is not None
        else np.eye(3)
    )
    # Guard against silently truncating a nonintegral transformation.
    if not np.allclose(transform, np.rint(transform), atol=1e-8):
        raise ValueError("Nonintegral primitive-to-unit-cell transformation")
    unitcell = make_supercell(primitive, np.rint(transform).astype(int))
    phonon = Phonopy(
        get_phonopy_structure(Structure.from_ase_atoms(unitcell)),
        reference_geometry.supercell_matrix,
        primitive_matrix=reference_geometry.primitive_matrix,
        symprec=1e-4,
    )
    phonon.generate_displacements(distance=0.01)
    phonon.forces = [
        calculator.get_forces(get_pmg_structure(cell).to_ase_atoms())
        for cell in phonon.supercells_with_displacements
    ]
    phonon.produce_force_constants(fc_calculator="traditional")
    # No post-fit force-constant symmetrization, matching the UMA protocol.
    diagnostics.update(
        n_displacements=len(phonon.supercells_with_displacements),
        supercell_atoms=len(phonon.supercell),
    )
    return {**phonon_observables(phonon), "diagnostics": diagnostics}


def elasticity(case, calculator):
    atoms = decode_atoms(case["atoms"])
    diagnostics = relax(atoms, calculator)
    equilibrium_stress = atoms.get_stress(voigt=False)
    deformations = DeformedStructureSet(
        Structure.from_ase_atoms(atoms),
        norm_strains=(-0.01, -0.005, 0.005, 0.01),
        shear_strains=(-0.06, -0.03, 0.03, 0.06),
    )
    stresses, relaxation_results = [], []
    for structure in deformations:
        strained = structure.to_ase_atoms()
        relaxation_results.append(relax(strained, calculator, cell=False))
        stresses.append(strained.get_stress(voigt=False))
    tensor = ElasticTensor.from_independent_strains(
        [Strain.from_deformation(d) for d in deformations.deformations],
        stresses,
        eq_stress=equilibrium_stress,
    )
    diagnostics["strained_relaxations"] = relaxation_results
    return {
        "bulk_modulus_vrh": float(tensor.k_vrh / units.GPa),
        "shear_modulus_vrh": float(tensor.g_vrh / units.GPa),
        "elastic_tensor_GPa": (tensor.voigt / units.GPa).tolist(),
        "diagnostics": diagnostics,
    }


CALCULATIONS = {"discovery": discovery, "phonons": phonons, "elasticity": elasticity}
