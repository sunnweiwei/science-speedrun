"""Check energy-derived forces and tensile-positive ASE stress after model export."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))
from common import decode_atoms, read_rows, write_json
from predict import load_calculator


def check(checkpoint, labels, device="cuda"):
    calc = load_calculator("uma", checkpoint, device)
    reports = []
    for row in read_rows(labels)[:4]:
        atoms = decode_atoms(row["atoms"])
        atoms.calc = calc
        calc.reset()
        forces = atoms.get_forces()
        stress = atoms.get_stress(voigt=False)
        i, j = np.unravel_index(np.argmax(np.abs(forces)), forces.shape)

        def energy(a):
            a.calc = calc
            calc.reset()
            return a.get_potential_energy()

        dx = 0.005
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions[i, j] += dx
        minus.positions[i, j] -= dx
        fd_force = -(energy(plus) - energy(minus)) / (2 * dx)
        eps = 0.001
        plus, minus = atoms.copy(), atoms.copy()
        strain = np.eye(3)
        strain[0, 0] += eps
        plus.set_cell(atoms.cell @ strain, scale_atoms=True)
        strain = np.eye(3)
        strain[0, 0] -= eps
        minus.set_cell(atoms.cell @ strain, scale_atoms=True)
        fd_stress = (energy(plus) - energy(minus)) / (2 * eps * atoms.get_volume())
        ferr = float(abs(fd_force - forces[i, j]))
        serr = float(abs(fd_stress - stress[0, 0]))
        reports.append(
            dict(
                id=row["id"],
                force_component=[int(i), int(j)],
                force_eV_per_A=float(forces[i, j]),
                force_fd_eV_per_A=float(fd_force),
                force_error_eV_per_A=ferr,
                stress_xx_eV_per_A3=float(stress[0, 0]),
                stress_fd_eV_per_A3=float(fd_stress),
                stress_error_eV_per_A3=serr,
                passed=bool(
                    ferr < 0.01 + 0.01 * abs(forces[i, j])
                    and serr < 0.0002 + 0.01 * abs(stress[0, 0])
                ),
            )
        )
    return dict(
        passed=all(x["passed"] for x in reports),
        cases=reports,
        dx_A=dx,
        strain=eps,
        scope="Four held-out geometries; central differences; FP32 tolerances; not a long-trajectory conservation test",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["checkpoint", "labels", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    r = check(a.checkpoint, a.labels)
    write_json(a.output, r)
    print(r)
    raise SystemExit(0 if r["passed"] else 2)
