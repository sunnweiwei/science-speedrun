"""Evaluate portable checkpoints on parent/composition-disjoint E/F/S labels."""

from pathlib import Path
import json
import sys
import numpy as np
from ase import units

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))
from common import decode_atoms, read_rows
from predict import load_calculator


def evaluate(checkpoint, labels, device="cuda"):
    calc = load_calculator("uma", Path(checkpoint), device)
    buckets = {}
    for row in read_rows(Path(labels)):
        a = decode_atoms(row["atoms"])
        a.calc = calc
        calc.reset()
        bucket = buckets.setdefault(
            row["source"], {"energy": [], "forces": [], "stress": []}
        )
        bucket["energy"].append(abs(a.get_potential_energy() - row["energy"]) / len(a))
        bucket["forces"].extend(np.abs(a.get_forces() - row["forces"]).ravel().tolist())
        bucket["stress"].extend(
            np.abs(a.get_stress(voigt=False) - row["stress"]).ravel().tolist()
        )
    return {
        key: {
            "energy_MAE_meV_per_atom": float(np.mean(v["energy"]) * 1000),
            "force_MAE_eV_per_A": float(np.mean(v["forces"])),
            "stress_MAE_GPa": float(np.mean(v["stress"]) / units.GPa),
            "frames": len(v["energy"]),
        }
        for key, v in buckets.items()
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = evaluate(a.checkpoint, a.labels)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result))
