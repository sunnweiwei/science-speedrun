"""Relax saved CIFs in independent GPU workers, without resampling or filtering."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import time

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoring import relaxation as protocol


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sample_timeout(signum, frame):
    raise TimeoutError("per-structure relaxation time limit exceeded")


def refine(structure, calculator) -> tuple:
    import numpy as np
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGSLineSearch
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = structure.to_ase_atoms()
    atoms.info = {}
    atoms.calc = calculator._get_ase_calculator()
    initial_energy = float(atoms.get_potential_energy())
    target = FrechetCellFilter(atoms)
    optimizer = BFGSLineSearch(target, logfile=None)
    converged = bool(optimizer.run(
        fmax=protocol.RECIPE["fmax_ev_per_angstrom"],
        steps=protocol.RECIPE["max_steps"],
    ))
    final_energy = float(atoms.get_potential_energy())
    forces = target.get_forces()
    if not all(np.isfinite(value).all() for value in (
        atoms.positions, atoms.cell.array, forces, [initial_energy, final_energy]
    )) or atoms.get_volume() <= 0:
        raise ValueError("relaxation produced nonfinite or nonpositive geometry")
    relaxed = AseAtomsAdaptor.get_structure(atoms)
    if list(relaxed.atomic_numbers) != list(structure.atomic_numbers):
        raise RuntimeError("relaxation changed species or atom ordering")
    return relaxed, {
        "converged": converged, "steps": optimizer.get_number_of_steps(),
        "initial_energy_ev": initial_energy, "final_energy_ev": final_energy,
        "max_generalized_force": float(np.linalg.norm(forces, axis=1).max()),
    }


def run_worker(output: Path, inventory: list[dict], index: int, workers: int, claims: Path | None = None) -> None:
    import torch
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifWriter
    from lemat_genbench.models.uma.calculator import create_uma_calculator

    torch.set_num_threads(2)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("each relaxation worker must see exactly one CUDA GPU")
    # Model-loading failures must stop the run, not become candidate failures.
    calculator = create_uma_calculator(model_name="uma-s-1", task="omat", device="cuda")
    signal.signal(signal.SIGALRM, sample_timeout)
    for ordinal, item in enumerate(inventory):
        if claims is None:
            if ordinal % workers != index:
                continue
        else:
            # Dynamic assignment: the first worker to create the claim file relaxes the sample.
            try:
                os.close(os.open(claims / f"{ordinal:06d}", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue
        source_path = Path(item["source_path"])
        destination = output / "generated" / item["structure_path"]
        record_path = output / "samples" / f"{ordinal:06d}.json"
        if protocol.sha256(source_path) != item["source_sha256"]:
            raise RuntimeError("source CIF changed after preflight")
        if record_path.exists():
            cached = json.loads(record_path.read_text())
            if (cached["input"] != item or cached["recipe"] != protocol.RECIPE
                    or not destination.is_file()
                    or protocol.sha256(destination) != cached["output_sha256"]):
                raise RuntimeError("cached relaxed sample changed")
            continue
        started = time.monotonic()
        record = {"input": item, "recipe": protocol.RECIPE}
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp.cif")
        try:
            signal.alarm(protocol.RECIPE["sample_timeout_seconds"])
            structure = Structure.from_file(source_path)
            relaxed, diagnostics = refine(structure, calculator)
            CifWriter(relaxed).write_file(temporary)
            # Check the actual exported bytes, not just the in-memory object.
            exported = Structure.from_file(temporary)
            if exported.composition != structure.composition:
                raise RuntimeError("CIF serialization changed composition")
            os.replace(temporary, destination)
            record.update(status="relaxed", **diagnostics)
        except Exception as error:
            if (isinstance(error, ImportError)
                    or (isinstance(error, OSError) and not isinstance(error, TimeoutError))) or any(token in str(error).lower() for token in (
                "cuda", "out of memory", "cublas", "cudnn", "changed species",
                "serialization changed composition", "huggingface", "checkpoint", "download",
            )):
                raise
            shutil.copyfile(source_path, destination)
            record.update(status="fallback_original", converged=False,
                          error=f"{type(error).__name__}: {error}")
        finally:
            signal.alarm(0)
        record.update(duration_seconds=time.monotonic() - started,
                      output_sha256=protocol.sha256(destination))
        write_json(record_path, record)
        print(json.dumps({"worker": index, "ordinal": ordinal,
                          "status": record["status"], "converged": record["converged"],
                          "seconds": record["duration_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--claims", type=Path, help="shared claim directory for dynamic sample assignment")
    args = parser.parse_args()
    if not 0 <= args.worker < args.workers:
        raise ValueError("invalid worker index")
    run_worker(args.output, json.loads(args.inventory.read_text()), args.worker, args.workers, args.claims)


if __name__ == "__main__":
    main()
