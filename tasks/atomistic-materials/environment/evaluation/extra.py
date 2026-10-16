"""Run the v2 off-equilibrium suites with a submitted calculator (one worker per GPU).

    python extra.py run --suite kappa|mof|amorphous|cleavage|neb --inputs SPLIT_DIR \
        --artifact SUBMISSION_DIR|MODEL.pt [--factory submission|uma] --output OUT_DIR [--gpus 0,1]

SPLIT_DIR is a `val` (development) or `test` (verifier) directory written by
environment/data/extra_suites.py. Inputs never contain reference values. Each item is written to
OUT_DIR/items/<id>.json; OUT_DIR/run.json binds the inputs manifest, model checksum and completeness.
Protocols:
- kappa: Matbench Discovery phonondb-v1 (FIRE + FrechetCellFilter with FixSymmetry, fmax 1e-4, 300
  steps; FC2/FC3 with 0.01 A displacements; imaginary-mode gate; Wigner RTA at 300 K, phono3py 3.30).
- mof: MOFSimBench (LBFGS + FrechetCellFilter fmax 1e-3, 1000 steps; 11 volumes in +/-4 %; FIRE fmax
  1e-3 at fixed cell; Birch-Murnaghan); the calculator is summed with D3(BJ, PBE, 40 Bohr).
- amorphous: single-point energy per atom and forces on AM26 structures.
- cleavage: single points, E = (E_slab - n_slab / n_bulk * E_bulk) / (2 A).
- neb: relax end points, climbing-image NEB (improved tangent, k = 5, FIRE, fmax 0.1, 300 steps),
  barrier = max(E) - min(E).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import digest, with_oom_retry  # noqa: E402

PROTOCOL = "atomistic-extra-suites-v1"


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            d.update(chunk)
    return d.hexdigest()


# ------------------------------------------------------------------------------------- items
def items(suite: str, root: Path):
    from ase.io import read
    if suite == "kappa":
        return [(a.info["material_id"], a) for a in read(root / "inputs.extxyz", index=":", format="extxyz")]
    if suite == "mof":
        files = json.loads((root / "manifest.json").read_text())["files"]
        return [(k, root / "structures" / v) for k, v in sorted(files.items())]
    if suite == "amorphous":
        return [(a.info["id"], a) for a in read(root / "inputs.extxyz", index=":", format="extxyz")]
    if suite == "cleavage":
        return [(r["id"], r) for r in json.load(gzip.open(root / "inputs.json.gz"))]
    if suite == "neb":
        ids = json.loads((root / "manifest.json").read_text())["ids"]
        return [(i, root / "paths" / f"{i}_init.xyz") for i in ids]
    raise ValueError(suite)


# ------------------------------------------------------------------------------------- physics
def run_kappa(atoms, calc):
    from ase.constraints import FixSymmetry
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from moyopy import MoyoDataset
    from moyopy.interface import MoyoAdapter
    from phono3py.api_phono3py import Phono3py
    from phonopy.structure.atoms import PhonopyAtoms

    def spg(a):
        return MoyoDataset(MoyoAdapter.from_atoms(a), symprec=1e-5).number

    info = dict(atoms.info)
    rec = {"init_spg_num": int(spg(atoms))}
    a = atoms.copy(); a.calc = calc
    a.set_constraint(FixSymmetry(a, symprec=0.01))
    opt = FIRE(FrechetCellFilter(a, mask=[True, True, True, False, False, False]), logfile=None)
    opt.run(fmax=1e-4, steps=300)
    a.calc = None; a.constraints = None
    rec.update(final_spg_num=int(spg(a)), relax_steps=int(opt.get_number_of_steps()))
    ph3 = Phono3py(PhonopyAtoms(a.get_chemical_symbols(), cell=a.cell[:], positions=a.positions),
                   supercell_matrix=np.asarray(info["fc3_supercell"]),
                   phonon_supercell_matrix=np.asarray(info["fc2_supercell"]), primitive_matrix="auto", symprec=1e-5)
    ph3.mesh_numbers = tuple(info["q_point_mesh"])
    ph3.generate_displacements(distance=0.01, is_plusminus="auto")
    ph3.phonon_forces = forces(ph3.phonon_supercells_with_displacements, calc)
    ph3.produce_fc2(symmetrize_fc2=True)
    ph3.init_phph_interaction(symmetrize_fc3q=False)
    ph3.run_phonon_solver()
    freqs, _, _ = ph3.get_phonon_data()
    f = np.asarray(freqs, dtype=float)
    imag = bool(np.all(np.isnan(f)) or np.any(f[0, 3:] < 0) or np.any(f[0, :3] < -0.01) or np.any(f[1:] < 0))
    rec["has_imag_ph_modes"] = imag
    if imag or rec["final_spg_num"] != rec["init_spg_num"]:
        rec["conductivity_skipped"] = True
        return rec
    sc = ph3.supercells_with_displacements
    present = forces([s for s in sc if s is not None], calc)
    full, j = [], 0
    for s in sc:
        if s is None:
            full.append(np.zeros((len(ph3.supercell), 3)))
        else:
            full.append(present[j]); j += 1
    ph3.forces = np.asarray(full)
    ph3.produce_fc3(symmetrize_fc3r=True)
    ph3.init_phph_interaction(symmetrize_fc3q=False)
    ph3.run_thermal_conductivity(temperatures=[300.0], is_isotope=True, transport_type="MS-SMM19", boundary_mfp=1e6)
    k = ph3.thermal_conductivity
    extra, cv = k.get_extra_kappa_output(), np.asarray(k.mode_heat_capacities)
    mp_, mc = np.asarray(extra["mode_kappa_P_RTA"][0]), np.asarray(extra["mode_kappa_C"][0])
    with np.errstate(divide="ignore", invalid="ignore"):
        c = 2 * ((mc * cv[:, :, :, None, None]) / (cv[:, :, :, None, None] + cv[:, :, None, :, None])).sum(axis=2)
    c[np.isnan(c)] = 0
    rec.update(kappa_tot_rta=np.asarray(extra["kappa_TOT_RTA"][0]).tolist(),
               mode_weights=np.asarray(k.grid_weights).tolist(), mode_kappa_tot_rta=(c + mp_).tolist())
    return rec


def forces(supercells, calc):
    from ase import Atoms
    out = []
    for s in supercells:
        a = Atoms(s.symbols, cell=s.cell, positions=s.positions, pbc=True); a.calc = calc
        out.append(np.asarray(a.get_forces(), dtype=float))
    return np.asarray(out)


def run_mof(path, calc):
    from ase import units
    from ase.eos import EquationOfState
    from ase.filters import FrechetCellFilter
    from ase.io import read
    from ase.optimize import FIRE, LBFGS
    atoms = read(path); atoms.calc = calc
    LBFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=1e-3, steps=1000)
    relaxed = atoms.copy(); vols, ens = [], []
    for s in np.linspace(0.96, 1.04, 11) ** (1 / 3):
        a = relaxed.copy(); a.calc = calc; a.set_cell(relaxed.get_cell() * s, scale_atoms=True)
        FIRE(a, logfile=None).run(fmax=1e-3, steps=1000)
        vols.append(float(a.get_volume())); ens.append(float(a.get_potential_energy()))
    v0, e0, B = EquationOfState(vols, ens, eos="birchmurnaghan").fit()
    return {"B_GPa": float(B / units.GPa), "v0": float(v0), "atoms": len(relaxed), "volumes": vols, "energies": ens}


def run_amorphous(atoms, calc):
    a = atoms.copy(); a.calc = calc
    return {"energy_per_atom": float(a.get_potential_energy() / len(a)), "forces": np.asarray(a.get_forces()).round(6).tolist()}


def run_cleavage(row, calc):
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    slab = AseAtomsAdaptor.get_atoms(Structure.from_dict(row["slab"]))
    bulk = AseAtomsAdaptor.get_atoms(Structure.from_dict(row["bulk"]))
    slab.calc = calc; e_slab = float(slab.get_potential_energy())
    bulk.calc = calc; e_bulk = float(bulk.get_potential_energy())
    return {"cleavage_energy": (e_slab - len(slab) / len(bulk) * e_bulk) / (2 * row["area"])}


def run_neb(path, calc, steps=300, fmax=0.1):
    from ase.io import read
    from ase.mep import NEB
    from ase.optimize import FIRE
    images = [im.copy() for im in read(path, index=":")]
    for im in images:
        im.calc = calc
    FIRE(images[0], logfile=None).run(fmax=0.05, steps=500)
    FIRE(images[-1], logfile=None).run(fmax=fmax, steps=steps)
    neb = NEB(images, k=5.0, climb=True, method="improvedtangent", allow_shared_calculator=True)
    opt = FIRE(neb, logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=steps))
    e = np.array([im.get_potential_energy() for im in images])
    return {"barrier_eV": float(e.max() - e.min()), "converged": converged, "neb_steps": int(opt.get_number_of_steps())}


RUNNERS = {"kappa": run_kappa, "mof": run_mof, "amorphous": run_amorphous, "cleavage": run_cleavage, "neb": run_neb}


def make_calculator(factory, artifact, suite):
    sys.path.insert(0, str(HERE))
    import predict
    calc = predict.load_calculator(factory, artifact, "cuda")
    if suite == "mof":
        from ase import units
        from ase.calculators.mixing import SumCalculator
        from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
        calc = SumCalculator([calc, TorchDFTD3Calculator(device="cuda", damping="bj", xc="pbe", cutoff=40.0 * units.Bohr)])
    return calc


def cost(suite, item):
    """Relative cost used to start the most expensive items first (MOF and NEB have long tails)."""
    if suite in ("mof", "neb"):
        from ase.io import read
        images = read(item, index=":")
        return len(images[0]) * len(images)
    if suite == "kappa":
        return len(item)
    return 0


def item_file(root: Path, key: str) -> Path:
    return root / "items" / (key.replace("|", "__").replace("/", "_") + ".json")


def _worker(gpu, queue, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    calc = make_calculator(args.factory, args.artifact, args.suite)
    todo = dict(items(args.suite, args.inputs))
    while (key := queue.get()) is not None:
        target = item_file(args.output, key)
        if target.exists():
            continue
        item = todo[key]
        t = time.monotonic()
        try:
            def calculate():
                if hasattr(calc, "reset"):
                    calc.reset()
                return RUNNERS[args.suite](item, calc)
            rec = {"id": key, "status": "ok", **with_oom_retry(calculate)}
        except Exception as exc:  # failures are scored as failures
            rec = {"id": key, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()[-1500:]}
        rec["seconds"] = time.monotonic() - t
        tmp = target.with_name(f".{target.name}.tmp")
        tmp.write_text(json.dumps(rec))
        tmp.rename(target)
        print(json.dumps({"suite": args.suite, "id": key, "status": rec["status"], "s": round(rec["seconds"], 1)}), flush=True)


def run(args):
    (args.output / "items").mkdir(parents=True, exist_ok=True)
    gpus = [g for g in args.gpus.split(",") if g != ""]
    per_gpu = max(1, args.workers_per_gpu)
    slots = [g for g in gpus for _ in range(per_gpu)]
    ctx = mp.get_context("spawn")
    keys = [k for k, _ in sorted(items(args.suite, args.inputs), key=lambda kv: -cost(args.suite, kv[1]))]
    queue = ctx.Queue()
    for k in keys:
        queue.put(k)
    for _ in slots:
        queue.put(None)
    procs = [ctx.Process(target=_worker, args=(g, queue, args)) for g in slots]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    queue.cancel_join_thread()  # items left by crashed workers must not block exit
    done = {k: json.loads(item_file(args.output, k).read_text()) for k in keys if item_file(args.output, k).exists()}
    run_info = {"protocol": PROTOCOL, "suite": args.suite, "inputs_manifest_sha256": sha256(args.inputs / "manifest.json"),
                "checkpoint_sha256": digest(args.artifact) if args.artifact.exists() else None,
                "factory": args.factory, "items": len(keys), "completed": len(done),
                "failed": sum(r["status"] != "ok" for r in done.values()), "complete": len(done) == len(keys)}
    (args.output / "run.json").write_text(json.dumps(run_info, indent=1))
    print(json.dumps(run_info))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--suite", choices=sorted(RUNNERS), required=True)
    r.add_argument("--inputs", type=Path, required=True)
    r.add_argument("--artifact", type=Path, required=True)
    r.add_argument("--factory", default="uma")
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    r.add_argument("--workers-per-gpu", type=int, default=1)
    a = ap.parse_args()
    run(a)


if __name__ == "__main__":
    main()
