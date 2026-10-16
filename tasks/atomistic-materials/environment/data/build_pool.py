"""Assemble the v2 training pool: four public DFT sources as uniform ASE-LMDB shards.

    python build_pool.py --raw RAW_DIR --omat OMAT_SEL_DIR --extra DEV_EXTRA PRIVATE_EXTRA \
        --output POOL_DIR [--workers 64]

RAW_DIR holds mptrj.json, salex-train.tar.gz and matpes-pbe.parquet (fetch_pool.sh); OMAT_SEL_DIR holds the hash-selected OMat24 shards. Every frame whose composition
(integer element ratio) or parent material ID belongs to any evaluation suite is dropped: the #49
discovery/phonon/elasticity split (validation and test), and the thermal-conductivity, MOF,
cleavage and Li-migration suites (validation and test). AM26 amorphous systems are not excluded.

Each frame keeps the source's own DFT labels (energy in eV, forces in eV/A, stress in eV/A^3 with
ASE's positive-tension sign) and data = {sid, source, parent, settings}. Energies of different
sources are NOT on one scale (see the pool README); nothing is corrected or re-referenced here.
Output: POOL_DIR/<source>/*.aselmdb (OMat24: POOL_DIR/omat24/<subset>/) plus metadata.npz (natoms,
in sorted file order) per directory and
POOL_DIR/manifest.json with frame counts, bytes and SHA256 of every file.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import sys
import tarfile
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "evaluation"))
sys.path.insert(0, str(HERE))
from common import digest, write_json  # noqa: E402
from splits import composition_key, load_split  # noqa: E402

VERSION = "atomistic-pool-v1"
SHARD = 100_000
SETTINGS = {
    "mptrj": "mp-pbe(+U), MP settings, uncorrected energy",
    "salex": "alexandria-pbe(+U), MP-compatible settings",
    "mplitrj": "mp-pbe(+U), MP settings (LiTraj recomputation)",
    "matpes": "matpes-pbe, MatPESStaticSet (no +U)",
    "omat24": "omat24-pbe(+U), OMat24 settings",
}


# --------------------------------------------------------------------------------- exclusions
_KEYS = {}


def counts_key(symbols) -> str:
    counts = tuple(sorted(Counter(symbols).items()))
    if counts not in _KEYS:
        _KEYS[counts] = composition_key(dict(counts))
    return _KEYS[counts]


def blocked_inventory(split_dir: Path | None, extra_roots: list[Path]):
    compositions, ids = set(), set()
    rows, _ = load_split(split_dir) if split_dir else load_split()
    for r in rows:
        compositions.add(r["composition"]); ids.add(r["material_id"])
    from ase.io import read
    for root in extra_roots:
        for split_dir_ in sorted(p for p in root.glob("*/*") if p.is_dir()):
            suite = split_dir_.parent.name
            if suite == "amorphous":
                continue
            if suite == "kappa":
                for a in read(split_dir_ / "inputs.extxyz", index=":"):
                    compositions.add(counts_key(a.get_chemical_symbols())); ids.add(a.info["material_id"])
            elif suite == "mof":
                for f in (split_dir_ / "structures").iterdir():
                    compositions.add(counts_key(read(f).get_chemical_symbols()))
            elif suite == "cleavage":
                from pymatgen.core import Structure
                for r in json.load(gzip.open(split_dir_ / "inputs.json.gz")):
                    s = Structure.from_dict(r["bulk"])
                    compositions.add(composition_key(s.composition.get_el_amt_dict())); ids.add(r["material_id"])
            elif suite == "neb":
                for f in (split_dir_ / "paths").glob("*_init.xyz"):
                    a = read(f, index=0)
                    compositions.add(counts_key(a.get_chemical_symbols()))
                    ids.add(f.name.split("_")[0])
    return compositions, ids


# --------------------------------------------------------------------------------- writing
class Writer:
    """Appends frames to numbered shards; returns [(file, natoms list)]."""

    def __init__(self, directory: Path, prefix: str):
        from ase.db import connect  # noqa: F401
        self.directory, self.prefix = directory, prefix
        self.db, self.n, self.index, self.natoms, self.done = None, 0, 0, [], []

    def write(self, atoms, energy, forces, stress, sid, source, parent):
        from ase.calculators.singlepoint import SinglePointCalculator
        from ase.db import connect
        if not (np.isfinite(energy) and np.isfinite(forces).all() and np.isfinite(stress).all()):
            return False
        if self.db is None:
            path = self.directory / f"{self.prefix}-{self.index:04d}.aselmdb"
            self.db, self.path = connect(path), path
        atoms.calc = SinglePointCalculator(atoms, energy=float(energy), forces=np.asarray(forces, float),
                                           stress=np.asarray(stress, float))
        self.db.write(atoms, data={"sid": sid, "source": source, "parent": parent,
                                   "settings": SETTINGS[source.split("/")[0]]})
        self.natoms.append(len(atoms)); self.n += 1
        if self.n == SHARD:
            self.close()
        return True

    def close(self):
        if self.db is not None:
            self.db.close()
            self.done.append((self.path.name, self.natoms))
            self.db, self.n, self.natoms = None, 0, []
            self.index += 1
        return self.done


def _keep(symbols, parent, blocked):
    comps, ids = blocked
    return parent not in ids and counts_key(symbols) not in comps


def filter_aselmdb(job):
    """Copy one released ASE-LMDB shard (sAlex or OMat24) without blocked frames."""
    src, out_dir, prefix, source, blocked_file = job
    from ase.db import connect
    blocked = load_blocked(blocked_file)
    w, kept, seen = Writer(out_dir, prefix), 0, 0
    with connect(src, readonly=True) as db:
        for row in db.select():
            seen += 1
            sid = str(row.data.get("sid", row.id))
            parent = sid.split("_")[0] if source == "salex" else str(row.data.get("parent_id", ""))
            if not _keep(row.symbols, parent, blocked):
                continue
            atoms = row.toatoms()
            kept += w.write(atoms, row.energy, row.forces, row.stress if row.stress.shape == (3, 3)
                            else _voigt(row.stress), sid=f"{source}/{sid}", source=source, parent=parent)
    return {"shard": str(src), "seen": seen, "kept": kept, "files": w.close()}


def _voigt(s):
    s = np.asarray(s, float)
    return np.array([[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]])


_BLOCKED = {}


def load_blocked(path):
    if path not in _BLOCKED:
        d = json.loads(Path(path).read_text())
        _BLOCKED[path] = (set(d["compositions"]), set(d["material_ids"]))
    return _BLOCKED[path]


def build_mptrj(raw: Path, out: Path, blocked_file):
    import ijson
    from records import mptrj_record
    from common import decode_atoms
    blocked = load_blocked(blocked_file)
    w, seen, kept = Writer(out, "mptrj"), 0, 0
    with (raw / "mptrj.json").open("rb") as stream:
        for parent, frames in ijson.kvitems(stream, "", use_float=True):
            for frame, row in frames.items():
                seen += 1
                r = mptrj_record(parent, frame, row)
                atoms = decode_atoms(r["atoms"])
                if not _keep(atoms.get_chemical_symbols(), parent, blocked):
                    continue
                kept += w.write(atoms, r["energy"], np.asarray(r["forces"]), np.asarray(r["stress"]),
                                sid=r["id"], source="mptrj", parent=parent)
    return {"shard": "mptrj.json", "seen": seen, "kept": kept, "files": w.close()}


MATPES_SLICES = 16


def build_matpes(raw: Path, out: Path, blocked_file, part: int):
    import pyarrow.parquet as pq
    from ase import Atoms
    blocked = load_blocked(blocked_file)
    table = pq.read_table(raw / "matpes-pbe.parquet", columns=[
        "energy", "atomic_forces", "cauchy_stress", "cell", "positions", "pbc", "atomic_numbers", "names", "metadata"])
    step = -(-table.num_rows // MATPES_SLICES)
    table = table.slice(part * step, step)
    w, seen, kept = Writer(out, f"matpes-p{part:02d}"), 0, 0
    for r in table.to_pylist():
        seen += 1
        name = r["names"][0].split("__")[-1]
        meta = json.loads(r["metadata"]) if r["metadata"] else {}
        parent = str(meta.get("mp_id") or meta.get("matpes_id") or name.rsplit("_", 1)[0])
        atoms = Atoms(numbers=r["atomic_numbers"], positions=r["positions"], cell=r["cell"], pbc=r["pbc"])
        if not _keep(atoms.get_chemical_symbols(), parent, blocked):
            continue
        # ColabFit's MatPES cauchy_stress is eV/A^3 with positive = compressive; ASE uses positive =
        # tensile. Checked against UMA-S-1.1 on 60 frames: pressure correlation -0.95 before the flip.
        kept += w.write(atoms, r["energy"], np.asarray(r["atomic_forces"]), -np.asarray(r["cauchy_stress"]),
                        sid=f"matpes/{name}", source="matpes", parent=parent)
    return {"shard": f"matpes:{part}", "seen": seen, "kept": kept, "files": w.close()}


def build_mplitrj(raw: Path, out: Path, blocked_file, member: str):
    from ase.io import read
    blocked = load_blocked(blocked_file)
    with zipfile.ZipFile(raw / "MPLiTrj_subsample.zip") as z:
        text = z.read(member).decode()
    tag = Path(member).stem.split("_")[-1]
    w, seen, kept = Writer(out, f"mplitrj-{tag}"), 0, 0
    for i, atoms in enumerate(read(io.StringIO(text), index=":", format="extxyz")):
        seen += 1
        parent = str(atoms.info.get("mp_id", atoms.info.get("material_id", "")))
        if not _keep(atoms.get_chemical_symbols(), parent, blocked):
            continue
        energy, forces = atoms.get_potential_energy(), atoms.get_forces()
        stress = atoms.get_stress(voigt=False)
        atoms.calc = None
        kept += w.write(atoms, energy, forces, stress, sid=f"mplitrj/{tag}/{i}", source="mplitrj", parent=parent)
    return {"shard": member, "seen": seen, "kept": kept, "files": w.close()}


# --------------------------------------------------------------------------------- driver
def finish(source_dir: Path, results):
    files = sorted((f, n) for r in results for f, n in r["files"])
    np.savez(source_dir / "metadata.npz", natoms=np.asarray([x for _, n in files for x in n], dtype=np.int64))
    return {"frames_seen": sum(r["seen"] for r in results), "frames": sum(r["kept"] for r in results),
            "files": len(files)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--omat", type=Path, required=True)
    ap.add_argument("--extra", type=Path, nargs="+", required=True)
    ap.add_argument("--split-dir", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=64)
    # MPLiTrj is supported but not in the release: its forces disagree with UMA-S-1.1 far more than
    # any other source (median per-frame force correlation 0.39 vs 0.995 for MatPES).
    ap.add_argument("--sources", default="mptrj,salex,matpes,omat24")
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    blocked_file = a.output / "blocked.json"
    if not blocked_file.exists():
        comps, ids = blocked_inventory(a.split_dir, a.extra)
        write_json(blocked_file, {"version": VERSION, "compositions": sorted(comps), "material_ids": sorted(ids)})
    blocked_file = str(blocked_file)
    summary = json.loads((a.output / "summary.json").read_text()) if (a.output / "summary.json").exists() else {}
    wanted = a.sources.split(",")
    with ProcessPoolExecutor(a.workers) as pool:
        futures = {}
        for source in wanted:
            if source in summary or source == "omat24":
                continue
            d = a.output / source
            if d.exists():
                shutil.rmtree(d)
            d.mkdir()
            if source == "mptrj":
                futures[source] = [pool.submit(build_mptrj, a.raw, d, blocked_file)]
            elif source == "matpes":
                futures[source] = [pool.submit(build_matpes, a.raw, d, blocked_file, i) for i in range(MATPES_SLICES)]
            elif source == "mplitrj":
                with zipfile.ZipFile(a.raw / "MPLiTrj_subsample.zip") as z:
                    members = [m for m in z.namelist() if m.endswith(".xyz")]
                futures[source] = [pool.submit(build_mplitrj, a.raw, d, blocked_file, m) for m in members]
            elif source == "salex":
                staging = a.output / ".salex-raw"
                if not staging.exists():
                    with tarfile.open(a.raw / "salex-train.tar.gz") as t:
                        t.extractall(staging, filter="data")
                shards = sorted(staging.rglob("*.aselmdb"))
                futures[source] = [pool.submit(filter_aselmdb, (s, d, f"salex-{s.stem}", "salex", blocked_file))
                                   for s in shards]
        if "omat24" in wanted:
            # One directory per OMat24 subset, built as soon as its download is complete (.done).
            for marker in sorted(a.omat.glob("*.done")):
                subset = marker.stem
                key = f"omat24/{subset}"
                if key in summary:
                    continue
                d = a.output / "omat24" / subset
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir(parents=True)
                shards = sorted((a.omat / subset).rglob("*.aselmdb"))
                futures[key] = [pool.submit(filter_aselmdb, (s_, d, s_.stem, key, blocked_file)) for s_ in shards]
        for source, fs in futures.items():
            results = [f.result() for f in fs]
            summary[source] = finish(a.output / source, results)
            write_json(a.output / "summary.json", summary)
            print(json.dumps({source: summary[source]}), flush=True)
    shutil.rmtree(a.output / ".salex-raw", ignore_errors=True)
    files = {str(p.relative_to(a.output)): {"bytes": p.stat().st_size, "sha256": digest(p)}
             for p in sorted(a.output.rglob("*")) if p.suffix in (".aselmdb", ".npz")}
    write_json(a.output / "manifest.json", {
        "version": VERSION, "sources": summary, "settings": SETTINGS,
        "blocked_sha256": digest(Path(blocked_file)), "bytes": sum(v["bytes"] for v in files.values()),
        "files": files})


if __name__ == "__main__":
    main()
