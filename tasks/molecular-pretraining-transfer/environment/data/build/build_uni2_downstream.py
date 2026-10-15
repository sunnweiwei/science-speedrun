#!/usr/bin/env python3
"""Rebuild the Uni-Mol2 downstream suite (QM9 v2, low-data QM9, COMPAS-1D) as Uni-Mol LMDBs.

Uni-Mol2 (arXiv 2406.14969) fine-tunes on QM9 (8 properties) and COMPAS-1D (4 properties),
split 8:1:1 by "scaffold similarity-based partitioning", but does not release those splits.
This script rebuilds them from public sources:

* QM9: the 133 885 molecules and DFT-optimised geometries shipped in Uni-Mol's
  `molecular_property_prediction/qm9dft` LMDBs (11 identical copies, as Uni-Mol does), with every
  target joined from `raw_data/qm9.csv` via `ori_index`.
* COMPAS-1D: COMPAS project (gitlab porannegroup/compas, CC BY 4.0), B3LYP/def2-SVP csv + xyz.

Split: Bemis-Murcko scaffolds -> Morgan fingerprints (r=2, 2048) -> Butina clusters at Tanimoto
distance 0.6, tightened until no cluster exceeds 10 % of the molecules -> clusters sorted by size (desc, ties by first member) and filled into train/valid/test
at 80/10/10. Deterministic; the split is written to `split.json` with a sha256.

Output layout (Uni-Mol / Uni-Mol2 `mol_finetune` convention):
  OUT/<task>/{train,valid,test}.lmdb with {atoms, coordinates[11], smi, target, ori_index}
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import pickle
import tarfile
from pathlib import Path

import lmdb
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

RDLogger.DisableLog("rdApp.*")
QM9_TASKS = {  # Uni-Mol2 task name -> csv column(s); units are those of qm9.csv (Hartree, bohr^3, ...)
    "qm9dft_v2": ["homo", "lumo", "gap"],
    "qm9dft_v2_alpha": ["alpha"],
    "qm9dft_v2_cv": ["cv"],
    "qm9dft_v2_mu": ["mu"],
    "qm9dft_v2_r2": ["r2"],
    "qm9dft_v2_zpve": ["zpve"],
}
COMPAS_TASKS = {"compas1d_aea": ["aEA_eV"], "compas1d_aip": ["aIP_eV"],
                "compas1d_disp": ["D3_disp_corr_eV"], "compas1d_dipmom": ["Dipmom_Debye"]}


def scaffold_similarity_split(smiles: list[str], frac=(0.8, 0.1, 0.1), cutoff=0.6) -> list[str]:
    scaffolds: dict[str, list[int]] = {}
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        key = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
        scaffolds.setdefault(key, []).append(i)
    keys = sorted(scaffolds, key=lambda k: (-len(scaffolds[k]), scaffolds[k][0]))
    fps = []
    for k in keys:
        m = Chem.MolFromSmiles(k) if k else None
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m is not None and m.GetNumAtoms() else None)
    # Butina on scaffolds that have a fingerprint; acyclic molecules (empty scaffold) form one group.
    idx = [i for i, f in enumerate(fps) if f is not None]
    dists = []
    for a in range(1, len(idx)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[idx[a]], [fps[idx[b]] for b in range(a)])
        dists.extend(1 - x for x in sims)
    # Adaptive cutoff: tighten until no single cluster exceeds 10 % of the molecules (homogeneous sets
    # such as COMPAS's polybenzenoids otherwise collapse into one cluster). Chosen cutoff is recorded.
    n_mol = len(smiles)
    for cut in (cutoff, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05):
        clusters = [[idx[j] for j in c] for c in Butina.ClusterData(dists, len(idx), cut, isDistData=True)]
        if max(sum(len(scaffolds[keys[k]]) for k in c) for c in clusters) <= 0.1 * n_mol:
            break
    scaffold_similarity_split.cutoff_used = cut
    clusters += [[i] for i, f in enumerate(fps) if f is None]
    groups = [sorted(m for k in c for m in scaffolds[keys[k]]) for c in clusters]
    groups.sort(key=lambda g: (-len(g), g[0]))
    n = len(smiles); split = [""] * n; filled = 0
    for g in groups:
        name = "train" if filled < frac[0] * n else "valid" if filled < (frac[0] + frac[1]) * n else "test"
        for m in g:
            split[m] = name
        filled += len(g)
    return split


def write_lmdb(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), subdir=False, map_size=2 ** 36, lock=False)
    with env.begin(write=True) as txn:
        for i, r in enumerate(rows):
            txn.put(str(i).encode(), pickle.dumps(r, protocol=4))
    env.sync(); env.close()


def qm9_records(unimol_ft: Path) -> list[dict]:
    raw = list(csv.DictReader(open(unimol_ft / "raw_data/qm9.csv")))
    recs = []
    for split in ("train", "valid", "test"):
        env = lmdb.open(str(unimol_ft / f"qm9dft/{split}.lmdb"), subdir=False, readonly=True, lock=False)
        with env.begin() as txn:
            for _, v in txn.cursor():
                d = pickle.loads(v)
                r = raw[d["ori_index"]]
                assert np.allclose(d["target"], (float(r["homo"]), float(r["lumo"]), float(r["gap"])), atol=1e-4)
                recs.append({"atoms": d["atoms"], "coordinates": d["coordinates"], "smi": r["smiles"],
                             "ori_index": d["ori_index"], "props": {c: float(r[c]) for cols in QM9_TASKS.values() for c in cols}})
    recs.sort(key=lambda r: r["ori_index"])
    assert len(recs) == len(raw) == 133885
    return recs


def compas_records(compas: Path) -> list[dict]:
    rows = list(csv.DictReader(open(compas / "compas-1D_b3lyp_def2-svp.csv")))
    geoms = {}
    with tarfile.open(compas / "compas-1D_b3lyp_def2-svp.tar.gz") as tf:
        for m in tf.getmembers():
            if m.name.endswith(".xyz"):
                lines = tf.extractfile(m).read().decode().splitlines()
                n = int(lines[0]); atoms, xyz = [], []
                for line in lines[2:2 + n]:
                    t = line.split(); atoms.append(t[0]); xyz.append([float(x) for x in t[1:4]])
                geoms[Path(m.name).stem] = (atoms, np.asarray(xyz, dtype=np.float32))
    recs = []
    for i, r in enumerate(rows):
        atoms, xyz = geoms[r["molecule"]]
        recs.append({"atoms": atoms, "coordinates": [xyz] * 11, "smi": r["smiles"], "ori_index": i,
                     "props": {c: float(r[c]) for cols in COMPAS_TASKS.values() for c in cols}})
    assert len(recs) == 8678
    return recs


def emit(out: Path, name: str, recs: list[dict], tasks: dict, split: list[str], report: dict) -> None:
    for task, cols in tasks.items():
        for s in ("train", "valid", "test"):
            rows = [{"atoms": r["atoms"], "coordinates": r["coordinates"], "smi": r["smi"], "ori_index": r["ori_index"],
                     "target": tuple(r["props"][c] for c in cols) if len(cols) > 1 else r["props"][cols[0]]}
                    for r, sp in zip(recs, split) if sp == s]
            write_lmdb(out / task / f"{s}.lmdb", rows)
            report.setdefault(task, {})[s] = len(rows)
            if s == "train":  # normalisation statistics (Uni-Mol2's task_metainfo convention), train split only
                t = np.asarray([[r["props"][c] for c in cols] for r, sp in zip(recs, split) if sp == "train"])
                report[task]["train_mean"] = t.mean(0).tolist(); report[task]["train_std"] = t.std(0).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unimol-ft", type=Path, required=True, help="Uni-Mol molecular_property_prediction dir")
    ap.add_argument("--compas", type=Path, required=True, help="dir with COMPAS-1D B3LYP csv + tar.gz")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    report: dict = {}
    qm9 = qm9_records(a.unimol_ft)
    qsplit = scaffold_similarity_split([r["smi"] for r in qm9])
    report["qm9_split_cutoff"] = scaffold_similarity_split.cutoff_used
    emit(a.out, "qm9", qm9, QM9_TASKS, qsplit, report)
    # low-data QM9 (Uni-Mol2 Table 6): train subsets of 50 / 100 / 200 % of the test size, stratified on gap
    rng = np.random.default_rng(0)
    train_idx = np.array([i for i, s in enumerate(qsplit) if s == "train"]); n_test = qsplit.count("test")
    gap = np.array([qm9[i]["props"]["gap"] for i in train_idx]); order = train_idx[np.argsort(gap, kind="stable")]
    for pct in (50, 100, 200):
        k = n_test * pct // 100; strata = np.array_split(order, k)
        pick = {int(rng.choice(s)) for s in strata if len(s)}
        sub = [("train" if i in pick else ("drop" if s == "train" else s)) for i, s in enumerate(qsplit)]
        emit(a.out, "qm9", qm9, {f"qm9dft_v2_train{pct}": QM9_TASKS["qm9dft_v2"]}, sub, report)
    comp = compas_records(a.compas)
    csplit = scaffold_similarity_split([r["smi"] for r in comp])
    report["compas1d_split_cutoff"] = scaffold_similarity_split.cutoff_used
    emit(a.out, "compas", comp, COMPAS_TASKS, csplit, report)
    for name, recs, split in (("qm9", qm9, qsplit), ("compas1d", comp, csplit)):
        blob = json.dumps({str(r["ori_index"]): s for r, s in zip(recs, split)}, sort_keys=True).encode()
        report[f"{name}_split_sha256"] = hashlib.sha256(blob).hexdigest()
        (a.out / f"{name}_split.json").write_bytes(blob)
    (a.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
