#!/usr/bin/env python3
"""Frontier ADMET tasks for the fast evaluation suite, one single-endpoint regression task per endpoint.

* OpenADMET-ExpansionRx (9 endpoints, `expansionrx/expansion_data_{train,test}.csv`): the official time split.
  Validation is the latest 10 % of each endpoint's training molecules by molecule ID (IDs increase with time;
  every test ID is later than every training ID). Test is the whole official test set.
* ASAP antiviral-ADMET-2025 (5 endpoints, `asap/antiviral-admet-2025-unblinded.csv`): the official Train/Test
  column. IDs are not chronological, so validation is a seeded random 10 % of each endpoint's training molecules.

Targets use the challenges' scoring space: log10(clip(y, 0) + 1) for every endpoint except LogD (official
`clip_and_log_transform`), so test MAE equals the official per-endpoint MAE and RAE = MAE / mean|y - mean(y)|
of the test targets (stored in the manifest as `test_mad`). Training sets stay within the suite's 5,000 cap.

Conformers follow Uni-Mol's smi2coords: explicit hydrogens, 10 ETKDG embeddings (seeds 0-9) with MMFF
optimisation (random-coordinate retry, 2D fallback) plus one 2D layout = 11 coordinate sets; heavy atoms in
SMILES order, then hydrogens. Writes OUT/<task>/{train,valid,test}.lmdb, OUT/task_metainfo.json (train mean/std
for target normalisation, read by ft-uni2/unimol2x) and merges the task entries into OUT/manifest.json.
"""
import csv, hashlib, json, math, pickle, random, sys
from multiprocessing import Pool
from pathlib import Path

import lmdb
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
EXRX = ["LogD", "KSOL", "HLM CLint", "MLM CLint", "Caco-2 Permeability Papp A>B", "Caco-2 Permeability Efflux",
        "MPPB", "MBPB", "MGMB"]
ASAP = ["HLM", "KSOL", "LogD", "MDR1-MDCKII", "MLM"]
TRAIN_CAP = 5000


def task_name(prefix, endpoint):
    return prefix + "_" + "".join(c if c.isalnum() else "_" for c in endpoint.lower()).strip("_").replace("__", "_")


def transform(endpoint, y):
    return y if endpoint.lower() == "logd" else math.log10(max(y, 0.0) + 1.0)


def conformers(smi):
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = []
    for seed in range(10):
        m = Chem.Mol(mol)
        ok = AllChem.EmbedMolecule(m, randomSeed=seed) == 0 or \
            AllChem.EmbedMolecule(m, randomSeed=seed, useRandomCoords=True, maxAttempts=1000) == 0
        if ok:
            try:
                AllChem.MMFFOptimizeMolecule(m)
            except Exception:
                pass
        else:
            AllChem.Compute2DCoords(m)
        coords.append(m.GetConformer().GetPositions().astype(np.float32))
    m = Chem.Mol(mol); AllChem.Compute2DCoords(m)
    coords.append(m.GetConformer().GetPositions().astype(np.float32))
    return smi, atoms, coords


def load(downstream):
    rows = []  # (source, split, id, smiles, {endpoint: value})
    for split in ("train", "test"):
        for r in csv.DictReader(open(downstream / "expansionrx" / f"expansion_data_{split}.csv")):
            rows.append(("exrx", split, r["Molecule Name"], r["SMILES"],
                         {e: float(r[e]) for e in EXRX if r[e] != ""}))
    for r in csv.DictReader(open(downstream / "asap" / "antiviral-admet-2025-unblinded.csv")):
        rows.append(("asap", r["Set"].lower(), r["Molecule Name"], r["CXSMILES"].split(" |")[0],
                     {e: float(r[e]) for e in ASAP if r[e] != ""}))
    return rows


def write(path, entries):
    path.unlink(missing_ok=True)
    env = lmdb.open(str(path), subdir=False, map_size=1 << 36, lock=False)
    digest = hashlib.sha256()
    with env.begin(write=True) as txn:
        for i, e in enumerate(entries):
            v = pickle.dumps(e, protocol=pickle.HIGHEST_PROTOCOL); txn.put(str(i).encode(), v); digest.update(v)
    env.sync(); env.close()
    return digest.hexdigest()


def main(downstream: Path, out: Path):
    rows = load(downstream)
    smiles = sorted({r[3] for r in rows})
    with Pool(64) as pool:
        conf = {s: (a, c) for s, a, c in pool.imap_unordered(conformers, smiles, chunksize=8)}
    print("conformers", len(conf), flush=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"tasks": {}}
    metainfo = {}
    for prefix, endpoints in (("exrx", EXRX), ("asap", ASAP)):
        for ept in endpoints:
            task = task_name(prefix, ept)
            labeled = [r for r in rows if r[0] == prefix and ept in r[4]]
            train = sorted((r for r in labeled if r[1] == "train"), key=lambda r: r[2])
            test = sorted((r for r in labeled if r[1] == "test"), key=lambda r: r[2])
            n_valid = max(1, round(0.1 * len(train)))
            if prefix == "exrx":
                train, valid = train[:-n_valid], train[-n_valid:]
            else:
                idx = set(random.Random(f"{task}/valid/0").sample(range(len(train)), n_valid))
                valid = [r for i, r in enumerate(train) if i in idx]; train = [r for i, r in enumerate(train) if i not in idx]
            assert len(train) <= TRAIN_CAP, (task, len(train))
            (out / task).mkdir(parents=True, exist_ok=True)
            info = {"source": f"{prefix}:{ept}"}
            for split, part in (("train", train), ("valid", valid), ("test", test)):
                entries = [{"atoms": conf[r[3]][0], "coordinates": conf[r[3]][1], "smi": r[3], "id": r[2],
                            "target": transform(ept, r[4][ept])} for r in part]
                info[split] = {"kept": len(entries), "sha256": write(out / task / f"{split}.lmdb", entries)}
                y = np.array([e["target"] for e in entries])
                if split == "train":
                    metainfo[task] = {"mean": float(y.mean()), "std": float(y.std())}
                if split == "test":
                    info["test_mad"] = float(np.abs(y - y.mean()).mean())
            manifest["tasks"][task] = info
            print(task, {k: v["kept"] for k, v in info.items() if isinstance(v, dict)}, flush=True)
    (out / "task_metainfo.json").write_text(json.dumps(metainfo, indent=1) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
