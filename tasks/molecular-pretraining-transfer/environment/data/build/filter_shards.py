#!/usr/bin/env python3
"""Remove every evaluation molecule from the Uni-Mol2 pretraining shards and write clean shards.

A pretraining molecule is dropped when the connectivity layer of its InChIKey (first 14 characters,
i.e. ignoring stereochemistry, isotopes and protonation-layer details) equals that of any molecule in
the evaluation sets (QM9, COMPAS-1D, OpenADMET-ExpansionRx, ASAP antiviral-ADMET-2025). Molecules whose
SMILES RDKit cannot parse are dropped too and counted separately.

Input  : SRC/train_part_{k}.lmdb (Uni-Mol2 format: 4-byte big-endian keys, pickled {smi, atoms, coordinates})
Output : OUT/train_part_{i}.lmdb (same format, contiguous keys) for i = 0..K-1, and OUT/valid_part_0.lmdb
         holding the last --valid-size kept molecules of the last shard (excluded from training).
Report : OUT/filter_report.json (per-shard counts and the evaluation key set sha256; the removed molecules are
         evaluation molecules, so they are counted but not listed).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from multiprocessing import Pool
from pathlib import Path

import lmdb
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def key14(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    k = Chem.MolToInchiKey(mol)
    return k[:14] if k else None


def eval_keys(downstream: Path) -> tuple[set[str], dict]:
    sources = {
        "qm9": [r["smiles"] for r in csv.DictReader(open(downstream / "qm9.csv"))],
        "compas1d": [r["smiles"] for r in csv.DictReader(open(downstream / "compas/compas-1D_b3lyp_def2-svp.csv"))],
        "expansionrx": [r["SMILES"] for f in ("expansion_data_train.csv", "expansion_data_test.csv", "expansion_data_raw.csv")
                        for r in csv.DictReader(open(downstream / "expansionrx" / f))],
        "asap_admet_2025": [r["CXSMILES"].split(" |")[0] for r in csv.DictReader(open(downstream / "asap/antiviral-admet-2025-unblinded.csv"))],
    }
    keys, stats = set(), {}
    for name, smis in sources.items():
        ks = {k for k in map(key14, smis) if k}
        stats[name] = {"molecules": len(smis), "unique_keys": len(ks)}
        keys |= ks
    return keys, stats


EVAL: set[str] = set()


def _init(keys):
    global EVAL
    EVAL = keys


def _scan(args):
    path, lo, hi = args
    env = lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    drop_eval, drop_parse, dropped = [], [], []
    with env.begin() as txn:
        for i in range(lo, hi):
            smi = pickle.loads(txn.get(i.to_bytes(4, "big")))["smi"]
            k = key14(smi)
            if k is None:
                drop_parse.append(i)
            elif k in EVAL:
                drop_eval.append(i); dropped.append(smi)
    env.close()
    return drop_eval, drop_parse, dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--downstream", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parts", type=int, nargs="+", required=True)
    ap.add_argument("--valid-size", type=int, default=100_000)
    ap.add_argument("--procs", type=int, default=128)
    a = ap.parse_args()
    keys, stats = eval_keys(a.downstream)
    report = {"eval_sources": stats, "eval_keys": len(keys),
              "eval_keys_sha256": hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest(), "shards": {}}
    a.out.mkdir(parents=True, exist_ok=True)
    with Pool(a.procs, initializer=_init, initargs=(keys,)) as pool:
        for out_i, part in enumerate(a.parts):
            src = str(a.src / f"train_part_{part}.lmdb")
            env = lmdb.open(src, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
            n = env.stat()["entries"]
            step = 20_000
            res = pool.map(_scan, [(src, lo, min(lo + step, n)) for lo in range(0, n, step)], chunksize=1)
            drop = {i for r in res for i in r[0] + r[1]}
            kept = [i for i in range(n) if i not in drop]
            valid = kept[-a.valid_size:] if out_i == len(a.parts) - 1 else []
            train = kept[:len(kept) - len(valid)]
            for name, idx in ((f"train_part_{out_i}", train), ("valid_part_0", valid)):
                if not idx:
                    continue
                dst = lmdb.open(str(a.out / f"{name}.lmdb"), subdir=False, map_size=2 ** 40, lock=False, writemap=True)
                with env.begin() as rt:
                    for start in range(0, len(idx), 200_000):
                        with dst.begin(write=True) as wt:
                            for j, i in enumerate(idx[start:start + 200_000], start):
                                wt.put(j.to_bytes(4, "big"), rt.get(i.to_bytes(4, "big")))
                dst.sync(); dst.close()
            env.close()
            report["shards"][f"source_part_{part}"] = {
                "out": f"train_part_{out_i}", "entries": n, "dropped_eval": sum(len(r[0]) for r in res),
                "dropped_unparsable": sum(len(r[1]) for r in res), "train": len(train), "valid": len(valid)}
            print(json.dumps(report["shards"][f"source_part_{part}"]), flush=True)
            (a.out / "filter_report.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
