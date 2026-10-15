#!/usr/bin/env python3
"""Fast evaluation suite: fixed seeded subsets of the rebuilt Uni-Mol2 downstream splits.

For each task, at most 5,000 training, 1,000 validation and 3,000 test molecules are drawn once with seed 0
(same splits, so no leakage change). With a third argument `full-test`, the whole test split is kept (v2 suite);
the train and valid subsets are the same either way. Entries are copied byte-for-byte and re-keyed 0..n-1, the key scheme of
the official LMDBDataset. Output: downstream/uni2-fast/<task>/{train,valid,test}.lmdb and manifest.json.
"""
import hashlib, json, random, sys
from pathlib import Path
import lmdb

CAPS = {"train": 5000, "valid": 1000, "test": 3000}
TASKS = ["qm9dft_v2", "qm9dft_v2_alpha", "qm9dft_v2_cv", "qm9dft_v2_mu", "qm9dft_v2_r2", "qm9dft_v2_zpve",
         "compas1d_aea", "compas1d_aip", "compas1d_disp", "compas1d_dipmom"]

def main(src: Path, dst: Path, full_test: bool = False):
    if full_test:
        CAPS["test"] = None
    manifest = {"seed": 0, "caps": CAPS, "tasks": {}}
    for task in TASKS:
        (dst / task).mkdir(parents=True, exist_ok=True)
        manifest["tasks"][task] = {}
        for split, cap in CAPS.items():
            env = lmdb.open(str(src / task / f"{split}.lmdb"), subdir=False, readonly=True, lock=False, readahead=False)
            with env.begin() as txn:
                keys = list(txn.cursor().iternext(values=False))
                chosen = sorted(random.Random(f"{task}/{split}/0").sample(range(len(keys)), len(keys) if cap is None else min(cap, len(keys))))
                values = [txn.get(keys[i]) for i in chosen]
            out = dst / task / f"{split}.lmdb"
            out.unlink(missing_ok=True)
            w = lmdb.open(str(out), subdir=False, map_size=1 << 36, lock=False)
            digest = hashlib.sha256()
            with w.begin(write=True) as txn:
                for i, v in enumerate(values):
                    txn.put(str(i).encode(), v); digest.update(v)
            w.sync(); w.close()
            manifest["tasks"][task][split] = {"source": len(keys), "kept": len(values), "sha256": digest.hexdigest()}
            print(task, split, len(keys), "->", len(values), flush=True)
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:] == ["full-test"])
