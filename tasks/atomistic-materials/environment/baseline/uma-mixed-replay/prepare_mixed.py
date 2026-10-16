"""Build the mixed-data reference recipe: MP-compatible E/F/S plus OMat24 forces/stress only.

    python prepare_mixed.py --pool /workspace/data/pool --checkpoint /workspace/data/models/uma-s-1p1.pt \
        --output RUN_DIR [--mp-frames 500000] [--omat-frames 500000] [--steps 46875] [--world-size 8]

Same model, optimiser and step budget as the #49 MP fine-tune (uma-mp-finetune: AdamW 4e-4,
cosine, loss weights 20/2/1, seed 41, FP32, no EMA, global batch 32, 46,875 updates), with half of
every batch drawn from OMat24 off-equilibrium frames (rattled and AIMD):
- dataset "omat": MPtrj + sAlex frames (MP-compatible raw energies) -> energy, forces, stress;
- dataset "offeq": OMat24 frames -> forces and stress only, because OMat24 energies use different
  DFT settings. The backbone maps "offeq" to the same "omat" dataset embedding.
Frames are chosen by SHA-256 priority of their IDs, with at most four frames per MP parent; the
labeled development split (2,500 + 2,500 frames) is composition-disjoint from training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from math import gcd
from functools import reduce
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE.parent / "uma-mp-finetune" / "templates"
VERSION = "mixed-replay-v2-reduced-compositions"
REF_FILE = HERE / "element-references.yaml"


def priority(key: str) -> int:
    return int(hashlib.sha256(f"{VERSION}:{key}".encode()).hexdigest()[:16], 16)


def composition_key(symbols):
    counts = Counter(symbols)
    divisor = reduce(gcd, counts.values())
    return "|".join(f"{k}:{v // divisor}" for k, v in sorted(counts.items()))


def scan(path: str):
    """(priority, file, row id, parent, composition) for every frame of one shard."""
    from ase.db import connect
    out = []
    with connect(path, readonly=True) as db:
        for row in db.select():
            comp = composition_key(row.symbols)
            out.append((priority(row.data["sid"]), path, row.id, row.data["parent"], comp))
    return out


def select(pool: Path, sources, n_train, n_dev, workers, parent_cap=4):
    files = sorted(str(f) for s in sources for f in (pool / s).glob("*.aselmdb"))
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for part in ex.map(scan, files, chunksize=1) for r in part]
    rows.sort()
    train, dev, per_parent = [], [], Counter()
    parent_splits = {}
    compositions = {False: set(), True: set()}
    for p, f, i, parent, comp in rows:
        is_dev = int(hashlib.sha256(f"{VERSION}:dev:{comp}".encode()).hexdigest(), 16) % 32 == 0
        if parent and parent in parent_splits and parent_splits[parent] != is_dev:
            continue
        target, quota = (dev, n_dev) if is_dev else (train, n_train)
        if len(target) >= quota or (parent and per_parent[parent] >= parent_cap):
            continue
        per_parent[parent] += 1
        target.append((f, i))
        compositions[is_dev].add(comp)
        if parent:
            parent_splits[parent] = is_dev
        if len(train) >= n_train and len(dev) >= n_dev:
            break
    assert not compositions[False] & compositions[True]
    return train, dev


class _AppendOnlyIDs(list):
    """ase-db-backends 0.11 checks its ID list on every insert (quadratic); keep a set alongside."""

    def __init__(self, values=()):
        super().__init__(values)
        self._members = set(self)

    def __contains__(self, value):
        return value in self._members

    def append(self, value):
        super().append(value)
        self._members.add(value)


def write(frames, destination: Path):
    from ase.db import connect
    destination.mkdir(parents=True)
    by_file, natoms, forces = {}, [], []
    for f, i in frames:
        by_file.setdefault(f, []).append(i)
    with connect(destination / "frames.aselmdb") as out:
        out.ids = _AppendOnlyIDs(out.ids)
        for f in sorted(by_file):
            with connect(f, readonly=True) as db:
                for i in by_file[f]:
                    row = db.get(id=i)
                    atoms = row.toatoms()
                    out.write(atoms, data=row.data)
                    natoms.append(len(atoms))
                    forces.append(np.asarray(row.forces).ravel())
    np.savez(destination / "metadata.npz", natoms=np.asarray(natoms, dtype=np.int64))
    return np.concatenate(forces)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mp-frames", type=int, default=500_000)
    ap.add_argument("--omat-frames", type=int, default=500_000)
    ap.add_argument("--dev-frames", type=int, default=2_500)
    ap.add_argument("--steps", type=int, default=46_875)
    ap.add_argument("--world-size", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--omat-subsets", default="all",
                    help="comma list of pool/omat24 subsets to sample from (default: every subset)")
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    subsets = (sorted(p.name for p in (a.pool / "omat24").iterdir() if p.is_dir()) if a.omat_subsets == "all"
               else a.omat_subsets.split(","))
    # Config names must not contain each other: FAIRChem 2.16 matches eval metrics by substring.
    groups = {"omat": ["mptrj", "salex"], "offeq": [f"omat24/{s}" for s in subsets]}
    quotas = {"omat": a.mp_frames, "offeq": a.omat_frames}
    rms, counts = {}, {}
    for name, sources in groups.items():
        train, dev = select(a.pool, sources, quotas[name], a.dev_frames, a.workers)
        f_train = write(train, a.output / name / "train")
        write(dev, a.output / name / "val")
        rms[name] = float(np.sqrt(np.mean(np.square(f_train))))
        counts[name] = {"train": len(train), "val": len(dev)}
        print(json.dumps({name: counts[name], "force_rms": rms[name]}), flush=True)

    refs = yaml.safe_load(REF_FILE.read_text())["mpa_elem_refs"]
    task = yaml.safe_load((TEMPLATES / "data/uma_conserving_data_task_energy_force_stress.yaml").read_text())
    task.update(dataset_name="omat", elem_refs=refs, normalizer_rmsd=rms["omat"])
    for split, entry in (("train", "train_dataset"), ("val", "val_dataset")):
        task[entry]["splits"][split]["src"] = str((a.output / "omat" / split).resolve())
    second = {k: json.loads(json.dumps(task[k])) for k in ("train_dataset", "val_dataset")}
    for split, entry in (("train", "train_dataset"), ("val", "val_dataset")):
        second[entry]["splits"][split]["src"] = str((a.output / "offeq" / split).resolve())
        for t in second[entry]["transforms"].values():
            t["dataset_name"] = "offeq"
        task[f"offeq_{entry}"] = second[entry]
    for t in task["tasks_list"]:
        t["datasets"] = ["omat"] if t["name"] == "energy" else ["omat", "offeq"]
    (a.output / "data").mkdir()
    (a.output / "data/efs.yaml").write_text(yaml.safe_dump(task, sort_keys=False))

    config = yaml.safe_load((TEMPLATES / "uma_sm_finetune_template.yaml").read_text())
    config["defaults"][0]["data"] = "efs"
    config.update(epochs=None, steps=a.steps, batch_size=a.batch_size, lr=4e-4,
                  evaluate_every_n_steps=1000, checkpoint_every_n_steps=5000)
    config["job"].update(run_dir=str((a.output / "runs").resolve()), run_name="mixed-replay", seed=41)
    config["job"]["scheduler"]["ranks_per_node"] = a.world_size
    for key in ("train_dataset", "val_dataset"):
        config[key]["dataset_configs"] = {"omat": f"${{data.{key}}}", "offeq": f"${{data.offeq_{key}}}"}
    model = config["runner"]["train_eval_unit"]["model"]
    model["checkpoint_location"] = str(a.checkpoint.resolve())
    # OMat24 frames get their own config name (so the energy loss can skip them) but share the
    # pretrained "omat" dataset embedding; see mixed_model.py.
    model["_target_"] = "mixed_model.initialize_mixed_model"
    model["aliases"] = {"offeq": "omat"}
    config["runner"]["train_eval_unit"]["ema_decay"] = None
    for callback in config["runner"]["callbacks"]:
        if "max_saved_checkpoints" in callback:
            callback["max_saved_checkpoints"] = 12
    config.pop("base_model_name")
    config["runner"]["callbacks"].append({"_target_": "learning_curves.LearningCurveAudit",
                                          "output": str((a.output / "gradient-audit.jsonl").resolve())})
    (a.output / "train.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (a.output / "preparation.json").write_text(json.dumps({
        "version": VERSION, "frames": counts, "omat24_subsets": subsets, "force_rms": rms, "steps": a.steps,
        "global_batch_size": a.world_size * a.batch_size, "pool_manifest_sha256": hashlib.sha256(
            (a.pool / "manifest.json").read_bytes()).hexdigest(),
        "initial_checkpoint_sha256": hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()}, indent=1))
    print(a.output / "train.yaml")


if __name__ == "__main__":
    main()
