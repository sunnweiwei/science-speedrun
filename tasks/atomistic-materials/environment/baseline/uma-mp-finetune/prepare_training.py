"""Build FAIRChem 2.16 E/F/S training inputs from audited MP-compatible labels."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))
from common import decode_atoms, digest, read_rows, write_json

TEMPLATES = Path(__file__).with_name("templates")


class _AppendOnlyIDs(list):
    """Index membership during bulk insertion into a new ASE LMDB database.

    ase-db-backends 0.11 checks its ID list on every insertion. A set alongside
    the ordered list avoids quadratic conversion time for the 500k dataset.
    This is used only while appending; normal database reads reconstruct IDs.
    """

    def __init__(self, values=()):
        super().__init__(values)
        self._members = set(self)

    def __contains__(self, value):
        return value in self._members

    def append(self, value):
        super().append(value)
        self._members.add(value)


def prepare(
    data: Path,
    output: Path,
    checkpoint: Path,
    steps=500,
    *,
    epochs=None,
    world_size=1,
    batch_size=2,
    evaluate_every=100,
):
    if (steps is None) == (epochs is None):
        raise ValueError("Specify exactly one of steps and epochs")
    if world_size < 1 or batch_size < 1:
        raise ValueError("Positive world size and batch size required")
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((data / "manifest.json").read_text())
    if (
        manifest["energy_convention"] != "mp-pbe-raw-total"
        or manifest["corrections_applied"]
    ):
        raise ValueError("Wrong training energy convention")
    all_forces = []
    train_count = 0
    for split in ["train", "val"]:
        source = data / f"{split}.jsonl"
        if digest(source) != manifest["files"][source.name]:
            raise ValueError("Training data checksum mismatch")
        rows = read_rows(source)
        if split == "train":
            train_count = len(rows)
        destination = output / split
        destination.mkdir(parents=True)
        sizes = []
        with connect(destination / "frames.aselmdb") as db:
            db.ids = _AppendOnlyIDs(db.ids)
            for row in rows:
                atoms = decode_atoms(row["atoms"])
                labels = {k: np.asarray(row[k]) for k in ["energy", "forces", "stress"]}
                if not all(np.isfinite(v).all() for v in labels.values()):
                    raise ValueError("Nonfinite label")
                atoms.calc = SinglePointCalculator(
                    atoms,
                    energy=float(labels["energy"]),
                    forces=labels["forces"],
                    stress=labels["stress"],
                )
                db.write(
                    atoms,
                    data={
                        "sid": row["id"],
                        "parent": row["parent"],
                        "source": row["source"],
                    },
                )
                sizes.append(len(atoms))
                if split == "train":
                    all_forces.extend(row["forces"])
        np.savez(destination / "metadata.npz", natoms=np.asarray(sizes))
    rms = float(np.sqrt(np.mean(np.square(all_forces))))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("Invalid force RMS")
    ref_file = data / "element-references.yaml"
    if digest(ref_file) != manifest["element_reference_resource"]["sha256"]:
        raise ValueError("Element-reference checksum mismatch")
    refs = yaml.safe_load(ref_file.read_text())["mpa_elem_refs"]
    task_config = yaml.safe_load(
        (
            TEMPLATES / "data/uma_conserving_data_task_energy_force_stress.yaml"
        ).read_text()
    )
    task_config.update(dataset_name="omat", elem_refs=refs, normalizer_rmsd=rms)
    for split, entry in [("train", "train_dataset"), ("val", "val_dataset")]:
        task_config[entry]["splits"][split]["src"] = str((output / split).resolve())
    (output / "data").mkdir()
    (output / "data/efs.yaml").write_text(yaml.safe_dump(task_config, sort_keys=False))
    config = yaml.safe_load((TEMPLATES / "uma_sm_finetune_template.yaml").read_text())
    config["defaults"][0]["data"] = "efs"
    epoch_steps = (train_count + world_size * batch_size - 1) // (
        world_size * batch_size
    )
    expected_steps = steps if steps is not None else epochs * epoch_steps
    config.update(
        epochs=epochs,
        steps=steps,
        batch_size=batch_size,
        lr=4e-4,
        evaluate_every_n_steps=evaluate_every,
        checkpoint_every_n_steps=5000 if epochs is not None else expected_steps + 1,
    )
    config["job"].update(
        run_dir=str((output / "runs").resolve()),
        run_name="adaptation-standard" if epochs is not None else "adaptation-smoke",
        seed=41,
    )
    config["job"]["scheduler"]["ranks_per_node"] = world_size
    for key in ["train_dataset", "val_dataset"]:
        config[key]["dataset_configs"]["omat"] = config[key]["dataset_configs"].pop(
            "DATASET_NAME"
        )
    config["runner"]["train_eval_unit"]["model"]["checkpoint_location"] = str(
        checkpoint.resolve()
    )
    if epochs is not None:
        config["runner"]["train_eval_unit"]["ema_decay"] = None
        for callback in config["runner"]["callbacks"]:
            if "max_saved_checkpoints" in callback:
                callback["max_saved_checkpoints"] = 12
    config.pop("base_model_name")  # local path replaces the template interpolation
    # The generic published recipe deliberately creates a new conservative EFS head.
    # Keep this visible, instead of silently dropping pretrained weights.
    config["runner"]["callbacks"].append(
        {
            "_target_": "learning_curves.LearningCurveAudit"
            if epochs is not None
            else "timing_training.DistributedTimingAudit"
            if world_size > 1
            else "audit_training.FiniteGradientAudit",
            "output": str((output / "gradient-audit.jsonl").resolve()),
        }
    )
    (output / "train.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(
        output / "preparation.json",
        dict(
            force_rms=rms,
            steps=steps,
            epochs=epochs,
            expected_steps=expected_steps,
            batch_size=batch_size,
            world_size=world_size,
            global_batch_size=world_size * batch_size,
            train_frames=train_count,
            initialization="pretrained backbone; newly initialized conservative EFS head",
            elemental_references="published MPA fitted references",
            training_manifest_sha256=digest(data / "manifest.json"),
            initial_checkpoint_sha256=digest(checkpoint),
        ),
    )
    return output / "train.yaml"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["data", "output", "checkpoint"]:
        p.add_argument("--" + name, type=Path, required=True)
    duration = p.add_mutually_exclusive_group()
    duration.add_argument("--steps", type=int)
    duration.add_argument("--epochs", type=int)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--evaluate-every", type=int, default=100)
    a = p.parse_args()
    steps = a.steps if a.steps is not None else (500 if a.epochs is None else None)
    print(
        prepare(
            a.data,
            a.output,
            a.checkpoint,
            steps,
            epochs=a.epochs,
            world_size=a.world_size,
            batch_size=a.batch_size,
            evaluate_every=a.evaluate_every,
        )
    )
