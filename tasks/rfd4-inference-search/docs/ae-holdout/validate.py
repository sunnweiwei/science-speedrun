"""Validate the published autoencoder on the frozen 640-structure reconstruction panel."""
import argparse
from collections import Counter
from functools import partial
import hashlib
import json
from pathlib import Path
import time

import hydra
import lightning as L
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import rfproteina
from rfproteina.datapipes.collate import collate_batch
from rfproteina.partial_autoencoder.autoencoder import AutoEncoder, VALIDATION_RESIDUE_GROUPS

DATA_REV = "95a7d244d64343de40784ecb9238ef9de4e0db37"
CHECKPOINT_SHA = "24536372f3fc9a5083b6b813baa80630d2287a5d61fbf2f126c9735802e1d21a"
HOLDOUT_SHA = "829866da16bf1a72267d64e9719edae6e56914d4b51d3eff4b88faf6926f507c"
PANELS = ("dna", "glycan", "protein_ligand", "ptm", "rna")


def source_files(directory):
    paths = sorted(directory.glob("*/*.cif.zst"))
    if Counter(p.parent.name for p in paths) != {group: 128 for group in PANELS}:
        raise ValueError("Expected the five published panels of 128 structures each")
    digest = hashlib.sha256()
    for p in paths:
        value = hashlib.sha256(p.read_bytes()).hexdigest()
        digest.update(f"{p.relative_to(directory).as_posix()}\t{value}\n".encode())
    if digest.hexdigest() != HOLDOUT_SHA:
        raise ValueError("Fixed validation set differs from the public release")
    return paths


class SeenExamples(L.Callback):
    def __init__(self):
        super().__init__()
        self.ids = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.ids.extend(batch["example_id"])


def run(args):
    paths = source_files(args.data.resolve())
    if args.smoke:
        paths = [next(p for p in paths if p.parent.name == group) for group in PANELS]
    with args.checkpoint.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != CHECKPOINT_SHA:
            raise ValueError("Unexpected AE checkpoint")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    L.seed_everything(43, workers=True)
    recipe = Path(rfproteina.__file__).parent / "configs/datasets/val/ae/pdb_temporal_holdout.yaml"
    config = OmegaConf.load(recipe).dataset
    config.dir_path_or_file_paths = [str(p) for p in paths]
    dataset = hydra.utils.instantiate(config)
    collate = partial(collate_batch, schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0 if args.smoke else 2,
                        multiprocessing_context=None if args.smoke else "spawn", collate_fn=collate)
    model = AutoEncoder.load_from_checkpoint(args.checkpoint, map_location="cpu", weights_only=False, strict=True,
                                             store_dir=str(output / "ae"))
    audit = SeenExamples()
    trainer = L.Trainer(accelerator="cpu" if args.smoke else "gpu", devices=1,
                        precision="32-true" if args.smoke else "bf16-mixed", logger=False,
                        enable_checkpointing=False, enable_model_summary=False, enable_progress_bar=False,
                        default_root_dir=output, callbacks=[audit])
    start = time.monotonic()
    metrics, = trainer.validate(model, dataloaders=loader, verbose=False)
    ids = {p.name.removesuffix(".cif.zst") for p in paths}
    if len(audit.ids) != len(paths) or set(audit.ids) != ids:
        raise RuntimeError(f"Expected exactly {len(paths)} unique structure IDs, observed {len(set(audit.ids))}")
    values = {}
    for group, (summed_rmsd_nm, residues) in zip(VALIDATION_RESIDUE_GROUPS, model._val_rmsd_totals[0].tolist(), strict=True):
        if residues == 0:
            continue
        key = f"validation_rmsd/pdb_temporal_holdout/{group}_ang"
        reported = float(metrics[key])
        if abs(reported - 10 * summed_rmsd_nm / residues) > 1e-7:
            raise RuntimeError(f"Inconsistent reported reconstruction RMSD for {group}")
        values[group] = {"supervised_residues": int(residues), "mean_per_residue_rmsd_A": reported}
    if len(values) != len(VALIDATION_RESIDUE_GROUPS):
        raise RuntimeError("Missing residue groups")
    report = {"examples": len(audit.ids), "input_classes": dict(Counter(p.parent.name for p in paths)),
              "checkpoint_sha256": CHECKPOINT_SHA, "dataset_revision": DATA_REV, "holdout_digest": HOLDOUT_SHA,
              "seed": 43, "precision": trainer.precision, "batch_size": 1,
              "device": "CPU" if args.smoke else torch.cuda.get_device_name(0),
              "elapsed_seconds": time.monotonic() - start, "residue_groups": values}
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="The official ae_pdb_temporal_holdout_2026_10_04 directory")
    parser.add_argument("checkpoint", type=Path, help="Pinned released native AE EMA checkpoint")
    parser.add_argument("output", type=Path, help="New work and output directory")
    parser.add_argument("--smoke", action="store_true", help="One example per panel on CPU")
    run(parser.parse_args())
