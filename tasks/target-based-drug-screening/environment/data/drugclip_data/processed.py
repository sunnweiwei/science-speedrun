from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import iter_lmdb, sha256_file, write_json, write_lmdb


def _array(data_root: Path, relative: str) -> dict[str, np.ndarray]:
    with np.load(data_root / relative, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def build_processed_data(data_root: Path) -> None:
    manifest_path = data_root / "processed" / "manifest.json"
    input_paths = [data_root / "canonical" / "biolip2.parquet",
                   data_root / "canonical" / "profsadb.parquet",
                   data_root / "splits" / "train.txt", data_root / "splits" / "dev.txt",
                   data_root / "splits" / "profsa_pretrain_500k.txt"]
    input_hashes = {str(path.relative_to(data_root)): sha256_file(path) for path in input_paths}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("inputs") == input_hashes and all(
            (data_root / item["path"]).exists()
            and sha256_file(data_root / item["path"]) == item["sha256"]
            for item in existing.get("datasets", {}).values()
        ):
            return
    biolip = pd.read_parquet(data_root / "canonical" / "biolip2.parquet").set_index("complex_id")
    profsa = pd.read_parquet(data_root / "canonical" / "profsadb.parquet").set_index("sample_id")

    def bio_records(split: str):
        selected = biolip[biolip.split.eq(split)].sort_index()
        for complex_id, row in selected.iterrows():
            arrays = _array(data_root, row.array_file)
            yield {
                "complex_id": complex_id, "smi": row.canonical_smiles,
                "pocket": complex_id, "atoms": arrays["ligand_atoms"].tolist(),
                "coordinates": [item.astype(np.float32) for item in arrays["ligand_coordinates"]],
                "pocket_atoms": arrays["pocket_atoms"].tolist(),
                "pocket_coordinates": arrays["pocket_coordinates"].astype(np.float32),
                "label": 1.0,
            }

    selected_profsa = [line.strip() for line in
                       (data_root / "splits" / "profsa_pretrain_500k.txt").read_text().splitlines()
                       if line.strip()]

    def profsa_records():
        for sample_id in selected_profsa:
            row = profsa.loc[sample_id]
            arrays = _array(data_root, row.array_file)
            ligand_atoms = arrays["ligand_atoms"].tolist()
            ligand_coordinates = arrays["ligand_coordinates"].astype(np.float32)
            yield {
                "sample_id": sample_id, "smi": sample_id, "pocket": sample_id,
                "lig_atoms_real": ligand_atoms, "lig_coord_real": ligand_coordinates,
                "pocket_atoms": arrays["pocket_atoms"].tolist(),
                "pocket_coordinates": arrays["pocket_coordinates"].astype(np.float32),
                # Aliases let the standard DrugCLIP loader consume the same view.
                "atoms": ligand_atoms, "coordinates": [ligand_coordinates], "label": 1.0,
            }

    outputs = [
        ("profsa_pretrain_500k", profsa_records()),
        ("biolip_train", bio_records("train")),
        ("biolip_dev", bio_records("dev")),
    ]
    manifest = {"schema_version": "drugclip-processed/v1", "inputs": input_hashes,
                "datasets": {}}
    for name, records in outputs:
        directory = data_root / "processed" / name
        directory.mkdir(parents=True, exist_ok=True)
        lmdb_path = directory / "data.lmdb"
        count = write_lmdb(lmdb_path, records)
        manifest["datasets"][name] = {
            "path": str(lmdb_path.relative_to(data_root)), "samples": count,
            "sha256": sha256_file(lmdb_path), "format": "ascii integer key -> pickle(dict)",
        }
    write_json(manifest_path, manifest)


def load_model_ready(path: Path) -> list[dict]:
    return [record for _, record in iter_lmdb(path)]
