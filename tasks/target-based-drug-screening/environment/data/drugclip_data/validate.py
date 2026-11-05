from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from .config import load_protocol, load_sources
from .io import iter_lmdb, sha256_file
from .testdata import _tree_digest


def _assert_record(record: dict, profsa: bool = False) -> None:
    atom_key = "lig_atoms_real" if profsa else "atoms"
    coord_key = "lig_coord_real" if profsa else "coordinates"
    atoms = record[atom_key]
    coordinates = np.asarray(record[coord_key])
    if not profsa:
        if coordinates.ndim != 3:
            raise AssertionError("ligand coordinates must be [conformer, atom, xyz]")
        atom_axis = 1
    else:
        if coordinates.ndim != 2:
            raise AssertionError("ProFSADB pseudo-ligand coordinates must be [atom, xyz]")
        atom_axis = 0
    assert coordinates.shape[atom_axis] == len(atoms) and coordinates.shape[-1] == 3
    assert len(record["pocket_atoms"]) == np.asarray(record["pocket_coordinates"]).shape[0]
    assert np.asarray(record["pocket_coordinates"]).shape[-1] == 3
    assert len(atoms) and len(record["pocket_atoms"])
    assert np.isfinite(coordinates).all()
    assert np.isfinite(np.asarray(record["pocket_coordinates"])).all()
    if not profsa and Chem.MolFromSmiles(record["smi"], sanitize=True) is None:
        raise AssertionError("invalid canonical ligand SMILES")


def validate_dataset(data_root: Path, smoke: bool = False) -> dict:
    protocol = load_protocol()
    errors = []
    test_manifest = json.loads((data_root / "test" / "manifest.json").read_text())
    if not smoke:
        expected_test = load_sources()["drugclip_benchmarks"]
        if test_manifest.get("revision") != expected_test["revision"]:
            errors.append("test manifest does not use the pinned Hugging Face revision")
        expected_archives = {item["name"]: item for item in expected_test["files"]}
        for archive in test_manifest.get("archives", []):
            expected_archive = expected_archives.get(archive["filename"])
            if not expected_archive or archive["size"] != expected_archive["size"] or archive["sha256"] != expected_archive["sha256"]:
                errors.append(f"test archive lock differs from protocol: {archive['filename']}")
            raw_archive = data_root / "raw" / "test_archives" / archive["filename"]
            if not raw_archive.exists() or raw_archive.stat().st_size != archive["size"] or sha256_file(raw_archive) != archive["sha256"]:
                errors.append(f"frozen test archive changed: {archive['filename']}")
            count, digest = _tree_digest(data_root / "test" / archive["benchmark"])
            if count != archive["extracted_file_count"] or digest != archive["extracted_tree_sha256"]:
                errors.append(f"extracted frozen test tree changed: {archive['benchmark']}")
        if {item["filename"] for item in test_manifest.get("archives", [])} != set(expected_archives):
            errors.append("test manifest archive set is incomplete")
    test_ligands = pd.read_parquet(data_root / "test" / "test_ligands.parquet")
    if test_ligands[["canonical_smiles", "inchikey", "source_smiles"]].eq("").any().any():
        errors.append("test ligand index has a missing molecular identity")
    if not set(test_ligands.parse_status).issubset({"sanitized", "unsanitized_source_valence"}):
        errors.append("test ligand index has an unknown parse status")
    if test_ligands.loc[test_ligands.label.eq(1), "parse_status"].ne("sanitized").any():
        errors.append("test active did not pass strict RDKit sanitization")
    profsa = pd.read_parquet(data_root / "canonical" / "profsadb.parquet")
    biolip = pd.read_parquet(data_root / "canonical" / "biolip2.parquet")
    selected = [line for line in (data_root / "splits" / "profsa_pretrain_500k.txt").read_text().splitlines()
                if line]
    expected = 2 if smoke else protocol["pretrain_samples"]
    if len(selected) != expected or len(set(selected)) != expected:
        errors.append(f"ProFSADB selection count/uniqueness is {len(selected)}, expected {expected}")
    selected_rows = profsa[profsa.sample_id.isin(selected)]
    if len(selected_rows) != expected:
        errors.append("selected ProFSADB IDs are absent or duplicated in canonical table")
    if (selected_rows.max_test_sequence_identity > protocol["protein_identity_threshold_percent"]).any():
        errors.append("ProFSADB test-protein leakage")
    if selected_rows.exclusion_reason.ne("").any():
        errors.append("excluded ProFSADB row was selected")

    clean = biolip[biolip.split.isin(["train", "dev"])]
    if clean.complex_id.duplicated().any(): errors.append("duplicate BioLiP2 complex IDs")
    train_ids, dev_ids = set(biolip.loc[biolip.split.eq("train"), "complex_id"]), set(biolip.loc[biolip.split.eq("dev"), "complex_id"])
    file_train = {line for line in (data_root / "splits" / "train.txt").read_text().splitlines() if line}
    file_dev = {line for line in (data_root / "splits" / "dev.txt").read_text().splitlines() if line}
    if file_train != train_ids or file_dev != dev_ids: errors.append("train/dev split files differ from canonical table")
    if not train_ids or not dev_ids: errors.append("BioLiP2 train or dev split is empty")
    if train_ids & dev_ids: errors.append("train/dev IDs overlap")
    train_clusters = set(biolip.loc[biolip.split.eq("train"), "protein_cluster"])
    dev_clusters = set(biolip.loc[biolip.split.eq("dev"), "protein_cluster"])
    if train_clusters & dev_clusters: errors.append("train/dev MMseqs2 clusters overlap")
    if "" in train_clusters or "" in dev_clusters: errors.append("train/dev row has no MMseqs2 cluster")
    if (clean.max_test_sequence_identity > protocol["protein_identity_threshold_percent"]).any(): errors.append("BioLiP2 protein leakage")
    if (clean.max_test_ligand_tanimoto > protocol["ligand_tanimoto_threshold"]).any(): errors.append("BioLiP2 ligand leakage")
    if clean.exact_test_ligand_match.any(): errors.append("BioLiP2 exact test-active match")
    if clean.exclusion_reason.ne("").any(): errors.append("excluded BioLiP2 row has a split")

    manifest = json.loads((data_root / "processed" / "manifest.json").read_text())
    for relative, digest in manifest.get("inputs", {}).items():
        if sha256_file(data_root / relative) != digest:
            errors.append(f"processed view is stale relative to canonical input: {relative}")
    expected_processed = {"profsa_pretrain_500k": expected,
                          "biolip_train": len(train_ids), "biolip_dev": len(dev_ids)}
    for name, profsa_mode in [("profsa_pretrain_500k", True), ("biolip_train", False), ("biolip_dev", False)]:
        item = manifest["datasets"][name]
        if item["samples"] != expected_processed[name]:
            errors.append(f"{name} manifest count differs from canonical split")
        path = data_root / item["path"]
        if sha256_file(path) != item["sha256"]: errors.append(f"processed checksum mismatch: {name}")
        count = 0
        try:
            for _, record in iter_lmdb(path):
                _assert_record(record, profsa_mode)
                count += 1
        except Exception as error:
            errors.append(f"{name} parse/shape error: {error}")
        if count != item["samples"]: errors.append(f"{name} LMDB count mismatch")
    if errors:
        raise SystemExit("dataset validation FAILED:\n- " + "\n- ".join(errors))
    receipt = {"status": "PASS", "smoke": smoke, "profsadb_selected": expected,
               "biolip_train": len(train_ids), "biolip_dev": len(dev_ids)}
    (data_root / "VALIDATION.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
