from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from tqdm import tqdm

from .config import load_sources
from .io import download_resumable, extract_zip, sha256_file, write_json
from .structure import ligand_identity, parse_mol2_protein, pdb_sequences, write_fasta


def _tree_digest(root: Path) -> tuple[int, str]:
    digest = __import__("hashlib").sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == ".extracted.json":
            continue
        relative = str(path.relative_to(root))
        digest.update(relative.encode() + b"\0")
        digest.update(str(path.stat().st_size).encode() + b"\0")
        digest.update(sha256_file(path).encode() + b"\n")
        count += 1
    return count, digest.hexdigest()


def download_test_data(data_root: Path) -> None:
    source = load_sources()["drugclip_benchmarks"]
    revision = source["revision"]
    raw = data_root / "raw" / "test_archives"
    test = data_root / "test"
    entries = []
    for item in source["files"]:
        url = (f"https://huggingface.co/datasets/{source['repository']}/resolve/"
               f"{revision}/{item['name']}")
        archive = raw / item["name"]
        record = download_resumable(url, archive, item["size"], item["sha256"])
        if item["name"] == "DUD-E.zip":
            extract_zip(archive, test / "dude", "DUD-E")
            benchmark = "dude"
        else:
            extract_zip(archive, test / "lit_pcba", "lit_pcba")
            benchmark = "lit_pcba"
        tree_count, tree_sha = _tree_digest(test / benchmark)
        record.update({"benchmark": benchmark, "repository_revision": revision})
        record.update({"extracted_file_count": tree_count, "extracted_tree_sha256": tree_sha})
        entries.append(record)
    write_json(test / "manifest.json", {
        "schema_version": "drugclip-test-manifest/v1", "immutable": True,
        "source_repository": source["repository"], "revision": revision,
        "archives": entries,
    })


def _canonicalize_smiles(smiles: str, label: int) -> tuple[str, str, str]:
    molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    if molecule is not None:
        canonical, inchikey = ligand_identity(molecule)
        return canonical, inchikey, "sanitized"
    # Four author-released DUD-E decoy rows have an over-valent neutral N.
    # Preserve their graph verbatim for indexing, but never relax parsing for
    # test actives because those molecules define the leakage fingerprints.
    if label != 0:
        raise ValueError(f"invalid test active SMILES: {smiles}")
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    if molecule is None:
        raise ValueError(f"unparseable test decoy/inactive SMILES: {smiles}")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    inchikey = Chem.MolToInchiKey(molecule)
    if not canonical or not inchikey:
        raise ValueError(f"test decoy/inactive has no stable identity: {smiles}")
    return canonical, inchikey, "unsanitized_source_valence"


def _molecule_rows(path: Path, benchmark: str, target: str, label: int,
                   label_name: str, data_root: Path):
    with path.open(errors="replace") as handle:
        for line_number, raw in enumerate(handle, 1):
            fields = raw.strip().split()
            if not fields:
                continue
            canonical, inchikey, parse_status = _canonicalize_smiles(fields[0], label)
            yield {
                "benchmark": benchmark, "target_id": target, "label": label,
                "label_name": label_name, "canonical_smiles": canonical,
                "inchikey": inchikey, "source_smiles": fields[0],
                "parse_status": parse_status,
                "source_compound_id": fields[1] if len(fields) > 1 else "",
                "source_file": str(path.relative_to(data_root)), "source_line": line_number,
            }


def build_test_index(data_root: Path) -> None:
    test = data_root / "test"
    manifest = test / "manifest.json"
    if not manifest.exists():
        raise RuntimeError("test archives are not frozen; run download_test_data.py first")
    target_rows, fasta = [], []
    ligand_path = test / "test_ligands.parquet"
    ligand_schema = pa.schema([
        ("benchmark", pa.string()), ("target_id", pa.string()), ("label", pa.int8()),
        ("label_name", pa.string()), ("canonical_smiles", pa.string()),
        ("inchikey", pa.string()), ("source_smiles", pa.string()),
        ("parse_status", pa.string()), ("source_compound_id", pa.string()),
        ("source_file", pa.string()), ("source_line", pa.int64()),
    ])
    writer = pq.ParquetWriter(ligand_path, ligand_schema, compression="zstd")
    ligand_buffer, ligand_count, active_count, fallback_count = [], 0, 0, 0
    ligand_progress = tqdm(desc="Indexing released test ligands", unit="mol",
                           unit_scale=True)

    def emit(records) -> None:
        nonlocal ligand_buffer, ligand_count, active_count, fallback_count
        for record in records:
            ligand_buffer.append(record)
            ligand_count += 1
            active_count += record["label"]
            fallback_count += record["parse_status"] != "sanitized"
            ligand_progress.update()
            if len(ligand_buffer) >= 100_000:
                writer.write_table(pa.Table.from_pylist(ligand_buffer, schema=ligand_schema))
                ligand_buffer = []

    for target_dir in sorted((test / "dude").iterdir()):
        if not target_dir.is_dir():
            continue
        target = target_dir.name
        structure = target_dir / "receptor.pdb"
        if not structure.exists():
            candidates = sorted(target_dir.glob("*_mergedprot.pdb"))
            if not candidates:
                raise RuntimeError(f"DUD-E target {target} has no receptor structure")
            structure = candidates[0]
        sequences = pdb_sequences(structure)
        if not sequences:
            raise RuntimeError(f"no protein sequence in {structure}")
        for chain, sequence in sorted(sequences.items()):
            target_id = target if len(sequences) == 1 else f"{target}:{chain}"
            target_rows.append({
                "benchmark": "dude", "target_id": target_id, "target_name": target,
                "structure_id": structure.stem, "protein_chain": chain,
                "protein_sequence": sequence,
                "structure_file": str(structure.relative_to(data_root)),
            })
            fasta.append((f"dude|{target_id}|{structure.stem}|chain={chain}", sequence))
        emit(_molecule_rows(target_dir / "actives_final.ism", "dude",
                            target, 1, "active", data_root))
        emit(_molecule_rows(target_dir / "decoys_final.ism", "dude",
                            target, 0, "decoy", data_root))

    for target_dir in sorted((test / "lit_pcba").iterdir()):
        if not target_dir.is_dir():
            continue
        target = target_dir.name
        structures = sorted(target_dir.glob("*_protein.mol2"))
        if not structures:
            raise RuntimeError(f"LIT-PCBA target {target} has no protein MOL2")
        for structure in structures:
            sequences, _ = parse_mol2_protein(structure)
            if not sequences:
                raise RuntimeError(f"no protein sequence in {structure}")
            for chain, sequence in sorted(sequences.items()):
                structure_id = structure.name.removesuffix("_protein.mol2")
                target_id = f"{target}:{structure_id}"
                target_rows.append({
                    "benchmark": "lit_pcba", "target_id": target_id,
                    "target_name": target, "structure_id": structure_id,
                    "protein_chain": chain, "protein_sequence": sequence,
                    "structure_file": str(structure.relative_to(data_root)),
                })
                fasta.append((f"lit_pcba|{target_id}|chain={chain}", sequence))
        emit(_molecule_rows(target_dir / "actives.smi", "lit_pcba",
                            target, 1, "active", data_root))
        emit(_molecule_rows(target_dir / "inactives.smi", "lit_pcba",
                            target, 0, "inactive", data_root))

    targets = pd.DataFrame(target_rows).sort_values(["benchmark", "target_id", "structure_id"])
    if ligand_buffer:
        writer.write_table(pa.Table.from_pylist(ligand_buffer, schema=ligand_schema))
    writer.close()
    ligand_progress.close()
    targets.to_parquet(test / "test_targets.parquet", index=False)
    write_fasta(test / "test_targets.fasta", fasta)
    summary = json.loads(manifest.read_text())
    summary["index"] = {
        "target_sequence_count": len(targets),
        "target_name_count": int(targets[["benchmark", "target_name"]].drop_duplicates().shape[0]),
        "ligand_count": ligand_count, "active_count": active_count,
        "inactive_or_decoy_count": ligand_count - active_count,
        "unsanitized_source_valence_count": fallback_count,
        "test_targets_parquet_sha256": sha256_file(test / "test_targets.parquet"),
        "test_ligands_parquet_sha256": sha256_file(test / "test_ligands.parquet"),
        "test_targets_fasta_sha256": sha256_file(test / "test_targets.fasta"),
    }
    write_json(manifest, summary)
