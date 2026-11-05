from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing
import os
import pickle
import random
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, DataStructs
from tqdm import tqdm

from .config import PACKAGE_PARENT, load_protocol, load_sources
from .io import download_resumable, extract_tar, sha256_file, write_immutable_text, write_json
from .mmseqs import cluster as mmseqs_cluster
from .mmseqs import search as mmseqs_search
from .structure import (AA3, crystal_coordinates, generate_conformers, ligand_identity,
                        load_ligand, parse_pdb_atoms, read_fasta,
                        select_pocket)


_BIOLIP_WORKER: dict = {}
_RECEPTOR_CACHE: OrderedDict[Path, tuple[list[dict], dict[str, str]]] = OrderedDict()


def download_biolip2(data_root: Path, lock_path: Path | None = None) -> None:
    source = load_sources()["biolip2"]
    raw = data_root / "raw" / "biolip2"
    downloads = raw / "downloads"
    locked = {}
    if lock_path:
        locked = {entry["source_name"]: entry for entry in json.loads(lock_path.read_text())["files"]}
    records = []
    for item in source["files"]:
        name = item["name"]
        destination = downloads / Path(name).name
        expected_hash = locked.get(name, {}).get("sha256")
        record = download_resumable(f"{source['base_url']}/{name}", destination,
                                    item["size"], expected_hash, item["etag"])
        record.update({"source_name": name, "etag": item["etag"]})
        records.append(record)
    for archive_name, subdir in [
        ("redund_rec.tar.gz", "receptors"), ("redund_lig.tar.gz", "ligands"),
        ("redund_fasta.tar.gz", "sequences")
    ]:
        extract_tar(downloads / archive_name, raw / subdir)
    write_json(raw / "download_manifest.json", {
        "schema_version": "biolip2-download-manifest/v1", "source": source["name"],
        "snapshot": source["snapshot"], "files": records,
    })


def _file_map(root: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    result = {}
    for path in root.rglob("*"):
        if path.is_file() and any(path.name.lower().endswith(s) for s in suffixes):
            result.setdefault(path.name.split(".")[0], path)
    return result


def _sequence_map(root: Path) -> dict[str, str]:
    result = {}
    for path in root.rglob("*"):
        if not path.is_file() or not path.name.lower().endswith((".fa", ".fasta", ".faa", ".fa.gz")):
            continue
        for header, sequence in read_fasta(path).items():
            keys = {header.split()[0], path.name.split(".")[0]}
            keys |= {token for token in re.split(r"[|\s]", header) if token}
            for key in keys:
                result.setdefault(key, sequence)
    return result


def _stable_seed(text: str, base_seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{base_seed}:{text}".encode()).digest()[:4], "big")


def _sequences_from_atoms(atoms: list[dict]) -> dict[str, str]:
    residues: dict[str, dict[tuple[int, str], str]] = {}
    for atom in atoms:
        amino_acid = AA3.get(atom["resname"])
        if amino_acid is None:
            continue
        key = (atom["residue_number"], atom["insertion"])
        residues.setdefault(atom["chain"], {}).setdefault(key, amino_acid)
    return {chain: "".join(chain_residues.values())
            for chain, chain_residues in residues.items() if chain_residues}


def _init_biolip_worker(context: dict) -> None:
    global _BIOLIP_WORKER, _RECEPTOR_CACHE
    _BIOLIP_WORKER = context
    _RECEPTOR_CACHE = OrderedDict()
    # Q-BioLiP contains many unsupported metal/cofactor structures. RDKit can
    # emit thousands of repeated parameter warnings without changing results.
    RDLogger.DisableLog("rdApp.warning")


def _load_receptor_cached(path: Path) -> tuple[list[dict], dict[str, str]]:
    global _RECEPTOR_CACHE
    cached = _RECEPTOR_CACHE.pop(path, None)
    if cached is not None:
        _RECEPTOR_CACHE[path] = cached
        return cached
    atoms = [atom for atom in parse_pdb_atoms(path, record_types=("ATOM  ",))
             if atom["resname"] in AA3]
    sequences = _sequences_from_atoms(atoms)
    value = (atoms, sequences)
    _RECEPTOR_CACHE[path] = value
    cache_size = int(_BIOLIP_WORKER.get("receptor_cache_size", 8))
    while len(_RECEPTOR_CACHE) > cache_size:
        _RECEPTOR_CACHE.popitem(last=False)
    return value


def _atomic_pickle(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=-1)
    os.replace(temporary, path)


def _base_biolip_row(source_row: dict[str, str]) -> dict:
    assembly = source_row["Assembly_ID"].strip()
    ligand_file = source_row["Ligand_file"].strip()
    ligand_id = source_row["Ligand_ID"].strip().upper()
    complex_id = f"{assembly}__{ligand_file}"
    return {
        "complex_id": complex_id, "pdb_id": assembly[:4].lower(),
        "protein_chain": "", "protein_sequence": "", "protein_sequences_json": "{}",
        "ligand_id": ligand_id, "canonical_smiles": "", "inchikey": "",
        "pocket_atom_count": 0, "pocket_residue_count": 0,
        "ligand_atom_count": 0, "ligand_heavy_atom_count": 0,
        "ligand_conformer_count": 0, "max_test_sequence_identity": 0.0,
        "test_alignment_coverage": 0.0, "nearest_test_target": "",
        "nearest_test_benchmark": "", "max_test_ligand_tanimoto": 0.0,
        "nearest_test_ligand": "", "nearest_test_ligand_target": "",
        "exact_test_ligand_match": False, "protein_cluster": "", "split": "",
        "exclusion_reason": "", "exclusion_detail": "", "array_file": "",
        "rdkit_version": rdBase.rdkitVersion,
    }


def _process_biolip_row(source_row: dict[str, str]) -> str:
    context = _BIOLIP_WORKER
    data_root: Path = context["data_root"]
    protocol = context["protocol"]
    row = _base_biolip_row(source_row)
    assembly = source_row["Assembly_ID"].strip()
    ligand_file = source_row["Ligand_file"].strip()
    ligand_id = row["ligand_id"]
    complex_id = row["complex_id"]
    cache_file = context["cache"] / f"{complex_id}.pkl"
    try:
        if ligand_id in context["artifacts"]:
            raise ValueError("artifact_ligand")
        receptor_path = context["receptors"].get(assembly)
        ligand_path = context["ligands"].get(ligand_file)
        if receptor_path is None or ligand_path is None:
            raise ValueError("missing_structure")
        try:
            molecule = load_ligand(ligand_path)
            smiles, inchikey = ligand_identity(molecule)
            ligand_pose = crystal_coordinates(molecule)
        except Exception as error:
            raise ValueError("invalid_ligand") from error
        try:
            protein_atoms, parsed_sequences = _load_receptor_cached(receptor_path)
            if not protein_atoms or not parsed_sequences:
                raise ValueError
        except Exception as error:
            raise ValueError("invalid_protein") from error
        try:
            pocket_atoms, pocket_coordinates, residue_count = select_pocket(
                protein_atoms, ligand_pose, protocol["pocket_cutoff_angstrom"])
        except Exception as error:
            raise ValueError("invalid_or_empty_pocket") from error
        binding_chains = sorted(set(re.findall(
            r"(?:^|\s)([^:\s]+):", source_row.get("Binding_residue", ""))))
        chain_sequences = {}
        for chain in binding_chains:
            sequence = (context["sequences"].get(f"{assembly}_{chain}")
                        or context["sequences"].get(f"{assembly}_{chain[:1]}")
                        or parsed_sequences.get(chain)
                        or parsed_sequences.get(chain[:1]))
            if sequence:
                chain_sequences[chain] = sequence
        if not chain_sequences and len(parsed_sequences) == 1:
            chain_sequences = parsed_sequences
        if not chain_sequences:
            raise ValueError("missing_protein_sequence")
        _, sequence = max(chain_sequences.items(), key=lambda item: len(item[1]))
        try:
            ligand_atoms, conformers, optimization = generate_conformers(
                smiles, protocol["max_ligand_conformers"],
                _stable_seed(complex_id, protocol["seed"]),
                protocol["conformer_prune_rms_angstrom"])
        except Exception as error:
            raise ValueError("invalid_ligand_conformer") from error
        array_rel = Path("canonical") / "arrays" / "biolip2" / f"{complex_id}.npz"
        array_path = data_root / array_rel
        array_temporary = array_path.with_name(f".{array_path.name}.{os.getpid()}.tmp")
        with array_temporary.open("wb") as handle:
            np.savez_compressed(handle,
                                ligand_atoms=np.asarray(ligand_atoms, dtype="U3"),
                                ligand_coordinates=np.asarray(conformers, dtype=np.float32),
                                pocket_atoms=np.asarray(pocket_atoms, dtype="U3"),
                                pocket_coordinates=pocket_coordinates.astype(np.float32))
        os.replace(array_temporary, array_path)
        row.update({
            "protein_chain": ",".join(chain_sequences), "protein_sequence": sequence,
            "protein_sequences_json": json.dumps(chain_sequences, sort_keys=True),
            "canonical_smiles": smiles, "inchikey": inchikey,
            "pocket_atom_count": len(pocket_atoms), "pocket_residue_count": residue_count,
            "ligand_atom_count": len(ligand_atoms),
            "ligand_heavy_atom_count": molecule.GetNumHeavyAtoms(),
            "ligand_conformer_count": len(conformers), "array_file": str(array_rel),
            "conformer_optimization": optimization,
            "source_receptor_file": str(receptor_path.relative_to(data_root)),
            "source_ligand_file": str(ligand_path.relative_to(data_root)),
        })
    except Exception as error:
        reason = str(error)
        known = {"artifact_ligand", "missing_structure", "invalid_ligand", "invalid_protein",
                 "invalid_or_empty_pocket", "missing_protein_sequence",
                 "invalid_ligand_conformer"}
        row["exclusion_reason"] = (reason if reason in known
                                   else f"parse_or_preprocess_error:{type(error).__name__}")
        row["exclusion_detail"] = repr(error.__cause__ or error)[:500]
    _atomic_pickle(cache_file, row)
    return complex_id


def _cache_is_complete(path: Path, complex_id: str, data_root: Path) -> bool:
    try:
        with path.open("rb") as handle:
            row = pickle.load(handle)
        if row.get("complex_id") != complex_id:
            return False
        return bool(row.get("exclusion_reason")) or (data_root / row["array_file"]).is_file()
    except Exception:
        return False


def _apply_manual_exclusions(unique_rows: dict, cache: Path, data_root: Path) -> int:
    """Publish auditable exclusion rows for a pinned, stalled source tail."""
    ledger = json.loads((PACKAGE_PARENT / "biolip2_manual_exclusions.json").read_text())
    if ledger.get("schema_version") != "biolip2-manual-exclusions/v1":
        raise ValueError("invalid BioLiP2 manual-exclusion ledger schema")
    if ledger.get("source_snapshot") != load_sources()["biolip2"]["snapshot"]:
        raise ValueError("BioLiP2 manual exclusions do not match the source snapshot")
    entries = ledger["excluded_complexes"]
    ids = [entry["complex_id"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate BioLiP2 manual-exclusion complex ID")
    if len(unique_rows) > 1000 and set(ids) - set(unique_rows):
        raise ValueError("BioLiP2 manual-exclusion ID is absent from the pinned annotation")
    written = 0
    for entry in entries:
        complex_id = entry["complex_id"]
        source_row = unique_rows.get(complex_id)
        if source_row is None:
            continue  # Synthetic smoke fixture, not the pinned full source.
        row = _base_biolip_row(source_row)
        if row["ligand_id"] != entry["ligand_id"]:
            raise ValueError(f"BioLiP2 manual-exclusion ligand ID changed: {complex_id}")
        cache_file = cache / f"{complex_id}.pkl"
        if cache_file.exists() and _cache_is_complete(cache_file, complex_id, data_root):
            with cache_file.open("rb") as handle:
                existing = pickle.load(handle)
            if existing.get("exclusion_reason") != "manual_stalled_tail_exclusion":
                raise ValueError(f"Manual exclusion conflicts with cached result: {complex_id}")
            continue
        row["exclusion_reason"] = "manual_stalled_tail_exclusion"
        row["exclusion_detail"] = entry["detail"]
        _atomic_pickle(cache_file, row)
        written += 1
    return written


def prepare_biolip2_structures(data_root: Path) -> None:
    """Parse, clean, extract pockets, generate conformers, and cache canonical rows."""
    protocol = load_protocol()
    raw = data_root / "raw" / "biolip2"
    annotation = raw / "downloads" / "PL_annotation.csv"
    artifacts = {line.strip().upper() for line in (raw / "downloads" / "artlist").read_text().splitlines()
                 if line.strip() and not line.startswith("#")}
    receptors = _file_map(raw / "receptors", (".pdb",))
    ligands = _file_map(raw / "ligands", (".pdb", ".mol2", ".sdf", ".mol"))
    sequences = _sequence_map(raw / "sequences")
    cache = data_root / "canonical" / "cache" / "biolip2"
    arrays = data_root / "canonical" / "arrays" / "biolip2"
    cache.mkdir(parents=True, exist_ok=True)
    arrays.mkdir(parents=True, exist_ok=True)

    with annotation.open(newline="", errors="replace") as handle:
        source_rows = list(csv.DictReader(handle))
    unique_rows = {}
    for source_row in source_rows:
        complex_id = (f"{source_row['Assembly_ID'].strip()}__"
                      f"{source_row['Ligand_file'].strip()}")
        unique_rows.setdefault(complex_id, source_row)
    manual_written = _apply_manual_exclusions(unique_rows, cache, data_root)
    if manual_written:
        print(f"Published {manual_written} pinned BioLiP2 manual-exclusion cache rows", flush=True)
    pending, cached = [], 0
    for complex_id, source_row in unique_rows.items():
        cache_file = cache / f"{complex_id}.pkl"
        if cache_file.exists() and _cache_is_complete(cache_file, complex_id, data_root):
            cached += 1
        else:
            pending.append(source_row)
    requested_workers = int(os.environ.get(
        "TARGET_BASED_DRUG_SCREENING_WORKERS",
        os.environ.get("DRUGCLIP_WORKERS", "24"),
    ))
    receptor_cache_size = int(os.environ.get(
        "TARGET_BASED_DRUG_SCREENING_RECEPTOR_CACHE_SIZE", "8"))
    task_chunk_size = int(os.environ.get(
        "TARGET_BASED_DRUG_SCREENING_TASK_CHUNK_SIZE", "16"))
    if requested_workers < 1:
        raise ValueError("TARGET_BASED_DRUG_SCREENING_WORKERS must be a positive integer")
    if receptor_cache_size < 1 or task_chunk_size < 1:
        raise ValueError("receptor cache and task chunk sizes must be positive integers")
    # Keep neighboring assemblies together. Seeds are derived from complex_id,
    # so scheduling order cannot affect generated conformers or output rows.
    pending.sort(key=lambda row: (row["Assembly_ID"].strip(), row["Ligand_file"].strip()))
    workers = min(requested_workers, len(pending)) if pending else 0
    context = {
        "data_root": data_root, "protocol": protocol, "artifacts": artifacts,
        "receptors": receptors, "ligands": ligands, "sequences": sequences,
        "cache": cache, "receptor_cache_size": receptor_cache_size,
    }
    progress = tqdm(total=len(unique_rows), initial=cached,
                    desc=f"Preparing BioLiP2 ({workers or 0} workers)", unit="complex")
    if workers == 1:
        _init_biolip_worker(context)
        for source_row in pending:
            _process_biolip_row(source_row)
            progress.update()
    elif workers > 1:
        fork_context = multiprocessing.get_context("fork")
        with fork_context.Pool(workers, initializer=_init_biolip_worker,
                               initargs=(context,)) as pool:
            for _ in pool.imap_unordered(_process_biolip_row, pending,
                                         chunksize=task_chunk_size):
                progress.update()
    progress.close()
    materialize_biolip_table(data_root)


def materialize_biolip_table(data_root: Path) -> pd.DataFrame:
    cache = data_root / "canonical" / "cache" / "biolip2"
    rows = []
    for path in sorted(cache.glob("*.pkl")):
        with path.open("rb") as handle:
            rows.append(pickle.load(handle))
    frame = pd.DataFrame(rows)
    output = data_root / "canonical" / "biolip2.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return frame


def filter_biolip_contamination(data_root: Path) -> None:
    protocol = load_protocol()
    frame = pd.read_parquet(data_root / "canonical" / "biolip2.parquet")
    valid = frame.exclusion_reason.eq("")
    sequence_to_query = {}
    for index in frame.index[valid]:
        for sequence in json.loads(frame.at[index, "protein_sequences_json"]).values():
            sequence_to_query.setdefault(sequence, f"protein_{len(sequence_to_query)}")
    hits = mmseqs_search([(query, sequence) for sequence, query in sequence_to_query.items()],
                         data_root / "test" / "test_targets.fasta",
                         data_root / "canonical" / "biolip2_vs_test.tsv",
                         protocol["protein_identity_threshold_percent"],
                         protocol["protein_coverage_threshold"], protocol["mmseqs_cov_mode"])
    target_index = pd.read_parquet(data_root / "test" / "test_targets.parquet").set_index("target_id")
    sequence_metrics = {}
    for sequence, query in sequence_to_query.items():
        hit = hits.get(query)
        if hit:
            raw_target = hit["target"].split("|", 2)[1]
            info = target_index.loc[raw_target]
            sequence_metrics[sequence] = (hit["pident"], min(hit["qcov"], hit["tcov"]),
                                          raw_target, info["benchmark"])
    for index in frame.index[valid]:
        metrics = [sequence_metrics[sequence]
                   for sequence in json.loads(frame.at[index, "protein_sequences_json"]).values()
                   if sequence in sequence_metrics]
        metric = max(metrics, default=None, key=lambda item: (item[0], item[1]))
        if metric:
            (frame.at[index, "max_test_sequence_identity"],
             frame.at[index, "test_alignment_coverage"],
             frame.at[index, "nearest_test_target"],
             frame.at[index, "nearest_test_benchmark"]) = metric
            if metric[0] > protocol["protein_identity_threshold_percent"]:
                frame.at[index, "exclusion_reason"] = "test_protein_similarity"

    actives = pd.read_parquet(data_root / "test" / "test_ligands.parquet",
                              filters=[("label", "==", 1)]).drop_duplicates("inchikey")
    active_mols = [Chem.MolFromSmiles(s) for s in actives.canonical_smiles]
    active_fps = [AllChem.GetMorganFingerprintAsBitVect(m, protocol["morgan_radius"],
                                                        nBits=protocol["morgan_nbits"])
                  for m in active_mols]
    active_rows = list(actives.itertuples(index=False))
    active_keys = set(actives.inchikey)
    for index in frame.index[valid]:
        molecule = Chem.MolFromSmiles(frame.at[index, "canonical_smiles"])
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            molecule, protocol["morgan_radius"], nBits=protocol["morgan_nbits"])
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, active_fps)
        best_index = int(np.argmax(similarities))
        best = active_rows[best_index]
        similarity = float(similarities[best_index])
        exact = frame.at[index, "inchikey"] in active_keys
        frame.at[index, "max_test_ligand_tanimoto"] = similarity
        frame.at[index, "nearest_test_ligand"] = best.inchikey
        frame.at[index, "nearest_test_ligand_target"] = f"{best.benchmark}:{best.target_id}"
        frame.at[index, "exact_test_ligand_match"] = exact
        if exact and frame.at[index, "exclusion_reason"] == "":
            frame.at[index, "exclusion_reason"] = "exact_test_active_match"
        elif (similarity > protocol["ligand_tanimoto_threshold"]
              and frame.at[index, "exclusion_reason"] == ""):
            frame.at[index, "exclusion_reason"] = "test_active_ecfp4_similarity"
    frame.to_parquet(data_root / "canonical" / "biolip2.parquet", index=False)


def make_split(data_root: Path) -> None:
    protocol = load_protocol()
    path = data_root / "canonical" / "biolip2.parquet"
    frame = pd.read_parquet(path)
    clean = frame[frame.exclusion_reason.eq("")]
    if clean.empty:
        raise RuntimeError("no clean BioLiP2 complexes remain after leakage filtering")
    all_sequences = sorted({sequence for value in clean.protein_sequences_json
                            for sequence in json.loads(value).values()})
    names = [(f"protein_{i}", sequence) for i, sequence in enumerate(all_sequences)]
    name_to_sequence = {name: sequence for name, sequence in names}
    clusters = mmseqs_cluster(names, data_root / "canonical" / "biolip2_30pct",
                              protocol["protein_identity_threshold_percent"],
                              protocol["protein_coverage_threshold"], protocol["mmseqs_cov_mode"])
    sequence_cluster = {name_to_sequence[name]: representative for name, representative in clusters.items()}
    # A multichain complex bridges all clusters represented by its binding
    # chains. Union them so no chain-level cluster can cross train/dev.
    parent = {representative: representative for representative in set(clusters.values())}
    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item
    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)
    row_representatives = {}
    for index in clean.index:
        representatives = sorted({sequence_cluster[sequence]
                                  for sequence in json.loads(frame.at[index, "protein_sequences_json"]).values()})
        for representative in representatives[1:]:
            union(representatives[0], representative)
        row_representatives[index] = representatives
    for index, representatives in row_representatives.items():
        frame.at[index, "protein_cluster"] = find(representatives[0])
    cluster_sizes = frame.loc[clean.index].groupby("protein_cluster").size().to_dict()
    ordered = sorted(cluster_sizes)
    random.Random(protocol["seed"]).shuffle(ordered)
    target = len(clean) * protocol["development_fraction"]
    dev_clusters, total = set(), 0
    for representative in ordered:
        if total < target:
            dev_clusters.add(representative)
            total += cluster_sizes[representative]
    frame.loc[clean.index, "split"] = [
        "dev" if frame.at[index, "protein_cluster"] in dev_clusters else "train"
        for index in clean.index
    ]
    split_dir = data_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        ids = sorted(frame.loc[frame.split.eq(split), "complex_id"])
        write_immutable_text(split_dir / f"{split}.txt", "".join(f"{item}\n" for item in ids))
    frame.to_parquet(path, index=False)
