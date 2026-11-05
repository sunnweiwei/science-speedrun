from __future__ import annotations

import hashlib
import multiprocessing
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import load_protocol, load_sources
from .io import download_resumable, extract_tar, write_immutable_text
from .mmseqs import search as mmseqs_search
from .structure import parse_pdb_atoms, pdb_sequences, read_fasta


_PROFSA_WORKER: dict = {}


def _download(data_root: Path) -> Path:
    source = load_sources()["profsadb"]
    item = source["file"]
    archive = data_root / "raw" / "profsadb" / "downloads" / item["name"]
    url = (f"https://huggingface.co/datasets/{source['repository']}/resolve/"
           f"{source['revision']}/{item['name']}")
    download_resumable(url, archive, item["size"], item["sha256"])
    extract_tar(archive, data_root / "raw" / "profsadb" / "complexes")
    # The released complexes contain only R/L fragments. Resolve full parent
    # chains from wwPDB SEQRES by the PDB identifiers encoded in sample paths.
    seqres = data_root / "raw" / "profsadb" / "pdb_seqres.txt.gz"
    download_resumable("https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz", seqres)
    return archive


def _sample_id(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("/", "__").removesuffix(".pdb")


def _source_key(path: Path) -> tuple[str, str | None]:
    """Infer PDB entry/chain from a ProFSADB path; unresolved data are excluded."""
    text = str(path)
    match = re.search(r"(?i)(?<![a-z0-9])([0-9][a-z0-9]{3})[_-]([a-z0-9])(?=[_.\-/])", text)
    if match:
        return match.group(1).lower(), match.group(2).upper()
    match = re.search(r"(?i)(?<![a-z0-9])([0-9][a-z0-9]{3})(?![a-z0-9])", text)
    return (match.group(1).lower(), None) if match else ("", None)


def _load_parent_sequences(raw: Path) -> dict[str, dict[str, str]]:
    candidates = (list(raw.rglob("*source*.fa*")) + list(raw.rglob("*seqres*.fa*"))
                  + list(raw.rglob("pdb_seqres.txt.gz")))
    mapping: dict[str, dict[str, str]] = {}
    for path in candidates:
        if not path.is_file():
            continue
        for header, sequence in read_fasta(path).items():
            match = re.search(r"(?i)([0-9][a-z0-9]{3})[_|:-]([a-z0-9])", header)
            if match:
                mapping.setdefault(match.group(1).lower(), {})[match.group(2).upper()] = sequence
    return mapping


def _choose_parent(entry: str, chain: str | None, fragment: str,
                   parents: dict[str, dict[str, str]]) -> tuple[str, str]:
    entry_chains = parents.get(entry, {})
    if chain and chain in entry_chains:
        return chain, entry_chains[chain]
    candidates = sorted(entry_chains.items())
    containing = [(candidate_chain, sequence) for candidate_chain, sequence in candidates
                  if fragment and fragment in sequence]
    if len(containing) == 1:
        return containing[0]
    if containing and len({sequence for _, sequence in containing}) == 1:
        return containing[0]
    if len(candidates) == 1:
        return candidates[0]
    return "", ""


def _requested_workers() -> int:
    workers = int(os.environ.get(
        "TARGET_BASED_DRUG_SCREENING_WORKERS",
        os.environ.get("DRUGCLIP_WORKERS", "24"),
    ))
    if workers < 1:
        raise ValueError("TARGET_BASED_DRUG_SCREENING_WORKERS must be a positive integer")
    return workers


def _init_profsa_worker(context: dict) -> None:
    global _PROFSA_WORKER
    _PROFSA_WORKER = context


def _profsa_row(path: Path, data_root: Path, complex_root: Path,
                parents: dict[str, dict[str, str]]) -> dict:
    sample_id = _sample_id(path, complex_root)
    row = {"sample_id": sample_id, "source_file": str(path.relative_to(data_root)),
           "source_pdb_id": "", "source_chain": "", "source_protein_sequence": "",
           "pocket_atom_count": 0, "pseudo_ligand_atom_count": 0,
           "max_test_sequence_identity": 0.0, "test_alignment_coverage": 0.0,
           "nearest_test_target": "", "nearest_test_benchmark": "",
           "array_file": "", "exclusion_reason": ""}
    try:
        receptor = [a for a in parse_pdb_atoms(path, {"R"}) if a["element"] not in {"H", "D"}]
        ligand = [a for a in parse_pdb_atoms(path, {"L"}) if a["element"] not in {"H", "D"}]
        if not receptor or not ligand:
            raise ValueError("invalid_pocket_or_pseudo_ligand")
        fragment = "".join(pdb_sequences(path, {"L"}).values())
        entry, chain_hint = _source_key(path)
        chain, sequence = _choose_parent(entry, chain_hint, fragment, parents)
        if not sequence:
            raise ValueError("unresolved_parent_protein")
        row.update({"source_pdb_id": entry, "source_chain": chain,
                    "source_protein_sequence": sequence,
                    "pocket_atom_count": len(receptor),
                    "pseudo_ligand_atom_count": len(ligand)})
    except Exception as error:
        reason = str(error)
        row["exclusion_reason"] = reason if reason in {
            "invalid_pocket_or_pseudo_ligand", "unresolved_parent_protein"
        } else f"parse_error:{type(error).__name__}"
    return row


def _process_profsa_shard(task: tuple[int, int, int]) -> tuple[int, str]:
    shard_index, start, stop = task
    context = _PROFSA_WORKER
    shard = context["cache_dir"] / f"part-{shard_index:06d}.parquet"
    rows = [_profsa_row(path, context["data_root"], context["complex_root"],
                        context["parents"])
            for path in context["pdb_files"][start:stop]]
    temporary = shard.with_name(f".{shard.name}.{os.getpid()}.tmp")
    try:
        pd.DataFrame(rows).to_parquet(temporary, index=False)
        os.replace(temporary, shard)
    finally:
        if temporary.exists():
            temporary.unlink()
    return shard_index, str(shard)


def _process_profsa_array(task: tuple[str, str]) -> tuple[str, str]:
    sample_id, source_file = task
    data_root: Path = _PROFSA_WORKER["data_root"]
    source_path = data_root / source_file
    digest = hashlib.sha256(sample_id.encode()).hexdigest()
    array_rel = Path("canonical") / "arrays" / "profsadb" / f"{digest}.npz"
    array_path = data_root / array_rel
    if not array_path.exists():
        receptor = [a for a in parse_pdb_atoms(source_path, {"R"})
                    if a["element"] not in {"H", "D"}]
        ligand = [a for a in parse_pdb_atoms(source_path, {"L"})
                  if a["element"] not in {"H", "D"}]
        if not receptor or not ligand:
            raise RuntimeError(f"selected ProFSADB sample became unparsable: {sample_id}")
        temporary = array_path.with_name(f".{array_path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    pocket_atoms=np.asarray([a["element"] for a in receptor], dtype="U3"),
                    pocket_coordinates=np.asarray([a["coord"] for a in receptor], dtype=np.float32),
                    ligand_atoms=np.asarray([a["element"] for a in ligand], dtype="U3"),
                    ligand_coordinates=np.asarray([a["coord"] for a in ligand], dtype=np.float32),
                )
            os.replace(temporary, array_path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return sample_id, str(array_rel)


def prepare_profsadb(data_root: Path, source_fasta: Path | None = None,
                     download: bool = True, smoke_count: int | None = None) -> None:
    protocol = load_protocol()
    raw = data_root / "raw" / "profsadb"
    if download:
        _download(data_root)
    complex_root = raw / "complexes"
    pdb_files = sorted(complex_root.rglob("*.pdb"))
    if not pdb_files:
        raise RuntimeError(f"no ProFSADB PDB files found below {complex_root}")
    parents = _load_parent_sequences(raw)
    if source_fasta:
        for header, sequence in read_fasta(source_fasta).items():
            match = re.search(r"(?i)([0-9][a-z0-9]{3})[_|:-]([a-z0-9])", header)
            if match:
                parents.setdefault(match.group(1).lower(), {})[match.group(2).upper()] = sequence

    cache_dir = data_root / "canonical" / "cache" / "profsadb_shards"
    arrays_dir = data_root / "canonical" / "arrays" / "profsadb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = 10_000
    shards = [cache_dir / f"part-{start // chunk_size:06d}.parquet"
              for start in range(0, len(pdb_files), chunk_size)]
    pending_shards = [(start // chunk_size, start,
                       min(start + chunk_size, len(pdb_files)))
                      for start in range(0, len(pdb_files), chunk_size)
                      if not shards[start // chunk_size].exists()]
    workers = min(_requested_workers(), len(pending_shards)) if pending_shards else 0
    shard_context = {"data_root": data_root, "complex_root": complex_root,
                     "parents": parents, "pdb_files": pdb_files,
                     "cache_dir": cache_dir}
    if workers == 1:
        _init_profsa_worker(shard_context)
        for task in tqdm(pending_shards, desc="Parsing ProFSADB shards", unit="shard"):
            _process_profsa_shard(task)
    elif workers > 1:
        fork_context = multiprocessing.get_context("fork")
        with fork_context.Pool(workers, initializer=_init_profsa_worker,
                               initargs=(shard_context,)) as pool:
            progress = tqdm(total=len(pending_shards), desc="Parsing ProFSADB shards",
                            unit="shard")
            for _ in pool.imap_unordered(_process_profsa_shard, pending_shards,
                                         chunksize=1):
                progress.update()
            progress.close()
    frames = [pd.read_parquet(shard) for shard in shards]

    frame = pd.concat(frames, ignore_index=True)
    valid = frame.exclusion_reason.eq("")
    proteins = frame.loc[valid, ["source_pdb_id", "source_chain", "source_protein_sequence"]].drop_duplicates()
    queries = [(f"{row.source_pdb_id}_{row.source_chain}", row.source_protein_sequence)
               for row in proteins.itertuples(index=False)]
    hits = mmseqs_search(queries, data_root / "test" / "test_targets.fasta",
                         data_root / "canonical" / "profsa_vs_test.tsv",
                         protocol["protein_identity_threshold_percent"],
                         protocol["protein_coverage_threshold"], protocol["mmseqs_cov_mode"])
    target_table = pd.read_parquet(data_root / "test" / "test_targets.parquet").set_index("target_id")
    for index in frame.index[valid]:
        query = f"{frame.at[index, 'source_pdb_id']}_{frame.at[index, 'source_chain']}"
        hit = hits.get(query)
        if hit:
            target = hit["target"].split("|", 2)[1]
            frame.at[index, "max_test_sequence_identity"] = hit["pident"]
            frame.at[index, "test_alignment_coverage"] = min(hit["qcov"], hit["tcov"])
            frame.at[index, "nearest_test_target"] = target
            frame.at[index, "nearest_test_benchmark"] = target_table.loc[target, "benchmark"]
            if hit["pident"] > protocol["protein_identity_threshold_percent"]:
                frame.at[index, "exclusion_reason"] = "test_protein_similarity"
    clean_ids = sorted(frame.loc[frame.exclusion_reason.eq(""), "sample_id"])
    required = smoke_count if smoke_count is not None else protocol["pretrain_samples"]
    if len(clean_ids) < required:
        raise RuntimeError(f"only {len(clean_ids)} clean ProFSADB samples remain; need {required}")
    selected = sorted(random.Random(protocol["seed"]).sample(clean_ids, required))
    frame["selected"] = frame.sample_id.isin(set(selected))
    id_to_index = pd.Series(frame.index, index=frame.sample_id).to_dict()
    array_tasks = [(sample_id, frame.at[id_to_index[sample_id], "source_file"])
                   for sample_id in selected]
    array_workers = min(_requested_workers(), len(array_tasks)) if array_tasks else 0
    array_context = {"data_root": data_root}
    progress = tqdm(total=len(array_tasks), desc="Materializing ProFSADB arrays",
                    unit="sample")
    if array_workers == 1:
        _init_profsa_worker(array_context)
        results = map(_process_profsa_array, array_tasks)
        for sample_id, array_rel in results:
            frame.at[id_to_index[sample_id], "array_file"] = array_rel
            progress.update()
    elif array_workers > 1:
        fork_context = multiprocessing.get_context("fork")
        with fork_context.Pool(array_workers, initializer=_init_profsa_worker,
                               initargs=(array_context,)) as pool:
            for sample_id, array_rel in pool.imap_unordered(
                    _process_profsa_array, array_tasks, chunksize=32):
                frame.at[id_to_index[sample_id], "array_file"] = array_rel
                progress.update()
    progress.close()
    output = data_root / "canonical" / "profsadb.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    split = data_root / "splits" / "profsa_pretrain_500k.txt"
    split.parent.mkdir(parents=True, exist_ok=True)
    write_immutable_text(split, "".join(f"{sample}\n" for sample in selected))
