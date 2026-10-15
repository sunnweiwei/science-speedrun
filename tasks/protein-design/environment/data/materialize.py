#!/usr/bin/env python3
"""Operator-only materializer. Builds the task data from a PDB snapshot.

Follows RFdiffusion3's approach: a frozen, dated PDB snapshot plus a precomputed
entry index, with every design condition derived from coordinates. Curated annotation
files (Catalytic Site Atlas, BioLiP) are deliberately not used - they lag the PDB by
years, which makes a temporal holdout impossible on the tracks that need one.

Runs on the host, never inside an agent or verifier image. Nothing it downloads is
redistributed by this repository.

    # 1. bulk entry metadata: id, deposition date, resolution, method, classification
    curl -O https://files.rcsb.org/pub/pdb/derived_data/index/entries.idx
    curl -O https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-30.txt

    # 2. entry-level filtering and the temporal split, no structure downloads
    python materialize.py --pdb-index entries.idx --clusters clusters-by-entity-30.txt \
        --plan-only

    # 3. full run: downloads assemblies and derives conditioning
    python materialize.py --pdb-index entries.idx --clusters clusters-by-entity-30.txt \
        --assembly-cache /secure/rcsb-cache --output-root /secure/protein-design
"""

import argparse
import collections
import datetime
import hashlib
import json
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assembly as A  # noqa: E402

TRACKS = ("unconditional", "sm_binder", "ppi_binder", "enzyme")

# Entry-level filters, mirroring RFdiffusion3's PDB dataset config
# (models/rf3/configs/datasets/train/pdb/*.yaml).
MAX_RESOLUTION = 3.5
ALLOWED_METHODS = {"X-RAY DIFFRACTION", "ELECTRON MICROSCOPY", "ELECTRON CRYSTALLOGRAPHY"}

# RFdiffusion3 trains on `deposition_date < 2024-01-01`, so anything deposited on or
# after that date is outside its training set. Evaluation targets are held out by
# deposition date because the harness contract permits pretrained design models and
# every current one is trained on the PDB; a sequence-identity split protects this
# benchmark's fit set, not theirs. Deposition date is the conservative field: release
# always follows deposition.
DEFAULT_EVAL_CUTOFF = "2024-01-01"

# The curated enzyme benchmark fixes 1-7 catalytic residues, not a whole pocket.
MAX_MOTIF_RESIDUES = 6
# The window tune must mirror is the curated benchmark's, whose enzyme targets are 180
# residues; an upper bound of 160 excluded every one of them and made the enzyme quota
# effectively unfillable (1 target in 800 entries, measured).
EVAL_MIN_LENGTH, EVAL_MAX_LENGTH = 60, 200
DEV_MIN_LENGTH, DEV_MAX_LENGTH = 40, 400
EVAL_TARGETS_PER_TRACK = 40
TUNE_TARGETS_PER_TRACK = 60
FIT_TARGETS_PER_TRACK = 4000
# Keep in step with ingest_targets.py::DESIGNS_PER_TARGET: the practice view an agent
# tunes against and the evaluation view it is graded on must request the same count.
DESIGNS_PER_TARGET = 20
MAX_PER_CLUSTER = 2

# Structures staged for RFdiffusion3-style training, which learns from whole assemblies
# rather than from the per-track design requests. Sized to be the same order as the fit
# view so the two are comparable, and because the agent's budget (1 GPU, 8 CPUs, 24 h)
# cannot consume more: at a measured ~2 s of preprocessing per drawn example, 8 workers
# saturate a single H100 well before the corpus is exhausted.
CORPUS_STRUCTURES = 16000


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_entry_index(path):
    """Parse RCSB entries.idx: IDCODE, HEADER, ACCESSION DATE, ..., RESOLUTION, METHOD."""
    entries = {}
    with Path(path).open(errors="replace") as stream:
        stream.readline()
        stream.readline()
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            pdb_id = fields[0].strip().lower()
            try:
                month, day, year = (int(x) for x in fields[2].strip().split("/"))
                # Two-digit years: the PDB predates 2000, so >50 is the twentieth century.
                deposited = datetime.date(1900 + year if year > 50 else 2000 + year,
                                          month, day)
            except ValueError:
                continue
            try:
                resolution = float(fields[6].strip())
            except ValueError:
                resolution = None
            entries[pdb_id] = {
                "pdb_id": pdb_id,
                "header": fields[1].strip(),
                "deposition_date": deposited,
                "resolution": resolution,
                "method": fields[7].strip() or "X-RAY DIFFRACTION",
            }
    return entries


def load_clusters(path):
    """RCSB 30% clusters: one whitespace-separated line of ENTITY ids per cluster."""
    mapping = {}
    for index, line in enumerate(Path(path).open(errors="replace")):
        for entity in line.split():
            mapping[entity.strip().upper()] = index
    return mapping


def entry_cluster(pdb_id, clusters):
    """Cluster for an entry, taken from its first clustered entity."""
    for entity in range(1, 9):
        key = f"{pdb_id.upper()}_{entity}"
        if key in clusters:
            return clusters[key]
    return None


def eligible_entries(entries, clusters):
    """Entry-level quality filter, before any structure is downloaded."""
    kept = []
    rejected = collections.Counter()
    for entry in entries.values():
        if entry["method"] not in ALLOWED_METHODS:
            rejected["method"] += 1
            continue
        if entry["resolution"] is None or entry["resolution"] > MAX_RESOLUTION:
            rejected["resolution"] += 1
            continue
        cluster = entry_cluster(entry["pdb_id"], clusters)
        if cluster is None:
            rejected["no_cluster"] += 1
            continue
        entry["cluster"] = cluster
        entry["is_enzyme"] = A.classify(entry["header"])
        kept.append(entry)
    return kept, rejected


def temporal_split(eligible, cutoff, seed=17):
    """Post-cutoff entries are evaluation; pre-cutoff are development.

    Cluster disjointness is layered on top: any cluster represented in evaluation is
    dropped from development entirely, so a held-out target has no close homologue in
    training even though the split itself is temporal.
    """
    evaluation_clusters = {e["cluster"] for e in eligible
                           if e["deposition_date"] >= cutoff}
    development, evaluation, excluded = [], [], 0
    for entry in eligible:
        if entry["deposition_date"] >= cutoff:
            entry["split"] = "eval"
            evaluation.append(entry)
        elif entry["cluster"] in evaluation_clusters:
            entry["split"] = "excluded_homologue"
            excluded += 1
        else:
            entry["split"] = "development"
            development.append(entry)
    random.Random(seed).shuffle(development)
    tune_clusters = {e["cluster"] for e in development[:len(development) // 10]}
    for entry in development:
        entry["split"] = "tune" if entry["cluster"] in tune_clusters else "fit"
    return evaluation, development, excluded


def download_assembly(pdb_id, cache, timeout=90):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / f"{pdb_id}-assembly1.cif.gz"
    if destination.is_file() and destination.stat().st_size:
        return destination
    url = f"https://files.rcsb.org/download/{pdb_id}-assembly1.cif.gz"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = response.read()
    if not payload:
        raise OSError(f"empty assembly download: {pdb_id}")
    destination.write_bytes(payload)
    return destination


def derive_tracks(entry, path, length_range):
    """Turn one assembly into at most one target per track."""
    chains, ligands = A.parse_assembly(path)
    if not chains:
        return {}
    low, high = length_range
    results = {}

    monomer = max(chains, key=len)
    if low <= len(monomer) <= high:
        results["unconditional"] = {"chain": monomer, "context": {}}

    for ligand in sorted(A.catalytic_ligands(ligands),
                         key=lambda l: -l["coordinates"].shape[0]):
        found = A.find_pocket(chains, ligand)
        if not found:
            continue
        chain, pocket = found
        if not low <= len(chain) <= high:
            continue
        # An sm_binder target is a bare ligand, matching RFdiffusion3's benchmark:
        # the design is built around it with no pocket template. Emitting the native
        # pocket here would make this track easier than the curated one it must match.
        ligand_arrays = {
            "ligand_coords": ligand["coordinates"],
            "ligand_ccd": np.array([ligand["name"]], dtype="<U5"),
            "ligand_atom_counts": np.array([ligand["coordinates"].shape[0]],
                                           dtype=np.int32),
            "ligand_elements": np.array(ligand["elements"], dtype="<U2"),
            "ligand_charges": np.array(ligand["charges"], dtype=np.float32),
        }
        results["sm_binder"] = {"chain": chain, "context": dict(ligand_arrays),
                                "ligand": ligand["name"]}
        # An EC-classified entry whose pocket holds a cofactor is an enzyme active
        # site. Only the residues closest to the cofactor are kept, so the motif is a
        # catalytic constellation rather than the whole pocket.
        if entry["is_enzyme"] and ligand["name"] in A.CATALYTIC_COFACTORS:
            core = A.closest_residues(chain, pocket, ligand["coordinates"],
                                      MAX_MOTIF_RESIDUES)
            results["enzyme"] = {
                "chain": chain,
                "context": {
                    "motif_coords": chain.coordinates[core],
                    "motif_mask": chain.mask[core],
                    "motif_restypes": np.array([chain.restypes[i] for i in core],
                                               dtype="<U3"),
                    **ligand_arrays,
                },
                "ligand": ligand["name"],
            }
        break

    interface = A.find_interface(chains)
    if interface:
        binder, binder_residues, target, target_residues, _ = interface
        if low <= len(binder) <= high:
            results["ppi_binder"] = {
                "chain": binder,
                "context": {
                    "target_coords": target.coordinates,
                    "target_mask": target.mask,
                    "target_restypes": np.array(target.restypes, dtype="<U3"),
                    "hotspot_indices": np.array(target_residues, dtype=np.int32),
                },
            }
    return results


def build_training_corpus(entries, root, cache, limit, log_every=200):
    """Stage assemblies for RFdiffusion3-style training, in the layout its loader expects.

    The four per-track `.npz` views are design *requests*: a length, a motif, a ligand.
    Upstream RFD3 does not train on requests - it trains on whole structures and samples
    its own crops and conditioning from them, which is where the tipatom and unindexed
    regimes come from. So a starter that is really RFD3 needs the structures themselves.

    They are written as `<root>/structures/<chars 2-3 of id>/<id>.cif.gz`, which is the
    RCSB divided layout that `atomworks`' `PNUnitsDFParser` already resolves with its
    default `path_template`. Indexing them is a separate step with a separate dependency
    (`build_training_index.py` needs atomworks; this file needs only gemmi), and the
    command to run is printed at the end.
    """
    structures = root / "structures"
    structures.mkdir(parents=True, exist_ok=True)
    staged, failed = [], 0
    for entry in entries:
        if len(staged) >= limit:
            break
        pdb_id = entry["pdb_id"]
        try:
            source = download_assembly(pdb_id, cache)
        except (OSError, ValueError, urllib.error.URLError) as error:
            failed += 1
            if failed <= 5:
                print(f"  skip {pdb_id}: {type(error).__name__}: {str(error)[:60]}",
                      file=sys.stderr)
            continue
        shard = structures / pdb_id[1:3]
        shard.mkdir(parents=True, exist_ok=True)
        destination = shard / f"{pdb_id}.cif.gz"
        if not destination.is_file():
            destination.write_bytes(Path(source).read_bytes())
        staged.append(pdb_id)
        if len(staged) % log_every == 0:
            print(f"  [{len(staged)}/{limit} structures staged]", file=sys.stderr)
    (root / "corpus_entries.json").write_text(
        json.dumps({"count": len(staged), "failed": failed,
                    "pdb_ids": sorted(staged)}, indent=2) + "\n")
    return len(staged), failed


def write_target(root, track, entry, payload, index, include_native):
    """Write one context .npz and return its JSONL row."""
    contexts = root / "contexts"
    contexts.mkdir(parents=True, exist_ok=True)
    example_id = f"{track}-{entry['pdb_id']}-{index:04d}"
    arrays = dict(payload["context"])
    chain = payload["chain"]
    if include_native:
        arrays["native_coords"] = chain.coordinates
        arrays["native_mask"] = chain.mask
        arrays["native_restypes"] = np.array(chain.restypes, dtype="<U3")
    path = contexts / f"{example_id}.npz"
    np.savez(path, design_length=np.array([len(chain)]), **arrays)
    return {
        "example_id": example_id,
        "track": track,
        "design_length": len(chain),
        "num_designs": DESIGNS_PER_TARGET,
        "context_path": f"contexts/{path.name}",
        "context_sha256": digest(path),
        "motif_size": int(arrays["motif_coords"].shape[0]) if "motif_coords" in arrays else 0,
        "hotspot_count": int(arrays["hotspot_indices"].size) if "hotspot_indices" in arrays else 0,
        "ligand_atom_count": int(arrays["ligand_coords"].shape[0]) if "ligand_coords" in arrays else 0,
    }


def build_view(entries, root, quotas, length_range, cache, include_native, log_every=25):
    """Download assemblies in order until every track meets its quota."""
    root.mkdir(parents=True, exist_ok=True)
    rows = {track: [] for track in TRACKS}
    per_cluster = collections.Counter()
    attempted = failed = 0
    for entry in entries:
        if all(len(rows[t]) >= quotas.get(t, 0) for t in TRACKS):
            break
        needed = [t for t in TRACKS if len(rows[t]) < quotas.get(t, 0)]
        if not needed:
            break
        if per_cluster[entry["cluster"]] >= MAX_PER_CLUSTER:
            continue
        attempted += 1
        try:
            path = download_assembly(entry["pdb_id"], cache)
            derived = derive_tracks(entry, path, length_range)
        except (OSError, ValueError, urllib.error.URLError) as error:
            failed += 1
            if failed <= 5:
                print(f"  skip {entry['pdb_id']}: {type(error).__name__}: "
                      f"{str(error)[:60]}", file=sys.stderr)
            continue
        used = False
        for track in needed:
            if track in derived:
                rows[track].append(write_target(root, track, entry, derived[track],
                                                len(rows[track]), include_native))
                used = True
        if used:
            per_cluster[entry["cluster"]] += 1
        if attempted % log_every == 0:
            counts = " ".join(f"{t}={len(rows[t])}" for t in TRACKS)
            print(f"  [{attempted} entries] {counts}", file=sys.stderr)

    files, counts = {}, {}
    for track in TRACKS:
        path = root / f"{track}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows[track]))
        files[path.name] = digest(path)
        counts[track] = len(rows[track])
    return files, counts, attempted, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb-index", required=True, type=Path,
                        help="RCSB derived_data/index/entries.idx")
    parser.add_argument("--clusters", required=True, type=Path,
                        help="RCSB clusters-by-entity-30.txt")
    parser.add_argument("--assembly-cache", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--eval-cutoff", default=DEFAULT_EVAL_CUTOFF)
    # Quotas are adjustable so the one-hour smoke profile can build a tiny real dataset.
    parser.add_argument("--eval-targets", type=int, default=EVAL_TARGETS_PER_TRACK)
    parser.add_argument("--tune-targets", type=int, default=TUNE_TARGETS_PER_TRACK)
    parser.add_argument("--fit-targets", type=int, default=FIT_TARGETS_PER_TRACK)
    parser.add_argument("--corpus-targets", type=int, default=CORPUS_STRUCTURES,
                        help="how many fit-split assemblies to stage as the RFD3 "
                             "training corpus; 0 disables it")
    parser.add_argument("--allow-short-eval", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="build only the development views; the evaluation view "
                             "comes from ingest_targets.py using the curated benchmark")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    arguments = parser.parse_args()

    cutoff = datetime.date.fromisoformat(arguments.eval_cutoff)
    entries = load_entry_index(arguments.pdb_index)
    clusters = load_clusters(arguments.clusters)
    eligible, rejected = eligible_entries(entries, clusters)
    evaluation, development, excluded = temporal_split(eligible, cutoff, arguments.seed)

    report = {
        "schema": "protein-design-materialization/v2",
        "source": "rcsb-pdb",
        "sources": {"pdb_index": digest(arguments.pdb_index),
                    "clusters": digest(arguments.clusters)},
        "eval_cutoff": arguments.eval_cutoff,
        "filters": {"max_resolution": MAX_RESOLUTION,
                    "methods": sorted(ALLOWED_METHODS)},
        "entries_total": len(entries),
        "entries_eligible": len(eligible),
        "entries_rejected": dict(rejected),
        "evaluation_pool": len(evaluation),
        "development_pool": len(development),
        "excluded_homologues": excluded,
        "enzyme_headers_in_eval_pool": sum(1 for e in evaluation if e["is_enzyme"]),
        "designs_per_target": DESIGNS_PER_TARGET,
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if arguments.plan_only:
        return
    if not (arguments.output_root and arguments.assembly_cache):
        parser.error("--output-root and --assembly-cache are required for a full run")

    random.Random(arguments.seed).shuffle(evaluation)
    root = arguments.output_root

    if not arguments.skip_evaluation:
        print("\nBuilding evaluation view ...", file=sys.stderr)
        eval_files, eval_counts, _, _ = build_view(
            evaluation, root / "evaluation",
            {t: arguments.eval_targets for t in TRACKS},
            (EVAL_MIN_LENGTH, EVAL_MAX_LENGTH), arguments.assembly_cache,
            include_native=False)
        short = [t for t, n in eval_counts.items() if n < arguments.eval_targets]
        if short and not arguments.allow_short_eval:
            raise SystemExit(
                "Evaluation quota unmet for: " + ", ".join(short) + ". Shipping a "
                "short track would make its reward noisy, so this fails closed. "
                "Loosen --eval-cutoff, relax the filters, or pass --allow-short-eval "
                "and record the shortfall in the task README.")
        write_manifest(root / "evaluation", eval_files, eval_counts, "verifier", report)

    # temporal_split already assigned every development entry to fit or tune by cluster.
    # Honour it: drawing both views from the whole pool put the same PDB entries in both.
    pools = {"fit": [e for e in development if e["split"] == "fit"],
             "tune": [e for e in development if e["split"] == "tune"]}
    for name, quota, length_range in (("fit", arguments.fit_targets, (DEV_MIN_LENGTH, DEV_MAX_LENGTH)),
                                      ("tune", arguments.tune_targets, (EVAL_MIN_LENGTH, EVAL_MAX_LENGTH))):
        print(f"\nBuilding {name} view ...", file=sys.stderr)
        files, counts, _, _ = build_view(
            pools[name], root / "development" / name, {t: quota for t in TRACKS},
            length_range, arguments.assembly_cache, include_native=True)
        write_manifest(root / "development" / name, files, counts, "development", report)

    if arguments.corpus_targets:
        print("\nStaging RFD3 training corpus ...", file=sys.stderr)
        staged, corpus_failed = build_training_corpus(
            pools["fit"], root / "development" / "fit", arguments.assembly_cache,
            arguments.corpus_targets)
        print(f"  staged {staged} structures ({corpus_failed} failed)", file=sys.stderr)
        index_root = root / "development" / "fit"
        print("\nNow build the training index (needs atomworks, which this script does "
              "not import):\n"
              f"    python {Path(__file__).resolve().parent / 'build_training_index.py'} \\\n"
              f"        --structures {index_root / 'structures'} \\\n"
              f"        --clusters {arguments.clusters} \\\n"
              f"        --entry-index {arguments.pdb_index} \\\n"
              f"        --output {index_root / 'index'} --workers $(nproc)",
              file=sys.stderr)

    print("\nMaterialization complete.", file=sys.stderr)


def write_manifest(root, files, counts, view, report):
    manifest = {"schema": "protein-design-input/v1", "view": view,
                "files": files, "counts": counts}
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (root / "provenance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
