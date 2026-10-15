#!/usr/bin/env python3
"""Build an AtomWorks training index over a directory of mmCIF structures.

RFdiffusion3 does not train from structure files directly. It trains from two parquet
tables that say which *example* to pull out of which file, and the structures are
resolved from a path template. This script builds those two tables:

    pn_units_df_train.parquet   one row per PN unit (polymer or non-polymer) per assembly
    interfaces_df_train.parquet one row per contacting pair of PN units per assembly

Everything after that is upstream's: the loader parses the mmCIF, builds the assembly,
crops it, samples a conditioning mode (tipatom / island / unconditional /
sequence_design), decides whether to unindex the motif and featurises it, re-rolled on
every draw. Nothing in this file touches features.

The schema is not ours - it is fixed by three consumers, and each column below exists
because one of them reads it:

  * `atomworks.ml.datasets.parsers.PNUnitsDFParser` / `InterfacesDFParser` need
    `example_id`, `pdb_id`, `assembly_id` and the `*_iid` columns to find the file and
    centre the crop.
  * `atomworks.ml.samplers.calculate_af3_example_weights` asserts on
    `n_prot`, `n_nuc`, `n_ligand`, `cluster_size` (the last is derived by the sampler
    from `cluster`) and additionally reads `n_peptide` and the LOI flag.
  * the dataset configs' `filters` read `deposition_date`, `resolution`, `method`,
    `num_polymer_pn_units`, `cluster` and the `*_non_polymer_res_names` columns.

Runs offline. `DataPreprocessor(from_rcsb=False)` is deliberate: with `from_rcsb=True`
atomworks POSTs to `data.rcsb.org/graphql` for ligand validity scores with no
try/except around the request, so the default raises `ConnectionError` without a
network rather than degrading. The cost of `False` is that `is_loi` ("ligand of
interest" / subject of investigation) is False everywhere, which switches off the
`a_loi` term in the AF-3 weighting - see README.md.

    python build_training_index.py --structures /data/structures \
        --clusters clusters-by-entity-30.txt --entry-index entries.idx \
        --output /data/index --workers 8
"""

import argparse
import collections
import datetime
import json
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path

# Polypeptides at or below this many residues are counted as peptides rather than
# proteins, so that the AF-3 weighting can damp them separately. AF-3's own value,
# mirrored from `atomworks.ml.preprocessing.constants.PEPTIDE_MAX_RESIDUES`; kept as a
# literal so this script states its own contract rather than drifting with the library.
PEPTIDE_MAX_RESIDUES = 20

# `atomworks.enums.ChainType`, by integer value. Duplicated here for the same reason.
PROTEIN_TYPES = {0, 5, 6}          # CYCLIC_PSEUDO_PEPTIDE, POLYPEPTIDE_D, POLYPEPTIDE_L
NUCLEIC_TYPES = {3, 4, 7}          # DNA, DNA_RNA_HYBRID, RNA
NON_POLYMER_TYPES = {8, 10, 11}    # NON_POLYMER, BRANCHED, MACROLIDE
WATER_TYPE = 9

# A single assembly can be pathologically expensive to preprocess (a ribosome or a
# whole capsid runs into minutes while the median entry is ~1.4 s). One slow structure
# must not stall a worker for the whole run, so each is given a wall-clock budget.
DEFAULT_TIMEOUT = 120


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def load_clusters(path):
    """RCSB 30% clusters: one whitespace-separated line of ENTITY ids per cluster.

    Same parse as materialize.py's, deliberately: the fit/evaluation split is made
    cluster-disjoint on these ids, so reusing them as the sampler's `cluster` column is
    what makes the homology balancing agree with the split instead of fighting it.
    """
    mapping = {}
    for index, line in enumerate(Path(path).open(errors="replace")):
        for entity in line.split():
            mapping[entity.strip().upper()] = f"rcsb30:{index}"
    return mapping


def load_entry_metadata(path):
    """Parse RCSB entries.idx for the fields the dataset filters read.

    Needed because an `-assembly1.cif.gz` download carries no entry metadata: its
    `metadata.id` is the literal `xxxx` and deposition date, release date and
    resolution all come back None. Method does survive the file, but taking all four
    from one source keeps them consistent.
    """
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
            # Underscored, as `atomworks` writes it and as the reference parquets store
            # it ("X-RAY_DIFFRACTION"). No dataset filter reads this column, but keeping
            # the spelling identical means a filter on it would behave the same.
            method = (fields[7].strip() or "X-RAY DIFFRACTION").upper().replace(" ", "_")
            entries[pdb_id] = {"deposition_date": deposited, "resolution": resolution,
                               "method": method}
    return entries


def unit_counts(chain_type, n_residues):
    """AF-3 example composition for one PN unit: exactly one of the four is 1.

    Verified against `pn_units_df_train.parquet`: a POLYPEPTIDE_L unit is
    (1, 0, 0, 0) unless it is short, in which case it is (0, 0, 0, 1); NON_POLYMER and
    BRANCHED are (0, 0, 1, 0); DNA/RNA are (0, 1, 0, 0).
    """
    if chain_type in PROTEIN_TYPES:
        if n_residues is not None and n_residues <= PEPTIDE_MAX_RESIDUES:
            return 0, 0, 0, 1
        return 1, 0, 0, 0
    if chain_type in NUCLEIC_TYPES:
        return 0, 1, 0, 0
    if chain_type in NON_POLYMER_TYPES:
        return 0, 0, 1, 0
    return 0, 0, 0, 0


def unit_cluster(record, chain_to_entity, clusters, pdb_id):
    """The `cluster` value for one PN unit.

    Upstream keys polymer clusters by a representative-sequence hash from a 40%-identity
    MMseqs2 clustering, and non-polymer clusters by CCD code. We keep the CCD convention
    exactly and substitute RCSB's 30% entity clusters for the polymer side, because that
    is the partition the task's own split is built on and it is a public, pinned input.
    Where an entity is absent from the cluster file we fall back to the exact-sequence
    hash atomworks computes, which under-clusters (one cluster per distinct sequence)
    but never collides across unrelated sequences.
    """
    chain_type = record.get("q_pn_unit_type")
    if chain_type in NON_POLYMER_TYPES:
        names = record.get("q_pn_unit_non_polymer_res_names") or ""
        return names or None
    entity = chain_to_entity.get(str(record.get("q_pn_unit_iid", "")).split("_")[0])
    if entity is not None:
        key = f"{pdb_id.upper()}_{entity}"
        if key in clusters:
            return clusters[key]
    return record.get("q_pn_unit_processed_entity_canonical_sequence_hash") or None


def records_for_structure(path, pdb_id, timeout=DEFAULT_TIMEOUT):
    """Preprocess one structure into PN unit records plus a chain -> entity map.

    Calls the two halves of `DataPreprocessor.get_rows` directly rather than the method
    itself, for two reasons: it reuses the single parse to read `chain_info`, which
    `get_rows` computes and discards but which holds the `rcsb_entity` ids the cluster
    join needs; and it skips the RCSB ligand-score request unconditionally.
    """
    from atomworks.ml.preprocessing.get_pn_unit_data_from_structure import DataPreprocessor

    previous = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        preprocessor = DataPreprocessor(from_rcsb=False)
        parsed = preprocessor._load_structure_with_atomworks(path)
        preprocessor.id = pdb_id
        chain_to_entity = {
            chain: info.get("rcsb_entity")
            for chain, info in (parsed.get("chain_info") or {}).items()
            if isinstance(info, dict)
        }
        records = []
        for assembly_id in parsed["assemblies"]:
            rows = preprocessor._generate_pn_unit_metadata_for_assembly(parsed, assembly_id, None)
            if rows:
                for row in rows:
                    row["path"] = str(path)
                records.extend(rows)
        return records, chain_to_entity
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def rows_from_records(records, chain_to_entity, clusters, meta, pdb_id):
    """Turn one structure's PN unit records into pn_unit and interface index rows."""
    from atomworks.ml.example_id import generate_example_id

    pn_rows, if_rows = [], []
    by_assembly = collections.defaultdict(list)
    for record in records:
        by_assembly[str(record.get("assembly_id"))].append(record)

    for assembly_id, group in by_assembly.items():
        units = {}
        for record in group:
            iid = record.get("q_pn_unit_iid")
            if iid is None:
                continue
            # atomworks hands these back as numpy str_; `generate_example_id` formats
            # the list with repr, so a numpy scalar would be written into the id as
            # "np.str_('A_1')" and `parse_example_id`'s eval() would then fail on it.
            iid = str(iid)
            chain_type = record.get("q_pn_unit_type")
            if chain_type == WATER_TYPE:
                continue
            n_res = record.get("q_pn_unit_num_resolved_residues")
            units[iid] = {
                "record": record,
                "type": chain_type,
                "counts": unit_counts(chain_type, n_res),
                "cluster": unit_cluster(record, chain_to_entity, clusters, pdb_id),
                "non_polymer": record.get("q_pn_unit_non_polymer_res_names") or None,
            }

        common = {
            "pdb_id": pdb_id,
            "assembly_id": assembly_id,
            "deposition_date": meta.get("deposition_date"),
            "resolution": meta.get("resolution"),
            "method": meta.get("method"),
        }

        for iid, unit in units.items():
            record = unit["record"]
            n_prot, n_nuc, n_ligand, n_peptide = unit["counts"]
            pn_rows.append({
                **common,
                "example_id": generate_example_id(["pdb", "pn_units"], pdb_id,
                                                  assembly_id, [iid]),
                "num_polymer_pn_units": record.get("num_polymer_pn_units"),
                "total_num_atoms_in_unprocessed_assembly":
                    record.get("total_num_atoms_in_unprocessed_assembly"),
                "cluster": unit["cluster"],
                "n_prot": n_prot, "n_nuc": n_nuc,
                "n_ligand": n_ligand, "n_peptide": n_peptide,
                "q_pn_unit_iid": iid,
                "q_pn_unit_non_polymer_res_names": unit["non_polymer"],
                "all_pn_unit_iids_after_processing":
                    record.get("all_pn_unit_iids_after_processing"),
                "q_pn_unit_is_loi": bool(record.get("q_pn_unit_is_loi", False)),
                "clash_severity": record.get("clash_severity"),
            })

        # An interface is an unordered pair of contacting units. Each side reports its
        # own partners, so every pair is seen twice; key on the sorted pair to keep one.
        seen = set()
        for iid, unit in units.items():
            try:
                partners = json.loads(unit["record"].get("q_pn_unit_contacting_pn_unit_iids") or "[]")
            except (TypeError, ValueError):
                partners = []
            for partner in partners:
                other = partner.get("pn_unit_iid") if isinstance(partner, dict) else partner
                other = str(other) if other is not None else None
                if other not in units:
                    continue
                pair = tuple(sorted((iid, other)))
                if pair in seen:
                    continue
                seen.add(pair)
                first, second = units[pair[0]], units[pair[1]]
                counts = [a + b for a, b in zip(first["counts"], second["counts"])]
                # Upstream keys an interface cluster on BOTH sides joined by '+', so
                # that a pair is balanced as a pair. Sorted, so the key does not depend
                # on which side happened to be listed first.
                if first["cluster"] is None or second["cluster"] is None:
                    cluster = None
                else:
                    cluster = "+".join(sorted((str(first["cluster"]), str(second["cluster"]))))
                if_rows.append({
                    **common,
                    "example_id": generate_example_id(["pdb", "interfaces"], pdb_id,
                                                      assembly_id, list(pair)),
                    "num_polymer_pn_units": first["record"].get("num_polymer_pn_units"),
                    "cluster": cluster,
                    "n_prot": counts[0], "n_nuc": counts[1],
                    "n_ligand": counts[2], "n_peptide": counts[3],
                    "pn_unit_1_iid": pair[0],
                    "pn_unit_2_iid": pair[1],
                    "pn_unit_1_non_polymer_res_names": first["non_polymer"],
                    "pn_unit_2_non_polymer_res_names": second["non_polymer"],
                    # Two units are always distinct molecules here: a PN unit is the
                    # covalent unit, so a contact between two of them is intermolecular.
                    "is_inter_molecule": True,
                    "all_pn_unit_iids_after_processing":
                        first["record"].get("all_pn_unit_iids_after_processing"),
                    "involves_loi": bool(first["record"].get("q_pn_unit_is_loi", False)
                                         or second["record"].get("q_pn_unit_is_loi", False)),
                })
    return pn_rows, if_rows


_WORKER = {}


def _init_worker(timeout):
    _WORKER["timeout"] = timeout
    # Each worker is one process on one core; the transform libraries default to
    # thread pools that would oversubscribe an 8-core box several times over.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"


def _process(path):
    """Worker half: parse one structure. Deliberately does NOT build the index rows.

    Row assembly needs the 30% cluster map, which is ~1.6M entries and costs ~2 GiB
    once materialised as a dict. Passing it through `initargs` gives every worker its
    own copy - 8 workers on the agent's 8-CPU box would spend ~16 GiB on nothing but
    lookup tables. The parse is what is actually expensive, so workers parse and the
    parent (which holds the map once) assembles.
    """
    path = Path(path)
    pdb_id = path.name.split("-")[0].split(".")[0].lower()
    try:
        records, chain_to_entity = records_for_structure(path, pdb_id, _WORKER["timeout"])
    except _Timeout:
        return pdb_id, None, None, "timeout"
    except Exception as error:  # noqa: BLE001 - one bad structure must not end the run
        return pdb_id, None, None, f"{type(error).__name__}: {str(error)[:120]}"
    return pdb_id, records, chain_to_entity, None


def find_structures(root):
    """Accept either a flat directory or the sharded `<xx>/<pdbid>.cif.gz` layout."""
    root = Path(root)
    found = sorted(root.rglob("*.cif.gz")) + sorted(root.rglob("*.cif"))
    return [path for path in found if path.is_file()]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--structures", required=True, type=Path,
                        help="directory of mmCIF files, flat or sharded")
    parser.add_argument("--clusters", type=Path,
                        help="RCSB clusters-by-entity-30.txt; without it every polymer "
                             "cluster falls back to an exact-sequence hash")
    parser.add_argument("--entry-index", type=Path,
                        help="RCSB entries.idx, for deposition date / resolution / method")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="per-structure wall-clock budget in seconds")
    parser.add_argument("--limit", type=int, help="process only the first N structures")
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    import pandas as pd

    structures = find_structures(args.structures)
    if args.limit:
        structures = structures[:args.limit]
    if not structures:
        sys.exit(f"no .cif/.cif.gz files under {args.structures}")

    clusters = load_clusters(args.clusters) if args.clusters else {}
    entries = load_entry_metadata(args.entry_index) if args.entry_index else {}
    print(f"structures {len(structures)}  clustered entities {len(clusters)}  "
          f"indexed entries {len(entries)}  workers {args.workers}", file=sys.stderr)

    pn_rows, if_rows, failures = [], [], collections.Counter()
    started = time.time()
    with multiprocessing.Pool(args.workers, initializer=_init_worker,
                              initargs=(args.timeout,)) as pool:
        for done, (pdb_id, records, chain_to_entity, error) in enumerate(
                pool.imap_unordered(_process, structures, chunksize=4), start=1):
            if error:
                failures[error.split(":")[0]] += 1
            else:
                try:
                    pn, interfaces = rows_from_records(
                        records, chain_to_entity, clusters,
                        entries.get(pdb_id, {}), pdb_id)
                    pn_rows.extend(pn)
                    if_rows.extend(interfaces)
                except Exception as error:  # noqa: BLE001
                    failures[type(error).__name__] += 1
            if done % args.log_every == 0:
                rate = done / max(1e-9, time.time() - started)
                print(f"  [{done}/{len(structures)}] {rate:.1f} structures/s  "
                      f"pn_units={len(pn_rows)} interfaces={len(if_rows)} "
                      f"failed={sum(failures.values())}", file=sys.stderr)

    args.output.mkdir(parents=True, exist_ok=True)
    pn_frame = pd.DataFrame(pn_rows)
    if_frame = pd.DataFrame(if_rows)
    for frame in (pn_frame, if_frame):
        if not frame.empty:
            frame["deposition_date"] = pd.to_datetime(frame["deposition_date"])
    pn_path = args.output / "pn_units_df_train.parquet"
    if_path = args.output / "interfaces_df_train.parquet"
    pn_frame.to_parquet(pn_path, index=False)
    if_frame.to_parquet(if_path, index=False)

    elapsed = time.time() - started
    report = {
        "structures_seen": len(structures),
        "structures_failed": sum(failures.values()),
        "failure_kinds": dict(failures),
        "pn_unit_rows": len(pn_frame),
        "interface_rows": len(if_frame),
        "pn_units_without_cluster": int(pn_frame["cluster"].isna().sum()) if not pn_frame.empty else 0,
        "interfaces_without_cluster": int(if_frame["cluster"].isna().sum()) if not if_frame.empty else 0,
        "elapsed_seconds": round(elapsed, 1),
        "seconds_per_structure": round(elapsed / max(1, len(structures)), 3),
        "workers": args.workers,
    }
    (args.output / "index_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), file=sys.stderr)
    print(f"wrote {pn_path} and {if_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
