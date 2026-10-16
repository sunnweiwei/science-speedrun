"""Materialize public references and label-free inputs (no GPU or credentials)."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import json
from pathlib import Path

import phonopy
import requests
from ase import Atoms
from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedEntry

from common import (
    ELASTIC_FIELDS,
    PHONON_FIELDS,
    PROTOCOL,
    digest,
    encode_atoms,
    indexed,
    write_json,
    write_rows,
)

CATALOG = Path(__file__).with_name("sources.json")
MDR_ROOT = "https://alexandria.icams.rub.de/data/phonon_benchmark/"


def download(url: str, path: Path, expected=None) -> Path:
    if path.exists() and path.stat().st_size > 0:
        if expected and digest(path) != expected:
            raise ValueError(f"Cached source checksum mismatch: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(1 << 20):
                stream.write(chunk)
    if temporary.stat().st_size == 0:
        raise ValueError(f"Empty download: {url}")
    if expected and digest(temporary) != expected:
        raise ValueError(f"Downloaded source checksum mismatch: {url}")
    temporary.replace(path)
    return path


def select_ids(rows, id_key, size, preset, limit):
    if preset == "pilot":
        # Label-blind size strata broaden the smoke while bounding GPU cost.
        # This deliberately differs from the full population's size distribution.
        selected = []
        for lower, upper in ((0, 4), (4, 8), (8, 16), (16, 32)):
            bucket = [r[id_key] for r in rows if lower < size(r) <= upper]
            bucket.sort(
                key=lambda value: hashlib.sha256(
                    f"pilot-v1:{value}".encode()
                ).hexdigest()
            )
            if len(bucket) < limit // 4:
                raise ValueError("Insufficient cases in pilot size stratum")
            selected.extend(bucket[: limit // 4])
        return selected
    eligible = rows if preset == "full" else [r for r in rows if size(r) <= 4]
    # Geometry-size cap is for smoke cost only; it is not an accuracy test panel.
    ids = [r[id_key] for r in eligible]
    if len(ids) != len(set(ids)):
        raise ValueError("Source contains duplicate material IDs")
    ids.sort(key=lambda value: hashlib.sha256(f"smoke-v1:{value}".encode()).hexdigest())
    return ids if preset == "full" else ids[:limit]


def structure_atoms(data):
    return encode_atoms(Structure.from_dict(data).to_ase_atoms())


def phonon_input(path):
    phonon = phonopy.load(path, produce_fc=False)
    cell = phonon.unitcell
    return {
        "atoms": encode_atoms(
            Atoms(
                numbers=cell.numbers,
                positions=cell.positions,
                cell=cell.cell,
                pbc=True,
            )
        ),
        "phonon": {
            "masses": cell.masses.tolist(),
            "supercell_matrix": phonon.supercell_matrix.tolist(),
            "primitive_matrix": (
                phonon.primitive_matrix.tolist()
                if phonon.primitive_matrix is not None
                else None
            ),
        },
    }


def prepare_phonon_row(job):
    """Independent CPU preparation; geometry is kept separate from targets."""
    row, cache = job
    name = f"{row['mp_id']}.yaml.bz2"
    url = MDR_ROOT + "pbe/" + name
    file = download(url, cache / "mdr" / name)
    return (
        {"id": f"phonons/{row['mp_id']}", "suite": "phonons", **phonon_input(file)},
        {
            "id": f"phonons/{row['mp_id']}",
            "suite": "phonons",
            **{key: float(row[key]) for key in PHONON_FIELDS},
        },
        f"mdr/{name}",
        {"url": url, "sha256": digest(file), "bytes": file.stat().st_size},
    )


def prepare(
    cache: Path,
    output: Path,
    suite="all",
    preset="smoke",
    limit=None,
    split_dir=None,
    workers=1,
    panel=None,
):
    if output.exists():
        raise FileExistsError(f"Refusing to mix/overwrite a prepared panel: {output}")
    catalog = json.loads(CATALOG.read_text())
    provenance, inputs, targets, counts = {}, [], [], {}
    split_rows, split_manifest = None, None
    panel_manifest = None
    if panel is not None:
        if preset not in ("val", "test") or limit is not None:
            raise ValueError("Frozen panel requires val/test preset and no limit")
        panel_manifest = json.loads(Path(panel).read_text())
    if preset in ("val", "test"):
        from splits import DEFAULT_SPLITS, load_split

        split_dir = split_dir or DEFAULT_SPLITS
        split_rows, split_manifest = load_split(split_dir)
        if panel_manifest and panel_manifest["parent_split_manifest_sha256"] != digest(split_dir / "manifest.json"):
            raise ValueError("Panel belongs to another split")
        # Reject materialization from sources other than the frozen split's sources.
        for name in ("summary.csv.gz", "mdr.csv", "elastic.json.gz"):
            if catalog[name]["sha256"] != split_manifest["sources"][name]["sha256"]:
                raise ValueError(f"Split/source version mismatch: {name}")

    def choose(rows, name, id_key, size, default_limit):
        if split_rows is None:
            return select_ids(rows, id_key, size, preset, limit or default_limit)
        if len(rows) != split_manifest["summary"][name]["all"]["cases"]:
            raise ValueError(f"Split/source population mismatch: {name}")
        wanted = [
            r["material_id"]
            for r in split_rows
            if r["suite"] == name and r["split"] == preset
        ]
        if not set(wanted) <= {r[id_key] for r in rows}:
            raise ValueError(f"Split IDs absent from source: {name}")
        if panel_manifest:
            ids = panel_manifest["panels"][preset][name]["ids"]
            available = {f"{name}/{key}":key for key in wanted}
            if len(ids) != len(set(ids)) or not set(ids) <= available.keys():
                raise ValueError("Panel IDs escape their frozen partition")
            return [available[key] for key in ids]
        # A bounded development run samples across all atom sizes, without using
        # target labels. Full val/test membership is fixed in the split manifest.
        wanted.sort(
            key=lambda key: hashlib.sha256(
                f"eval-subset-v1:{name}/{key}".encode()
            ).hexdigest()
        )
        return wanted[:limit] if limit else wanted

    def source(name):
        item = catalog[name]
        file = download(item["url"], cache / name, item["sha256"])
        provenance[name] = {**item, "bytes": file.stat().st_size}
        return file

    if suite in ("all", "discovery"):
        with gzip.open(source("summary.csv.gz"), "rt") as stream:
            summary = list(csv.DictReader(stream))
        counts["discovery"] = len(summary)
        ids = choose(
            summary,
            "discovery",
            "material_id",
            lambda r: int(float(r["n_sites"])),
            128 if preset == "pilot" else 8,
        )
        id_set = set(ids)
        selected = {r["material_id"]: r for r in summary if r["material_id"] in id_set}
        entries, structures = {}, {}
        for file, key, result in (
            (source("initial.jsonl.gz"), "initial_structure", structures),
            (source("cse.jsonl.gz"), "computed_structure_entry", entries),
        ):
            with gzip.open(file, "rt") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["material_id"] in selected:
                        result[row["material_id"]] = row[key]
        with gzip.open(source("elements.json.gz"), "rt") as stream:
            refs = {
                el: ComputedEntry.from_dict(entry).energy_per_atom
                for el, entry in json.load(stream).items()
            }
        for material_id in ids:
            row = selected[material_id]
            inputs.append(
                {
                    "id": f"discovery/{material_id}",
                    "suite": "discovery",
                    "atoms": structure_atoms(structures[material_id]),
                }
            )
            targets.append(
                {
                    "id": f"discovery/{material_id}",
                    "suite": "discovery",
                    "entry": entries[material_id],
                    "elemental_refs": refs,
                    "e_form": float(row["e_form_per_atom_mp2020_corrected"]),
                    "e_hull": float(row["e_above_hull_mp2020_corrected_ppd_mp"]),
                    "unique_prototype": row["unique_prototype"].lower() == "true",
                }
            )

    if suite in ("all", "phonons"):
        with source("mdr.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        counts["phonons"] = len(rows)
        ids = set(
            choose(
                rows,
                "phonons",
                "mp_id",
                lambda r: int(r["nsites"]),
                16 if preset == "pilot" else 2,
            )
        )
        jobs = [(r, cache) for r in rows if r["mp_id"] in ids]
        pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        try:
            results = (
                pool.map(prepare_phonon_row, jobs)
                if pool
                else map(prepare_phonon_row, jobs)
            )
            for i, (case, target, name, entry) in enumerate(results, 1):
                inputs.append(case)
                targets.append(target)
                provenance[name] = entry
                if i % 100 == 0 or i == len(jobs):
                    print(f"Prepared phonons: {i}/{len(jobs)}", flush=True)
        finally:
            if pool:
                pool.shutdown(cancel_futures=True)

    if suite in ("all", "elasticity"):
        with gzip.open(source("elastic.json.gz"), "rt") as stream:
            rows = json.load(stream)
        counts["elasticity"] = len(rows)
        ids = set(
            choose(
                rows,
                "elasticity",
                "mp_id",
                lambda r: len(r["structure"]["sites"]),
                16 if preset == "pilot" else 2,
            )
        )
        for row in rows:
            if row["mp_id"] in ids:
                inputs.append(
                    {
                        "id": f"elasticity/{row['mp_id']}",
                        "suite": "elasticity",
                        "atoms": structure_atoms(row["structure"]),
                    }
                )
                targets.append(
                    {
                        "id": f"elasticity/{row['mp_id']}",
                        "suite": "elasticity",
                        **{key: float(row[key]) for key in ELASTIC_FIELDS},
                    }
                )

    inputs.sort(key=lambda row: row["id"])
    targets.sort(key=lambda row: row["id"])
    if set(indexed(inputs)) != set(indexed(targets)):
        raise ValueError("Input/target ID mismatch")
    if split_rows is not None:
        from splits import atoms_key

        split_index = {r["id"]: r for r in split_rows}
        for row in inputs:
            expected = split_index[row["id"]]
            if (
                expected["split"] != preset
                or atoms_key(row["atoms"]) != expected["composition"]
            ):
                raise ValueError(f"Frozen split/geometry mismatch: {row['id']}")
    write_rows(output / "inputs" / "cases.jsonl", inputs)
    write_rows(output / "references" / "targets.jsonl", targets)
    manifest = {
        "protocol": PROTOCOL,
        "preset": preset,
        "source_population_counts": counts,
        "case_ids": [r["id"] for r in inputs],
        "sources": provenance,
        "inputs_sha256": digest(output / "inputs" / "cases.jsonl"),
        "targets_sha256": digest(output / "references" / "targets.jsonl"),
        "leaderboard_comparable": False,
        "selection": "Equal counts in 1–4, 5–8, 9–16, 17–32 atom strata; SHA256 pilot-v1 order; no target labels"
        if preset == "pilot"
        else preset,
        "note": "Public-data evaluator.",
    }
    if split_manifest is not None:
        manifest["split"] = {
            "version": split_manifest["version"],
            "partition": preset,
            "manifest_sha256": digest(split_dir / "manifest.json"),
            "assignments_sha256": split_manifest["files"]["assignments.jsonl.gz"],
            "limit_per_suite": limit,
            "complete_partition": limit is None and panel_manifest is None,
        }
        manifest["selection"] = split_manifest["assignment"]
    if panel_manifest:
        manifest["panel"] = {"version":panel_manifest["version"], "manifest_sha256":digest(Path(panel))}
        manifest["selection"] = panel_manifest["selection"]
    write_json(output / "references" / "manifest.json", manifest)
    print(
        json.dumps(
            {"cases": len(inputs), "population_counts": counts, "preset": preset}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".data/sources"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--panel", type=Path, help="Frozen compact panel manifest (val/test only)"
    )
    parser.add_argument(
        "--suite", choices=["all", "discovery", "phonons", "elasticity"], default="all"
    )
    parser.add_argument(
        "--preset", choices=["smoke", "pilot", "full", "val", "test"], default="smoke"
    )
    parser.add_argument(
        "--split-dir", type=Path, help="Frozen split directory; defaults to HF-materialized .data/splits/v1"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Bound each suite in smoke/val/test; omitted val/test means entire partition",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="CPU processes for phonon preparation"
    )
    args = parser.parse_args()
    if args.limit is not None and (
        args.limit <= 0 or args.preset not in ("smoke", "val", "test")
    ):
        parser.error(
            "--limit must be positive and is only valid with --preset smoke/val/test"
        )
    if args.split_dir and args.preset not in ("val", "test"):
        parser.error("--split-dir requires --preset val/test")
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    prepare(
        args.cache,
        args.output,
        args.suite,
        args.preset,
        args.limit,
        args.split_dir,
        args.workers,
        args.panel,
    )
