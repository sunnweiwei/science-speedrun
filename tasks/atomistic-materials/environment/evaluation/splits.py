"""Freeze composition-grouped validation/test IDs across the three public suites.

Only identifiers, compositions and geometry sizes enter assignment. Public targets
remain evaluation data; this is an AI4Sci split, not an upstream leaderboard split.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from fractions import Fraction
import gzip
import hashlib
import json
from math import gcd, lcm
from pathlib import Path

import numpy as np
from pymatgen.core import Composition

from common import decode_atoms, digest, read_rows, write_json

VERSION = "atomistic-eval-split-v1"
DEFAULT_SPLITS = Path(__file__).resolve().parents[2] / ".data/splits/v1"
SUITES = ("discovery", "phonons", "elasticity")
PHONON_GEOMETRIES = {
    "url": "https://huggingface.co/datasets/materialyze/matcalc-bench/resolve/54b2b336cccca933f51c82f47f2dbeed7bab054f/alexandria-pbe-phonon-2025.3.json.gz",
    "sha256": "edbd5e2e638197f2c3863e990a51ca27d4c151f9f67b95f55348006a8f8e36b8",
}


def composition_key(value):
    """Canonical integer element ratios, including dimers and supercells."""
    amounts = Composition(value).get_el_amt_dict()
    fractions = {
        k: Fraction(str(v)).limit_denominator(1000000) for k, v in amounts.items()
    }
    denominator = lcm(*(x.denominator for x in fractions.values()))
    integers = {k: int(v * denominator) for k, v in fractions.items()}
    factor = gcd(*integers.values())
    if not integers or factor <= 0 or any(x <= 0 for x in integers.values()):
        raise ValueError("Invalid composition")
    return "|".join(f"{k}:{integers[k] // factor}" for k in sorted(integers))


def atoms_key(data):
    return composition_key(Counter(decode_atoms(data).get_chemical_symbols()))


def stable_hash(value, seed=VERSION):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def source_inventory(cache):
    # Imported here to keep manifest readers independent of Phonopy/downloads.
    from prepare import CATALOG, MDR_ROOT, download, phonon_input

    catalog = json.loads(CATALOG.read_text())
    provenance = {}

    def source(name, item=None):
        item = catalog[name] if item is None else item
        path = download(item["url"], cache / name, item["sha256"])
        provenance[name] = {**item, "bytes": path.stat().st_size}
        return path

    with gzip.open(source("summary.csv.gz"), "rt") as stream:
        wbm = list(csv.DictReader(stream))
    with source("mdr.csv").open() as stream:
        mdr = list(csv.DictReader(stream))
    with gzip.open(source("elastic.json.gz"), "rt") as stream:
        elastic = json.load(stream)
    with gzip.open(
        source("phonon-structures.json.gz", PHONON_GEOMETRIES), "rt"
    ) as stream:
        mirrored = json.load(stream)
    phonon_formulas = {}
    for row in mirrored:
        if row["mp_id"] in phonon_formulas:
            raise ValueError("Duplicate MP ID in phonon geometry mirror")
        # Derive composition from occupancies, not the formatting of formula strings.
        amounts = Counter()
        for site in row["structure"]["sites"]:
            for species in site["species"]:
                amounts[species["element"]] += species["occu"]
        phonon_formulas[row["mp_id"]] = composition_key(amounts)

    missing = sorted({r["mp_id"] for r in mdr} - phonon_formulas.keys())

    def missing_formula(mp_id):
        name = f"mdr/{mp_id}.yaml.bz2"
        url = MDR_ROOT + f"pbe/{mp_id}.yaml.bz2"
        path = download(url, cache / name)
        return (
            mp_id,
            atoms_key(phonon_input(path)["atoms"]),
            name,
            {"url": url, "sha256": digest(path), "bytes": path.stat().st_size},
        )

    # The geometry mirror has 9,865/9,958 cases; retain the remaining 93 by
    # reading their original PBE YAMLs. Never silently shrink the evaluation pool.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for mp_id, key, name, entry in pool.map(missing_formula, missing):
            phonon_formulas[mp_id] = key
            provenance[name] = entry

    records = []
    for suite, rows in (("discovery", wbm), ("phonons", mdr), ("elasticity", elastic)):
        for row in rows:
            material_id = row["material_id"] if suite == "discovery" else row["mp_id"]
            key = (
                phonon_formulas[material_id]
                if suite == "phonons"
                else composition_key(row["formula"])
            )
            n_sites = (
                int(float(row["n_sites"]))
                if suite == "discovery"
                else int(row["nsites"])
                if suite == "phonons"
                else len(row["structure"]["sites"])
            )
            records.append(
                {
                    "id": f"{suite}/{material_id}",
                    "suite": suite,
                    "material_id": material_id,
                    "composition": key,
                    "n_sites": n_sites,
                }
            )
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate evaluation case")
    return sorted(records, key=lambda r: r["id"]), provenance


def collect_exposure(panel_paths, adaptation_paths):
    """Store only metadata needed to keep previously used groups out of test."""
    groups, ids, sources = set(), set(), []
    for kind, paths in (
        ("prior_evaluation", panel_paths),
        ("prior_adaptation", adaptation_paths),
    ):
        for path in paths:
            rows = read_rows(path)
            sources.append(
                {
                    "kind": kind,
                    "name": path.name,
                    "sha256": digest(path),
                    "rows": len(rows),
                }
            )
            for row in rows:
                groups.add(atoms_key(row["atoms"]))
                value = (
                    row["id"].split("/")[-1]
                    if kind == "prior_evaluation"
                    else row.get("parent", "")
                )
                if value:
                    ids.add(value)
    return {
        "sources": sources,
        "compositions": sorted(groups),
        "material_ids": sorted(ids),
    }


def assign(records, exposure, val_fraction=0.2, seed=VERSION):
    if not 0 < val_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Empty or duplicate evaluation inventory")
    # Union composition groups when an MP ID has different stoichiometric metadata
    # across releases. This guarantees shared material IDs never cross splits.
    parent = {r["composition"]: r["composition"] for r in records}

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    by_id = {}
    for row in sorted(records, key=lambda r: r["id"]):
        key = row["composition"]
        previous = by_id.setdefault(row["material_id"], key)
        a, b = root(key), root(previous)
        parent[max(a, b)] = min(a, b)
    forced = {root(x) for x in exposure["compositions"] if x in parent}
    forced.update(root(by_id[x]) for x in exposure["material_ids"] if x in by_id)
    result = []
    for row in sorted(records, key=lambda r: r["id"]):
        group = root(row["composition"])
        is_val = (
            group in forced or int(stable_hash(group, seed), 16) / 2**256 < val_fraction
        )
        result.append({**row, "group": group, "split": "val" if is_val else "test"})
    return result, forced


def validate_assignment(records, exposure):
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Empty or duplicate split IDs")
    if any(
        r["split"] not in ("val", "test") or r["suite"] not in SUITES for r in records
    ):
        raise ValueError("Unknown split or suite")
    for field in ("composition", "group", "material_id"):
        sides = defaultdict(set)
        for row in records:
            sides[row[field]].add(row["split"])
        if any(len(x) != 1 for x in sides.values()):
            raise ValueError(f"Validation/test overlap in {field}")
    test = [r for r in records if r["split"] == "test"]
    if set(exposure["compositions"]) & {r["composition"] for r in test}:
        raise ValueError("Previously exposed composition in test")
    if set(exposure["material_ids"]) & {r["material_id"] for r in test}:
        raise ValueError("Previously exposed material ID in test")


def describe(records):
    summary = {}
    for suite in SUITES:
        summary[suite] = {}
        for part in ("all", "val", "test"):
            rows = [
                r
                for r in records
                if r["suite"] == suite and (part == "all" or r["split"] == part)
            ]
            sizes = [r["n_sites"] for r in rows]
            summary[suite][part] = {
                "cases": len(rows),
                "composition_groups": len({r["group"] for r in rows}),
                "n_sites_median": float(np.median(sizes)),
                "n_sites_p95": float(np.quantile(sizes, 0.95)),
                "n_sites_max": max(sizes),
                "fraction_above_32_sites": sum(n > 32 for n in sizes) / len(sizes),
            }
    return summary


def load_split(directory=DEFAULT_SPLITS):
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Split checksum mismatch: {name}")
    with gzip.open(directory / "assignments.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    exposure = json.loads((directory / "exposure.json").read_text())
    validate_assignment(rows, exposure)
    counts = Counter(r["suite"] for r in rows)
    if counts != {s: x["all"]["cases"] for s, x in manifest["summary"].items()}:
        raise ValueError("Split population counts mismatch")
    return rows, manifest


def build(
    cache,
    output,
    panel_paths=(),
    adaptation_paths=(),
    exposure_path=None,
    val_fraction=0.2,
):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen split: {output}")
    records, provenance = source_inventory(cache)
    exposure = (
        json.loads(exposure_path.read_text())
        if exposure_path
        else collect_exposure(panel_paths, adaptation_paths)
    )
    assigned, forced = assign(records, exposure, val_fraction)
    validate_assignment(assigned, exposure)
    output.mkdir(parents=True)
    payload = "".join(
        json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in assigned
    ).encode()
    (output / "assignments.jsonl.gz").write_bytes(gzip.compress(payload, mtime=0))
    write_json(output / "exposure.json", exposure)
    manifest = {
        "version": VERSION,
        "seed": VERSION,
        "target_validation_fraction": val_fraction,
        "assignment": "SHA256 of cross-suite composition/MP-ID group; exposed groups forced to validation",
        "target_labels_used_for_assignment": False,
        "forced_validation_groups": len(forced),
        "summary": describe(assigned),
        "sources": provenance,
        "files": {
            n: digest(output / n) for n in ("assignments.jsonl.gz", "exposure.json")
        },
        "leaderboard_comparable": False,
        "limitations": [
            "Custom AI4Sci partition of public evaluation data, not an official upstream split",
            "20% is a group-hash target; case fractions differ and prior exposure increases validation",
            "Previously used adaptation materials can occur in validation; test excludes their groups",
            "No guarantee of independence from UMA foundation pretraining or public elemental references",
            "Freeze future training exclusions against both evaluation partitions before benchmark launch",
            "Only IDs/metadata are materialized here; full geometry/reference preparation is on demand",
        ],
    }
    write_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "summary": manifest["summary"],
                "forced_validation_groups": len(forced),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-panels", type=Path, nargs="*", default=[])
    parser.add_argument("--prior-adaptation", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--exposure",
        type=Path,
        help="Reuse a frozen exposure inventory for exact rebuilding",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if args.exposure and (args.prior_panels or args.prior_adaptation):
        parser.error("Use --exposure or prior source paths, not both")
    build(
        args.cache,
        args.output,
        args.prior_panels,
        args.prior_adaptation,
        args.exposure,
        args.val_fraction,
    )
