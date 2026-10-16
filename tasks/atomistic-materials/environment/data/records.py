"""MP-compatible label conversion and protected full-release assembly."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import requests
from ase import units
from pymatgen.core import Structure

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from common import decode_atoms, digest, encode_atoms, read_rows, write_rows
from splits import DEFAULT_SPLITS, composition_key, load_split

MPTRJ_URL = "https://ndownloader.figshare.com/files/41619375"
SALEX_URL = "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/sAlex/train.tar.gz"
REF_URL = "https://huggingface.co/datasets/facebook/OMAT24/resolve/ca00b8aac581497cd0a13146fb3708ad8e2e2fe8/references/element-references.yaml"


def formula(atoms):
    return Structure.from_ase_atoms(atoms).composition.reduced_formula


def mptrj_record(parent, frame, row):
    atoms = Structure.from_dict(row["structure"]).to_ase_atoms()
    if not np.isclose(
        row["corrected_total_energy"] / len(atoms),
        row["energy_per_atom"],
        atol=1e-7,
        rtol=0,
    ):
        raise ValueError("MPTrj corrected/per-atom energy identity failed")
    stress = np.asarray(row["stress"], dtype=float)
    # VASP kbar, positive compression -> ASE eV/A^3, positive tension.
    stress = -0.1 * units.GPa * (stress + stress.T) / 2
    return dict(
        id=f"mptrj/{frame}",
        parent=parent,
        source="mptrj",
        atoms=encode_atoms(atoms),
        energy=float(row["uncorrected_total_energy"]),
        forces=row["force"],
        stress=stress.tolist(),
        audit={
            "raw_energy_eV": row["uncorrected_total_energy"],
            "corrected_energy_eV": row["corrected_total_energy"],
            "corrected_energy_per_atom_eV": row["energy_per_atom"],
            "stress_source_kbar": row["stress"],
        },
    )


def validate(row):
    atoms = decode_atoms(row["atoms"])
    for name, shape in [("forces", (len(atoms), 3)), ("stress", (3, 3))]:
        value = np.asarray(row[name])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid {name}: {row['id']}")
    if not np.isfinite(row["energy"]):
        raise ValueError("Invalid energy")
    return formula(atoms)


def key_hash(value):
    # Preserve the released record order, originally named by the timing pilot.
    return hashlib.sha256(f"adaptation-timing-100k-v1:{value}".encode()).hexdigest()


def protected_inventory():
    rows, _ = load_split(DEFAULT_SPLITS)
    return {r["composition"] for r in rows}, {r["material_id"] for r in rows}


def structure_key(structure):
    amounts = Counter()
    for site in structure["sites"]:
        for species in site["species"]:
            amounts[species["element"]] += species["occu"]
    return composition_key(amounts)


def assemble(root, *, quotas):
    root = Path(root)
    output = root / "data"
    if output.exists():
        raise FileExistsError(output)
    blocked, blocked_ids = protected_inventory()
    sources = {
        source: json.loads((root / "sources" / source / "manifest.json").read_text())
        for source in ("mptrj", "salex")
    }
    groups, parents, ids, statistics = {}, {}, {}, {}
    for split in quotas:
        rows = []
        for source in sources:
            path = root / "sources" / source / f"{split}.jsonl"
            if digest(path) != sources[source]["files"][path.name]:
                raise ValueError("Source subset checksum mismatch")
            selected = read_rows(path)
            if len(selected) != quotas[split]:
                raise ValueError("Source quota mismatch")
            rows.extend(selected)
        ids[split] = {r["id"] for r in rows}
        groups[split] = {r["composition_group"] for r in rows}
        parents[split] = {(r["source"], r["parent"]) for r in rows}
        if len(ids[split]) != len(rows) or groups[split] & blocked:
            raise ValueError("Duplicate data or evaluation composition overlap")
        if any(r["parent"] in blocked_ids for r in rows):
            raise ValueError("Evaluation parent overlap")
        sizes = [len(decode_atoms(r["atoms"])) for r in rows]
        statistics[split] = dict(
            frames=len(rows),
            unique_parents=len(parents[split]),
            unique_compositions=len(groups[split]),
            total_atoms=sum(sizes),
            atom_quantiles=dict(
                zip(
                    ["min", "median", "p95", "max"],
                    np.quantile(sizes, [0, 0.5, 0.95, 1]).tolist(),
                )
            ),
        )
        write_rows(
            output / f"{split}.jsonl", sorted(rows, key=lambda r: key_hash(r["id"]))
        )
    if (
        ids["train"] & ids["val"]
        or groups["train"] & groups["val"]
        or parents["train"] & parents["val"]
    ):
        raise ValueError("Training/development overlap")
    response = requests.get(REF_URL, timeout=60)
    response.raise_for_status()
    (output / "element-references.yaml").write_bytes(response.content)
    manifest = dict(
        sources=sources,
        files={f"{split}.jsonl": digest(output / f"{split}.jsonl") for split in quotas},
        counts={
            f"{split}/{source}": n for split, n in quotas.items() for source in sources
        },
        statistics=statistics,
        protected_compositions=len(blocked),
        split_manifest_sha256=digest(DEFAULT_SPLITS / "manifest.json"),
        energy_convention="mp-pbe-raw-total",
        corrections_applied=False,
        element_reference_resource={
            "url": REF_URL,
            "sha256": digest(output / "element-references.yaml"),
        },
        overlap_checks={
            "all_eval_compositions_excluded": True,
            "all_eval_parent_ids_excluded": True,
            "train_dev_composition_disjoint": True,
            "unique_frames": True,
        },
    )
    return manifest
