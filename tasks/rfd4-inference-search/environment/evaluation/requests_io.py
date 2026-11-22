"""Create versioned requests for conditioned four-track design and length-only monomers."""
import json
from pathlib import Path
import shutil

import gemmi
import zstandard

from policy import LENGTHS

# Balanced 4-track development panel (3 targets per track = 12 targets) for rapid multi-track iteration.
DEV_4TRACK_TARGETS = {
    "ppi_backbone": ("01_PD1", "15_DerF7", "rfd3_pdl1"),
    "studio_figure_2c_unfixed": ("BODIPY_final_0", "biotin_final_0", "retinal_final_0"),
    "mcsa_fixed": ("M0097_1ctt", "M0157_1qh5", "M0630_1j79"),
    "dna_unfixed": ("7m5w", "7n5u", "7rte"),
}
FOUR_TRACK_DATASETS = ("ppi_backbone", "studio_figure_2c_unfixed", "mcsa_fixed", "dna_unfixed")


def conditioned_suite_requests(path):
    """Load a versioned four-track target roster supplied by the controller."""
    suite = json.loads(Path(path).read_text())
    if not isinstance(suite, dict) or suite.get("version") != 1:
        raise ValueError("Expected a version 1 conditioned suite")
    targets = suite.get("targets")
    if not isinstance(targets, dict) or set(targets) != set(FOUR_TRACK_DATASETS):
        raise ValueError("Conditioned suite must specify target lists for all four Figure 2 datasets")
    roster = {}
    for ds in FOUR_TRACK_DATASETS:
        items = targets[ds]
        if not isinstance(items, list) or not items:
            raise ValueError(f"Expected a nonempty target list for {ds}")
        seen = set()
        cleaned = []
        for tgt in items:
            if not isinstance(tgt, str) or not tgt.strip() or tgt in seen:
                raise ValueError(f"Invalid or duplicate target id in {ds}")
            seen.add(tgt)
            cleaned.append(tgt)
        roster[ds] = cleaned
    return roster


def suite_requests(path):
    suite = json.loads(Path(path).read_text())
    if not isinstance(suite, dict) or suite.get("version") != 1:
        raise ValueError("Expected a version 1 suite")
    requests = suite.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("Expected a nonempty list of requests")
    by_length = {}
    ids = set()
    for item in requests:
        if not isinstance(item, dict) or item.get("kind") != "unconditional_monomer":
            raise ValueError("No evaluator for the requested kind")
        payload = item.get("input")
        length = payload.get("length") if isinstance(payload, dict) else None
        if not isinstance(payload, dict) or set(payload) != {"length"}:
            raise ValueError("The unconditional monomer grader accepts a length-only input")
        if type(length) is not int or length < 1 or length in by_length:
            raise ValueError("The monomer suite requires distinct positive integer lengths")
        request_id = item.get("id", f"monomer-{length}")
        if not isinstance(request_id, str) or not request_id or request_id in ids:
            raise ValueError("Each request must have a unique nonempty string id")
        if set(item) - {"kind", "input", "id", "metadata"}:
            raise ValueError("Unknown suite request fields")
        ids.add(request_id)
        by_length[length] = dict(item, id=request_id)
    return by_length


def write_monomer_request(directory, assets, length, slots, seed, seconds, descriptor=None):
    if type(length) is not int or length < 1 or type(slots) is not int or slots < 1:
        raise ValueError("length and output count must be positive integers")
    directory.mkdir(parents=True, exist_ok=False)
    source = assets / "monomers"
    target = directory / f"monomer-{LENGTHS.get(length, length)}.cif.zst"
    if length in LENGTHS:
        shutil.copyfile(source / f"monomer-{LENGTHS[length]}.cif.zst", target)
    else:
        with (source / "monomer-short.cif.zst").open("rb") as f:
            with zstandard.ZstdDecompressor().stream_reader(f) as reader:
                doc = gemmi.cif.read_string(reader.read().decode())
        block = doc.sole_block()
        segment = f"{length}-{length}"
        for tag in ("_atom_site.label_comp_id", "_atom_site.auth_comp_id", "_entity_poly_seq.mon_id",
                    "_chem_comp.id", "_chem_comp.name", "_rfd_design.sampled_contig", "_chem_comp_atom.comp_id"):
            if block.find_value(tag) != "100-100":
                raise ValueError(f"Unexpected monomer template field {tag}")
            block.set_pair(tag, segment)
        for tag in ("_atom_site.annotation_expsegmin_1_residue", "_atom_site.annotation_expsegmax_1_residue"):
            block.set_pair(tag, str(length))
        block.name = f"monomer-{length}"
        block.set_pair("_entry.id", f"monomer-{length}")
        target.write_bytes(zstandard.ZstdCompressor().compress(doc.as_string().encode()))
    request = {"id": f"monomer-{length}", **(descriptor or {}), "version": 1, "kind": "unconditional_monomer",
               "input": {"length": length, "file": target.name}, "num_outputs": slots,
               "seed": seed, "budget": {"wall_seconds": seconds}}
    path = directory / "request.json"
    path.write_text(json.dumps(request, indent=2) + "\n")
    return path


def write_conditioned_request(directory, input_cif, request_id, slots, seed, seconds, metadata=None):
    """Write a track-agnostic conditioned biomolecular design request for any of the four Figure 2 suites."""
    if type(slots) is not int or slots < 1:
        raise ValueError("output count must be a positive integer")
    input_cif = Path(input_cif).resolve()
    if not input_cif.is_file():
        raise FileNotFoundError(f"Missing conditioned benchmark input: {input_cif}")
    directory.mkdir(parents=True, exist_ok=False)
    staged = directory / input_cif.name
    shutil.copyfile(input_cif, staged)
    request = {
        "version": 1,
        "id": str(request_id),
        "kind": "conditioned_design",
        "input": {"file": staged.name},
        "num_outputs": slots,
        "seed": int(seed),
        "budget": {"wall_seconds": float(seconds)},
    }
    if metadata is not None:
        request["metadata"] = metadata
    path = directory / "request.json"
    path.write_text(json.dumps(request, indent=2) + "\n")
    return path
