#!/usr/bin/env python3
"""Convert the curated RFdiffusion3 benchmark targets into the task's input view.

The evaluation set is RFdiffusion3's own benchmark: M-CSA-derived enzyme active sites,
small-molecule binder sites, and the canonical binder targets (insulin receptor, PD-L1,
IL-7Ra, IL-2Ra, TIE2). Using them keeps scores comparable to published numbers.

Each source `.cif` holds only the motif residues and the ligand, already in the
conditioning frame. `select_fixed_atoms` names the atoms that are actually constrained;
those become the context's `motif_mask`, so the constraint is expressed at atom level
exactly as RFdiffusion3 poses it.

    python ingest_targets.py --output-root OUT          # uses the vendored targets
    python ingest_targets.py --results-root DIR --output-root OUT

Requires `gemmi` (structure parsing) and `pyyaml` (the ppi target spec) on the host.

`unconditional` is the one track with no upstream target, because RFdiffusion3's
benchmark has none: a request there is a LENGTH and nothing else. Its specs are
synthesised from UNCONDITIONAL_TARGETS below rather than read from a `.cif`, which is
the same thing `benchmark/standard/make_specs.py` does for the operator harness - both
paths end up emitting `{"length": "100-100"}`, so the graded and published numbers stay
comparable on this track too. It was an empty file until 2026-09-28; that contradicted
`instruction.md`, which tells the agent to serve all four tracks in the mounted data,
and `materialize.py`, whose practice view has always populated it.
"""

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline" / "rfd3-small"))
from residues import ATOM14, ATOM14_WIDTH, atom_slot  # noqa: E402

TRACKS = ("unconditional", "sm_binder", "ppi_binder", "enzyme")
# 20 designs per target (RFdiffusion3's protocol is 100): the pass rate is a per-target fraction, so it
# estimates the same quantity, and grading fits ~2 h on eight GPUs instead of ~9 h. Must not exceed
# tests/common.py::MAX_DESIGNS_PER_TARGET, and must match materialize.py's copy of this
# constant or the evaluation and practice views disagree on the sample size.
DESIGNS_PER_TARGET = 20
# `unconditional` has no upstream target to ingest, so the request is a bare design
# length. Mirrors the frozen operator definition, `benchmark/standard/targets.json`
# ["tracks"]["unconditional"]["targets"], and must keep matching it or the graded and
# published unconditional numbers stop being comparable.
UNCONDITIONAL_TARGETS = {"uncond_L100": 100}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_specs(results_root):
    """Parse the three target specifications, each in its own near-JSON dialect."""
    root = Path(results_root)
    enzyme = json.loads((root / "enzyme" / "input.yaml").read_text())
    # The small-molecule file lists entries without separating commas.
    raw = (root / "sm" / "input.yaml").read_text().strip().rstrip(",")
    small = json.loads("{" + re.sub(r"\}\s*\n\s*\"", '},\n"', raw) + "}")
    # Only the ppi file is real YAML; the other two parse as JSON dialects. Report the
    # missing dependency by name rather than letting a bare ImportError surface here.
    try:
        import yaml
    except ImportError:                                     # pragma: no cover
        raise SystemExit("pyyaml is required to read the ppi targets: "
                         "pip install pyyaml")
    binder = yaml.safe_load((root / "ppi" / "input.yaml").read_text())
    return enzyme, small, binder


def parse_length(spec):
    """'180-180' or '40-120,/0,E6-155' -> a single design length (range midpoint)."""
    head = str(spec).split(",")[0]
    match = re.match(r"^(\d+)-(\d+)$", head)
    if not match:
        raise ValueError(f"unparsable design length: {spec}")
    low, high = int(match.group(1)), int(match.group(2))
    return (low + high) // 2, (low, high)


def read_structure(path):
    """Protein residues in atom14 form plus ligand atoms, keyed by (chain, seqid)."""
    import gemmi

    structure = gemmi.read_structure(str(path))
    structure.remove_alternative_conformations()
    structure.remove_hydrogens()
    structure.setup_entities()
    tabulated = gemmi.find_tabulated_residue

    residues, ligands = collections.OrderedDict(), []
    for chain in structure[0]:
        for residue in chain:
            info = tabulated(residue.name)
            is_protein = info is not None and info.is_amino_acid()
            if is_protein and residue.name in ATOM14:
                coordinates = np.zeros((ATOM14_WIDTH, 3))
                present = np.zeros(ATOM14_WIDTH, dtype=bool)
                names = {}
                for atom in residue:
                    slot = atom_slot(residue.name, atom.name)
                    if slot is None:
                        continue
                    if not np.isfinite([atom.pos.x, atom.pos.y, atom.pos.z]).all():
                        continue
                    coordinates[slot] = [atom.pos.x, atom.pos.y, atom.pos.z]
                    present[slot] = True
                    names[atom.name] = slot
                residues[(chain.name, residue.seqid.num)] = {
                    "name": residue.name, "coordinates": coordinates,
                    "present": present, "slots": names}
            elif not residue.is_water():
                # Reached by anything that is not a standard residue we can put in
                # atom14 - INCLUDING a residue gemmi calls an amino acid that atom14
                # has no row for. The condition used to be `not is_protein`, which sent
                # those to neither branch and dropped them silently: M0110_1c0p lost
                # DAL and M0717_1x7d lost ORN, both ligands upstream pins
                # (`select_fixed_atoms` names 5 DAL atoms; the graded context had none).
                # A non-standard amino acid sitting in one of these files is a substrate,
                # not backbone to scaffold, so the ligand branch is where it belongs.
                #
                # Real entries carry unresolved atoms: BCA in M0024_1nzy has four
                # phosphate atoms with no coordinates. Drop them rather than emit a
                # context the grader must reject as non-finite.
                kept = [a for a in residue
                        if np.isfinite([a.pos.x, a.pos.y, a.pos.z]).all()]
                positions = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in kept])
                if positions.size:
                    ligands.append({
                        "name": residue.name, "coordinates": positions,
                        # RFD3 conditions on ref_element; a ligand's chemistry is not
                        # recoverable from coordinates, so it has to travel with them.
                        "elements": [a.element.name.upper() for a in kept],
                        "charges": [float(getattr(a, "charge", 0) or 0) for a in kept],
                        # Deposition atom names, so `select_fixed_atoms` can be carried
                        # through verbatim instead of guessed back from the CCD.
                        "atom_names": [a.name for a in kept],
                    })
    return residues, ligands


def build_motif(residues, fixed_atoms):
    """Assemble the motif context, masking to the atoms actually constrained."""
    coordinates, mask, restypes, order = [], [], [], []
    for key, atoms in fixed_atoms.items():
        match = re.match(r"^([A-Za-z])(-?\d+)$", key)
        if not match:                      # a ligand entry, handled separately
            continue
        chain, number = match.group(1), int(match.group(2))
        residue = residues.get((chain, number))
        if residue is None:
            continue
        selected = np.zeros(ATOM14_WIDTH, dtype=bool)
        for name in str(atoms).split(","):
            slot = residue["slots"].get(name.strip())
            if slot is not None:
                selected[slot] = True
        if not selected.any():
            continue
        coordinates.append(residue["coordinates"])
        mask.append(selected)
        restypes.append(residue["name"])
        order.append(key)
    if not coordinates:
        return None
    return (np.stack(coordinates), np.stack(mask),
            np.array(restypes, dtype="<U3"), order)


def ingest(results_root, output_root):
    enzyme_spec, small_spec, binder_spec = load_specs(results_root)
    results_root, output_root = Path(results_root), Path(output_root)
    contexts = output_root / "contexts"
    contexts.mkdir(parents=True, exist_ok=True)
    rows = {track: [] for track in TRACKS}
    skipped = collections.Counter()

    def emit(track, name, length, arrays):
        path = contexts / f"{name}.npz"
        np.savez(path, design_length=np.array([length]), **arrays)
        rows[track].append({
            "example_id": name, "track": track, "design_length": length,
            "num_designs": DESIGNS_PER_TARGET,
            "context_path": f"contexts/{path.name}",
            "context_sha256": digest(path),
            "motif_size": int(arrays["motif_coords"].shape[0]) if "motif_coords" in arrays else 0,
            "hotspot_count": int(arrays["hotspot_indices"].size) if "hotspot_indices" in arrays else 0,
            "ligand_atom_count": int(arrays["ligand_coords"].shape[0]) if "ligand_coords" in arrays else 0,
        })

    for source, track, folder in ((enzyme_spec, "enzyme", "enzyme"),
                                  (small_spec, "sm_binder", "sm")):
        for name, spec in source.items():
            path = results_root / folder / spec["input"]
            if not path.is_file():
                skipped[f"{track}:missing_cif"] += 1
                continue
            residues, ligands = read_structure(path)
            length, _ = parse_length(spec.get("length") or spec.get("contig"))
            fixed = spec.get("select_fixed_atoms")
            arrays = {}
            if isinstance(fixed, dict):
                motif = build_motif(residues, fixed)
                if motif is None:
                    skipped[f"{track}:no_motif"] += 1
                    continue
                arrays.update(motif_coords=motif[0], motif_mask=motif[1],
                              motif_restypes=motif[2])
            elif residues:
                # Some targets fix every residue present in the file.
                coordinates = np.stack([r["coordinates"] for r in residues.values()])
                mask = np.stack([r["present"] for r in residues.values()])
                names = np.array([r["name"] for r in residues.values()], dtype="<U3")
                arrays.update(motif_coords=coordinates, motif_mask=mask,
                              motif_restypes=names)
            if ligands:
                arrays["ligand_coords"] = np.concatenate(
                    [l["coordinates"] for l in ligands])
                # A complex predictor needs the ligand's identity, not just its
                # coordinates, to model the holo complex at all.
                arrays["ligand_ccd"] = np.array(
                    [l["name"] for l in ligands], dtype="<U5")
                arrays["ligand_atom_counts"] = np.array(
                    [l["coordinates"].shape[0] for l in ligands], dtype=np.int32)
                arrays["ligand_elements"] = np.array(
                    [e for l in ligands for e in l["elements"]], dtype="<U2")
                arrays["ligand_charges"] = np.array(
                    [c for l in ligands for c in l["charges"]], dtype=np.float32)
                arrays["ligand_atom_names"] = np.array(
                    [n for l in ligands for n in l["atom_names"]], dtype="<U6")
                # `select_fixed_atoms` names the ligand atoms RFdiffusion3 actually pins,
                # and carrying only coordinates threw that away: rfd3_inputs.py then
                # rebuilt the full CCD atom list and pinned EVERY atom, which on 12 of the
                # 41 enzyme targets is 2.21x what upstream pins (up to 3.67x). The motif's
                # half of the same `fixed` dict was already honoured, via motif_mask - the
                # ligand's half was the asymmetry. Matched by NAME, as the spec expresses
                # it, so repeated copies of one code share a selection.
                #
                # PER LIGAND, not unioned. The first version built one `wanted` set over
                # every ligand and tested all atoms against it, so an atom name pinned
                # for one code was pinned on every other code that happened to use the
                # same name - and PDB ligand atom names collide constantly (C1', PA, O2B
                # are nucleotide-wide). That left 16 of 41 enzyme targets disagreeing
                # with upstream: on M0092_1dli, NAD pins 4 atoms (O2D,C2D,C3D,C1D) but
                # UDX's 25 names bled in and pinned 14, while UDX itself gained O2D for
                # 26 of 25. It is the residual behind "1.06x upstream's atom count".
                if isinstance(fixed, dict):
                    mask, any_entry = [], False
                    for ligand in ligands:
                        entry = fixed.get(ligand["name"])
                        want = ({a.strip() for a in entry.split(",") if a.strip()}
                                if isinstance(entry, str) else set())
                        any_entry = any_entry or bool(want)
                        mask.extend(n in want for n in ligand["atom_names"])
                    if any_entry:
                        arrays["ligand_fixed_mask"] = np.array(mask, dtype=bool)
            # An sm_binder target is a bare ligand: RFdiffusion3 designs a protein
            # around it with no pocket template, so a motif is not required there.
            # An enzyme target is the reverse: the catalytic motif is the whole point.
            if "ligand_coords" not in arrays:
                skipped[f"{track}:no_ligand"] += 1
                continue
            if track == "enzyme" and "motif_coords" not in arrays:
                skipped[f"{track}:no_motif"] += 1
                continue
            emit(track, name, length, arrays)

    for name, spec in binder_spec.items():
        path = results_root / "ppi" / spec["input"]
        if not path.is_file():
            skipped["ppi:missing_cif"] += 1
            continue
        residues, _ = read_structure(path)
        if not residues:
            skipped["ppi:no_target"] += 1
            continue
        length, _ = parse_length(spec["contig"])
        keys = list(residues)
        coordinates = np.stack([residues[k]["coordinates"] for k in keys])
        mask = np.stack([residues[k]["present"] for k in keys])
        index_of = {k: i for i, k in enumerate(keys)}
        # Local residue numbers that PRESERVE the published numbering gaps. Shifting so
        # the first target residue is 1 hides the absolute PDB ids, but collapsing to
        # `index + 1` also erased the gaps, and a gap is real structure: rfd3_il2ra's
        # target is `B0-47,B53-64,B104-165`, three ranges with breaks at 48-52 and
        # 65-103, which contiguous numbering asserts are adjacent. The other four ppi
        # targets are gapless, so their numbering is unchanged by this. enzyme and
        # sm_binder are unaffected either way - upstream `unindex`es their motif
        # residues, so numbering carries no adjacency there.
        offset = 1 - min(number for _, number in keys)
        resnums = np.array([number + offset for _, number in keys], dtype=np.int32)
        # `select_hotspots` names two SIDECHAIN ATOMS per hotspot, not a residue:
        # `{"A56": "CG,OH", ...}`. Iterating the dict yields keys only, so the atom
        # names were being thrown away and rfd3_inputs.py rebuilt a token-level
        # selection - which `canonicalize_` expands to EVERY atom of the residue, so
        # `is_atom_level_hotspot` was set on ~8-11 atoms where upstream sets exactly 2.
        # Same shape of bug as the ligand half of `select_fixed_atoms`: a dict whose
        # values carried the specification. Carry the atom names, keyed to the sorted
        # index order so the two arrays stay aligned.
        hotspots = []
        for hotspot, atoms in (spec.get("select_hotspots") or {}).items():
            match = re.match(r"^([A-Za-z])(-?\d+)$", str(hotspot))
            if match:
                key = (match.group(1), int(match.group(2)))
                if key in index_of:
                    # Only the atoms actually resolved into atom14, or the selection
                    # would name an atom absent from the conditioning structure and
                    # silently select nothing at all.
                    present = [a.strip() for a in str(atoms).split(",")
                               if a.strip() in residues[key]["slots"]]
                    if present:
                        hotspots.append((index_of[key], ",".join(present)))
        if not hotspots:
            skipped["ppi:no_hotspots"] += 1
            continue
        emit("ppi_binder", name, length,
             {"target_coords": coordinates, "target_mask": mask,
              # The target's sequence is needed to fold the binder-target complex.
              "target_restypes": np.array([residues[k]["name"] for k in keys],
                                          dtype="<U3"),
              "hotspot_indices": np.array([i for i, _ in sorted(hotspots)],
                                          dtype=np.int32),
              # Parallel to hotspot_indices: the atoms `input.yaml` names for each.
              "hotspot_atoms": np.array([a for _, a in sorted(hotspots)],
                                        dtype="<U32"),
              # Parallel to target_coords: gap-preserving local residue numbers.
              "target_resnums": resnums})

    # No structure to read and nothing to condition on: the context holds the design
    # length alone, which is what `tests/common.py` demands of this track (it rejects an
    # unconditional row carrying any motif, hotspot or ligand) and all
    # `rfd3_inputs.py` reads back for it.
    for name, length in UNCONDITIONAL_TARGETS.items():
        emit("unconditional", name, length, {})

    files, counts = {}, {}
    for track in TRACKS:
        path = output_root / f"{track}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows[track]))
        files[path.name] = digest(path)
        counts[track] = len(rows[track])
    (output_root / "manifest.json").write_text(json.dumps(
        {"schema": "protein-design-input/v1", "view": "verifier",
         "files": files, "counts": counts}, indent=2, sort_keys=True) + "\n")
    return counts, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path,
                        default=Path(__file__).resolve().parent / "targets",
                        help="benchmark target directory; defaults to the copy "
                             "vendored in this repository")
    parser.add_argument("--output-root", required=True, type=Path)
    arguments = parser.parse_args()
    counts, skipped = ingest(arguments.results_root, arguments.output_root)
    print(json.dumps({"counts": counts, "skipped": dict(skipped)},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
