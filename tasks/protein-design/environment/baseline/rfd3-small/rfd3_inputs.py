#!/usr/bin/env python
"""Turn any protein-design input view into RFdiffusion3 inference specs.

The task hands out a request as a `<track>.jsonl` row plus an `.npz` conditioning
context. Upstream RFD3 wants an `InputSpecification`: an input structure, a contig, and
selections over that structure. This script writes both - a conditioning structure
rebuilt from the context, and the spec that points at it.

It works on **fit and tune**, which is what an agent actually has: `docker-compose.yaml`
mounts only `development`, so `/data/fit` and `/data/tune` are the whole world inside the
container. That is the difference between this and `environment/evaluation/benchmark/make_inputs.py`, which
reuses upstream's published benchmark specs and therefore only works on the evaluation
view the verifier holds.

    python rfd3_inputs.py --view /data/tune --out /workspace/research/inputs

Then design with upstream inference, and convert back with `rfd3_collect.py`:

    rfd3 design out_dir=... inputs=.../enzyme.json ckpt_path=... \
        n_batches=25 diffusion_batch_size=4 prevalidate_inputs=True
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from residues import ATOM14  # noqa: E402

# Chains in the conditioning structure we write. The designed chain is generated, so it
# is not in this file at all; these hold only what the request fixes.
MOTIF_CHAIN = "A"
TARGET_CHAIN = "B"
# Each ligand needs its OWN chain: atomworks refuses two ligand residues on one chain
# ("Multiple ligand residues on chain L ... set 'allow_ligand_on_existing_chain: true'
# to restore the old behaviour"). 29 of the 41 enzyme targets carry 2 or 3 ligands, so
# hand out successive letters from L, skipping the motif and target chains.
LIGAND_CHAINS = [c for c in "LMNOPQRSTUVWXYZ" if c not in (MOTIF_CHAIN, TARGET_CHAIN)]


def ligand_atom_names(code, elements):
    """Name a ligand's atoms by embedding its element sequence into the CCD's.

    The context stores ligand coordinates and elements but no atom names, and RFD3 needs
    names that match the chemical component so bonds and the reference conformer resolve.
    Materialization keeps deposition order, which follows the CCD, so for a fully
    resolved ligand the names are just the CCD's heavy atoms in order (checked: 22 of 24
    sampled ligands match element-for-element).

    The other two had *fewer* atoms than the CCD - unresolved density, not a reordering -
    so positional assignment would slip after the first gap. Embedding the observed
    element sequence as a subsequence of the CCD's, earliest match first, handles both.
    Returns None if no embedding exists, so the caller can skip the target rather than
    invent chemistry.
    """
    from biotite.structure.info import residue as ccd_residue

    try:
        component = ccd_residue(code)
    except Exception:  # noqa: BLE001 - an unknown code is a skip, not a crash
        return None
    heavy = [(str(atom.atom_name), str(atom.element).upper())
             for atom in component if str(atom.element).upper() != "H"]
    names, cursor = [], 0
    for element in (str(e).upper() for e in elements):
        while cursor < len(heavy) and heavy[cursor][1] != element:
            cursor += 1
        if cursor == len(heavy):
            return None
        names.append(heavy[cursor][0])
        cursor += 1
    return names


def split_ligands(context):
    """Un-concatenate a context's ligands into
    (code, names, elements, coordinates, pinned) tuples.

    `ingest_targets.py` stores ALL of a target's ligands in one flat array each -
    `ligand_coords` and `ligand_elements` concatenated, with `ligand_ccd` and
    `ligand_atom_counts` saying where the boundaries are. 29 of the 41 enzyme targets
    carry more than one ligand (a cofactor plus a substrate, often plus a metal:
    M0040_13pk is MG + ADP + 3PG), so reading `ligand_ccd[0]` and handing it the whole
    element array asks the CCD for a component that explains every ligand at once. That
    never embeds, so the target was silently skipped - which cost 29 of 41 enzyme
    targets, i.e. the agent-facing path could reach only 12 of them.

    Returns None if any ligand's names cannot be resolved, so the caller still skips
    rather than inventing chemistry.
    """
    codes = [str(code) for code in context["ligand_ccd"]]
    counts = [int(n) for n in context["ligand_atom_counts"]]
    elements = [str(e) for e in context["ligand_elements"]]
    coordinates = context["ligand_coords"]
    # Prefer the deposition names the context carries. Reconstructing them from the CCD
    # by element subsequence works but is a guess, and it cannot recover WHICH atoms
    # upstream pins - see `ligand_fixed_mask` below. Older contexts lack both keys, so
    # the CCD path stays as the fallback.
    supplied = ([str(n) for n in context["ligand_atom_names"]]
                if "ligand_atom_names" in context else None)
    pinned = (context["ligand_fixed_mask"].astype(bool).tolist()
              if "ligand_fixed_mask" in context else None)
    ligands, offset = [], 0
    for code, count in zip(codes, counts):
        chunk = elements[offset:offset + count]
        names = (supplied[offset:offset + count] if supplied is not None
                 else ligand_atom_names(code, chunk))
        if names is None:
            return None
        keep = (pinned[offset:offset + count] if pinned is not None
                else [True] * len(names))
        ligands.append((code, names, chunk, coordinates[offset:offset + count], keep))
        offset += count
    return ligands


class PDBWriter:
    """Minimal PDB writer - only the columns atomworks reads back."""

    def __init__(self):
        self.lines = []
        self.serial = 1

    def residue(self, chain, number, name, coordinates, mask, record="ATOM  "):
        """One amino acid, written in atom14 slot order."""
        for slot, atom in enumerate(ATOM14[name]):
            if not mask[slot]:
                continue
            self.atom(chain, number, name, atom, coordinates[slot], atom[0], record)

    def atom(self, chain, number, residue_name, atom_name, position, element,
             record="ATOM  "):
        x, y, z = (float(value) for value in position)
        # PDB column rule: a one- or two-letter atom name starts in column 14, so it is
        # padded with a leading space. Four-character names (none in atom14) start at 13.
        field = atom_name if len(atom_name) >= 4 else f" {atom_name:<3s}"
        self.lines.append(
            f"{record}{self.serial:>5d} {field} {residue_name:>3s} "
            f"{chain}{number:>4d}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {element:>2s}")
        self.serial += 1

    def write(self, path):
        path.write_text("\n".join(self.lines + ["END"]) + "\n")


def fixed_atom_names(name, mask):
    """The atom14 atoms a context actually carries for one residue."""
    return [atom for slot, atom in enumerate(ATOM14[name]) if mask[slot]]


def build_enzyme(context, length, structure_path):
    """Motif geometry with no positions: upstream's unindexed / guidepost setting.

    The request says which catalytic residues exist and where they sit relative to each
    other and the ligand, but not which design positions they occupy - that is the
    model's decision, and `motif_map` reports it back. So every motif residue goes into
    `unindex`, and `select_fixed_atoms` pins exactly the atoms the context supplies,
    which for this benchmark is a few tip atoms per residue rather than a backbone.
    """
    writer = PDBWriter()
    types = [str(name) for name in context["motif_restypes"]]
    masks = context["motif_mask"].astype(bool)
    fixed = {}
    for index, name in enumerate(types):
        number = index + 1
        writer.residue(MOTIF_CHAIN, number, name, context["motif_coords"][index],
                       masks[index])
        fixed[f"{MOTIF_CHAIN}{number}"] = ",".join(fixed_atom_names(name, masks[index]))

    # Every ligand, each as its own residue with its own CCD code. Upstream's published
    # specs for these same targets use exactly this shape - `"ligand": "ADP,MG,3PG"`
    # with one select_fixed_atoms key per code (see data/targets/enzyme/input.yaml).
    ligands = split_ligands(context)
    if ligands is None:
        return None
    for chain, (code, names, elements, coordinates, keep) in zip(LIGAND_CHAINS, ligands):
        for atom, position, element in zip(names, coordinates, elements):
            writer.atom(chain, 1, code, atom, position, element, record="HETATM")
        # Only the atoms upstream's spec pins, not every atom present. Every atom is
        # still written to the PDB - the ligand's geometry is the conditioning - but
        # pinning all of them posed a different, more constrained problem than the
        # published benchmark does.
        selected = [atom for atom, wanted in zip(names, keep) if wanted]
        fixed[code] = ",".join(selected or names)

    writer.write(structure_path)
    return {"dialect": 2, "input": str(structure_path),
            "ligand": ",".join(ligand[0] for ligand in ligands),
            "unindex": ",".join(f"{MOTIF_CHAIN}{i + 1}" for i in range(len(types))),
            "length": f"{length}-{length}", "select_fixed_atoms": fixed}


def build_sm_binder(context, length, structure_path):
    """A bare ligand: build a chain around it, with no pocket template."""
    writer = PDBWriter()
    ligands = split_ligands(context)
    if ligands is None:
        return None
    for chain, (code, names, elements, coordinates, _) in zip(LIGAND_CHAINS, ligands):
        for atom, position, element in zip(names, coordinates, elements):
            writer.atom(chain, 1, code, atom, position, element, record="HETATM")
    writer.write(structure_path)
    # All four shipped sm_binder targets are single-ligand, but the concatenation is the
    # same, so this goes through split_ligands too rather than keeping a second code path
    # that would break the first time a multi-ligand sm target appears.
    return {"dialect": 2, "input": str(structure_path),
            "ligand": ",".join(ligand[0] for ligand in ligands),
            "contig": f"{length}-{length}", "select_fixed_atoms": True}


def build_ppi_binder(context, length, structure_path):
    """A fixed target chain plus hotspots the binder should contact.

    The contig is `<designed>,/0,<target range>`: `/0` is upstream's chain break, so the
    designed chain and the target are separate chains rather than one fused polymer.
    """
    writer = PDBWriter()
    types = [str(name) for name in context["target_restypes"]]
    masks = context["target_mask"].astype(bool)
    # Gap-preserving residue numbers where the context carries them: a break in the
    # target's numbering is real structure, and numbering it `index + 1` would assert
    # that residues either side of an unresolved stretch are adjacent. The development
    # views have no such gaps to preserve, so they number sequentially.
    resnums = ([int(n) for n in context["target_resnums"]]
               if "target_resnums" in context else list(range(1, len(types) + 1)))
    for index, name in enumerate(types):
        writer.residue(TARGET_CHAIN, resnums[index], name,
                       context["target_coords"][index], masks[index])
    writer.write(structure_path)

    # One contig range per contiguous run, so the spec states the same range structure
    # `input.yaml` does (`B0-47,B53-64,B104-165` -> `B1-48,B54-65,B105-166`).
    ranges, start = [], 0
    for index in range(1, len(resnums) + 1):
        if index == len(resnums) or resnums[index] != resnums[index - 1] + 1:
            ranges.append(f"{TARGET_CHAIN}{resnums[start]}-{resnums[index - 1]}")
            start = index
    target_contig = ",".join(ranges)

    # Atom-level hotspots where the context carries them, matching `input.yaml`, which
    # names exactly two sidechain atoms per hotspot. A contig string is accepted too,
    # but `canonicalize_` expands it to EVERY atom of the residue, so the per-atom
    # `is_atom_level_hotspot` feature would be set on ~8-11 atoms instead of 2. The
    # development views built by materialize.py derive hotspots from coordinates and
    # have no published atom selection, so they fall back to the token-level form.
    if "hotspot_atoms" in context:
        hotspots = {f"{TARGET_CHAIN}{resnums[int(i)]}": str(atoms)
                    for i, atoms in zip(context["hotspot_indices"],
                                        context["hotspot_atoms"])}
    else:
        hotspots = ",".join(f"{TARGET_CHAIN}{resnums[int(i)]}"
                            for i in context["hotspot_indices"])
    return {"dialect": 2, "input": str(structure_path),
            "contig": f"{length}-{length},/0,{target_contig}",
            "select_fixed_atoms": True, "select_hotspots": hotspots,
            # NOT optional. With both `ori_token` and `infer_ori_strategy` unset,
            # `set_com` (`rfd3/utils/inference.py`) falls through to "COM of fixed
            # motif"; here the whole target chain IS the fixed motif, so the origin
            # becomes the TARGET's centre of mass and the binder is diffused from
            # inside it. Measured on a run that omitted this: median minimum
            # binder-target atom distance 0.25-0.45 A on all five targets, binder COM
            # 2.4-6.7 A from the target COM, ipTM 0.19 and pass rate 0/500 - the track
            # was not measuring the model at all. `protein_binder_design.md` recommends
            # "hotspots", which puts the origin 10 A along the outward normal from the
            # hotspot centroid. `benchmark/standard/make_specs.py` has always set this;
            # the graded path did not, which is the whole discrepancy on this track.
            "infer_ori_strategy": "hotspots"}


BUILDERS = {"enzyme": build_enzyme, "sm_binder": build_sm_binder,
            "ppi_binder": build_ppi_binder}


def build(view, out, tracks):
    out.mkdir(parents=True, exist_ok=True)
    structures = out / "conditioning"
    structures.mkdir(exist_ok=True)
    counts, skipped = {}, {}

    for track in tracks:
        if track != "unconditional" and track not in BUILDERS:
            raise SystemExit(f"unknown track {track!r}; expected one of "
                             f"unconditional, {', '.join(sorted(BUILDERS))}")
        source = view / f"{track}.jsonl"
        if not source.is_file():
            continue
        rows = [json.loads(line) for line in source.read_text().splitlines()
                if line.strip()]
        if not rows:
            continue
        specs = {}
        for request in rows:
            example = request["example_id"]
            length = int(request["design_length"])
            if track == "unconditional":
                # A length is the whole request - no structure, nothing to condition on.
                specs[example] = {"length": f"{length}-{length}"}
                continue
            context = dict(np.load(view / request["context_path"], allow_pickle=False))
            spec = BUILDERS[track](context, length, structures / f"{example}.pdb")
            if spec is None:
                skipped[track] = skipped.get(track, 0) + 1
                continue
            specs[example] = spec
        if specs:
            (out / f"{track}.json").write_text(json.dumps(specs, indent=2) + "\n")
            counts[track] = len(specs)
    return counts, skipped


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", required=True, type=Path,
                        help="an input view directory: /data/fit or /data/tune")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tracks", nargs="*",
                        default=["unconditional", "sm_binder", "ppi_binder", "enzyme"])
    arguments = parser.parse_args()
    counts, skipped = build(arguments.view, arguments.out, set(arguments.tracks))
    for track, number in sorted(counts.items()):
        note = f"  ({skipped[track]} skipped)" if track in skipped else ""
        print(f"{track:15s} {number:3d} targets -> "
              f"{arguments.out / (track + '.json')}{note}")
    if not counts:
        raise SystemExit(f"no targets found under {arguments.view}")


if __name__ == "__main__":
    main()
