#!/usr/bin/env python
"""Convert RFdiffusion3 inference output into a protein-design submission.

Pairs with `rfd3_inputs.py`: that builds the specs, `rfd3 design` runs upstream
inference, this turns the result back into what the grader reads. View-agnostic -
it works on fit, tune or the evaluation view alike.

RFD3 writes one gzipped mmCIF per design, holding everything it was given: the designed
chain, the fixed target chain for a binder, and the ligand. The task's grader wants the
opposite - one PDB per design containing ONLY the designed chain, exactly
`design_length` residues, every atom name valid for its residue in the atom14 table
(`tests/geometry.py::parse_allatom_pdb`) - plus a `<track>.jsonl` index.

For the enzyme track it also wants `motif_map`: which design position each motif residue
landed on. That is the whole point of the unindexed setting - the model chooses - so the
mapping is recovered from the output by matching the motif's fixed atoms, which RFD3
holds at noise level zero and therefore reproduces exactly.
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

# The starter's `residues.py` is asserted byte-identical to the grader's
# (tests/test_task.py::test_starter_and_grader_share_one_atom14_table), so the
# atom14 table used to write a submission is the one used to read it back.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from residues import ATOM14, ATOM14_WIDTH, atom_slot          # noqa: E402

BACKBONE = (0, 1, 2, 3)


def read_cif(path):
    """Parse one RFD3 output into per-chain residue lists.

    Returns {chain: [(residue_number, residue_name, {atom_name: (x, y, z)}), ...]}.
    """
    from biotite.structure.io.pdbx import CIFFile, get_structure
    with gzip.open(path, "rt") as handle:
        structure = get_structure(CIFFile.read(handle), model=1)
    chains = {}
    for atom in structure:
        if atom.element == "H":
            continue
        key = str(atom.chain_id)
        residues = chains.setdefault(key, {})
        entry = residues.setdefault(int(atom.res_id), [str(atom.res_name), {}])
        entry[1][str(atom.atom_name)] = tuple(float(value) for value in atom.coord)
    return {chain: [(number, *residues[number]) for number in sorted(residues)]
            for chain, residues in chains.items()}


def design_chain(chains, design_length):
    """The designed chain is the amino-acid chain with exactly `design_length` residues.

    Cheap and unambiguous: a binder's target chain is a different length by construction,
    and the ligand is not an amino-acid chain at all.
    """
    candidates = [(chain, residues) for chain, residues in chains.items()
                  if len(residues) == design_length
                  and all(name in ATOM14 for _, name, _ in residues)]
    if len(candidates) != 1:
        raise ValueError(f"expected one chain of {design_length} residues, "
                         f"found {[(c, len(r)) for c, r in chains.items()]}")
    return candidates[0][1]


def to_atom14(residues):
    """[L, 14, 3] coordinates and an [L, 14] mask, dropping atoms atom14 has no slot for."""
    length = len(residues)
    coordinates = np.zeros((length, ATOM14_WIDTH, 3))
    mask = np.zeros((length, ATOM14_WIDTH), dtype=bool)
    names = []
    for index, (_, residue_name, atoms) in enumerate(residues):
        names.append(residue_name)
        for atom, position in atoms.items():
            slot = atom_slot(residue_name, atom)
            if slot is None:                 # OXT, virtual slots, anything non-atom14
                continue
            coordinates[index, slot] = position
            mask[index, slot] = True
    return coordinates, mask, names


def recover_motif_map(metadata, names, context, coordinates, mask):
    """Which design residue each motif position landed on.

    Upstream reports this itself: the output metadata carries `diffused_index_map`,
    input residue -> output residue, which for an unindexed motif is precisely the
    placement decision the model made. The context lists its motif residues in the same
    order as the spec's `unindex` field, and the result is checked against the residue
    identities before it is trusted; if anything disagrees, fall back to matching the
    motif's own atoms in the output.
    """
    motif_types = [str(name) for name in context["motif_restypes"]]
    index_map = metadata.get("diffused_index_map") or {}
    unindexed = str(metadata.get("specification", {}).get("unindex", "")).split(",")
    unindexed = [item.strip() for item in unindexed if item.strip()]
    if len(unindexed) == len(motif_types) and index_map:
        mapping = []
        for source in unindexed:
            target = lookup(index_map, source)
            position = int("".join(c for c in str(target) if c.isdigit())) - 1 if target else -1
            mapping.append(position)
        if (all(0 <= position < len(names) for position in mapping)
                and len(set(mapping)) == len(mapping)
                and [names[position] for position in mapping] == motif_types):
            return mapping
    return match_motif_by_coordinates(coordinates, mask, names, context)


def lookup(index_map, source):
    """`diffused_index_map` keyed by input residue, tolerating upstream's truncation.

    Upstream writes four-digit residue ids short - a motif residue `B1029` comes back as
    the key `B102` - so an exact lookup silently misses and the design gets dropped.
    Fall back to a prefix match, and only when it is unambiguous.
    """
    if source in index_map:
        return index_map[source]
    hits = [value for key, value in index_map.items()
            if source.startswith(key) or key.startswith(source)]
    return hits[0] if len(hits) == 1 else None


def match_motif_by_coordinates(coordinates, mask, names, context):
    """Fallback placement: the closest residue of the right identity, greedily assigned."""
    motif = context["motif_coords"]
    motif_mask = context["motif_mask"].astype(bool)
    motif_types = [str(name) for name in context["motif_restypes"]]
    scores = np.full((len(motif_types), len(names)), np.inf)
    for i, residue_name in enumerate(motif_types):
        shared = motif_mask[i][None, :] & mask                  # [L, 14]
        for j in range(len(names)):
            # A motif position carries only the atoms the context kept - for these
            # benchmarks that is the 2-3 fixed tip atoms, NOT a whole backbone - so the
            # overlap is whatever both sides have, and one atom is too few to place on.
            if names[j] != residue_name or shared[j].sum() < 2:
                continue
            selected = shared[j]
            difference = coordinates[j][selected] - motif[i][selected]
            scores[i, j] = float(np.sqrt((difference ** 2).sum(axis=-1).mean()))
    assignment, taken = [None] * len(motif_types), set()
    for i in np.argsort(scores.min(axis=1)):
        order = np.argsort(scores[i])
        choice = next((int(j) for j in order
                       if int(j) not in taken and np.isfinite(scores[i, j])), None)
        if choice is None:
            return None
        assignment[int(i)] = choice
        taken.add(choice)
    return assignment


def write_pdb(path, coordinates, mask, names, chain="A"):
    """Canonical atom14 PDB, the same columns `tests/geometry.py` parses."""
    lines, serial = [], 1
    for index, residue_name in enumerate(names):
        for slot, atom in enumerate(ATOM14[residue_name]):
            if not mask[index, slot]:
                continue
            x, y, z = coordinates[index, slot]
            lines.append(f"ATOM  {serial:>5d} {(' ' + atom):<4s} {residue_name:<3s} "
                         f"{chain}{index + 1:>4d}    "
                         f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00" + " " * 10
                         + f"{atom[0]:>2s}")
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def outputs_for(track_dir, example):
    """RFD3 names a design `<prefix><example>_<batch>_model_<i>.cif.gz`."""
    return sorted(track_dir.glob(f"*{example}_*_model_*.cif.gz"))


def collect(view, designs, out, tracks):
    predictions = out / "predictions"
    structures = predictions / "structures"
    structures.mkdir(parents=True, exist_ok=True)
    summary = {}

    for track in tracks:
        source = view / f"{track}.jsonl"
        track_dir = designs / track
        if not source.is_file() or not track_dir.is_dir():
            continue
        rows, kept, dropped = [], 0, {}
        for line in source.read_text().splitlines():
            if not line.strip():
                continue
            request = json.loads(line)
            example, length = request["example_id"], int(request["design_length"])
            context = dict(np.load(view / request["context_path"], allow_pickle=False))
            records = []
            for index, path in enumerate(outputs_for(track_dir, example)):
                if index >= request["num_designs"]:
                    break
                try:
                    residues = design_chain(read_cif(path), length)
                    coordinates, mask, names = to_atom14(residues)
                    if not mask[:, BACKBONE].all():
                        raise ValueError("a residue is missing backbone atoms")
                    record = {"design_index": index,
                              "ranking_score": 1.0 - 0.01 * index,
                              "structure_path": f"{example}-{index}.pdb"}
                    if "motif_coords" in context:
                        metadata = json.loads(path.with_suffix("").with_suffix(".json")
                                              .read_text())
                        motif_map = recover_motif_map(metadata, names, context,
                                                      coordinates, mask)
                        if motif_map is None:
                            raise ValueError("could not locate the motif in the design")
                        record["motif_map"] = motif_map
                except (OSError, ValueError, KeyError) as error:
                    dropped[str(error)[:90]] = dropped.get(str(error)[:90], 0) + 1
                    continue
                write_pdb(structures / record["structure_path"],
                          coordinates, mask, names)
                records.append(record)
                kept += 1
            rows.append({"example_id": example, "designs": records})
        (predictions / f"{track}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows))
        summary[track] = {"targets": len(rows), "designs": kept, "dropped": dropped}
        print(f"{track:15s} {len(rows):3d} targets, {kept:4d} designs"
              + (f", dropped {dropped}" if dropped else ""))
    (out / "collect.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", required=True, type=Path)
    parser.add_argument("--designs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tracks", nargs="*",
                        default=["unconditional", "sm_binder", "ppi_binder", "enzyme"])
    arguments = parser.parse_args()
    collect(arguments.view, arguments.designs, arguments.out, arguments.tracks)


if __name__ == "__main__":
    main()
