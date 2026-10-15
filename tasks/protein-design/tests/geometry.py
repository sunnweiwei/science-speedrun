"""All-atom backbone parsing, superposition, and geometric validity checks.

Structures use the atom14 layout (see residues.py): [L, 14, 3] coordinates with an
[L, 14] occupancy mask, backbone N/CA/C/O always in slots 0-3.

Every function here consumes output produced by an untrusted design system, so parsing
is strict and allocation is bounded before any array is built.
"""

import numpy as np

from residues import (ATOM14, ATOM14_WIDTH, IGNORED_ATOMS, SYMMETRIC_SWAPS,
                      atom_slot)

CA_SLOT = 1
MAX_PDB_BYTES = 32 * 1024 * 1024
MAX_DESIGN_LENGTH = 600

# Bond-length and clash thresholds lived here for `validate_geometry`, the task's own
# pre-fold sanity gate. That gate was deleted on 2026-09-26 - designability is the only
# metric now - and the thresholds went with it rather than sitting here advertising limits
# nothing enforces. See environment/evaluation/RESULTS.md for what the gate had cost.
DIVERSITY_RMSD = 2.0


def parse_allatom_pdb(path, expected_length):
    """Parse a single-chain all-atom PDB into (coords [L,14,3], mask [L,14], restypes)."""
    size = path.stat().st_size
    if path.is_symlink() or not path.is_file() or size == 0 or size > MAX_PDB_BYTES:
        raise ValueError("missing, empty, oversized, or symlinked structure")
    if not 0 < expected_length <= MAX_DESIGN_LENGTH:
        raise ValueError("unsupported design length")

    residues, order, chains = {}, [], set()
    with path.open("r", errors="strict") as stream:
        for line in stream:
            if not line.startswith("ATOM"):
                continue
            if len(line) < 54:
                raise ValueError("truncated ATOM record")
            atom = line[12:16].strip()
            if atom in IGNORED_ATOMS or atom.startswith("H") or line[76:78].strip() == "H":
                continue
            if line[16] not in (" ", "A"):
                continue
            residue_name = line[17:20].strip()
            if residue_name not in ATOM14:
                raise ValueError(f"non-standard residue: {residue_name}")
            chains.add(line[21])
            if len(chains) > 1:
                raise ValueError("design must contain exactly one chain")
            try:
                number = int(line[22:26])
                coordinate = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError as error:
                raise ValueError("unparsable residue number or coordinate") from error
            slot = atom_slot(residue_name, atom)
            if slot is None:
                raise ValueError(f"atom {atom} is not valid for residue {residue_name}")
            if number not in residues:
                if len(residues) >= MAX_DESIGN_LENGTH:
                    raise ValueError("design exceeds the maximum supported length")
                residues[number] = (residue_name, {})
                order.append(number)
            stored_name, atoms = residues[number]
            if stored_name != residue_name:
                raise ValueError(f"residue {number} has conflicting identities")
            if slot in atoms:
                raise ValueError(f"duplicate {atom} in residue {number}")
            atoms[slot] = coordinate

    if len(residues) != expected_length:
        raise ValueError(f"expected {expected_length} residues, found {len(residues)}")

    coordinates = np.zeros((expected_length, ATOM14_WIDTH, 3), dtype=np.float64)
    mask = np.zeros((expected_length, ATOM14_WIDTH), dtype=bool)
    restypes = []
    for index, number in enumerate(sorted(order)):
        residue_name, atoms = residues[number]
        restypes.append(residue_name)
        if not set((0, 1, 2, 3)) <= set(atoms):
            raise ValueError(f"residue {number} is missing backbone atoms")
        for slot, coordinate in atoms.items():
            coordinates[index, slot] = coordinate
            mask[index, slot] = True
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("non-finite coordinate")
    if np.abs(coordinates[mask]).max(initial=0.0) > 1e4:
        raise ValueError("coordinate outside a plausible envelope")
    return coordinates, mask, restypes


def write_allatom_pdb(path, coordinates, mask, restypes, chain="A"):
    """Write an atom14 structure in canonical PDB columns."""
    lines, serial = [], 1
    for index, residue_name in enumerate(restypes):
        for slot, atom in enumerate(ATOM14[residue_name]):
            if not mask[index, slot]:
                continue
            x, y, z = coordinates[index, slot]
            # Four-character atom field: one-character elements start at column 14.
            field = f" {atom:<3s}" if len(atom) < 4 else atom
            lines.append(
                f"ATOM  {serial:>5d} {field} {residue_name:<3s} {chain}{index + 1:>4d}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00" + " " * 10 + f"{atom[0]:>2s}")
            serial += 1
    path.write_text("\n".join(lines + ["TER", "END"]) + "\n")


def present_atoms(coordinates, mask):
    """Flatten to the [K, 3] set of atoms that actually exist."""
    return coordinates[mask]


def kabsch_rotation(mobile, target):
    correlation = mobile.T @ target
    u, _, vt = np.linalg.svd(correlation)
    reflection = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, reflection]) @ u.T


def superposed_rmsd(mobile, target):
    """RMSD after optimal rigid superposition of two equal-length point sets.

    Fewer than three points do not define a rigid frame, so superposition is not
    meaningful. One real benchmark target constrains only two motif atoms, so this is
    reached in practice: fall back to the centroid-aligned RMSD, which is the tightest
    well-defined answer, rather than raising.
    """
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("superposition needs matching [N, 3] point sets")
    if mobile.shape[0] == 0:
        raise ValueError("superposition needs at least one point")
    if mobile.shape[0] < 3:
        centred_mobile = mobile - mobile.mean(axis=0)
        centred_target = target - target.mean(axis=0)
        return float(np.sqrt(np.mean(np.sum(
            (centred_mobile - centred_target) ** 2, axis=1))))
    left = mobile - mobile.mean(axis=0)
    right = target - target.mean(axis=0)
    rotated = left @ kabsch_rotation(left, right).T
    return float(np.sqrt(np.mean(np.sum((rotated - right) ** 2, axis=1))))


def backbone_aligned_motif_rmsd(design, design_mask, reference, reference_mask, restypes):
    """RFdiffusion3's motif criterion: superimpose on motif N/CA/C, score all atoms.

    Aligning on every motif atom (including sidechains) lets a design trade sidechain
    error against backbone error and flatters a scaffold that placed the backbone
    badly. RFdiffusion3 aligns on the motif backbone only, then measures all-atom RMSD,
    which is what actually has to be right for catalysis.
    """
    shared = design_mask & reference_mask
    backbone = shared.copy()
    backbone[:, 4:] = False           # slots 0-3 are N, CA, C, O
    backbone[:, 3] = False            # align on N, CA, C as RFdiffusion3 does
    if backbone.sum() < 3:
        # Too few backbone atoms to define a frame; fall back to all shared atoms.
        backbone = shared
    if shared.sum() == 0:
        raise ValueError("motif has no shared atoms to compare")

    best = None
    for variant in _symmetry_variants(reference, shared, restypes):
        mobile = design[backbone]
        target = variant[backbone]
        offset_mobile = mobile.mean(axis=0)
        offset_target = target.mean(axis=0)
        rotation = kabsch_rotation(mobile - offset_mobile, target - offset_target)
        moved = (design[shared] - offset_mobile) @ rotation.T + offset_target
        value = float(np.sqrt(np.mean(np.sum((moved - variant[shared]) ** 2, axis=1))))
        best = value if best is None else min(best, value)
    return best


def _symmetry_variants(reference, shared, restypes):
    """Reference motif under every indistinguishable sidechain relabeling."""
    variants = [reference]
    for index, residue_name in enumerate(restypes):
        for left_name, right_name in SYMMETRIC_SWAPS.get(residue_name, ()):
            left = atom_slot(residue_name, left_name)
            right = atom_slot(residue_name, right_name)
            if left is None or right is None:
                continue
            if not (shared[index, left] and shared[index, right]):
                continue
            swapped = []
            for variant in variants:
                copy = variant.copy()
                copy[index, [left, right]] = copy[index, [right, left]]
                swapped.append(copy)
            variants = variants + swapped
            if len(variants) > 64:
                variants = variants[:64]
    return variants


def symmetric_motif_rmsd(design, design_mask, reference, reference_mask, restypes):
    """Motif RMSD over shared atoms, minimised across symmetric sidechain relabelings.

    ASP OD1/OD2, GLU OE1/OE2, ARG NH1/NH2 and the PHE/TYR ring pairs are physically
    indistinguishable, so a naive atom-by-atom RMSD penalises a correct design purely
    for the order the atoms were written in.
    """
    shared = design_mask & reference_mask
    if shared.sum() < 3:
        raise ValueError("motif has too few shared atoms to superimpose")

    variants = [reference]
    for index, residue_name in enumerate(restypes):
        for left_name, right_name in SYMMETRIC_SWAPS.get(residue_name, ()):
            left, right = atom_slot(residue_name, left_name), atom_slot(residue_name, right_name)
            if left is None or right is None:
                continue
            if not (shared[index, left] and shared[index, right]):
                continue
            swapped = []
            for variant in variants:
                copy = variant.copy()
                copy[index, [left, right]] = copy[index, [right, left]]
                swapped.append(copy)
            variants = variants + swapped
            if len(variants) > 64:      # bound the combinatorics on large motifs
                variants = variants[:64]
    return min(superposed_rmsd(design[shared], variant[shared]) for variant in variants)


def minimum_distance(left, right):
    if left.size == 0 or right.size == 0:
        return float("inf")
    return float(np.sqrt(((left[:, None, :] - right[None, :, :]) ** 2).sum(-1)).min())


def cluster_designs(traces):
    """Greedy within-target clustering by superposed CA-RMSD; returns the cluster count."""
    centers = []
    for alpha in traces:
        if not any(superposed_rmsd(alpha, other) < DIVERSITY_RMSD for other in centers):
            centers.append(alpha)
    return len(centers)
