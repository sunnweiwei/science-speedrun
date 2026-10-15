"""Derive design conditioning directly from PDB assembly coordinates.

Conditioning is computed from structure, not read from curated annotation files. This
follows RFdiffusion3's approach of building datasets from a frozen PDB snapshot: the
PDB is current, curated annotation sets (Catalytic Site Atlas, BioLiP) lag it by years,
and a benchmark that must hold out recent structures cannot depend on a stale index.

Requires `gemmi` on the host. Never imported by an agent or verifier image.
"""

import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline" / "rfd3-small"))
from residues import ATOM14, ATOM14_WIDTH, IGNORED_ATOMS, atom_slot  # noqa: E402

# CCD codes excluded by AlphaFold3 (Supp. Table 10), as used by RFdiffusion3 through
# atomworks.constants.AF3_EXCLUDED_LIGANDS. Crystallisation aids, buffers, and cryo-
# protectants; note that biologically meaningful metals such as ZN, MG and FE are
# deliberately NOT excluded.
AF3_EXCLUDED_LIGANDS = {
    "144", "15P", "1PE", "2F2", "2JC", "3HR", "3SY", "7N5", "7PE", "9JE", "AAE", "ABA",
    "ACE", "ACN", "ACT", "ACY", "AZI", "BAM", "BCN", "BCT", "BDN", "BEN", "BME", "BO3",
    "BTB", "BTC", "BU1", "C8E", "CAD", "CAQ", "CBM", "CCN", "CIT", "CL", "CLR", "CM",
    "CMO", "CO3", "CPT", "CXS", "D10", "DEP", "DIO", "DMS", "DN", "DOD", "DOX", "EDO",
    "EEE", "EGL", "EOH", "EOX", "EPE", "ETF", "FCY", "FJO", "FLC", "FMT", "FW5", "GOL",
    "GSH", "GTT", "GYF", "HED", "IHP", "IHS", "IMD", "IOD", "IPA", "IPH", "LDA", "MB3",
    "MEG", "MES", "MLA", "MLI", "MOH", "MPD", "MRD", "MSE", "MYR", "N", "NA", "NH2",
    "NH4", "NHE", "NO3", "O4B", "OHE", "OLA", "OLC", "OMB", "OME", "OXA", "P6G", "PE3",
    "PE4", "PEG", "PEO", "PEP", "PG0", "PG4", "PGE", "PGR", "PLM", "PO4", "POL", "POP",
    "PVO", "SAR", "SCN", "SEO", "SIN", "SO4", "SPD", "SPM", "SR", "STE", "STO", "STU",
    "TAR", "TBU", "TME", "TRS", "UNK", "UNL", "UNX", "UPL", "URE",
}

# Cofactors and substrate analogues that mark a genuine catalytic site. The enzyme
# track conditions on the reactive centre rather than on a curated catalytic triad,
# which is how RFdiffusion3 and DISCO actually pose enzyme design.
CATALYTIC_COFACTORS = {
    "NAD", "NAI", "NAP", "NDP", "FAD", "FMN", "FDA", "SAM", "SAH", "COA", "ACO",
    "TPP", "TDP", "PLP", "PMP", "B12", "COB", "HEM", "HEC", "HEA", "HEB", "SF4",
    "FES", "F3S", "MGD", "MOO", "PQQ", "BTN", "THG", "FOL", "GDP", "GTP", "ADP",
    "ATP", "AMP", "UDP", "UTP", "CTP", "GSH", "ASC", "MQ7", "UQ1",
}

EC_CLASS_HEADERS = ("OXIDOREDUCTASE", "TRANSFERASE", "HYDROLASE", "LYASE",
                    "ISOMERASE", "LIGASE", "TRANSLOCASE")

POCKET_RADIUS = 5.0
INTERFACE_RADIUS = 5.0
MIN_POCKET_RESIDUES = 4
MIN_INTERFACE_RESIDUES = 6
MIN_LIGAND_ATOMS = 6


class ProteinChain:
    """One polymer chain in atom14 form."""

    def __init__(self, chain_id, coordinates, mask, restypes):
        self.chain_id = chain_id
        self.coordinates = coordinates
        self.mask = mask
        self.restypes = restypes

    def __len__(self):
        return len(self.restypes)

    def atoms(self):
        return self.coordinates[self.mask]

    def residue_atoms(self, index):
        return self.coordinates[index][self.mask[index]]


def _is_amino_acid(name, tabulated):
    info = tabulated(name)
    return info is not None and info.is_amino_acid()


def parse_assembly(path, max_chains=64, max_residues=4000):
    """Parse an assembly mmCIF into protein chains and non-excluded ligands."""
    import gemmi

    structure = gemmi.read_structure(str(path))
    structure.remove_alternative_conformations()
    structure.remove_hydrogens()
    structure.remove_waters()
    structure.setup_entities()
    tabulated = gemmi.find_tabulated_residue

    model = structure[0]
    if len(model) > max_chains:
        raise ValueError(f"assembly has {len(model)} chains, above the {max_chains} cap")

    chains, ligands, total = [], [], 0
    for chain in model:
        residues = [r for r in chain if _is_amino_acid(r.name, tabulated) and r.name in ATOM14]
        if len(residues) >= 20:
            total += len(residues)
            if total > max_residues:
                raise ValueError("assembly exceeds the residue cap")
            length = len(residues)
            coordinates = np.zeros((length, ATOM14_WIDTH, 3))
            mask = np.zeros((length, ATOM14_WIDTH), dtype=bool)
            restypes = []
            for index, residue in enumerate(residues):
                restypes.append(residue.name)
                for atom in residue:
                    if atom.name in IGNORED_ATOMS or atom.element.name == "H":
                        continue
                    if not np.isfinite([atom.pos.x, atom.pos.y, atom.pos.z]).all():
                        continue
                    slot = atom_slot(residue.name, atom.name)
                    if slot is None:
                        continue
                    coordinates[index, slot] = [atom.pos.x, atom.pos.y, atom.pos.z]
                    mask[index, slot] = True
            # A residue missing backbone atoms cannot anchor a motif or a design.
            keep = mask[:, :4].all(axis=1)
            if keep.sum() >= 20:
                chains.append(ProteinChain(chain.name, coordinates[keep], mask[keep],
                                           [r for r, k in zip(restypes, keep) if k]))
        for residue in chain:
            if _is_amino_acid(residue.name, tabulated):
                continue
            info = tabulated(residue.name)
            if info is not None and (info.is_nucleic_acid() or info.is_water()):
                continue
            if residue.name in AF3_EXCLUDED_LIGANDS:
                continue
            kept = [a for a in residue if a.element.name != "H"
                    and np.isfinite([a.pos.x, a.pos.y, a.pos.z]).all()]
            positions = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in kept])
            if positions.shape[0] >= 1:
                ligands.append({"name": residue.name, "coordinates": positions,
                                "elements": [a.element.name.upper() for a in kept],
                                "charges": [float(getattr(a, "charge", 0) or 0)
                                            for a in kept],
                                "chain": chain.name})
    return chains, ligands


def contacting_residues(chain, points, radius):
    """Indices of chain residues with any heavy atom within `radius` of `points`."""
    if points.size == 0 or len(chain) == 0:
        return []
    hits = []
    for index in range(len(chain)):
        atoms = chain.residue_atoms(index)
        if atoms.size == 0:
            continue
        distances = np.sqrt(((atoms[:, None, :] - points[None, :, :]) ** 2).sum(-1))
        if distances.min() <= radius:
            hits.append(index)
    return hits


def find_pocket(chains, ligand, radius=POCKET_RADIUS):
    """The chain that best encloses a ligand, plus its contacting residue indices."""
    best = None
    for chain in chains:
        residues = contacting_residues(chain, ligand["coordinates"], radius)
        if len(residues) >= MIN_POCKET_RESIDUES and (best is None or len(residues) > len(best[1])):
            best = (chain, residues)
    return best


def find_interface(chains, radius=INTERFACE_RADIUS):
    """The largest inter-chain protein-protein interface in an assembly."""
    best = None
    for i, left in enumerate(chains):
        for right in chains[i + 1:]:
            # Cheap rejection before the full residue sweep.
            if np.linalg.norm(left.atoms().mean(0) - right.atoms().mean(0)) > 120.0:
                continue
            left_residues = contacting_residues(left, right.atoms(), radius)
            if len(left_residues) < MIN_INTERFACE_RESIDUES:
                continue
            right_residues = contacting_residues(right, left.atoms(), radius)
            if len(right_residues) < MIN_INTERFACE_RESIDUES:
                continue
            size = len(left_residues) + len(right_residues)
            if best is None or size > best[-1]:
                best = (left, left_residues, right, right_residues, size)
    return best


def classify(header):
    """True when the PDB HEADER classification names an EC enzyme class."""
    upper = (header or "").upper()
    return any(name in upper for name in EC_CLASS_HEADERS)


def catalytic_ligands(ligands):
    return [l for l in ligands if l["name"] in CATALYTIC_COFACTORS
            or l["coordinates"].shape[0] >= MIN_LIGAND_ATOMS]


def closest_residues(chain, candidates, points, limit):
    """The `limit` candidate residues whose atoms come nearest to `points`."""
    scored = []
    for index in candidates:
        atoms = chain.residue_atoms(index)
        if atoms.size == 0:
            continue
        distance = np.sqrt(((atoms[:, None, :] - points[None, :, :]) ** 2).sum(-1)).min()
        scored.append((distance, index))
    scored.sort()
    return sorted(index for _, index in scored[:limit])
