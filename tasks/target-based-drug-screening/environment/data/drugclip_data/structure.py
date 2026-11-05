from __future__ import annotations

import gzip
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1
from rdkit import Chem
from rdkit.Chem import AllChem

AA3 = {key.upper(): value.upper() for key, value in protein_letters_3to1.items()}
AA3.update({"MSE": "M", "SEC": "U", "PYL": "O"})


def element_from_pdb_line(line: str) -> str:
    element = line[76:78].strip().title()
    if element:
        return element
    name = re.sub(r"^[0-9]+", "", line[12:16].strip())
    return (name[:2].title() if len(name) > 1 and name[:2].title() in {"Cl", "Br"}
            else name[:1].upper())


def parse_pdb_atoms(path: Path, chains: set[str] | None = None,
                    record_types: tuple[str, ...] = ("ATOM  ", "HETATM")) -> list[dict]:
    atoms = []
    with path.open(errors="replace") as handle:
        for line in handle:
            if not line.startswith(record_types) or len(line) < 54:
                continue
            chain = line[21].strip() or "_"
            if chains is not None and chain not in chains:
                continue
            alt = line[16].strip()
            if alt not in {"", "A", "1"}:
                continue
            try:
                coord = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                residue_number = int(line[22:26])
            except ValueError:
                continue
            atoms.append({
                "element": element_from_pdb_line(line), "coord": coord,
                "chain": chain, "resname": line[17:20].strip().upper(),
                "residue_number": residue_number, "insertion": line[26].strip(),
                "atom_name": line[12:16].strip(),
            })
    return atoms


def pdb_sequences(path: Path, chains: set[str] | None = None) -> dict[str, str]:
    residues: dict[str, OrderedDict] = {}
    for atom in parse_pdb_atoms(path, chains, ("ATOM  ",)):
        aa = AA3.get(atom["resname"])
        if aa is None:
            continue
        key = (atom["residue_number"], atom["insertion"])
        residues.setdefault(atom["chain"], OrderedDict()).setdefault(key, aa)
    return {chain: "".join(values.values()) for chain, values in residues.items() if values}


def parse_mol2_protein(path: Path) -> tuple[dict[str, str], list[dict]]:
    atoms, residues = [], OrderedDict()
    in_atoms = False
    with path.open(errors="replace") as handle:
        for line in handle:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atoms = True
                continue
            if line.startswith("@<TRIPOS>"):
                in_atoms = False
            if not in_atoms or not line.strip():
                continue
            fields = line.split()
            if len(fields) < 8:
                continue
            try:
                coord = [float(fields[2]), float(fields[3]), float(fields[4])]
                subst_id = int(fields[6])
            except ValueError:
                continue
            subst_name = fields[7]
            match = re.match(r"([A-Za-z]{3})(-?\d+)?", subst_name)
            resname = match.group(1).upper() if match else "UNK"
            element = re.match(r"[A-Za-z]+", fields[5]).group(0).split(".")[0].title()
            residues.setdefault(subst_id, AA3.get(resname, "X"))
            atoms.append({"element": element, "coord": coord, "chain": "_",
                          "resname": resname, "residue_number": subst_id,
                          "insertion": "", "atom_name": fields[1]})
    sequence = "".join(residues.values()).replace("X", "")
    return ({"_": sequence} if sequence else {}), atoms


def read_fasta(path: Path) -> dict[str, str]:
    opener = gzip.open if path.suffix == ".gz" else open
    records, header, sequence = {}, None, []
    with opener(path, "rt", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith(">"):
                if header is not None:
                    records[header] = "".join(sequence).upper()
                header, sequence = line[1:].strip(), []
            elif header is not None:
                sequence.append(line)
    if header is not None:
        records[header] = "".join(sequence).upper()
    return records


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n")
            for offset in range(0, len(sequence), 80):
                handle.write(sequence[offset:offset + 80] + "\n")


def load_ligand(path: Path):
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".mol2"):
        mol = Chem.MolFromMol2File(str(path), sanitize=True, removeHs=False)
    elif suffixes.endswith(".sdf") or suffixes.endswith(".mol"):
        supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
        mol = next((item for item in supplier if item is not None), None)
    else:
        mol = Chem.MolFromPDBFile(str(path), sanitize=True, removeHs=False,
                                  proximityBonding=True)
    if mol is None:
        raise ValueError(f"RDKit could not parse ligand: {path}")
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(fragments) > 1:
        mol = max(fragments, key=lambda item: item.GetNumHeavyAtoms())
    if mol.GetNumHeavyAtoms() < 2:
        raise ValueError("isolated ion or one-heavy-atom ligand")
    Chem.SanitizeMol(mol)
    return mol


def ligand_identity(mol) -> tuple[str, str]:
    heavy = Chem.RemoveHs(mol)
    fragments = Chem.GetMolFrags(heavy, asMols=True, sanitizeFrags=True)
    if fragments:
        heavy = max(fragments, key=lambda item: item.GetNumHeavyAtoms())
    smiles = Chem.MolToSmiles(heavy, canonical=True, isomericSmiles=True)
    inchikey = Chem.MolToInchiKey(heavy)
    if not smiles or not inchikey:
        raise ValueError("canonical identity generation failed")
    return smiles, inchikey


def molecule_atoms(mol) -> list[str]:
    return [atom.GetSymbol() for atom in Chem.RemoveHs(mol).GetAtoms()]


def crystal_coordinates(mol) -> np.ndarray:
    heavy = Chem.RemoveHs(mol)
    if not heavy.GetNumConformers():
        raise ValueError("ligand has no crystallographic coordinates")
    return heavy.GetConformer().GetPositions().astype(np.float32)


def generate_conformers(smiles: str, max_conformers: int, seed: int,
                        prune_rms: float) -> tuple[list[str], list[np.ndarray], str]:
    """Closest compact variant of DrugCLIP single_conf_gen (Embed + MMFF)."""
    base = Chem.MolFromSmiles(smiles, sanitize=True)
    if base is None:
        raise ValueError("invalid SMILES")
    mol = Chem.AddHs(base)
    # Force-field availability depends only on molecular topology.  Check it
    # before ETKDG so unsupported metal/cofactor structures fail with the same
    # result without first paying to embed up to ``max_conformers`` geometries.
    has_mmff = AllChem.MMFFHasAllMoleculeParams(mol)
    has_uff = False if has_mmff else AllChem.UFFHasAllMoleculeParams(mol)
    if not has_mmff and not has_uff:
        raise ValueError("neither MMFF nor UFF parameters available")
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed & 0x7FFFFFFF)
    params.numThreads = 1
    params.pruneRmsThresh = float(prune_rms)
    params.clearConfs = True
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=max_conformers, params=params))
    if not ids:
        raise ValueError("RDKit embedding generated no conformer")
    method = "MMFF94s"
    if has_mmff:
        for conformer_id in ids:
            AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s",
                                         confId=conformer_id, maxIters=500)
    else:
        method = "UFF"
        for conformer_id in ids:
            AllChem.UFFOptimizeMolecule(mol, confId=conformer_id, maxIters=500)
    heavy = Chem.RemoveHs(mol)
    atoms = [atom.GetSymbol() for atom in heavy.GetAtoms()]
    coordinates = [conf.GetPositions().astype(np.float32) for conf in heavy.GetConformers()]
    return atoms, coordinates, method


def select_pocket(protein_atoms: list[dict], ligand_coordinates: np.ndarray,
                  cutoff: float) -> tuple[list[str], np.ndarray, int]:
    if not protein_atoms or ligand_coordinates.size == 0:
        raise ValueError("empty protein or ligand")
    protein_coordinates = np.asarray([atom["coord"] for atom in protein_atoms], dtype=np.float32)
    distances = np.linalg.norm(
        protein_coordinates[:, None, :] - ligand_coordinates[None, :, :], axis=2
    )
    close_atom = distances.min(axis=1) <= cutoff
    residue_keys = {
        (atom["chain"], atom["residue_number"], atom["insertion"])
        for atom, close in zip(protein_atoms, close_atom) if close
    }
    chosen = [atom for atom in protein_atoms
              if (atom["chain"], atom["residue_number"], atom["insertion"]) in residue_keys
              and atom["element"] not in {"H", "D"}]
    if not chosen:
        raise ValueError("empty pocket")
    return ([atom["element"] for atom in chosen],
            np.asarray([atom["coord"] for atom in chosen], dtype=np.float32),
            len(residue_keys))
