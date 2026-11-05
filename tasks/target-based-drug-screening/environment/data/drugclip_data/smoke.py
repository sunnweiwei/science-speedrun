from __future__ import annotations

import json
import os
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from .io import write_json


def _protein_pdb(sequence: list[tuple[str, str]], chain: str = "A", offset: float = 0.0) -> str:
    lines, serial = [], 1
    for index, (resname, _) in enumerate(sequence, 1):
        x = offset + index * 1.5
        for atom, element, delta in [("N", "N", 0.0), ("CA", "C", .4), ("C", "C", .8), ("O", "O", 1.1)]:
            lines.append(f"ATOM  {serial:5d} {atom:^4s} {resname:>3s} {chain}{index:4d}    "
                         f"{x + delta:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00          {element:>2s}")
            serial += 1
    return "\n".join(lines) + "\nEND\n"


def _profsa_pdb() -> str:
    receptor = _protein_pdb([("GLY", "G"), ("ALA", "A"), ("SER", "S")], "R")
    ligand = _protein_pdb([("LEU", "L"), ("VAL", "V")], "L", 1.0)
    return receptor.replace("END\n", "") + ligand.replace("END\n", "") + "END\n"


def _protein_mol2() -> str:
    return """@<TRIPOS>MOLECULE
smoke
 8 0 2 0 0
BIOPOLYMER
NO_CHARGES
@<TRIPOS>ATOM
1 N 0.0 0.0 0.0 N.am 1 ALA1 0.0
2 CA 0.4 0.0 0.0 C.3 1 ALA1 0.0
3 C 0.8 0.0 0.0 C.2 1 ALA1 0.0
4 O 1.1 0.0 0.0 O.2 1 ALA1 0.0
5 N 1.5 0.0 0.0 N.am 2 GLY2 0.0
6 CA 1.9 0.0 0.0 C.3 2 GLY2 0.0
7 C 2.3 0.0 0.0 C.2 2 GLY2 0.0
8 O 2.6 0.0 0.0 O.2 2 GLY2 0.0
@<TRIPOS>BOND
"""


def _ligand(path: Path, smiles: str, seed: int) -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.UFFOptimizeMolecule(mol)
    Chem.MolToPDBFile(mol, str(path))


def create_smoke_inputs(data_root: Path) -> None:
    test = data_root / "test"
    dude = test / "dude" / "smoke_dude"
    lit = test / "lit_pcba" / "SMOKE_PCBA"
    dude.mkdir(parents=True, exist_ok=True)
    lit.mkdir(parents=True, exist_ok=True)
    (dude / "receptor.pdb").write_text(_protein_pdb([("TYR", "Y"), ("TYR", "Y"), ("TYR", "Y")]))
    (dude / "actives_final.ism").write_text("c1ccccc1 active1\n")
    (dude / "decoys_final.ism").write_text(
        "C1CCCCC1 decoy1\n"
        "c1ccc(c(c1)C=N(=Cc2ccccc2O)N(CCO)C3=[NH+]CCN3)O released_bad_valence\n"
    )
    (lit / "1xyz_protein.mol2").write_text(_protein_mol2())
    (lit / "actives.smi").write_text("CC(=O)Oc1ccccc1C(=O)O active2\n")
    (lit / "inactives.smi").write_text("N#N inactive1\n")
    write_json(test / "manifest.json", {
        "schema_version": "drugclip-test-manifest/v1", "immutable": True,
        "smoke": True, "warning": "SYNTHETIC FIXTURE; NOT RELEASED BENCHMARK DATA",
        "archives": [],
    })

    profsa = data_root / "raw" / "profsadb"
    complexes = profsa / "complexes"
    complexes.mkdir(parents=True, exist_ok=True)
    for index, entry in enumerate(["1abc", "2abc", "3abc"]):
        (complexes / f"{entry}_A_sample{index}.pdb").write_text(_profsa_pdb())
    (profsa / "source_sequences.fasta").write_text(
        ">1abc_A\nGASLVGASLV\n>2abc_A\nGASLVGASLA\n>3abc_A\nGASLVGASLI\n")

    raw = data_root / "raw" / "biolip2"
    receptors, ligands, sequences = raw / "receptors", raw / "ligands", raw / "sequences"
    downloads = raw / "downloads"
    for directory in (receptors, ligands, sequences, downloads): directory.mkdir(parents=True, exist_ok=True)
    smiles = ["CCO", "CCN", "CCC", "COC"]
    terminal_residues = [("LEU", "L"), ("VAL", "V"), ("TYR", "Y"), ("THR", "T")]
    annotations = ["Assembly_ID,Ligand_file,Ligand_ID,Binding_residue"]
    for index, smi in enumerate(smiles, 1):
        assembly, ligand_file = f"{index}smk_1", f"{index}smk_1_LIG_Z"
        terminal = terminal_residues[index - 1]
        (receptors / f"{assembly}.pdb").write_text(
            _protein_pdb([("GLY", "G"), ("ALA", "A"), ("SER", "S"), terminal]))
        _ligand(ligands / f"{ligand_file}.pdb", smi, 100 + index)
        (sequences / f"{assembly}.fasta").write_text(f">{assembly}\nGAS{terminal[1]}\n")
        annotations.append(f"{assembly},{ligand_file},L{index:02d},A:G1 A2 S3 L4")
    (downloads / "PL_annotation.csv").write_text("\n".join(annotations) + "\n")
    (downloads / "artlist").write_text("HOH\nSO4\n")

    fake_bin = data_root / ".smoke-bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    fake = fake_bin / "mmseqs"
    fake.write_text("""#!/usr/bin/env python3
import sys
from pathlib import Path
args=sys.argv
if args[1]=='easy-search':
 Path(args[4]).write_text('')
elif args[1]=='easy-cluster':
 fasta=Path(args[2]).read_text().splitlines()
 names=[line[1:] for line in fasta if line.startswith('>')]
 Path(args[3]+'_cluster.tsv').write_text(''.join(f'{name}\\t{name}\\n' for name in names))
else:
 raise SystemExit(2)
""")
    fake.chmod(0o755)
    os.environ["PATH"] = f"{fake_bin}:{os.environ['PATH']}"
    write_json(data_root / "SMOKE_DATA.json", {"synthetic": True, "for_training": False})
