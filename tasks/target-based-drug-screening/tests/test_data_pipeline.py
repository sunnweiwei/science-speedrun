from __future__ import annotations

import shutil
import tempfile
import unittest
import pickle
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from drugclip_data.cli import (clear_materialization_root, run_all,
                               validate_materialization_root)
from drugclip_data.biolip import _apply_manual_exclusions
from drugclip_data.structure import generate_conformers, select_pocket
from drugclip_data.validate import validate_dataset


class DataPipelineTest(unittest.TestCase):
    def test_pinned_stalled_tail_exclusion_is_cached_and_reusable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "canonical" / "cache" / "biolip2"
            cache.mkdir(parents=True)
            complex_id = "9wat_1__9wat_1_HEM_B1"
            rows = {complex_id: {"Assembly_ID": "9wat_1",
                                 "Ligand_file": "9wat_1_HEM_B1", "Ligand_ID": "HEM"}}
            self.assertEqual(_apply_manual_exclusions(rows, cache, root), 1)
            with (cache / f"{complex_id}.pkl").open("rb") as handle:
                record = pickle.load(handle)
            self.assertEqual(record["complex_id"], complex_id)
            self.assertEqual(record["exclusion_reason"], "manual_stalled_tail_exclusion")
            self.assertIn("heme", record["exclusion_detail"])
            self.assertEqual(_apply_manual_exclusions(rows, cache, root), 0)

    def test_container_data_mount_is_allowed(self) -> None:
        validate_materialization_root(Path("/data"))
        with self.assertRaises(SystemExit):
            validate_materialization_root(Path("/tmp"))

    def test_overwrite_preserves_materialization_mount_point(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "mounted-data"
            nested = root / "old" / "nested"
            nested.mkdir(parents=True)
            (nested / "cached.bin").write_bytes(b"old")
            (root / ".complete.json").write_text("old\n")
            root_inode = root.stat().st_ino

            clear_materialization_root(root)

            self.assertTrue(root.is_dir())
            self.assertEqual(root.stat().st_ino, root_inode)
            self.assertEqual(list(root.iterdir()), [])

    def test_smoke_materialization_and_fail_closed_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            run_all(root, smoke=True, overwrite=False, biolip_lock=None,
                    profsa_source_fasta=None)
            receipt = validate_dataset(root, smoke=True)
            self.assertEqual(receipt["status"], "PASS")
            test_ligands = pd.read_parquet(root / "test" / "test_ligands.parquet")
            fallback = test_ligands[test_ligands.parse_status.eq("unsanitized_source_valence")]
            self.assertEqual(len(fallback), 1)
            self.assertEqual(int(fallback.iloc[0].label), 0)
            self.assertTrue(fallback.iloc[0].inchikey)
            self.assertEqual(len((root / "splits" / "profsa_pretrain_500k.txt").read_text().splitlines()), 2)
            frame_path = root / "canonical" / "biolip2.parquet"
            frame = pd.read_parquet(frame_path)
            train_index = frame.index[frame.split.eq("train")][0]
            frame.at[train_index, "max_test_sequence_identity"] = 30.0001
            frame.to_parquet(frame_path, index=False)
            with self.assertRaises(SystemExit):
                validate_dataset(root, smoke=True)

    def test_pocket_selects_whole_residue(self):
        atoms = [
            {"element": "C", "coord": [0., 0., 0.], "chain": "A", "residue_number": 1, "insertion": ""},
            {"element": "O", "coord": [20., 0., 0.], "chain": "A", "residue_number": 1, "insertion": ""},
            {"element": "N", "coord": [30., 0., 0.], "chain": "A", "residue_number": 2, "insertion": ""},
        ]
        elements, coordinates, residues = select_pocket(atoms, __import__("numpy").array([[5.9, 0., 0.]]), 6.0)
        self.assertEqual(elements, ["C", "O"])
        self.assertEqual(coordinates.shape, (2, 3))
        self.assertEqual(residues, 1)

    def test_conformer_generation_is_deterministic(self):
        first = generate_conformers("CCCO", 3, 12345, .5)
        second = generate_conformers("CCCO", 3, 12345, .5)
        self.assertEqual(first[0], second[0])
        self.assertTrue(__import__("numpy").array_equal(first[1], second[1]))

    def test_unsupported_force_field_fails_before_embedding(self):
        with (patch("drugclip_data.structure.AllChem.MMFFHasAllMoleculeParams",
                    return_value=False),
              patch("drugclip_data.structure.AllChem.UFFHasAllMoleculeParams",
                    return_value=False),
              patch("drugclip_data.structure.AllChem.EmbedMultipleConfs") as embed):
            with self.assertRaisesRegex(ValueError,
                                       "neither MMFF nor UFF parameters available"):
                generate_conformers("CCCO", 10, 12345, .5)
            embed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
