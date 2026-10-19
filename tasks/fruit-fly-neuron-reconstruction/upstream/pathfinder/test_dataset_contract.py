"""Dataset replacement checks: coordinates, leakage, integrity and label centers."""
import copy
import json
from pathlib import Path
from test_fs_support import TemporaryDirectory
import unittest
import numpy as np
from dataset_contract import ContractError, sha256, validate_manifest


class DatasetContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = dict(schema='pathfinder-dataset/v1', dataset_id='synthetic-other-em', axes='zyx',
            coordinate_convention='voxel_outer_corner', voxel_size_nm=[32, 8, 8],
            native_voxel_size_nm=[32, 8, 8], source_receipt_sha256='a' * 64,
            paper_comparable=False, split_kind='spatial', splits={})
        for i, split in enumerate(('train', 'validation', 'test')):
            folder = self.root / split
            folder.mkdir()
            for key, array in dict(raw=np.zeros((4, 4, 4), np.uint8), labels=np.ones((4, 4, 4), np.uint16),
                                   valid=np.ones((4, 4, 4), bool), centers=np.array([[1, 1, 1]], np.int16)).items():
                np.save(folder / (key + '.npy'), array)
            self.manifest['splits'][split] = [dict(name=split, folder=split, shape=[4, 4, 4],
                origin_nm=[-128 + i * 128, 100, 200], dense_core_zyx=[[0, 4], [0, 4], [0, 4]],
                raw_source_sha256='b' * 64, labels_source_sha256='c' * 64,
                files={p.name: sha256(p) for p in folder.iterdir()})]

    def validate(self, **kwargs):
        return validate_manifest(self.manifest, base_dir=self.root, **kwargs)

    def test_different_spacing_negative_origin_touching_regions_and_hashes(self):
        result = self.validate(verify_hashes=True)
        self.assertTrue(result['hashes_verified'])
        self.assertFalse(result['paper_protocol_verified'])

    def test_development_manifest_excludes_test_and_still_checks_overlap(self):
        self.manifest['visibility']='development'
        del self.manifest['splits']['test']
        self.assertEqual(self.validate(verify_hashes=True)['counts'],{'train':1,'validation':1})
        self.manifest['splits']['validation'][0]['origin_nm'][0]-=1
        with self.assertRaisesRegex(ContractError,'leakage'):self.validate()

    def test_rejects_physical_split_overlap(self):
        self.manifest['splits']['validation'][0]['origin_nm'][0] -= 1
        with self.assertRaisesRegex(ContractError, 'leakage'):
            self.validate()

    def test_detects_mutation_even_when_header_unchanged(self):
        np.save(self.root / 'train' / 'raw.npy', np.ones((4, 4, 4), np.uint8))
        with self.assertRaisesRegex(ContractError, 'SHA256 mismatch'):
            self.validate(verify_hashes=True)

    def test_rejects_unlabeled_center(self):
        np.save(self.root / 'train' / 'labels.npy', np.zeros((4, 4, 4), np.uint16))
        with self.assertRaisesRegex(ContractError, 'positive instance'):
            self.validate(check_files=True)

    def test_requires_explicit_coordinates_and_resampling(self):
        del self.manifest['coordinate_convention']
        with self.assertRaisesRegex(ContractError, 'coordinate_convention'):
            self.validate()
        self.manifest['coordinate_convention'] = 'voxel_outer_corner'
        self.manifest['voxel_size_nm'] = [16, 16, 16]
        with self.assertRaisesRegex(ContractError, 'interpolation'):
            self.validate()

    def test_rejects_claimed_paper_parity_and_nonfinite_resolution(self):
        self.manifest['paper_comparable'] = True
        with self.assertRaisesRegex(ContractError, 'paper parity'):
            self.validate()
        self.manifest['paper_comparable'] = False
        self.manifest['voxel_size_nm'][0] = float('nan')
        with self.assertRaisesRegex(ContractError, 'non-finite'):
            self.validate()



class CommittedManifestTest(unittest.TestCase):
    def test_current_committed_manifest_is_compatible(self):
        manifest = json.loads((Path(__file__).parent / 'receipts' / 'axonem-prepared-manifest.json').read_text())
        self.assertEqual(validate_manifest(manifest)['counts'], {'train': 4, 'validation': 1, 'test': 1})


if __name__ == '__main__':
    unittest.main()
