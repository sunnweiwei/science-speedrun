import copy
import json
from pathlib import Path
from test_fs_support import TemporaryDirectory
import unittest
import numpy as np
from import_continuous import import_image,validate
from common import sha256


class ImportTests(unittest.TestCase):
    def fixture(self):
        raw=np.arange(120,dtype=np.uint8).reshape(4,5,6);valid=np.array([True,False,True,True])
        config=dict(schema='pathfinder-continuous-source/v1',axes='zyx',coordinate_convention='voxel_outer_corner',
            shape=[4,5,6],origin_nm=[-16,32,48],voxel_size_nm=[16]*3,native_voxel_size_nm=[40,8,8],
            dataset_id='test-only',source='local regression fixture',source_revision='v1',
            conversion='explicit test intensities; no biological claims',image_valid_scope='known missing plane',ground_truth_included=False)
        return config,raw,valid

    def test_roundtrip_preserves_pixels_coordinates_and_missing_planes(self):
        config,raw,valid=self.fixture()
        with TemporaryDirectory() as temp:
            root=Path(temp);np.save(root/'raw.npy',raw);np.save(root/'valid.npy',valid)
            config.update(raw=dict(path='raw.npy',sha256=sha256(root/'raw.npy')),
                          image_valid_z=dict(path='valid.npy',sha256=sha256(root/'valid.npy')))
            source=root/'source.json';source.write_text(json.dumps(config))
            result=import_image(source,root,root/'imported')
            np.testing.assert_array_equal(np.load(root/'imported/raw.npy'),raw)
            np.testing.assert_array_equal(np.load(root/'imported/image_valid_z.npy'),valid)
            self.assertEqual(result['origin_nm'],[-16,32,48]);self.assertEqual(result['image_valid']['invalid_output_planes'],[1])
            self.assertFalse(result['paper_comparable']);self.assertFalse(result['full_context_image_validity_verified'])
            with self.assertRaises(FileExistsError):import_image(source,root,root/'imported')
            config['raw']['sha256']='bad';source.write_text(json.dumps(config))
            with self.assertRaises(ValueError):import_image(source,root,root/'bad')
            self.assertFalse((root/'bad').exists())

    def test_rejects_ambiguous_axes_mask_and_metadata(self):
        config,raw,valid=self.fixture()
        for key,value in [('axes','xyz'),('origin_nm',[0,0]),('voxel_size_nm',[0,16,16]),('conversion',''),('ground_truth_included',True)]:
            wrong=copy.deepcopy(config);wrong[key]=value
            with self.assertRaises(ValueError):validate(wrong,raw,valid)
        with self.assertRaises(ValueError):validate(config,raw,valid.astype(np.uint8))
        config['voxel_size_nm']=[8,8,8]
        validate(config,raw,valid)  # An honest different grid is valid input, not a compatible16nm model recipe.

if __name__=='__main__':unittest.main()
