import json
from pathlib import Path
from test_fs_support import TemporaryDirectory
import unittest
import numpy as np
from prediction_contract import seal,validate


class PredictionContractTests(unittest.TestCase):
    def test_relocation_expected_roi_and_mutation(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);out=root/'result';out.mkdir()
            np.save(out/'segmentation.npy',np.array([[[0,4_000_000_000]]],np.uint32))
            meta=dict(axes='zyx',coordinate_convention='voxel_outer_corner',shape=[1,1,2],origin_nm=[-8,16,32],voxel_size_nm=[16]*3)
            source=root/'expected.json';source.write_text(json.dumps(meta))
            seal(out/'segmentation.npy',source)
            moved=root/'relocated';out.rename(moved)
            self.assertEqual(validate(moved,meta)['geometry'],meta)
            wrong=dict(meta,origin_nm=[0,16,32])
            with self.assertRaises(ValueError):validate(moved,wrong)
            np.save(moved/'segmentation.npy',np.zeros((1,1,2),np.uint32))
            with self.assertRaises(ValueError):validate(moved,meta)

if __name__=='__main__':unittest.main()
