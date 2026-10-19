import unittest
import numpy as np
from run_continuous_ffn import validate_geometry


class FFNGeometryTests(unittest.TestCase):
    def test_default_halo_explicit_roi_and_physical_contract(self):
        raw=np.zeros((70,80,90),np.uint8);valid=np.ones(70,bool)
        meta=dict(schema='pathfinder-continuous-image/v1',axes='zyx',coordinate_convention='voxel_outer_corner',
            ground_truth_included=False,voxel_size_nm=[16]*3,shape=list(raw.shape),origin_nm=[-100,200,500],image_valid={})
        self.assertEqual(validate_geometry(meta,raw,valid),[16,54,16,64,16,74])
        roi=[20,40,20,60,25,70]
        self.assertEqual(validate_geometry(meta,raw,valid,roi),roi)
        for bad in ([0,54,16,64,16,74],[16,55,16,64,16,74],[20,20,20,60,25,70]):
            with self.assertRaises(ValueError):validate_geometry(meta,raw,valid,bad)
        meta['voxel_size_nm']=[8]*3
        with self.assertRaises(ValueError):validate_geometry(meta,raw,valid)

if __name__=='__main__':unittest.main()
