import unittest
import numpy as np
from prediction_core import core_slices
from evaluate_physical import assignments


class PredictionCoreTests(unittest.TestCase):
    def test_anisotropic_xyz_crop_retains_every_assignment_and_large_ids(self):
        pred=dict(axes='xyz',coordinate_convention='voxel_outer_corner',
            origin_nm=[100,200,300],voxel_size_nm=[9,9,20],shape=[20,20,10])
        ref=dict(axes='zyx',coordinate_convention='voxel_outer_corner',
            origin_nm=[320,216,116],voxel_size_nm=[16]*3,shape=[8]*3)
        a=np.arange(4000,dtype=np.uint64).reshape(20,20,10)+2**60
        slices,meta=core_slices(pred,ref)
        coords=np.stack(np.meshgrid(*[o+(np.arange(8)+.5)*16 for o in ref['origin_nm']],indexing='ij'),axis=-1).reshape(-1,3)
        np.testing.assert_array_equal(assignments(a,pred,ref,coords),assignments(a[slices],meta,ref,coords))
        self.assertEqual(meta['axes'],'xyz')
        self.assertEqual(meta['origin_nm'],[109,209,320])

    def test_zyx_order_and_noncoverage(self):
        grid=dict(axes='zyx',coordinate_convention='voxel_outer_corner',origin_nm=[0]*3,voxel_size_nm=[16]*3,shape=[20]*3)
        ref=dict(grid,scored_core_origin_nm=[32,48,64],scored_core_shape=[4]*3)
        slices,meta=core_slices(grid,ref)
        self.assertEqual(slices,(slice(2,6),slice(3,7),slice(4,8)))
        self.assertEqual(meta['origin_nm'],[32,48,64])
        with self.assertRaises(ValueError):core_slices(dict(grid,shape=[3]*3),ref)


if __name__=='__main__':unittest.main()
