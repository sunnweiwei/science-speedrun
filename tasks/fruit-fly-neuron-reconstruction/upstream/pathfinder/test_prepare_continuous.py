import unittest
import numpy as np
from scipy.ndimage import map_coordinates
from prepare_continuous import geometry, plane_is_filler, xy_average, interpolate_planes


class ContinuousPreparationTest(unittest.TestCase):
    def test_geometry_default_and_bounds(self):
        start,shape=geometry([3000,4384,4384],24000)
        np.testing.assert_array_equal(start,[95,1060,1060]);self.assertEqual(shape,(1500,)*3)
        for origin,size in [([3000,4384,4384],30000),([3001,4384,4384],24000),([3000,4384,4384],24001)]:
            with self.assertRaises(ValueError):geometry(origin,size)

    def test_interpolation_matches_training_voxel_center_rule(self):
        rng=np.random.default_rng(42)
        raw=rng.integers(0,256,size=(4,12,12),dtype=np.uint8)
        yy,xx=np.meshgrid((np.arange(4)+.5)*2-.5+2,(np.arange(4)+.5)*2-.5+2,indexing='ij')
        for weight in [.1,.3,.5,.7,.9]:
            expected=map_coordinates(raw,np.array([np.full(yy.shape,1+weight),yy,xx]),order=1,mode='nearest',prefilter=False)
            actual,valid=interpolate_planes(xy_average(raw[1],[2,2],[4,4]),xy_average(raw[2],[2,2],[4,4]),weight,True,True)
            np.testing.assert_array_equal(actual,expected);self.assertTrue(valid)

    def test_large_global_z_has_only_single_gray_level_rounding_difference(self):
        raw=np.random.default_rng(22).integers(0,256,(4,128,128),dtype=np.uint8)
        yy,xx=np.meshgrid((np.arange(64)+.5)*2-.5,(np.arange(64)+.5)*2-.5,indexing='ij')
        for output_z in [0,100,333,700,1499]:
            native_z=95+(output_z+.5)*16/40-.5
            weight=native_z-np.floor(native_z)
            expected=map_coordinates(raw,np.array([np.full(yy.shape,1+weight),yy,xx]),order=1,mode='nearest',prefilter=False)
            actual,_=interpolate_planes(xy_average(raw[1],[0,0],[64,64]),xy_average(raw[2],[0,0],[64,64]),weight,True,True)
            self.assertLessEqual(np.abs(actual.astype(int)-expected).max(),1)

    def test_filler_preserved_but_validity_propagates(self):
        filler=np.full((8,8),128,np.uint8);filler[0]=0
        self.assertTrue(plane_is_filler(filler))
        real=filler.copy();real[1,1]=129
        self.assertFalse(plane_is_filler(real))
        output,valid=interpolate_planes(np.full((4,4),128,np.float32),np.full((4,4),200,np.float32),.5,False,True)
        self.assertFalse(valid);self.assertTrue((output==164).all())
        _,valid=interpolate_planes(np.zeros((4,4)),np.ones((4,4)),0,True,False)
        self.assertTrue(valid)


if __name__=='__main__':unittest.main()
